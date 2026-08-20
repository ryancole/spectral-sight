"""Watch the whole pipeline run: detect, identify, track.

Live, against a window -- this is the real-time path:

    # read the kilrogg receiver as it plays
    python tools/watch.py --window kilrogg

    # ...and keep the timeline while you watch
    python tools/watch.py --window kilrogg --export session.jsonl

    # ...or stream frame envelopes to another program as they happen
    python tools/watch.py --window kilrogg --quiet --export - | your-tool

Or offline, against a recorded clip -- the development path:

    # play a clip in a window
    python tools/watch.py --input "data/my clip.mp4"

    # every frame instead of 10 Hz, and write an annotated video out
    python tools/watch.py --input clip.mp4 --stride 1 --save out.mp4

    # print the tracked state instead of showing it
    python tools/watch.py --input clip.mp4 --quiet

    # extract the whole clip to a timeline, as fast as it will go
    python tools/watch.py --input clip.mp4 --quiet --export clip.jsonl

The two differ in how they fall behind, and only in that. A clip waits; a window
does not. Frames arrive from a window whether or not the pipeline is ready for
them, so the ones it cannot keep up with are dropped on arrival rather than
queued -- see `Mailbox`. The run reports the drop count at the end, which is the
number to watch if the overlay looks like it is lagging the game.

Keys: Q or ESC to quit, SPACE to pause, any key to step while paused.

Champions currently visible are drawn solid; champions in fog are drawn hollow
at their last known position with the time since they were seen, which is the
readout that actually matters on player-perspective footage. A champion the HUD
confirms is dead is crossed out rather than dimmed, since a champion who cannot
walk out of the fog at you is a different thing from one who can.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from spectral_sight.capture import (
    FrameSizeChanged,
    FrameSource,
    WindowSource,
    open_source,
)
from spectral_sight.debug import draw_tracks
from spectral_sight.debug.overlay import CastMark
from spectral_sight.events import EventDeriver
from spectral_sight.feed import FanOut, FrameState, JsonlSink, RateMeter, StdoutSink
from spectral_sight.calibration import (
    MISSING_CLOCK,
    Reference,
    derive,
    fit_layout,
    missing,
)
from spectral_sight.perception.minimap.locate import locate_panel
from spectral_sight.perception.minimap.region import REGION_DIR, MinimapRegion
from spectral_sight.pipeline import Pipeline
from spectral_sight.types import Frame, Team

DEFAULT_ICONS = Path(__file__).resolve().parents[1] / "etc" / "icons"


def newest_icon_set() -> Path:
    """Most recently fetched icon set, so --icons is usually unnecessary."""
    if not DEFAULT_ICONS.exists():
        raise FileNotFoundError(
            f"no icon sets in {DEFAULT_ICONS}. Run: python tools/fetch_icons.py"
        )
    versions = sorted(p for p in DEFAULT_ICONS.iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(
            f"no icon sets in {DEFAULT_ICONS}. Run: python tools/fetch_icons.py"
        )
    return versions[-1]


class Session:
    """A source's frames, absorbing the two things that end a live run.

    Ctrl+C is how a live session is meant to be stopped rather than a failure,
    and a resized window invalidates every calibration at once. Neither should
    unwind past the block that owns the timeline file, or a run gets thrown away
    by the way it ended -- which for an hour of VOD review is the whole session.
    """

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self.interrupted = False
        self.error: str | None = None

    def __iter__(self) -> Iterator[Frame]:
        try:
            yield from self.source.frames()
        except KeyboardInterrupt:
            self.interrupted = True
        except FrameSizeChanged as exc:
            self.error = str(exc)


def open_target(args: argparse.Namespace) -> FrameSource:
    """The clip or the window, whichever was asked for."""
    if args.window:
        return WindowSource(args.window, target_fps=args.fps)
    return open_source(args.input, stride=args.stride, start=args.start)


SAMPLE_FRAMES = 8
"""Frames pulled for calibration. Geometry needs one; the clock check wants
several, since a reader that lands on the single frame it was derived from is
not yet a reader that works."""


def _derive_everything(frames, width: int, height: int) -> bool:
    """Derive the whole calibration set from the reference layout.

    Everything under `etc/` is the same HUD at one scale, so finding the minimap
    panel fixes all of it -- see `spectral_sight/calibration.py`. This is the
    path that makes starting the tool the only step: no drags, and the optional
    calibrations arrive with the required one instead of being a list of four
    more commands to go and run.
    """
    try:
        reference = Reference.load()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return False

    fit = fit_layout(frames[0], reference)
    if fit is None:
        return False

    written = derive(frames, fit, reference)
    if "minimap" not in written:
        return False
    print(f"derived {', '.join(sorted(written))} from the {reference.width}x"
          f"{reference.height} layout, match {fit.score:.2f}")
    if "game time" not in written:
        print(MISSING_CLOCK, file=sys.stderr)
    return True


def _locate(image) -> MinimapRegion | None:
    """The panel by recognition, or None to fall back to a human.

    A weak match is reported with its score rather than swallowed, because the
    next thing that happens is someone being asked to drag a box and the useful
    thing to know is whether the answer was nearly there or nowhere near.
    """
    try:
        match = locate_panel(image)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return None
    if match is None:
        return None
    if not match.confident:
        print(f"the map art did not match well enough to trust "
              f"({match.score:.2f}); asking instead")
        return None
    print(f"found the minimap panel by its art, match {match.score:.2f}. "
          f"Run tools/calibrate_minimap.py to override it.")
    return match.region


def calibrate(source: FrameSource, width: int, height: int) -> bool:
    """Complete the calibration set for this frame size, however it can.

    Derivation first, since it settles all six at once from the map art. The
    drag is what is left when that declines -- on a frame with no panel in it,
    or a window shaped so oddly the search cannot express the panel's aspect --
    and it can only produce the minimap region, so a run that reaches it is a
    run with no game time, world units, deaths or casts.

    Runs whenever *anything* is absent, not only the required region. A setup
    calibrated before any of this existed has a minimap region and nothing else,
    and it should end up with the rest rather than being left alone for having
    the one file that stops the pipeline erroring.

    Calibration used to mean going and finding a screenshot first, which for a
    live tool is a recording step in the middle of the workflow whose whole
    point is that there is no recording. The frames are already here.
    """
    absent = missing(width, height)
    if not absent:
        return True

    frames = [f.image for _, f in zip(range(SAMPLE_FRAMES), source.frames())]
    if not frames:
        print("the source ended before it produced a frame to calibrate against",
              file=sys.stderr)
        return False

    print(f"{width}x{height} has no {', '.join(absent)} calibration yet.")
    if _derive_everything(frames, width, height):
        return True
    if "minimap" not in absent:
        return True   # the optional pieces could not be derived; the run stands

    region = _locate(frames[0])
    if region is None:
        print("Drag a box around the minimap panel, then ENTER to accept, C to "
              "cancel.")
        region = MinimapRegion.select(frames[0])
        if region is None:
            print("cancelled; there is nothing to run without a minimap region",
                  file=sys.stderr)
            return False
        # Only a hand-drawn region is checked for shape. A heavily stretched
        # window really does make the panel oblong -- 264x332 on one measured
        # size -- so warning about the locator's answer would be scolding it for
        # being right. A drag has no such excuse.
        if not region.looks_square:
            print(f"warning: {region.width}x{region.height} is not square, and "
                  "the minimap panel usually is", file=sys.stderr)

    path = REGION_DIR / f"{width}x{height}.json"
    region.save(path)
    print(f"saved {region} -> {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--input", help="video or image path")
    target.add_argument("--window",
                        help="capture a live window whose title contains this")
    parser.add_argument("--icons", help="icon set directory; defaults to newest")
    parser.add_argument("--stride", type=int, default=3,
                        help="process every Nth frame (3 = 10 Hz on 30 fps); "
                             "--input only")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for; --window only")
    parser.add_argument("--zoom", type=float, default=2.5, help="preview upscale")
    parser.add_argument("--save", help="write an annotated video here")
    parser.add_argument("--export",
                        help="write a JSONL timeline here, or '-' to stream "
                             "frame envelopes to stdout for another program")
    parser.add_argument("--quiet", action="store_true",
                        help="print state instead of opening a window")
    parser.add_argument("--limit", type=int, help="stop after N processed frames")
    parser.add_argument("--start", type=int, default=0,
                        help="skip to this source frame before starting; --input only")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="run with whatever calibration already exists instead "
                             "of deriving what is missing")
    args = parser.parse_args()

    if args.window and args.start:
        parser.error("--start is a seek into a file; a live window has no past")

    # When stdout is the data channel, everything said *about* the run moves to
    # stderr, or the consumer's JSON parser meets a status line.
    console = sys.stderr if args.export == "-" else sys.stdout

    try:
        icons = Path(args.icons) if args.icons else newest_icon_set()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    with contextlib.ExitStack() as stack:
        # Opening a window and learning its size are one step from the caller's
        # side: a window that cannot be found and one that never paints are the
        # same failure to report, and neither is worth a traceback.
        try:
            source = stack.enter_context(open_target(args))
            width, height = source.size
        except (RuntimeError, TimeoutError) as exc:
            print(exc, file=sys.stderr)
            return 1
        # Calibration comes before the pipeline rather than after it fails to
        # build, because the pipeline only objects to the *minimap* being
        # absent -- it starts quite happily with no clock, no world units and no
        # deaths, which is not what anyone asked for.
        if not args.no_calibrate and missing(width, height):
            if args.window:
                print(f"The window is {width}x{height}, and that size is part of "
                      "the calibration -- leave it there once this is done.",
                      file=console)
                calibrated = calibrate(source, width, height)
            else:
                # A file gets a throwaway reader of its own. Calibrating costs
                # several frames, and taking them from the run would quietly
                # skip the start of the clip -- a live window has no such
                # problem, since those frames are seconds that really passed.
                with open_target(args) as scratch:
                    calibrated = calibrate(scratch, width, height)
            if not calibrated:
                return 1
        try:
            pipeline = Pipeline.for_resolution(width, height, icons)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

        extras = []
        if pipeline.clock is not None:
            extras.append("clock")
        if pipeline.world is not None:
            ux, _ = pipeline.world.units_per_pixel
            extras.append(f"world {ux:.0f}u/px")
        rate = (f"live {args.fps:g} fps" if args.window
                else f"every {args.stride} frames")
        print(f"{width}x{height} | minimap {pipeline.region.width}px | "
              f"{len(pipeline.gallery)} champion icons | {rate}"
              + (f" | {', '.join(extras)}" if extras else ""), file=console)

        # The optional calibrations are skipped quietly, which is right for the
        # run and wrong for the person watching it: without them there is no
        # game time, no world coordinates, no deaths and no casts, and nothing
        # would say so. Naming the command that fixes each is only useful now
        # that these tools can be pointed at a live window -- before the
        # `window:` scheme the answer was still "go and find a screenshot".
        spec = f"window:{args.window}" if args.window else args.input
        absent = [(what, tool) for what, got, tool in (
            ("game time", pipeline.clock, "calibrate_clock.py"),
            ("world units", pipeline.world, "calibrate_world.py"),
            ("deaths", pipeline.liveness, "calibrate_hud.py"),
            ("nameplates", pipeline.plate_reader, "calibrate_nameplates.py"),
        ) if got is None]
        if absent:
            print(f"no {', '.join(what for what, _ in absent)}. Add with:",
                  file=console)
            for _, tool in absent:
                print(f"  python tools/{tool} --input \"{spec}\"", file=console)

        # A live run has no stride -- it takes whichever frames it can keep up
        # with -- so the timeline records 1, meaning "no frames deliberately
        # skipped", rather than a number that would read as a decimation the
        # run did not perform.
        stride = 1 if args.window else args.stride
        origin = args.window or args.input

        timeline: JsonlSink | None = None
        sinks: list[JsonlSink | StdoutSink] = []
        if args.export:
            meta = pipeline.timeline_meta(origin, stride, (width, height))
            if args.export == "-":
                sinks.append(StdoutSink(meta))
            else:
                timeline = JsonlSink(args.export, meta)
                sinks.append(timeline)
            unkeyed = [name for name, calibration in (("clock", pipeline.clock),
                                                      ("world", pipeline.world))
                       if calibration is None]
            if unkeyed:
                print(f"warning: no calibrated {' and '.join(unkeyed)}; the "
                      "timeline will be missing the keys that join this clip to "
                      "anything else", file=sys.stderr)
        feed = stack.enter_context(FanOut(sinks))

        writer: cv2.VideoWriter | None = None
        paused = False
        processed = 0
        meter = RateMeter()
        deriver = EventDeriver()
        started = time.perf_counter()
        # When each track last cast, so the overlay can mark it for a moment
        # rather than for the single frame the cast settles on.
        last_cast: dict[int, tuple[float, bool]] = {}

        session = Session(source)
        for frame in session:
            result = pipeline.process(frame.image, frame.timestamp)
            processed += 1

            if len(feed):
                state = FrameState.of(
                    result, frame,
                    seq=processed - 1,
                    fps=meter.tick(),
                    dropped=getattr(source, "dropped", 0),
                )
                feed.publish(state)
                # After the frame, so a consumer holds the state an event
                # describes before being told about the change.
                for event in deriver.update(state):
                    feed.publish_event(event)

            for observation in result.observations:
                if observation.cast_drop is not None:
                    last_cast[observation.track_id] = (
                        frame.timestamp, bool(observation.cast_continuous)
                    )

            visible = [t for t in result.tracks
                       if t.age(frame.timestamp) < pipeline.tracker.config.lost_after]
            named = result.named()
            dead = frozenset(o.champion for o in result.observations
                             if o.alive is False and o.champion)

            clock = f"{result.clock}" if result.clock else "--:--"
            if result.clock is not None and not result.clock.observed:
                clock += "*"

            down = ""
            if result.liveness is not None and result.liveness.dead_count:
                # Named casualties where the pipeline could attribute them, a
                # bare count where it could not.
                down = f"  down={','.join(sorted(dead)) or result.liveness.dead_count}"

            if args.quiet:
                allies = sorted(n for n, t in named.items() if t.team is Team.BLUE)
                enemies = sorted(n for n, t in named.items() if t.team is Team.RED)
                where = ""
                if result.self_track is not None:
                    position = pipeline.world_position(
                        result.self_track.x, result.self_track.y
                    )
                    if position is not None:
                        where = f"  self=({position[0]:5.0f},{position[1]:5.0f})"
                print(f"{clock:>7}  t={frame.timestamp:7.2f}s  visible={len(visible):2d}"
                      f"{where}  allies={','.join(allies) or '-':40s} "
                      f"enemies={','.join(enemies) or '-'}{down}", file=console)
            else:
                minimap = pipeline.region.crop(frame.image)
                canvas = draw_tracks(
                    minimap, result.tracks, frame.timestamp,
                    scale=args.zoom, self_track=result.self_track,
                    lost_after=pipeline.tracker.config.lost_after,
                    dead=dead,
                    casts={
                        track_id: CastMark(frame.timestamp - when, continuous)
                        for track_id, (when, continuous) in last_cast.items()
                    },
                )
                cv2.putText(
                    canvas,
                    f"{clock}  ({frame.timestamp:.1f}s)  tracked {len(result.tracks)}  "
                    f"visible {len(visible)}  named {len(named)}{down}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )

                if args.save:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps if args.window else 30.0 / max(args.stride, 1),
                            (w, h),
                        )
                    writer.write(canvas)
                else:
                    cv2.imshow("spectral-sight", canvas)
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = not paused

            if args.limit and processed >= args.limit:
                break

        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    # Frames dropped on arrival, which is the number that says whether the
    # pipeline kept up: a high count means the overlay is describing a moment
    # the game has already moved on from, and the answer is a lower --fps.
    behind = ""
    if isinstance(source, WindowSource) and source.dropped:
        share = source.dropped / max(processed + source.dropped, 1)
        behind = f", dropped {source.dropped} ({share:.0%}) to keep up"
    print(f"\n{processed} frames in {elapsed:.1f}s "
          f"({processed / max(elapsed, 1e-9):.1f} fps{behind})", file=console)
    if args.save:
        print(f"wrote {args.save}", file=console)
    if timeline is not None:
        print(f"wrote {timeline.path} ({timeline.rows} observations)", file=console)
    if session.error is not None:
        print(session.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
