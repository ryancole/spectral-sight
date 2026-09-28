"""Watch the whole pipeline run: detect, identify, track.

Always live, against the kilrogg receiver as it plays:

    # read the receiver, serve the feed on 127.0.0.1:8723
    python tools/watch.py

    # ...and keep the timeline while you watch
    python tools/watch.py --export session.jsonl

    # ...or stream frame envelopes to another program as they happen
    python tools/watch.py --export - | your-tool

    # the feed is served over HTTP by default, to as many programs as care
    # to listen; --serve PORT moves it, --no-serve turns it off
    curl http://127.0.0.1:8723/stream

    # a receiver whose title has changed
    python tools/watch.py --window "some other title"

    # more frames for the world view; the minimap stays at 10 Hz regardless
    python tools/watch.py --fps 30


Frames arrive from the window whether or not the pipeline is ready for them, so
the ones it cannot keep up with are dropped on arrival rather than queued -- see
`Mailbox`. The run reports the drop count at the end, which is the number to
watch if the printed state looks like it is lagging the game.

Ctrl+C ends the run.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from spectral_sight.capture import (
    DEFAULT_WINDOW,
    FrameSizeChanged,
    FrameSource,
    WindowSource,
)
from spectral_sight.events import EventDeriver
from spectral_sight.feed import (
    FanOut, FrameState, JsonlSink, KnownRoster, RateMeter, StdoutSink,
)
from spectral_sight.serve import DEFAULT_PORT, FeedServer
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
from spectral_sight.profiling import StageTimer
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

    The resize surfaces from the source, so the iterator handles it. Ctrl+C
    lands wherever the process happens to be, which is nearly always inside the
    pipeline rather than the frame grab, so it is absorbed by the ``with``
    block that wraps the whole loop rather than by the iterator.
    """

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self.interrupted = False
        self.error: str | None = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            self.interrupted = True
            return True
        return False

    def __iter__(self) -> Iterator[Frame]:
        try:
            yield from self.source.frames()
        except FrameSizeChanged as exc:
            self.error = str(exc)


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


def port_taken(exc: OSError) -> bool:
    """Whether a bind was refused because something else holds the port.

    Windows says so in `winerror`, and reports a port held exclusively (or
    reserved by the OS) as access denied rather than in use."""
    return (exc.errno == errno.EADDRINUSE
            or getattr(exc, "winerror", None) in (10048, 10013))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default=DEFAULT_WINDOW,
                        help="capture the window whose title contains this "
                             f"(default {DEFAULT_WINDOW!r})")
    parser.add_argument("--icons", help="icon set directory; defaults to newest")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for")
    parser.add_argument("--export",
                        help="write a JSONL timeline here, or '-' to stream "
                             "frame envelopes to stdout for another program")
    parser.add_argument("--serve", nargs="?", const=DEFAULT_PORT,
                        default=DEFAULT_PORT, type=int, metavar="PORT",
                        help="serve the feed over HTTP on 127.0.0.1, on by "
                             f"default (port {DEFAULT_PORT}): /meta, /state, "
                             "/stream, /events. Composes with --export")
    parser.add_argument("--no-serve", dest="serve", action="store_const",
                        const=None, help="do not serve the feed over HTTP")
    parser.add_argument("--limit", type=int, help="stop after N processed frames")
    parser.add_argument("--timings", action="store_true",
                        help="time each pipeline stage and print the table at "
                             "the end -- see tools/profile_frames.py for the "
                             "pipeline alone")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="run with whatever calibration already exists instead "
                             "of deriving what is missing")
    args = parser.parse_args()

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
            source = stack.enter_context(
                WindowSource(args.window, target_fps=args.fps))
            width, height = source.size
        except (RuntimeError, TimeoutError) as exc:
            print(exc, file=sys.stderr)
            return 1
        # Calibration comes before the pipeline rather than after it fails to
        # build, because the pipeline only objects to the *minimap* being
        # absent -- it starts quite happily with no clock, no world units and no
        # deaths, which is not what anyone asked for.
        if not args.no_calibrate and missing(width, height):
            print(f"The window is {width}x{height}, and that size is part of "
                  "the calibration -- leave it there once this is done.",
                  file=console)
            if not calibrate(source, width, height):
                return 1
        try:
            pipeline = Pipeline.for_resolution(width, height, icons)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

        if args.timings:
            pipeline.timer = StageTimer()

        extras = []
        if pipeline.clock is not None:
            extras.append("clock")
        if pipeline.world is not None:
            ux, _ = pipeline.world.units_per_pixel
            extras.append(f"world {ux:.0f}u/px")
        print(f"{width}x{height} | minimap {pipeline.region.width}px | "
              f"{len(pipeline.gallery)} champion icons | live {args.fps:g} fps"
              + (f" | {', '.join(extras)}" if extras else ""), file=console)

        # The optional calibrations are skipped quietly, which is right for the
        # run and wrong for the person watching it: without them there is no
        # game time, no world coordinates, no deaths and no casts, and nothing
        # would say so. Naming the command that fixes each is only useful now
        # that these tools can be pointed at a live window -- before that the
        # answer was still "go and find a screenshot".
        target = ("" if args.window == DEFAULT_WINDOW
                  else f' --window "{args.window}"')
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
                print(f"  python tools/{tool}{target}", file=console)
        if pipeline.self_reader is None:
            # Not optional like the others: the ability slots are the only
            # thing that says who the player is, so without them no row is
            # ever the player and every self-only reading goes unpublished.
            print("warning: the player cannot be identified -- no spell "
                  "icons in the icon set or no ability calibration. Run:\n"
                  "  python tools/fetch_icons.py", file=sys.stderr)

        origin = args.window

        timeline: JsonlSink | None = None
        server: FeedServer | None = None
        sinks: list[JsonlSink | StdoutSink | FeedServer] = []
        if args.export or args.serve is not None:
            meta = pipeline.timeline_meta(origin, (width, height))
            if args.export == "-":
                sinks.append(StdoutSink(meta))
            elif args.export:
                timeline = JsonlSink(args.export, meta)
                sinks.append(timeline)
            if args.serve is not None:
                server = FeedServer(meta, port=args.serve)
                sinks.append(server)
            unkeyed = [name for name, calibration in (("clock", pipeline.clock),
                                                      ("world", pipeline.world))
                       if calibration is None]
            if unkeyed:
                print(f"warning: no calibrated {' and '.join(unkeyed)}; the "
                      "feed will be missing the keys that join this session to "
                      "anything else", file=sys.stderr)
        try:
            feed = stack.enter_context(FanOut(sinks))
        except OSError as exc:
            # Serving is the default, so a second run or a replay already on
            # the port is the likely way to get here, not a real fault.
            if server is None or not port_taken(exc):
                raise
            print(f"cannot serve on port {args.serve}: {exc.strerror or exc}. "
                  "Pick another with --serve PORT, or --no-serve.",
                  file=sys.stderr)
            return 1
        if server is not None:
            print(f"serving {server.url}/stream", file=console)

        processed = 0
        meter = RateMeter()
        deriver = EventDeriver()
        known = KnownRoster()
        started = time.perf_counter()
        hud_scale = 1.0
        game = 0

        with Session(source) as session:
            for frame in session:
                result = pipeline.process(frame.image, frame.timestamp)
                if (pipeline.hud_scale is not None
                        and pipeline.hud_scale.scale != hud_scale):
                    hud_scale = pipeline.hud_scale.scale
                    print(f"HUD scale {hud_scale:.3f} of the calibration; "
                          "player panel readers moved to match",
                          file=console)
                if result.game != game:
                    game = result.game
                    print(f"new game (game {game + 1} this run): the clock went "
                          "back to the start of a match; tracks, roster and "
                          "identities start over", file=console)
                if not result.sampled:
                    # A frame between samples: the world-view stages saw it, the
                    # minimap stages did not, and there is nothing to publish.
                    continue
                processed += 1

                if len(feed):
                    state = FrameState.of(
                        result, frame,
                        seq=processed - 1,
                        fps=meter.tick(),
                        dropped=getattr(source, "dropped", 0),
                        roster=known.update(result.observations, result.game),
                    )
                    feed.publish(state)
                    # After the frame, so a consumer holds the state an event
                    # describes before being told about the change.
                    for event in deriver.update(state):
                        feed.publish_event(event)

                lost_after = pipeline.tracker.config.lost_after
                visible = [t for t in result.tracks
                           if t.age(frame.timestamp) < lost_after]
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
                    who = ",".join(sorted(dead)) or result.liveness.dead_count
                    down = f"  down={who}"

                # The player is a blue track too; name them on the self field
                # rather than among the allies, so a self row that has latched
                # onto a teammate reads as wrong at a glance.
                me = result.self_track
                allies = sorted(n for n, t in named.items()
                                if t.team is Team.BLUE and t is not me)
                enemies = sorted(n for n, t in named.items() if t.team is Team.RED)
                where = ""
                if me is not None:
                    where = f"  self={me.identity or '?'}"
                    position = pipeline.world_position(me.x, me.y)
                    if position is not None:
                        where += (f"({position[0]:5.0f},"
                                  f"{position[1]:5.0f})")
                print(f"{clock:>7}  t={frame.timestamp:7.2f}s  "
                      f"visible={len(visible):2d}"
                      f"{where}  allies={','.join(allies) or '-':40s} "
                      f"enemies={','.join(enemies) or '-'}{down}", file=console)

                if args.limit and processed >= args.limit:
                    break

    elapsed = time.perf_counter() - started
    # Frames dropped on arrival, which is the number that says whether the
    # pipeline kept up: a high count means the printed state is describing a
    # moment the game has already moved on from, and the answer is a lower --fps.
    behind = ""
    if source.dropped:
        share = source.dropped / max(processed + source.dropped, 1)
        behind = f", dropped {source.dropped} ({share:.0%}) to keep up"
    print(f"\n{processed} frames in {elapsed:.1f}s "
          f"({processed / max(elapsed, 1e-9):.1f} fps{behind})", file=console)
    if pipeline.timer is not None:
        print(pipeline.timer.report(), file=console)
    if timeline is not None:
        print(f"wrote {timeline.path} ({timeline.rows} observations)", file=console)
    if session.error is not None:
        print(session.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
