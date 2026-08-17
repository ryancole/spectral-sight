"""Watch the whole pipeline run on a clip: detect, identify, track.

    # play a clip in a window
    python tools/watch.py --input "data/my clip.mp4"

    # every frame instead of 10 Hz, and write an annotated video out
    python tools/watch.py --input clip.mp4 --stride 1 --save out.mp4

    # print the tracked state instead of showing it
    python tools/watch.py --input clip.mp4 --quiet

Keys: Q or ESC to quit, SPACE to pause, any key to step while paused.

Champions currently visible are drawn solid; champions in fog are drawn hollow
at their last known position with the time since they were seen, which is the
readout that actually matters on player-perspective footage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from spectral_sight.capture import open_source
from spectral_sight.debug import draw_tracks
from spectral_sight.pipeline import Pipeline
from spectral_sight.types import Team

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="video or image path")
    parser.add_argument("--icons", help="icon set directory; defaults to newest")
    parser.add_argument("--stride", type=int, default=3,
                        help="process every Nth frame (3 = 10 Hz on 30 fps)")
    parser.add_argument("--zoom", type=float, default=2.5, help="preview upscale")
    parser.add_argument("--save", help="write an annotated video here")
    parser.add_argument("--quiet", action="store_true",
                        help="print state instead of opening a window")
    parser.add_argument("--limit", type=int, help="stop after N processed frames")
    parser.add_argument("--start", type=int, default=0,
                        help="skip to this source frame before starting")
    args = parser.parse_args()

    try:
        icons = Path(args.icons) if args.icons else newest_icon_set()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    with open_source(args.input, stride=args.stride, start=args.start) as source:
        width, height = source.size
        try:
            pipeline = Pipeline.for_resolution(width, height, icons)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            print("Calibrate first: python tools/calibrate_minimap.py --image <frame>",
                  file=sys.stderr)
            return 1

        print(f"{width}x{height} | minimap {pipeline.region.width}px | "
              f"{len(pipeline.gallery)} champion icons | every {args.stride} frames")

        writer: cv2.VideoWriter | None = None
        paused = False
        processed = 0
        started = time.perf_counter()

        for frame in source.frames():
            result = pipeline.process(frame.image, frame.timestamp)
            processed += 1

            visible = [t for t in result.tracks
                       if t.age(frame.timestamp) < pipeline.tracker.config.lost_after]
            named = result.named()

            if args.quiet:
                allies = sorted(n for n, t in named.items() if t.team is Team.BLUE)
                enemies = sorted(n for n, t in named.items() if t.team is Team.RED)
                print(f"t={frame.timestamp:7.2f}s  visible={len(visible):2d}  "
                      f"allies={','.join(allies) or '-':40s} enemies={','.join(enemies) or '-'}")
            else:
                minimap = pipeline.region.crop(frame.image)
                canvas = draw_tracks(
                    minimap, result.tracks, frame.timestamp,
                    scale=args.zoom, self_track=result.self_track,
                    lost_after=pipeline.tracker.config.lost_after,
                )
                cv2.putText(
                    canvas,
                    f"{frame.timestamp:6.1f}s  tracked {len(result.tracks)}  "
                    f"visible {len(visible)}  named {len(named)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )

                if args.save:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                            30.0 / max(args.stride, 1), (w, h),
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
    print(f"\n{processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-9):.1f} fps)")
    if args.save:
        print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
