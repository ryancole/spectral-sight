"""Time the pipeline on the live window, stage by stage, with nothing else running.

Answers two questions about a run that drops frames:

1. **Which stage is slow.** Every stage of `Pipeline.process` is lapped, and
   the table ranks them by what they cost per frame, with each one's worst
   single run -- a mean hides the spikes that cause the drops.

2. **Whether that time is OpenCV or Python.** With `--split`, the run goes
   under cProfile instead, and each function's own time is charged to OpenCV,
   NumPy, other native code or Python. Native time only gets faster by doing
   less image work; Python time is the part a compiled language would win back.

The two are separate runs on purpose: cProfile charges a fixed cost to every
Python call, which inflates exactly the Python-heavy stages being measured, so
its absolute numbers are not comparable with the timer's. Read the stage table
for *where*, and the split for *what kind*.

There is no feed, events or printing, so these are the pipeline's numbers
alone; `watch.py --timings` gives the same table for a real run. Time spent
waiting for the window's next frame is reported separately and excluded.

Usage:
    python tools/profile_frames.py --limit 600
    python tools/profile_frames.py --fps 30 --split
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

from spectral_sight.calibration import missing
from spectral_sight.capture import DEFAULT_WINDOW, WindowSource
from spectral_sight.pipeline import Pipeline
from spectral_sight.profiling import StageTimer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watch import newest_icon_set  # noqa: E402


def classify(key: tuple[str, int, str]) -> str:
    """Which kind of code a cProfile entry's own time was spent in."""
    filename, _, name = key
    if filename == "~":
        # Builtins: cProfile names them by the object they are bound to.
        # OpenCV 5's functions carry no module and appear bare, `<inRange>`,
        # where everything else reads `<built-in method ...>` or `<method ...>`.
        if "numpy" in name:
            return "NumPy (native)"
        if "cv2" in name or " " not in name:
            return "OpenCV"
        return "other native"
    if "cv2" in filename.replace("\\", "/").split("/"):
        return "OpenCV"
    if "numpy" in filename:
        return "NumPy (Python layer)"
    if "spectral_sight" in filename:
        return "spectral_sight Python"
    return "other Python"


def split(profile: cProfile.Profile, frames: int, top: int) -> str:
    stats = pstats.Stats(profile).stats  # type: ignore[attr-defined]
    kinds: dict[str, float] = defaultdict(float)
    ours: dict[str, float] = defaultdict(float)
    native: dict[str, float] = defaultdict(float)
    for key, (_, _, tottime, _, _) in stats.items():
        kind = classify(key)
        kinds[kind] += tottime
        if kind == "spectral_sight Python":
            filename = key[0].replace("\\", "/")
            module = filename.split("spectral_sight/", 1)[-1]
            ours[f"{module}:{key[2]}"] += tottime
        elif kind in ("OpenCV", "NumPy (native)"):
            native[key[2]] += tottime
    total = sum(kinds.values()) or 1e-9

    def table(title: str, rows: dict[str, float], limit: int) -> list[str]:
        ranked = sorted(rows.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        width = max((len(n) for n, _ in ranked), default=5)
        lines = [title]
        for name, spent in ranked:
            lines.append(f"  {name:<{width}}  {1000 * spent / frames:8.2f}ms/frame"
                         f"  {spent / total:6.1%}")
        return lines

    lines = [f"cProfile, {frames} frames, {1000 * total / frames:.1f} ms/frame "
             "(inflated by profiling overhead)"]
    lines += table("by kind of code:", kinds, len(kinds))
    lines += table(f"top {top} native calls:", native, top)
    lines += table(f"top {top} spectral_sight functions (own time):", ours, top)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default=DEFAULT_WINDOW,
                        help="capture the window whose title contains this "
                             f"(default {DEFAULT_WINDOW!r})")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for")
    parser.add_argument("--icons", help="icon set directory; defaults to newest")
    parser.add_argument("--stride", type=int, default=3,
                        help="with coaching, run the minimap stages every Nth "
                             "frame, as watch.py does")
    parser.add_argument("--coach", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="read the world view on every frame, as watch.py "
                             "does by default")
    parser.add_argument("--limit", type=int, default=600,
                        help="stop after N calls to process (default 600)")
    parser.add_argument("--split", action="store_true",
                        help="run under cProfile and split time by kind of code "
                             "instead of timing stages")
    parser.add_argument("--top", type=int, default=15,
                        help="rows in each --split ranking")
    args = parser.parse_args()

    icons = Path(args.icons) if args.icons else newest_icon_set()
    with WindowSource(args.window, target_fps=args.fps) as source:
        width, height = source.size
        # A stage with no calibration does not run, and so costs nothing --
        # which would flatter the numbers without saying why.
        absent = missing(width, height)
        if absent:
            print(f"warning: no {', '.join(absent)} calibration for "
                  f"{width}x{height}; those stages will not be timed. Run "
                  "watch.py once to derive them.", file=sys.stderr)
        try:
            pipeline = Pipeline.for_resolution(
                width, height, icons, every=args.stride if args.coach else 1,
                coach=args.coach,
            )
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        timer = StageTimer()
        profile = cProfile.Profile() if args.split else None
        if profile is None:
            pipeline.timer = timer

        # Waiting for the window is the capture's time, not the pipeline's, so
        # it stays outside both the timer and the profile.
        calls = 0
        wait = 0.0
        began = time.perf_counter()
        frames = iter(source.frames())
        while calls < args.limit:
            mark = time.perf_counter()
            frame = next(frames, None)
            wait += time.perf_counter() - mark
            if frame is None:
                break
            if profile is not None:
                profile.enable()
            pipeline.process(frame.image, frame.timestamp)
            if profile is not None:
                profile.disable()
            calls += 1
        wall = time.perf_counter() - began

    if not calls:
        print("no frames read", file=sys.stderr)
        return 1
    print(f"{width}x{height}, {calls} calls, {wall:.1f}s wall, "
          f"waiting on the window {1000 * wait / calls:.1f} ms/frame")
    print(split(profile, calls, args.top) if profile is not None
          else timer.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
