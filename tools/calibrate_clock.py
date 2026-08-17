"""Teach the clock reader its digits from a clip, with no hand labelling.

    # drag a rough box around the timer, then let it learn the glyphs
    python tools/calibrate_clock.py --input "data/my clip.mp4"

    # skip the drag if you already know where the timer is
    python tools/calibrate_clock.py --input clip.mp4 --region 2044,43,47,13

    # check an existing calibration without rebuilding it
    python tools/calibrate_clock.py --input clip.mp4 --validate-only

The box does not have to be tight -- it is shrunk to the glyphs, and the gold
clock icon beside them is dropped by saturation, so including it is harmless.

**Where the labels come from.** Nobody types them. The seconds ones-digit counts
0-9 in order once per second, and it returns to zero exactly on the frame the
tens-digit changes. So watching which glyph cells change from frame to frame
pins the whole alphabet: the first frame where both seconds glyphs change at
once is a zero, and every ones-change after that is the next digit up. Twenty
seconds of footage covers all ten.

**And nobody has to check them either.** The clock advances one second per
second of video, so `clock - video_time` is a constant for a correct reader. The
validation pass measures that constant's spread across thousands of frames,
which is a real accuracy number obtained for free.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict

import cv2
import numpy as np

from spectral_sight.capture import open_source
from spectral_sight.perception.hud.clock import (
    COLON,
    ClockConfig,
    ClockReader,
    ClockRegion,
    GlyphSet,
    glyph_boxes,
    glyph_similarity,
    load_clock_reader,
    save_calibration,
    segment_glyphs,
    tighten,
)

DIGITS = [str(d) for d in range(10)]

MMSS = 5
"""Glyph count for a well-formed MM:SS reading: four digits and a separator."""

SECONDS_ONES = 4
SECONDS_TENS = 3
SEPARATOR = 2

STABLE = 0.97
"""Correlation above which two consecutive glyph images are the same digit.

A static digit re-renders identically frame to frame, so this only has to sit
below 1.0 by enough to absorb video compression noise.
"""

MIN_SAMPLES = 4
"""Frames of a digit to average into its template. Each digit is on screen for
about a second, so this is a small fraction of what is available and just
protects against a transient overlay landing on the one frame we kept."""


def _sizing_pass(source, region: ClockRegion, config: ClockConfig,
                 frames: int = 60) -> tuple[int, int] | None:
    """Measure the largest glyph, which fixes the canvas everything shares."""
    width = height = 0
    seen = 0
    for frame in source.frames():
        for _, _, w, h in glyph_boxes(region.crop(frame.image), config):
            width, height = max(width, w), max(height, h)
        seen += 1
        if seen >= frames:
            break
    if width == 0:
        return None
    return width + 2, height + 2


def _learn(source, region: ClockRegion, size: tuple[int, int],
           config: ClockConfig) -> GlyphSet | None:
    """Walk the clip and label glyphs off the seconds counter."""
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    previous: list[np.ndarray] | None = None
    ones: int | None = None

    for frame in source.frames():
        canvases = segment_glyphs(region.crop(frame.image), size, config)
        if len(canvases) != MMSS:
            previous = None
            continue

        if previous is not None:
            ones_changed = (
                glyph_similarity(canvases[SECONDS_ONES], previous[SECONDS_ONES])
                < STABLE
            )
            tens_changed = (
                glyph_similarity(canvases[SECONDS_TENS], previous[SECONDS_TENS])
                < STABLE
            )
            if ones_changed and tens_changed:
                # The tens digit moves only when the ones digit wraps, so the
                # two changing together is unambiguously a zero.
                ones = 0
            elif ones_changed and ones is not None:
                ones = (ones + 1) % 10

        if ones is not None:
            samples[str(ones)].append(canvases[SECONDS_ONES])
            samples[COLON].append(canvases[SEPARATOR])

        previous = canvases
        if all(len(samples[d]) >= MIN_SAMPLES for d in DIGITS):
            break

    missing = [d for d in DIGITS if len(samples[d]) < MIN_SAMPLES]
    if missing:
        print(
            f"only saw {10 - len(missing)}/10 digits (missing {', '.join(missing)}). "
            "Try a longer stretch, or --start past the loading screen.",
            file=sys.stderr,
        )
        return None

    # Median over samples rather than mean: a ping or damage flash landing on
    # the strip biases an average and does not move a median.
    glyphs = {
        label: np.median(np.stack(images), axis=0).astype(np.float32)
        for label, images in sorted(samples.items())
    }
    return GlyphSet(glyphs=glyphs, size=size)


def _validate(source, reader: ClockReader, limit: int) -> bool:
    """Check the reader against video time, which advances at the same rate.

    Two different things are measured, and conflating them would be a mistake.

    *Monotonicity* is a hard invariant: a match timer never runs backwards. A
    single misclassified glyph moves the clock by seconds or minutes, so almost
    any misread trips this. Violations are errors, full stop.

    *Offset drift* is not an error. `clock - video_time` is constant while the
    game runs at real speed, so a step in it means the game stopped and the
    video did not -- a pause, or a client stall during which the HUD is not
    repainted. That is a true reading of a frozen screen. It is tempting to also
    require that the clock never gains on video time, but the catch-up after a
    stall does exactly that, so the sample clip's one real hitch would be
    charged as an error. Stalls are reported instead of counted.
    """
    readings: list[tuple[float, int]] = []
    total = 0
    for frame in source.frames():
        total += 1
        reading = reader.read(frame.image)
        if reading is not None:
            readings.append((frame.timestamp, reading.total_seconds))
        if total >= limit:
            break

    if not readings:
        print("read the clock in 0 frames", file=sys.stderr)
        return False

    backwards = sum(1 for (_, c0), (_, c1) in zip(readings, readings[1:]) if c1 < c0)

    offsets = [c - t for t, c in readings]
    median = statistics.median(offsets)
    # One second of quantisation is expected: the clock is an integer while
    # video time is not.
    off_median = [abs(o - median) > 1.0 for o in offsets]
    stalled = sum(off_median)
    episodes = sum(1 for a, b in zip([False, *off_median], off_median) if b and not a)

    print(f"\nvalidated over {total} frames")
    print(f"  read           {len(readings)}/{total} ({len(readings) / total:.1%})")
    print(f"  ran backwards  {backwards}")
    print(f"  offset median  {median:+.2f}s  (clock minus video time)")
    if stalled:
        worst = max(offsets, key=lambda o: abs(o - median)) - median
        print(f"  stalls         {episodes} episode(s) over {stalled} frames, "
              f"drifting to {worst:+.1f}s (frozen HUD, not a misread)")
    return backwards == 0 and len(readings) == total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="clip containing a running game")
    parser.add_argument("--region", help="skip the drag: x,y,width,height")
    parser.add_argument("--start", type=int, default=0,
                        help="skip to this source frame first")
    parser.add_argument("--validate-only", action="store_true",
                        help="test the saved calibration instead of rebuilding it")
    parser.add_argument("--validate-frames", type=int, default=2000,
                        help="how many frames to validate over")
    args = parser.parse_args()

    config = ClockConfig()

    with open_source(args.input, start=args.start) as source:
        width, height = source.size

        if args.validate_only:
            try:
                reader = load_clock_reader(width, height, config)
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
                return 1
            print(f"{width}x{height} | clock at {reader.region} | "
                  f"{len(reader.glyphs)} glyphs")
            return 0 if _validate(source, reader, args.validate_frames) else 1

        frame = next(iter(source.frames()), None)
        if frame is None:
            print(f"no frames in {args.input}", file=sys.stderr)
            return 1

        if args.region:
            parts = [int(p) for p in args.region.split(",")]
            box = ClockRegion(*parts)
        else:
            print(f"frame is {width}x{height}. Drag a box around the match timer, "
                  "ENTER to accept. It will be tightened for you.")
            x, y, w, h = (int(v) for v in
                          cv2.selectROI("calibrate clock", frame.image,
                                        showCrosshair=False))
            cv2.destroyAllWindows()
            if w == 0 or h == 0:
                print("cancelled", file=sys.stderr)
                return 1
            box = ClockRegion(x=x, y=y, width=w, height=h)

        region = tighten(frame.image, box, config)
        if region is None:
            print(f"no timer glyphs inside {box}. Is the box on the clock?",
                  file=sys.stderr)
            return 1
        print(f"tightened {box} -> {region}")

    with open_source(args.input, start=args.start) as source:
        size = _sizing_pass(source, region, config)
    if size is None:
        print("found no glyphs while sizing", file=sys.stderr)
        return 1
    print(f"glyph canvas {size[0]}x{size[1]}px")

    with open_source(args.input, start=args.start) as source:
        glyphs = _learn(source, region, size, config)
    if glyphs is None:
        return 1
    print(f"learned {len(glyphs)} glyphs: {' '.join(sorted(glyphs.glyphs))}")

    region_path, glyph_path = save_calibration(region, glyphs, width, height)
    print(f"saved {region_path}\n      {glyph_path}")

    with open_source(args.input, start=args.start) as source:
        ok = _validate(source, ClockReader(region, glyphs, config),
                       args.validate_frames)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
