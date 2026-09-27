"""Run the stage 1 detector over the live window and show what it found.

    # region loaded from etc/regions/
    python tools/detect_blips.py

    # explicit region, save one annotated frame instead of opening a window
    python tools/detect_blips.py --region 1610,790,290,290 --save out.png

    # how fast is it, really, on the current frame
    python tools/detect_blips.py --benchmark 2000

Press Q to quit, SPACE to pause.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time

import cv2
import numpy as np

from spectral_sight.capture import DEFAULT_WINDOW, WindowSource
from spectral_sight.debug import draw_blips, stack_masks
from spectral_sight.perception.minimap import (
    BlipDetector,
    BlipDetectorConfig,
    MinimapRegion,
)
from spectral_sight.perception.minimap.blips import scaled_config


def resolve_region(spec: str | None, size: tuple[int, int]) -> MinimapRegion:
    if spec:
        return MinimapRegion.parse(spec)
    return MinimapRegion.for_resolution(*size)


def build_detector(region: MinimapRegion, *, autoscale: bool) -> BlipDetector:
    config = BlipDetectorConfig()
    if autoscale:
        config = scaled_config(config, minimap_width=region.width)
    return BlipDetector(config)


def benchmark(detector: BlipDetector, minimap: np.ndarray, iterations: int) -> None:
    for _ in range(min(50, iterations)):  # warm up caches and the morphology kernel
        detector.detect(minimap)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        detector.detect(minimap)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    height, width = minimap.shape[:2]
    print(f"minimap {width}x{height}, {iterations} iterations")
    print(f"  mean   {statistics.fmean(samples):6.3f} ms")
    print(f"  median {samples[len(samples) // 2]:6.3f} ms")
    print(f"  p95    {samples[int(len(samples) * 0.95)]:6.3f} ms")
    print(f"  max    {samples[-1]:6.3f} ms")
    print(f"  -> {1000.0 / statistics.fmean(samples):,.0f} minimap reads/sec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default=DEFAULT_WINDOW,
                        help="capture the window whose title contains this "
                             f"(default {DEFAULT_WINDOW!r})")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for")
    parser.add_argument("--region", help="x,y,width,height; else load by resolution")
    parser.add_argument("--save", help="write one annotated frame here and stop")
    parser.add_argument("--zoom", type=float, default=2.0, help="preview upscale")
    parser.add_argument("--masks", action="store_true", help="show the colour masks")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N iterations")
    parser.add_argument(
        "--no-autoscale",
        action="store_true",
        help="do not rescale radii to the minimap size",
    )
    args = parser.parse_args()

    with (WindowSource(args.window, target_fps=args.fps) as source,
          contextlib.suppress(KeyboardInterrupt)):
        try:
            region = resolve_region(args.region, source.size)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

        detector = build_detector(region, autoscale=not args.no_autoscale)
        paused = False
        counts: list[int] = []

        for frame in source.frames():
            minimap = region.crop(frame.image)

            if args.benchmark:
                benchmark(detector, minimap, args.benchmark)
                return 0

            blips, debug = detector.detect_with_debug(minimap)
            counts.append(len(blips))

            canvas = draw_blips(minimap, blips, scale=args.zoom)
            if args.masks:
                masks = cv2.resize(
                    stack_masks(debug.masks),
                    (canvas.shape[1], canvas.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas = np.hstack([canvas, masks])

            if args.save:
                cv2.imwrite(args.save, canvas)
                break
            cv2.imshow("stage 1 - blips", canvas)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

    cv2.destroyAllWindows()

    if counts:
        print(
            f"{len(counts)} frames | blips per frame: "
            f"mean {statistics.fmean(counts):.2f}, "
            f"min {min(counts)}, max {max(counts)}"
        )
        full = sum(1 for c in counts if c == 10)
        print(f"frames with all 10 champions: {full}/{len(counts)} ({full / len(counts):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
