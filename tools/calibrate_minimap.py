"""Interactively mark the minimap panel in a frame and save the region.

The minimap's size is driven by an in-game scale slider, so it cannot be derived
from resolution alone. Drag a box around the panel once per (resolution, scale)
and everything downstream reads the saved JSON.

    python tools/calibrate_minimap.py --image data/frame.png
    python tools/calibrate_minimap.py --image window:kilrogg

Drag the box, then press ENTER to accept or C to cancel.

`watch.py` asks for this itself when it finds no calibration for the size it is
looking at, so reaching for this tool directly is for recalibrating, or for a
non-default minimap scale via `--profile`.
"""

from __future__ import annotations

import argparse
import sys

from spectral_sight.capture import open_source
from spectral_sight.perception.minimap.region import REGION_DIR, MinimapRegion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="screenshot or clip to mark up")
    parser.add_argument(
        "--profile",
        default="default",
        help="suffix for a non-default minimap scale, e.g. 'large'",
    )
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()

    with open_source(args.image) as source:
        frame = next(iter(source.frames()), None)
    if frame is None:
        print(f"no frames in {args.image}", file=sys.stderr)
        return 1

    width, height = frame.size
    print(f"frame is {width}x{height}. Drag the minimap panel, ENTER to accept.")

    region = MinimapRegion.select(frame.image)
    if region is None:
        print("cancelled", file=sys.stderr)
        return 1

    if not region.looks_square:
        print(f"warning: {region.width}x{region.height} is not square; the "
              "minimap panel should be", file=sys.stderr)

    suffix = "" if args.profile == "default" else f".{args.profile}"
    path = args.out or REGION_DIR / f"{width}x{height}{suffix}.json"
    region.save(path)
    print(f"saved {region} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
