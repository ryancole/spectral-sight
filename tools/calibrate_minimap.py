"""Interactively mark the minimap panel in a screenshot and save the region.

The minimap's size is driven by an in-game scale slider, so it cannot be derived
from resolution alone. Drag a box around the panel once per (resolution, scale)
and everything downstream reads the saved JSON.

    python tools/calibrate_minimap.py --image data/frame.png

Drag the box, then press ENTER to accept or C to cancel.
"""

from __future__ import annotations

import argparse
import sys

import cv2

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

    box = cv2.selectROI("calibrate minimap", frame.image, showCrosshair=False)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v) for v in box)
    if w == 0 or h == 0:
        print("cancelled", file=sys.stderr)
        return 1

    if abs(w - h) > max(w, h) * 0.08:
        print(f"warning: {w}x{h} is not square; the minimap panel should be", file=sys.stderr)

    region = MinimapRegion(x=x, y=y, width=w, height=h)
    suffix = "" if args.profile == "default" else f".{args.profile}"
    path = args.out or REGION_DIR / f"{width}x{height}{suffix}.json"
    region.save(path)
    print(f"saved {region} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
