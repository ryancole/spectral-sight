"""Build the map-art reference that the panel locator recognises.

    python tools/build_reference.py --input "data/your clip.mp4"

The panel is found by correlating this picture against the frame, so what it
has to be is the map with nothing on it. Averaging many frames does that
without any masking: terrain and structures are in the same place every frame
and survive, while champions, wards, pings and the camera box are somewhere
different each time and wash out. Fog washes out the same way, which is why the
source should span enough of a game for vision to have moved around.

The source needs a calibrated minimap region already, because this reads the
panel out of the frame using it -- the one bootstrap in the project. Run it
against a clip you have calibrated by hand, or against a live window after
`watch.py` has calibrated it, and the reference then works at every other
resolution and window shape.

Rebuild it when the map art changes -- a new season, a new map. The locator
degrades into asking a human rather than into a wrong answer, so the symptom is
`watch.py` starting to ask for a drag it used to skip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.capture import open_source
from spectral_sight.perception.minimap.locate import REFERENCE_PATH
from spectral_sight.perception.minimap.region import MinimapRegion

SIZE = 256
"""Side of the stored reference.

It is resized to whatever the frame needs anyway, so this only has to be fine
enough to carry the map's layout -- and small enough that the coarse pass's
downscale of it is still a picture rather than a smudge.
"""

MIN_FRAMES = 20
"""Below this the average still has champions visible in it, which would make
the reference match a particular game rather than the map."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="clip, still or window:name with a calibrated region")
    parser.add_argument("--stride", type=int, default=10,
                        help="sample every Nth frame; the map does not move, so "
                             "neighbouring frames add nothing")
    parser.add_argument("--limit", type=int, default=300,
                        help="stop after N samples")
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()

    try:
        source = open_source(args.input, stride=args.stride)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    with source:
        width, height = source.size
        try:
            region = MinimapRegion.for_resolution(width, height)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            print("This reads the panel out of frames using an existing "
                  "calibration, so that has to come first.", file=sys.stderr)
            return 1

        total = np.zeros((SIZE, SIZE, 3), np.float64)
        seen = 0
        for frame in source.frames():
            crop = cv2.resize(region.crop(frame.image), (SIZE, SIZE))
            total += crop
            seen += 1
            if seen >= args.limit:
                break

    if seen < MIN_FRAMES:
        print(f"only {seen} frames; need at least {MIN_FRAMES} for champions and "
              "fog to average out", file=sys.stderr)
        return 1

    reference = (total / seen).astype(np.uint8)
    out = Path(args.out) if args.out else REFERENCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), reference):
        print(f"could not write {out}", file=sys.stderr)
        return 1

    print(f"averaged {seen} frames of {region.width}x{region.height} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
