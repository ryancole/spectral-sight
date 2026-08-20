"""Save one frame from a live window as a PNG.

    python tools/grab.py --window kilrogg --out etc/grabs/kilrogg.png

Nothing in the workflow needs this: every tool here accepts `window:kilrogg` as
a source directly, and `watch.py` calibrates itself from a live frame. This is
for when you want the still itself -- to look at, to diff against, to keep as
the reference frame for a size, or to hand to something outside this project.

`--delay` is there because the frame worth keeping is one with a game in it, and
the moment this is run from a terminal is not that moment.

The size printed is the one that matters: it is the key every calibration is
filed under, and the window has to stay at it. The receiver stretches its stream
to fill the window without preserving aspect (`DXGI_SCALING_STRETCH`), so a
resize does not merely move the minimap, it reshapes it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from spectral_sight.capture import WindowSource


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--window", required=True,
                        help="capture a window whose title contains this")
    parser.add_argument("--out", required=True, help="where to write the PNG")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to wait before grabbing")
    args = parser.parse_args()

    if args.delay:
        print(f"grabbing in {args.delay:g}s...")
        time.sleep(args.delay)

    try:
        with WindowSource(args.window) as source:
            width, height = source.size
            frame = next(iter(source.frames()))
    except (RuntimeError, TimeoutError) as exc:
        print(exc, file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame.image):
        print(f"could not write {out}", file=sys.stderr)
        return 1

    print(f"wrote {out} ({width}x{height})")
    print(f"Calibrations are filed under {width}x{height}; keep the window at "
          "that size or they will describe the wrong pixels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
