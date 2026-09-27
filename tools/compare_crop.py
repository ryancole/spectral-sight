"""Check that cropping the bar readers to the playfield changes nothing.

The nameplate and minion readers build their masks over the playfield only --
the frame outside the HUD's `exclude` regions, grown by how far a reader looks
from a bar -- rather than the whole frame. By construction that should find
exactly the bars the whole-frame reading finds. This is the check that it does:
both readings are run on every frame, and every difference is printed.

A bar touching the HUD's edge is the case to watch. On the whole frame it can
join up with HUD art into one component that starts inside a panel and is
rejected; cropped, the panel reads as empty and the bar stands alone.

Usage:
    python tools/compare_crop.py --input "data/live-lane-20260925.mp4"
    python tools/compare_crop.py --window kilrogg --limit 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from spectral_sight.capture import WindowSource, open_source
from spectral_sight.perception.nameplates import MinionReader, NameplateReader
from spectral_sight.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watch import newest_icon_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--input", help="video path")
    target.add_argument("--window", help="capture a live window whose title "
                                         "contains this")
    parser.add_argument("--stride", type=int, default=1,
                        help="read every Nth frame; --input only")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for")
    parser.add_argument("--limit", type=int, help="stop after N frames")
    parser.add_argument("--show", type=int, default=10,
                        help="differences to print in full (default 10)")
    args = parser.parse_args()

    source = (WindowSource(args.window, target_fps=args.fps) if args.window
              else open_source(args.input, stride=args.stride))
    with source:
        width, height = source.size
        pipeline = Pipeline.for_resolution(width, height, newest_icon_set())
        cropped_plates = pipeline.plate_reader
        cropped_minions = pipeline.minion_reader
        if cropped_plates is None and cropped_minions is None:
            print(f"no nameplate calibration for {width}x{height}",
                  file=sys.stderr)
            return 1
        whole_plates = whole_minions = None
        if cropped_plates is not None:
            whole_plates = NameplateReader(
                cropped_plates.layout, cropped_plates.glyphs,
                cropped_plates.config, crop=False,
            )
            field = cropped_plates.playfield(width, height)
            print(f"{width}x{height}: playfield covers {field.coverage:.0%} of "
                  f"the frame in {len(field.rects)} rectangles, components "
                  f"over rows {field.rows[0]}-{field.rows[1]}")
        if cropped_minions is not None:
            whole_minions = MinionReader(
                cropped_minions.layout, cropped_minions.config, crop=False
            )

        frames = differing = 0
        counts = {"plates": 0, "minions": 0}
        spent = {"whole": 0.0, "cropped": 0.0}
        shown = 0
        try:
            for frame in source.frames():
                image = frame.image
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                readings = {}
                for kind, plates, minions in (
                    ("whole", whole_plates, whole_minions),
                    ("cropped", cropped_plates, cropped_minions),
                ):
                    began = time.perf_counter()
                    readings[kind] = (
                        plates.read(image, hsv) if plates is not None else [],
                        minions.read(image, hsv) if minions is not None else [],
                    )
                    spent[kind] += time.perf_counter() - began
                frames += 1
                counts["plates"] += len(readings["whole"][0])
                counts["minions"] += len(readings["whole"][1])
                if readings["whole"] != readings["cropped"]:
                    differing += 1
                    if shown < args.show:
                        shown += 1
                        print(f"\nframe {frames - 1} (t={frame.timestamp:.2f}s):")
                        for index, name in enumerate(("plates", "minions")):
                            whole = set(readings["whole"][index])
                            cropped = set(readings["cropped"][index])
                            for bar in sorted(whole - cropped, key=repr):
                                print(f"  {name} whole only:   {bar}")
                            for bar in sorted(cropped - whole, key=repr):
                                print(f"  {name} cropped only: {bar}")
                if args.limit and frames >= args.limit:
                    break
        except KeyboardInterrupt:
            pass

    if not frames:
        print("no frames read", file=sys.stderr)
        return 1
    print(f"\n{frames} frames, {counts['plates']} plates and "
          f"{counts['minions']} minions read on the whole frame")
    print(f"frames that differ: {differing} ({differing / frames:.2%})")
    print(f"both readers, per frame: whole {1000 * spent['whole'] / frames:.1f} "
          f"ms, cropped {1000 * spent['cropped'] / frames:.1f} ms")
    return 0 if differing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
