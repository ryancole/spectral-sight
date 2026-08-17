"""Mark the friendly HUD portraits, which is what makes death readable.

The portrait row and the local player's own portrait sit at fixed positions for
a given resolution, but not at positions derivable from it, so they are marked
once by hand like the minimap panel:

    python tools/calibrate_hud.py --input "data/your clip.mp4"

Three boxes: the leftmost teammate portrait, the rightmost teammate portrait,
and your own. The spacing between the four teammate slots is interpolated from
the first and last rather than asked for, since the row is evenly spaced and
three of the four numbers would otherwise be a chance to make a typo.

Then check it against footage, which reports what each slot looks like alive
and any deaths it finds:

    python tools/calibrate_hud.py --input "data/your clip.mp4" --validate

That check is worth running. A box placed a few pixels off still reads a
portrait and still produces a baseline, so a bad calibration does not announce
itself -- what it does instead is fail to notice deaths, which is invisible
until you go looking for one.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from spectral_sight.capture import open_source
from spectral_sight.perception.hud.alive import AliveReader
from spectral_sight.perception.hud.portraits import LAYOUT_DIR, PortraitLayout


def circle_from_box(box: tuple[int, ...]) -> tuple[float, float, int] | None:
    """Centre and radius of the circle a dragged box encloses."""
    x, y, w, h = (int(v) for v in box)
    if w == 0 or h == 0:
        return None
    return x + w / 2, y + h / 2, int(round(min(w, h) / 2))


def mark(image, prompt: str) -> tuple[float, float, int] | None:
    print(prompt)
    box = cv2.selectROI("calibrate hud", image, showCrosshair=False)
    return circle_from_box(box)


def validate(path: str, layout: PortraitLayout, stride: int) -> int:
    """Run the reader over a clip and report what it learned and found."""
    reader = AliveReader(layout)
    history: dict[str, list[tuple[float, bool | None]]] = {}

    with open_source(path, stride=stride) as source:
        for frame in source.frames():
            for state in reader.read(frame.image).slots:
                history.setdefault(state.slot, []).append(
                    (frame.timestamp, state.alive)
                )

    if not history:
        print(f"no frames in {path}", file=sys.stderr)
        return 1

    frames = len(next(iter(history.values())))
    print(f"\n{frames} frames at stride {stride}\n")
    print(f"{'slot':>6} {'baseline':>9} {'readable':>9}  deaths")
    total_deaths = 0
    for slot, rows in history.items():
        readable = sum(1 for _, alive in rows if alive is not None)
        spans = death_spans(rows)
        total_deaths += len(spans)
        summary = ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in spans) or "-"
        print(f"{slot:>6} {reader.baselines[slot]:9.0f} "
              f"{readable / frames:8.0%}  {summary}")

    if total_deaths == 0:
        print("\nNo deaths found. If nobody died in this clip that is the right "
              "answer; if somebody did, the boxes are probably off.",
              file=sys.stderr)
    return 0


def death_spans(rows: list[tuple[float, bool | None]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = last = None
    for timestamp, alive in rows:
        if alive is False:
            start = timestamp if start is None else start
            last = timestamp
        elif start is not None:
            spans.append((start, last))
            start = None
    if start is not None:
        spans.append((start, last))
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="clip or screenshot")
    parser.add_argument("--validate", action="store_true",
                        help="check the saved layout against this clip")
    parser.add_argument("--stride", type=int, default=15,
                        help="frames per sample when validating")
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()

    with open_source(args.input) as source:
        frame = next(iter(source.frames()), None)
    if frame is None:
        print(f"no frames in {args.input}", file=sys.stderr)
        return 1
    width, height = frame.size
    path = args.out or LAYOUT_DIR / f"{width}x{height}.json"

    if args.validate:
        try:
            layout = PortraitLayout.load(path)
        except FileNotFoundError:
            print(f"no layout at {path}; run without --validate first",
                  file=sys.stderr)
            return 1
        return validate(args.input, layout, args.stride)

    print(f"frame is {width}x{height}. Drag each portrait, ENTER to accept.")
    first = mark(frame.image, "1/3: the LEFTMOST teammate portrait")
    last = mark(frame.image, "2/3: the RIGHTMOST teammate portrait")
    me = mark(frame.image, "3/3: your OWN portrait, in the centre HUD")
    cv2.destroyAllWindows()

    if first is None or last is None or me is None:
        print("cancelled", file=sys.stderr)
        return 1
    if last[0] <= first[0]:
        print("the rightmost portrait must be right of the leftmost one",
              file=sys.stderr)
        return 1

    layout = PortraitLayout(
        ally_first_center_x=first[0],
        ally_center_y=(first[1] + last[1]) / 2,
        ally_spacing=(last[0] - first[0]) / 3,
        ally_radius=round((first[2] + last[2]) / 2),
        self_center_x=me[0],
        self_center_y=me[1],
        self_radius=me[2],
    )
    layout.save(path)
    print(f"saved -> {path}")
    print(f"  teammates every {layout.ally_spacing:.1f}px from "
          f"({layout.ally_first_center_x:.0f}, {layout.ally_center_y:.0f}), "
          f"r={layout.ally_radius}")
    print(f"  you at ({layout.self_center_x:.0f}, {layout.self_center_y:.0f}), "
          f"r={layout.self_radius}")
    print(f"\nNow check it:\n"
          f"  python tools/calibrate_hud.py --input \"{args.input}\" --validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
