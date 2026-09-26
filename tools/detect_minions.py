"""Read minion health bars off the world view, and draw what was found.

    python tools/detect_minions.py --input "data/your clip.mp4" --from 150 --to 330
    python tools/detect_minions.py --input "data/your clip.mp4" --at 260 --overlay out/

Prints one line per sampled frame -- how many ally and enemy minions, how many
of them had a readable health, and how many of those are low enough to be a
last hit -- then a summary. `--overlay` writes each sampled frame with every
bar boxed and its fill printed over it, which is how the reader is checked:
there is no automatic ground truth for minions, so the check is by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from spectral_sight.capture import open_source
from spectral_sight.perception.nameplates import NameplateLayout
from spectral_sight.perception.nameplates.minions import MinionReader
from spectral_sight.types import Team

COLOURS = {Team.BLUE: (255, 160, 0), Team.RED: (60, 60, 255)}


def draw(image, minions, reader):
    out = image.copy()
    for minion in minions:
        colour = COLOURS[minion.team]
        cv2.rectangle(
            out, (minion.x - 2, minion.y - 2),
            (minion.x + minion.width + 1, minion.y + reader.height + 1),
            colour, 1,
        )
        label = ("occl" if minion.occluded else "clip" if minion.clipped
                 else f"{minion.health:.2f}")
        cv2.putText(out, label, (minion.x, minion.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def sampled(args):
    """The frames to read: each `--at` time, or one per `--every` seconds."""
    if args.at:
        for at in sorted(args.at):
            with open_source(args.input, start=round(at * 30)) as source:
                frame = next(iter(source.frames()), None)
            if frame is not None:
                yield frame
        return
    next_at = args.start
    with open_source(args.input, start=int(args.start * 30)) as source:
        for frame in source.frames():
            if args.end is not None and frame.timestamp > args.end:
                break
            if frame.timestamp + 1e-6 >= next_at:
                next_at = frame.timestamp + args.every
                yield frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, default=None)
    parser.add_argument("--at", type=float, nargs="*", default=None,
                        help="read only these video times")
    parser.add_argument("--every", type=float, default=1.0,
                        help="seconds between sampled frames")
    parser.add_argument("--low", type=float, default=0.25,
                        help="health at or below which a minion counts as low")
    parser.add_argument("--overlay", type=Path, default=None)
    args = parser.parse_args()

    reader = None
    if args.overlay:
        args.overlay.mkdir(parents=True, exist_ok=True)

    totals = {Team.BLUE: 0, Team.RED: 0}
    readable = low = frames = 0
    for frame in sampled(args):
        t = frame.timestamp
        if reader is None:
            width, height = frame.size
            reader = MinionReader(NameplateLayout.for_resolution(width, height))
        minions = reader.read(frame.image)
        frames += 1
        counts = {team: sum(m.team is team for m in minions) for team in totals}
        read = [m for m in minions if m.health is not None]
        weak = [m for m in read if m.health <= args.low]
        for team in totals:
            totals[team] += counts[team]
        readable += len(read)
        low += len(weak)
        print(f"{t:8.2f}  blue {counts[Team.BLUE]:2d}  red {counts[Team.RED]:2d}"
              f"  readable {len(read):2d}  low {len(weak):2d}")
        if args.overlay:
            cv2.imwrite(str(args.overlay / f"minions_{t:08.2f}.png"),
                        draw(frame.image, minions, reader))

    if frames:
        total = totals[Team.BLUE] + totals[Team.RED]
        print(f"\n{frames} frames: {totals[Team.BLUE] / frames:.1f} ally and "
              f"{totals[Team.RED] / frames:.1f} enemy minions per frame; "
              f"{readable / max(total, 1):.0%} with readable health, "
              f"{low} low readings")


if __name__ == "__main__":
    main()
