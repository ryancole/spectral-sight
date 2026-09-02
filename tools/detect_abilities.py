"""Read the local player's ability casts off a clip and report them.

    # run the reader over a clip
    python tools/detect_abilities.py --input "data/your clip.mp4"

    # score against the player's printed mana, the free ground truth
    python tools/detect_abilities.py --input "data/your clip.mp4" --validate

The HUD draws the local player's cooldowns, so a cast is the slot's ability art
being replaced by the cooldown veil. That names the button -- Q, W, E, R, or a
summoner spell -- which the resource-drop cast detector never can, and it sees
summoner spells and zero-mana casts the resource route is blind to.

Whether the reader is seeing casts or a noise generator is not something it can
assert about itself, so `--validate` checks it the way the resource cast
detector is checked: against the player's own mana, printed as text and read
exactly, on every frame. An ability that costs mana must coincide with a fall
in that number, and a fall with no ability nearby is a miss. Both halves are
measurable and neither needs a label.

Needs the ability calibration (derived automatically from the minimap fit, or
`etc/abilities/<WxH>.json`) and, for the countdown digits, the clock's glyph
set -- a run with no clock reads casts without their seconds.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from spectral_sight.capture import open_source
from spectral_sight.perception.hud.abilities import load_ability_reader
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.hud.resources import load_resource_reader


def _readers(width: int, height: int):
    try:
        clock = load_clock_reader(width, height)
    except FileNotFoundError:
        clock = None
    glyphs = None if clock is None else clock.glyphs
    reader = load_ability_reader(width, height, glyphs)
    resources = load_resource_reader(width, height, glyphs)
    return reader, resources


def run(path: str, stride: int, limit: int) -> tuple[list, list]:
    """Every ability cast, and the player's mana series, over the clip."""
    with open_source(path) as probe:
        first = next(iter(probe.frames()), None)
    if first is None:
        raise SystemExit(f"no frames in {path}")
    width, height = first.size

    reader, resources = _readers(width, height)
    if reader is None:
        raise SystemExit(
            f"no ability calibration for {width}x{height}; it derives from the "
            f"minimap fit on a normal run, or drag one with the receiver open"
        )

    casts: list = []
    mana: list[tuple[float, int, int]] = []
    with open_source(path, stride=stride) as source:
        for sampled, frame in enumerate(source.frames()):
            if limit and sampled >= limit:
                break
            casts.extend(reader.read(frame.image, frame.timestamp))
            if resources is not None:
                reading = resources.read_line(frame.image, resources.layout.mana)
                if reading is not None:
                    mana.append((frame.timestamp, reading.current, reading.maximum))
    casts.extend(reader.flush())
    return casts, mana


def report(casts: list, list_all: bool) -> None:
    by_slot = collections.Counter(c.slot for c in casts)
    print(f"\n{len(casts)} casts")
    for slot in ("Q", "W", "E", "R", "D", "F"):
        if by_slot[slot]:
            seconds = collections.Counter(
                c.countdown for c in casts if c.slot == slot
            )
            read = sum(n for cd, n in seconds.items() if cd is not None)
            print(f"  {slot}: {by_slot[slot]:3d}  "
                  f"countdown read on {read}/{by_slot[slot]}  {dict(seconds)}")
    if list_all:
        print()
        for c in casts:
            cd = "  ?" if c.countdown is None else f"{c.countdown:3d}"
            flag = "" if c.confirmed else "  (unconfirmed)"
            print(f"  {c.at:8.1f}  {c.slot}  cd={cd}{flag}")


def validate(casts: list, mana: list[tuple[float, int, int]]) -> int:
    """Score the casts against the player's printed mana.

    Recall: of genuine mana falls -- a consecutive readable pair whose current
    drops by a real ability cost, death resets excluded -- how many a cast
    lands on. A clean precision figure is harder, because rapid casts merge
    their falls and E/R fire at low mana, so this reports the falls caught and
    the casts left uncorroborated rather than a single ratio that would flatter
    or malign the reader depending on the clip.
    """
    if not mana:
        print("no mana readings; cannot validate", file=sys.stderr)
        return 1

    falls = []
    for (t0, c0, m0), (t1, c1, m1) in zip(mana, mana[1:]):
        if t1 - t0 <= 1.0 and m0 == m1 and 15 <= c0 - c1 <= 0.6 * m0 and c1 > 0:
            falls.append((t1, c0 - c1))

    qwer = [c for c in casts if c.slot in "QWER"]
    caught = sum(1 for f in falls
                 if any(abs(c.at - f[0]) <= 0.7 for c in qwer))
    corroborated = sum(1 for c in qwer
                       if any(abs(c.at - f[0]) <= 0.7 for f in falls))

    print(f"\nmana falls (death resets excluded): {len(falls)}")
    if falls:
        print(f"  caught by a cast within 0.7s: {caught}  "
              f"(recall {caught / len(falls):.0%})")
    print(f"ability casts (QWER): {len(qwer)}")
    print(f"  landing on a mana fall: {corroborated}")
    print(f"  no mana fall nearby: {len(qwer) - corroborated}  "
          f"(rapid combos merge falls, and E/R can fire at low mana, so this "
          f"is an upper bound on false positives, not a count of them)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="clip or window:name")
    parser.add_argument("--validate", action="store_true",
                        help="score against the player's printed mana")
    parser.add_argument("--list", action="store_true",
                        help="print every cast, not just the summary")
    parser.add_argument("--stride", type=int, default=3,
                        help="source frames per sample; 3 is 10 Hz on 30 fps")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N sampled frames; 0 walks the whole "
                             "source, which a live window never finishes")
    args = parser.parse_args()

    casts, mana = run(args.input, args.stride, args.limit)
    report(casts, args.list)
    if args.validate:
        return validate(casts, mana)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
