"""Read the local player's ability casts off the game and report them.

    # run the reader over the game as it plays; Ctrl+C ends it and reports
    python tools/detect_abilities.py

    # score against the player's printed mana, the free ground truth
    python tools/detect_abilities.py --validate --seconds 300

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
import contextlib
import sys
from pathlib import Path

from spectral_sight.capture import DEFAULT_WINDOW, WindowSource, lasting
from spectral_sight.perception.hud.abilities import load_ability_reader
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.hud.resources import load_resource_reader
from spectral_sight.perception.hud.skill_points import load_skill_point_reader


def _readers(width: int, height: int):
    try:
        clock = load_clock_reader(width, height)
    except FileNotFoundError:
        clock = None
    glyphs = None if clock is None else clock.glyphs
    reader = load_ability_reader(width, height, glyphs)
    resources = load_resource_reader(width, height, glyphs)
    points = None if reader is None else load_skill_point_reader(reader.layout)
    return reader, resources, points


def run(window: str, fps: float, seconds: float) -> tuple[list, list, list]:
    """Every ability cast, the player's mana series, and every change in the
    level-up chevrons, for as long as it watched."""
    casts: list = []
    mana: list[tuple[float, int, int]] = []
    changes: list[tuple[float, tuple[str, ...]]] = []
    learnable: tuple[str, ...] | None = None
    with WindowSource(window, target_fps=fps) as source:
        width, height = source.size
        reader, resources, points = _readers(width, height)
        if reader is None:
            raise SystemExit(
                f"no ability calibration for {width}x{height}; it derives from "
                f"the minimap fit on a normal run, or drag one with the receiver "
                f"open"
            )
        # Ctrl+C ends the watching, not the report.
        with contextlib.suppress(KeyboardInterrupt):
            for frame in lasting(source.frames(), seconds):
                casts.extend(reader.read(frame.image, frame.timestamp))
                if resources is not None:
                    reading = resources.read_line(frame.image,
                                                  resources.layout.mana)
                    if reading is not None:
                        mana.append((frame.timestamp, reading.current,
                                     reading.maximum))
                if points is not None:
                    now = points.read(frame.image, frame.timestamp)
                    if now is not None and now != learnable:
                        changes.append((frame.timestamp, now))
                        learnable = now
    casts.extend(reader.flush())
    return casts, mana, changes


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


def report_points(changes: list[tuple[float, tuple[str, ...]]]) -> None:
    """Each stretch the level-up chevrons were lit: when, what, how long."""
    windows = []
    opened: tuple[float, tuple[str, ...]] | None = None
    for at, slots in changes:
        if slots and opened is None:
            opened = (at, slots)
        elif slots and opened is not None and slots != opened[1]:
            windows.append((opened[0], at, opened[1]))
            opened = (at, slots)
        elif not slots and opened is not None:
            windows.append((opened[0], at, opened[1]))
            opened = None
    if opened is not None:
        windows.append((opened[0], None, opened[1]))
    print(f"\n{len(windows)} skill-point windows")
    for start, end, slots in windows:
        held = ("  (still lit)" if end is None
                else f"  spent after {end - start:5.1f}s")
        print(f"  {start:8.1f}  {''.join(slots):4s}{held}")


def validate(casts: list, mana: list[tuple[float, int, int]]) -> int:
    """Score the casts against the player's printed mana.

    Recall: of genuine mana falls -- a consecutive readable pair whose current
    drops by a real ability cost, death resets excluded -- how many a cast
    lands on. A clean precision figure is harder, because rapid casts merge
    their falls and E/R fire at low mana, so this reports the falls caught and
    the casts left uncorroborated rather than a single ratio that would flatter
    or malign the reader depending on the game.
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
    parser.add_argument("--window", default=DEFAULT_WINDOW,
                        help="capture the window whose title contains this "
                             f"(default {DEFAULT_WINDOW!r})")
    parser.add_argument("--validate", action="store_true",
                        help="score against the player's printed mana")
    parser.add_argument("--list", action="store_true",
                        help="print every cast, not just the summary")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="frames per second to ask the window for")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this long; 0 watches until Ctrl+C")
    args = parser.parse_args()

    casts, mana, changes = run(args.window, args.fps, args.seconds)
    report(casts, args.list)
    report_points(changes)
    if args.validate:
        return validate(casts, mana)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
