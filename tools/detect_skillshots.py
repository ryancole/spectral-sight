"""Follow the player's own casts to their bolts, and report what landed.

    python tools/detect_skillshots.py --seconds 600

Watches the window until `--seconds` or Ctrl+C. Joins the three stages a skillshot needs: the ability HUD says a button was
pressed, the world view says a bolt left the player's model, and an enemy
nameplate's health bar says whether it reached anyone. Every frame feeds the
projectile tracker; the plates are read at 10 Hz, on the same clock the
pipeline samples them on (`MINIMAP_INTERVAL`).

The report has two halves. The first is the census -- how many casts, how many
launched a bolt, how many had an enemy on screen to aim at, and how they came
out -- which is what a coaching session actually needs, and which says plainly
how often the footage can say nothing.

The second, under `--sweep`, is the check on whether the geometry means
anything: the bolt's line comes from the stabilised residual and the target's
health from the printed bar, neither knowing about the other, so grouping the
shots by how near the bolt passed and reporting the share followed by a fall
is those two observers agreeing, or failing to. Read the shares against the
baseline in `AimConfig.hit_radius` -- on the footage this project has, an
enemy's bar falls in half of *all* windows of that length, which is why the
verdict does not rest on it.
"""

from __future__ import annotations

import argparse
import contextlib

import numpy as np

from spectral_sight.capture import DEFAULT_WINDOW, WindowSource, lasting
from spectral_sight.perception.hud.abilities import load_ability_reader
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.nameplates import NameplateLayout, NameplateReader, Side
from spectral_sight.perception.screen import (
    AimConfig,
    AimDetector,
    EnemyPlate,
    ProjectileConfig,
    ProjectileTracker,
    WorldView,
)
from spectral_sight.pipeline import MINIMAP_INTERVAL, MINIMAP_SLACK

PLATE_ABOVE_MODEL = 95.0
"""Pixels from the nameplate's bar down to the champion model -- the same
constant the pipeline uses; the model is what a bolt comes from and goes to."""


def run(args) -> tuple[list, float]:
    """Every shot, and how many seconds were watched to find them."""
    source = WindowSource(args.window, target_fps=args.fps)
    width, height = source.size
    try:
        clock = load_clock_reader(width, height)
    except FileNotFoundError:
        clock = None
    glyphs = None if clock is None else clock.glyphs
    abilities = load_ability_reader(width, height, glyphs)
    if abilities is None:
        raise SystemExit("no ability calibration for this resolution")
    plates = NameplateReader(NameplateLayout.for_resolution(width, height), glyphs)
    tracker = ProjectileTracker()
    config = AimConfig(hit_radius=args.radius) if args.radius else AimConfig()
    aim = AimDetector(config=config)
    view = WorldView()
    vx, vy, _, _ = view.box(width, height)

    shots, anchor = [], None
    last_read: float | None = None
    elapsed = 0.0
    # Ctrl+C ends the watching, not the report.
    with source, contextlib.suppress(KeyboardInterrupt):
        for frame in lasting(source.frames(), args.seconds):
            t = elapsed = frame.timestamp
            for cast in abilities.read(frame.image, t):
                aim.observe_cast(cast.slot, cast.at)
            if (last_read is None or t - last_read
                    >= MINIMAP_INTERVAL * (1 - MINIMAP_SLACK)):
                last_read = t
                read = plates.read(frame.image)
                mine = [p for p in read if p.side is Side.SELF]
                anchor = None
                if len(mine) == 1:
                    cx, cy = mine[0].center
                    anchor = (cx - vx, cy - vy + PLATE_ABOVE_MODEL)
                aim.observe_enemies(t, [
                    EnemyPlate(p.center[0] - vx,
                               p.center[1] - vy + PLATE_ABOVE_MODEL,
                               p.health)
                    for p in read if p.hostile
                ])
            candidates = [
                track for track in tracker.update(frame.image, t)
                if track.is_projectile(tracker.config)
            ]
            motion = tracker.last_motion
            if motion is not None:
                aim.observe_motion(t, motion)
            aim.consider(candidates, anchor)
            shots.extend(aim.resolve(t))
    tracker.flush()
    shots.extend(aim.flush())
    return shots, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default=DEFAULT_WINDOW,
                        help="capture the window whose title contains this "
                             f"(default {DEFAULT_WINDOW!r})")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="frames per second to ask the window for; a bolt "
                             "is gone in a few tenths of a second, so keep it high")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this long; 0 watches until Ctrl+C")
    parser.add_argument("--radius", type=float, default=None,
                        help="override AimConfig.hit_radius")
    parser.add_argument("--list", action="store_true", help="print every shot")
    parser.add_argument("--sweep", action="store_true",
                        help="the validation: health falls grouped by miss distance")
    args = parser.parse_args()

    shots, elapsed = run(args)

    launched = [s for s in shots if s.launched is not None]
    aimed = [s for s in launched if s.miss is not None]
    minutes = max(1e-9, elapsed / 60)
    print(f"{len(shots)} casts in ability slots ({len(shots)/minutes:.1f}/min)")
    print(f"  launched a bolt: {len(launched)} ({len(launched)/max(1,len(shots)):.0%})")
    print(f"  with an enemy in front of it: {len(aimed)} "
          f"({len(aimed)/max(1,len(launched)):.0%} of the bolts)")
    by_slot: dict[str, list] = {}
    for s in shots:
        by_slot.setdefault(s.slot, []).append(s)
    for slot in sorted(by_slot):
        group = by_slot[slot]
        hit = sum(1 for s in group if s.outcome == "hit")
        missed = sum(1 for s in group if s.outcome == "missed")
        unknown = sum(1 for s in group if s.outcome == "unknown")
        bolts = [s for s in group if s.launched is not None]
        speed = (f"{np.median([s.speed for s in bolts]):.0f} px/s"
                 if bolts else "-")
        print(f"  {slot}: {len(group):3d} cast, {len(bolts):3d} launched at {speed:>10}"
              f" -- {hit} hit, {missed} missed, {unknown} unknown")
    if aimed:
        miss = [s.miss for s in aimed]
        print(f"  miss distance px: p10={np.percentile(miss,10):.0f} "
              f"p50={np.median(miss):.0f} p90={np.percentile(miss,90):.0f}")
        lag = [s.lead for s in aimed if s.lead is not None]
        if lag:
            print(f"  lead measured on {len(lag)}: "
                  f"{sum(1 for x in lag if x > 0)} ahead of the target, "
                  f"{sum(1 for x in lag if x < 0)} behind")

    if args.sweep:
        # The verdict is geometric, so this is not a scoring of it -- it is
        # the check on whether the geometry means anything, run against the
        # one other observer the game has. `fall` is the target's bar
        # moving in the arrival window, and the shares below are only
        # readable against the baseline rate at which it moves in any window
        # of that length -- which is recorded in `AimConfig.hit_radius`,
        # because it takes a whole game's plate tracks to compute and not
        # just the shots.
        print()
        print("does the bolt's geometry agree with the target's health bar?")
        print(f"{'miss px':>12} {'shots':>6} {'fell':>5} {'share':>6}")
        edges = [(0, 50), (50, 100), (100, 130), (130, 200), (200, 400), (400, 1e9)]
        for lo, hi in edges:
            group = [s for s in aimed if lo <= s.miss < hi]
            fell = [s for s in group if s.fall is not None]
            share = f"{len(fell)/len(group):.0%}" if group else "-"
            label = f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+"
            print(f"{label:>12} {len(group):6d} {len(fell):5d} {share:>6}")
        for name, group in (("hit", [s for s in aimed if s.outcome == "hit"]),
                            ("missed", [s for s in aimed if s.outcome == "missed"])):
            fell = [s for s in group if s.fall is not None]
            share = f"{len(fell)/len(group):.0%}" if group else "-"
            print(f"  called {name:>6}: {len(group):3d} shots, "
                  f"the bar fell after {len(fell):3d} ({share})")

    if args.list:
        print()
        for s in shots:
            bolt = ("no bolt" if s.launched is None
                    else f"{s.speed:5.0f}px/s ({s.heading[0]:+.2f},{s.heading[1]:+.2f})")
            aimed_at = "" if s.miss is None else f" miss={s.miss:5.0f} flight={s.flight:.2f}"
            extra = "" if s.fall is None else f" fall={s.fall:.3f}"
            lead = "" if s.lead is None else f" lead={s.lead:+.0f}"
            print(f"  {s.slot}@{s.at:8.2f} {bolt:>28}{aimed_at}{extra}{lead} -> {s.outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
