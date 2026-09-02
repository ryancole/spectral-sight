"""Find projectile candidates on the world view and report what they are worth.

    python tools/detect_projectiles.py --input "data/your clip.mp4" --from 150 --to 330

Runs at every frame (repeats skipped), stabilises the camera, segments what
moved relative to the ground, tracks it, and keeps the fast, straight, brief
tracks. Then scores the one thing the footage can score without labels: the
local player's own Q and W, timestamped by the ability HUD reader, each launch
a bolt from the player's own nameplate -- so every cast should have a
candidate born beside the plate within half a second. That is recall. The
candidates nobody's cast explains are reported as a rate, not as errors: in
lane most of them are minion and turret bolts, which are projectiles too.

Also reports the stabiliser's own health -- repeat share, camera speed, the
terrain's inlier vote -- and how many candidates were heading for the player,
which is the raw material of a threat.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from spectral_sight.capture import open_source
from spectral_sight.perception.hud.abilities import load_ability_reader
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.nameplates import NameplateLayout, NameplateReader, Side
from spectral_sight.perception.screen import (
    CameraTracker,
    MotionConfig,
    ProjectileConfig,
    ProjectileTracker,
    WorldView,
)

PLATE_ABOVE_MODEL = 95.0
"""Pixels from the nameplate's bar down to the champion model, measured on
the 2026-08-30 clip; the model is what a bolt comes from and goes to."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="seconds")
    parser.add_argument("--to", dest="end", type=float, default=0.0, help="seconds; 0 = end")
    parser.add_argument("--min-speed", type=float, default=None)
    parser.add_argument("--repeat-pixels", type=int, default=None,
                        help="override the repeat floor (changed pixels)")
    parser.add_argument("--no-ghosts", action="store_true",
                        help="disable ghost suppression, to measure its cost")
    parser.add_argument("--list", action="store_true", help="print every candidate")
    parser.add_argument("--sweep", action="store_true",
                        help="re-draw the gate over a grid of thresholds from the same run")
    args = parser.parse_args()

    with open_source(args.input) as probe:
        first = next(iter(probe.frames()), None)
    if first is None:
        raise SystemExit(f"no frames in {args.input}")
    width, height = first.size
    try:
        clock = load_clock_reader(width, height)
    except FileNotFoundError:
        clock = None
    glyphs = None if clock is None else clock.glyphs
    abilities = load_ability_reader(width, height, glyphs)
    layout = NameplateLayout.for_resolution(width, height)
    plates = NameplateReader(layout, glyphs)

    overrides = {}
    if args.min_speed is not None:
        overrides["min_speed"] = args.min_speed
    if args.no_ghosts:
        overrides["suppress_ghosts"] = False
    config = ProjectileConfig(**overrides)
    motion = MotionConfig() if args.repeat_pixels is None else MotionConfig(repeat_pixels=args.repeat_pixels)
    tracker = ProjectileTracker(config=config, camera=CameraTracker(config=motion))
    view = WorldView()
    vx, vy, _, _ = view.box(width, height)

    tracks, casts, anchors = [], [], {}
    repeats = distinct = 0
    speeds, inliers, blobs = [], [], []
    fps = 30.0
    with open_source(args.input, start=int(args.start * fps)) as source:
        for frame in source.frames():
            t = frame.timestamp
            if args.end and t > args.end:
                break
            if abilities is not None:
                for cast in abilities.read(frame.image, t):
                    if cast.slot in ("Q", "W"):
                        casts.append(cast)
                        friendly = [p for p in plates.read(frame.image) if p.side is Side.SELF]
                        if len(friendly) == 1:
                            cx, cy = friendly[0].center
                            anchors[cast.at] = (cx - vx, cy - vy + PLATE_ABOVE_MODEL)
            tracks.extend(tracker.update(frame.image, t))
            m = tracker.last_motion
            if m is not None:
                if m.repeat:
                    repeats += 1
                else:
                    distinct += 1
                    if m.estimated:
                        speeds.append(m.speed)
                        inliers.append(m.inliers)
                    blobs.append(tracker.last_blobs)
    tracks.extend(tracker.flush())
    if abilities is not None:
        casts.extend(c for c in abilities.flush() if c.slot in ("Q", "W"))

    minutes = max(1e-9, (distinct + repeats) / fps / 60)
    print(f"{distinct} distinct frames, {repeats} repeats ({repeats/max(1,distinct+repeats):.0%}), {minutes:.1f} min")
    if speeds:
        print(f"camera: speed px/s p50={np.median(speeds):.0f} p90={np.percentile(speeds,90):.0f}; "
              f"terrain inliers p50={np.median(inliers):.2f} p10={np.percentile(inliers,10):.2f}")
    print(f"blobs/frame p50={np.median(blobs):.0f} p90={np.percentile(blobs,90):.0f}")
    candidates = [t for t in tracks if t.is_projectile(config)]
    print(f"{len(tracks)} tracks, {len(candidates)} candidates ({len(candidates)/minutes:.0f}/min)")
    if candidates:
        sp = [t.speed for t in candidates]
        print(f"  candidate speed p10/p50/p90 = {np.percentile(sp,10):.0f}/{np.median(sp):.0f}/{np.percentile(sp,90):.0f}; "
              f"points p50={np.median([len(t.points) for t in candidates]):.0f}")

    if anchors:
        hits = 0
        misses = []
        for cast in casts:
            a = anchors.get(cast.at)
            if a is None:
                continue
            born = [t for t in candidates
                    if -0.15 <= t.start[0] - cast.at <= 0.6
                    and np.hypot(t.start[1] - a[0], t.start[2] - a[1]) <= 400]
            if born:
                hits += 1
            else:
                misses.append(f"{cast.slot}@{cast.at:.1f}")
        print(f"\nself Q/W casts with an anchor: {len(anchors)}; a candidate launched beside the player: {hits} "
              f"(recall {hits/len(anchors):.0%})")
        if misses:
            print("  missed:", ", ".join(misses))
        anchor = tuple(np.median(list(anchors.values()), axis=0))
        toward = [t for t in candidates
                  if t.approaches(anchor, within=140.0) is not None
                  and np.hypot(t.start[1] - anchor[0], t.start[2] - anchor[1]) > 250]
        print(f"candidates heading for the player from >250px away: {len(toward)} ({len(toward)/minutes:.0f}/min)")

    if anchors and args.sweep:
        # Every track is in hand, so the gate can be re-drawn without another
        # run -- the same move detect_casts makes with --min-drop.
        print("\nsweep: candidates/min and self-cast recall by gate")
        print(f"{'min_speed':>10} {'max_rms':>8} {'cand/min':>9} {'recall':>7}")
        for min_speed in (800.0, 900.0, 1100.0):
            for max_rms in (15.0, 20.0, 25.0, 30.0, 40.0):
                gate = ProjectileConfig(min_speed=min_speed, max_rms=max_rms)
                cands = [t for t in tracks if t.is_projectile(gate)]
                hit = sum(
                    1 for cast in casts if cast.at in anchors and any(
                        -0.15 <= t.start[0] - cast.at <= 0.6
                        and np.hypot(t.start[1] - anchors[cast.at][0],
                                     t.start[2] - anchors[cast.at][1]) <= 400
                        for t in cands)
                )
                print(f"{min_speed:10.0f} {max_rms:8.0f} {len(cands)/minutes:9.0f} "
                      f"{hit:3d}/{len(anchors)}")

    if args.list:
        for t in candidates:
            print(f"  {t.start[0]:8.2f}s ({t.start[1]:5.0f},{t.start[2]:4.0f})->({t.end[1]:5.0f},{t.end[2]:4.0f}) "
                  f"speed={t.speed:5.0f} n={len(t.points)} rms={t.rms:4.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
