"""Measure the cast detector against the numbers the client prints.

    python tools/validate_casts.py --input "data/clip.mp4" --timeline clip.jsonl

Every other check on cast detection argues from the shape of its own errors,
because a fill fraction has nothing to be compared against. The local player is
the exception: the HUD prints their mana as `488 / 488`, so their resource is
known exactly on every frame, and a cast is simply a fall in that number.

That gives both halves of the score. A fall the HUD saw and the detector did not
is a miss; a cast the detector reported with no fall behind it is a false
positive. Neither is available for any other champion, and nothing about this
route generalises to one -- the client never draws an enemy's numbers.

Two kinds of fall are not casts and are excluded. A level-up moves the maximum
and carries the current value with it. Death empties the pool, which looks like
the largest cast imaginable.
"""

from __future__ import annotations

import argparse
import collections
import sys

import cv2

from spectral_sight.export import Observation, read_timeline
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.hud.resources import Reading, load_resource_reader
from spectral_sight.perception.nameplates import Cast, CastBook, CastConfig

MIN_COST = 20
"""Smallest fall in mana counted as a cast, in absolute units. Every ability in
the game costs more than this; regeneration only ever adds."""

DEATH_FRACTION = 0.4
"""A fall of more than this much of the pool in one frame is the champion dying,
not casting. Real abilities top out well below it."""

MATCH_WINDOW = 0.6
"""How far apart a detected cast and a real one may sit and still be the same
event. Generous, because a detected cast is stamped at the reading it was
measured to and the plate is not read every frame."""


def hud_series(path: str, stride: int) -> list[tuple[float, Reading]]:
    """The player's mana off the HUD, one entry per processed frame."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    clock = load_clock_reader(width, height)
    reader = load_resource_reader(
        width, height, None if clock is None else clock.glyphs
    )
    if reader is None:
        capture.release()
        raise SystemExit(
            f"no resource calibration for {width}x{height}, and no clock glyphs "
            "to read one with. Run tools/calibrate_clock.py first."
        )

    out: list[tuple[float, Reading]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            mana = reader.read(frame).mana
            if mana is not None:
                out.append((index / fps, mana))
        index += 1
    capture.release()
    return out


def true_casts(series: list[tuple[float, Reading]]) -> list[tuple[float, int]]:
    """Falls in the player's mana that are neither a level-up nor a death."""
    casts = []
    for (_, before), (time, after) in zip(series, series[1:]):
        if after.maximum != before.maximum:
            continue
        fall = before.current - after.current
        if fall < MIN_COST:
            continue
        if fall >= DEATH_FRACTION * after.maximum:
            continue
        casts.append((time, fall))
    return casts


def detected_casts(rows: list[Observation], config: CastConfig) -> list[Cast]:
    """What the detector makes of the same champion's nameplate series."""
    book = CastBook(config=config)
    found = []
    for row in rows:
        if not row.is_self or row.resource is None:
            continue
        cast = book.update(
            row.track_id, row.video_time, row.resource, row.health, row.level
        )
        if cast is not None:
            found.append(cast)
    for track_id in list(book.detectors):
        cast = book.forget(track_id)
        if cast is not None:
            found.append(cast)
    return sorted(found, key=lambda c: c.at)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="the clip the timeline came from")
    parser.add_argument("--timeline", required=True, help="JSONL from watch.py --export")
    parser.add_argument("--min-drop", type=float, help="override the cast threshold")
    args = parser.parse_args()

    try:
        meta, rows = read_timeline(args.timeline)
    except (OSError, ValueError) as exc:
        print(f"could not read {args.timeline}: {exc}", file=sys.stderr)
        return 1

    series = hud_series(args.input, meta.stride)
    if not series:
        print("the HUD reader found no mana on any frame. Either the player is on "
              "a champion with no mana, or the calibration is wrong.", file=sys.stderr)
        return 1

    truth = true_casts(series)
    config = CastConfig(
        min_drop=args.min_drop if args.min_drop is not None else CastConfig().min_drop
    )
    found = detected_casts(rows, config)

    plates = sum(1 for r in rows if r.is_self and r.resource is not None)
    self_rows = sum(1 for r in rows if r.is_self)
    print(f"{meta.source}: HUD mana read on {len(series)} frames; "
          f"the player's plate on {plates} of {self_rows} rows")
    print(f"real casts (HUD): {len(truth)}     detected (nameplate): {len(found)}")
    if truth:
        costs = collections.Counter(fall for _, fall in truth)
        print("  costs the HUD saw:", ", ".join(
            f"{cost} mana x{n}" for cost, n in sorted(costs.items())))

    matched, used = [], set()
    for cast in found:
        hit = next(
            (i for i, (time, _) in enumerate(truth)
             if i not in used and abs(time - cast.at) <= MATCH_WINDOW),
            None,
        )
        if hit is None:
            continue
        used.add(hit)
        matched.append((cast, truth[hit]))

    if not truth and not found:
        # Not a perfect score -- no score at all. Printing 0/0 as a percentage
        # invites a clip where nobody cast to be quoted as a result.
        print("\nnothing to score: the player did not cast in this clip, and "
              "the detector agreed. That is consistent, not accurate.")
        return 0

    precision = len(matched) / len(found) if found else 0.0
    recall = len(matched) / len(truth) if truth else 0.0
    print(f"\nprecision {len(matched)}/{len(found)} = {precision:.0%}"
          f"     recall {len(matched)}/{len(truth)} = {recall:.0%}")
    if len(truth) < 10:
        print(f"  {len(truth)} real casts is a sample, not a rate. Do not "
              "quote these as accuracy figures.")

    if matched:
        print("\nmatched, detector against HUD:")
        for cast, (time, fall) in matched:
            print(f"  {time:7.1f}s  HUD {fall:4d} mana   detector {cast.drop:5.1%}")

    missed = [truth[i] for i in range(len(truth)) if i not in used]
    if missed:
        # A miss is far more often the plate saying nothing, or saying somebody
        # else's number, than the drop being too small to clear the threshold --
        # so print what the plate actually held at the time.
        by_time = {round(r.video_time, 1): r for r in rows if r.is_self}
        print("\nmissed, and what the plate said:")
        for time, fall in missed:
            row = by_time.get(round(time, 1))
            before = by_time.get(round(time - meta.stride / 30.0, 1))
            state = "no row for this frame"
            if row is not None and row.resource is None:
                state = "plate not readable"
            elif row is not None:
                previous = "?" if before is None or before.resource is None \
                    else f"{before.resource:.3f}"
                state = f"plate held {previous} -> {row.resource:.3f}"
            print(f"  {time:7.1f}s  HUD {fall:4d} mana   {state}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
