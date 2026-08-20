"""Read casts out of an exported timeline and report what they look like.

    # what the pipeline recorded
    python tools/detect_casts.py --timeline clip.jsonl

    # re-derive with different thresholds, no vision re-run
    python tools/detect_casts.py --timeline clip.jsonl --min-drop 0.05

    # every cast, not just the summary
    python tools/detect_casts.py --timeline clip.jsonl --list

A cast is a step down in a champion's resource bar that holds. Whether that is a
real reading of ability usage or a noise generator is not something the code can
assert about itself, so this prints the things that would show it either way:
how large the drops are per champion, how much of the evidence survives a
continuous follow-up, and how much of it arrives across a gap instead.

The check with teeth is the last one. A champion the HUD says is dead cannot
cast, so any cast landing on a dead champion is a false positive that cost
nothing to find.

Timelines written before casts existed carry no `cast_*` fields. Those are
re-derived from the raw `resource` series, which is the same path any consumer
takes who wants different thresholds -- the series is kept in the file for
exactly that reason.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from spectral_sight.export import Observation, read_timeline
from spectral_sight.perception.nameplates import Cast, CastBook, CastConfig

DEFAULT_ICONS = Path(__file__).resolve().parents[1] / "etc" / "icons"

NO_RESOURCE = {"", "None", "Rage", "Fury", "Energy", "Ferocity", "Heat",
               "Shield", "Blood Well", "Grit", "Flow", "Crimson Rush", "Courage"}
"""Resource types that draw no blue bar, so a champion on one is *expected* to
produce no casts.

Held as a set of names rather than a rule because Data Dragon spells these
however it likes. Checked against patch 16.16.1, where 145 of 173 champions are
on Mana and the other 28 are spread across these thirteen names."""


def resource_types() -> dict[str, str]:
    """Champion -> resource type from the newest icon manifest, if recorded.

    Returns empty when the manifest predates the field, which is not an error:
    it only costs the report its ability to say *why* a champion is silent.
    """
    if not DEFAULT_ICONS.exists():
        return {}
    versions = sorted(p for p in DEFAULT_ICONS.iterdir() if p.is_dir())
    for version in reversed(versions):
        manifest = version / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("resources"):
            return dict(data["resources"])
    return {}


def recorded(rows: list[Observation]) -> list[tuple[Observation, Cast]]:
    """Casts the pipeline already found, lifted back off the rows.

    A cast is recorded on the row where it *settled*, a frame or more after the
    fall, so the levels either side of it are read from the row at `cast_at`
    rather than from the row carrying the fields. Taking them from the carrying
    row would report the value after it had been held and drifted, which is
    close but not the number the detector actually compared.
    """
    at_time = {(row.track_id, row.video_time): row for row in rows}
    found = []
    for row in rows:
        if row.cast_drop is None:
            continue
        at = row.cast_at if row.cast_at is not None else row.video_time
        source = at_time.get((row.track_id, at), row)
        after = source.resource if source.resource is not None else 0.0
        found.append((source, Cast(
            track_id=row.track_id,
            drop=row.cast_drop,
            resource_before=after + row.cast_drop,
            resource_after=after,
            at=at,
            span=row.cast_span or 0.0,
            continuous=bool(row.cast_continuous),
            confirmed=bool(row.cast_confirmed),
            level=row.level,
        )))
    return found


def rederive(
    rows: list[Observation], config: CastConfig
) -> list[tuple[Observation, Cast]]:
    """Run the detector over the raw resource series in the file.

    Rows are fed in time order per track, which the file already is, and the
    row a cast is reported against is the one the drop was measured *to* --
    matching where the pipeline would have put it.
    """
    book = CastBook(config=config)
    at_time: dict[tuple[int, float], Observation] = {}
    found = []
    for row in rows:
        if row.resource is None:
            continue
        at_time[(row.track_id, row.video_time)] = row
        cast = book.update(
            row.track_id, row.video_time, row.resource, row.health, row.level
        )
        if cast is not None:
            found.append((at_time[(cast.track_id, cast.at)], cast))
    for track_id in list(book.detectors):
        cast = book.forget(track_id)
        if cast is not None:
            found.append((at_time[(cast.track_id, cast.at)], cast))
    return found


def name_of(row: Observation) -> str:
    return row.champion or f"track {row.track_id}"


def report(
    rows: list[Observation],
    casts: list[tuple[Observation, Cast]],
    source: str,
) -> int:
    """Print the summary, and return the number of casts on dead champions."""
    readings = sum(1 for row in rows if row.resource is not None)
    span = max((row.video_time for row in rows), default=0.0)
    print(f"{source}: {len(rows)} rows over {span:.1f}s, "
          f"{readings} resource readings, {len(casts)} casts")

    if not casts:
        print("\nNo casts. Either the clip has no readable plates or the "
              "threshold is above every drop in it.")
        return 0

    continuous = sum(1 for _, c in casts if c.continuous)
    confirmed = sum(1 for _, c in casts if c.confirmed)
    print(f"  {continuous} continuous ({continuous / len(casts):.0%}), "
          f"{len(casts) - continuous} across a gap")
    print(f"  {confirmed} confirmed ({confirmed / len(casts):.0%}), "
          f"{len(casts) - confirmed} with no follow-up reading")

    # -- per champion: does the drop size cluster the way repeated use of one
    #    ability should? That clustering is the whole claim.
    print("\ndrop sizes, continuous casts only:")
    by_name: dict[str, list[float]] = collections.defaultdict(list)
    for row, cast in casts:
        if cast.continuous:
            by_name[name_of(row)].append(cast.drop)
    if not by_name:
        print("  (none -- every cast in this clip spans a gap)")
    for name, drops in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        drops.sort()
        median = statistics.median(drops)
        print(f"  {name:<14} n={len(drops):<4} "
              f"{min(drops):5.1%} - {max(drops):5.1%}   median {median:5.1%}")

    # -- silence, split by whether it is expected
    types = resource_types()
    seen = {name_of(row) for row in rows if row.resource is not None}
    cast_names = {name_of(row) for row, _ in casts}
    quiet = sorted(seen - cast_names)
    if quiet:
        print("\nseen but never cast:")
        for name in quiet:
            # An unnamed track has no resource type rather than an empty one,
            # and saying "expected, no blue bar" about it would be an answer
            # made up out of a missing key.
            kind = types.get(name)
            why = ("resource type unknown" if kind is None
                   else "expected, no blue bar" if kind in NO_RESOURCE
                   else "unexplained")
            print(f"  {name:<14} {kind or '?':<10} {why}")

    never = sorted({row.champion for row in rows if row.champion} - seen)
    if never:
        print(f"\nnever had a readable plate: {', '.join(never)}")

    # -- the falsification check
    dead_at: dict[tuple[str, float], bool] = {}
    for row in rows:
        if row.champion and row.alive is not None:
            dead_at[(row.champion, row.video_time)] = row.alive
    impossible = [
        (row, cast) for row, cast in casts
        if row.champion and dead_at.get((row.champion, cast.at)) is False
    ]
    print(f"\ncasts by a champion the HUD called dead: {len(impossible)}"
          + ("  <- these are false positives" if impossible else "  (as it should be)"))
    for row, cast in impossible[:10]:
        print(f"  {name_of(row):<14} {cast.drop:5.1%} at {cast.at:.1f}s")
    return len(impossible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeline", required=True, help="JSONL written by watch.py --export")
    parser.add_argument("--min-drop", type=float, help="override the cast threshold")
    parser.add_argument("--continuity", type=float,
                        help="override the gap treated as consecutive, in seconds")
    parser.add_argument("--rederive", action="store_true",
                        help="ignore recorded casts and run the detector again")
    parser.add_argument("--list", action="store_true", help="print every cast")
    args = parser.parse_args()

    try:
        meta, rows = read_timeline(args.timeline)
    except (OSError, ValueError) as exc:
        print(f"could not read {args.timeline}: {exc}", file=sys.stderr)
        return 1

    if not meta.has_nameplates:
        print("warning: this timeline was extracted without a nameplate "
              "calibration, so it has no resource readings to find casts in",
              file=sys.stderr)

    tuned = args.min_drop is not None or args.continuity is not None
    casts = [] if args.rederive or tuned else recorded(rows)
    if not casts:
        defaults = CastConfig()
        config = CastConfig(
            min_drop=args.min_drop if args.min_drop is not None else defaults.min_drop,
            continuity=(args.continuity if args.continuity is not None
                        else defaults.continuity),
        )
        why = ("re-derived with min_drop=%.3f continuity=%.2fs"
               % (config.min_drop, config.continuity)) if tuned or args.rederive \
            else "re-derived: this timeline has no recorded casts"
        print(f"({why})")
        casts = rederive(rows, config)

    casts.sort(key=lambda pair: pair[1].at)
    if args.list:
        print()
        for row, cast in casts:
            flags = ("continuous" if cast.continuous else f"over {cast.span:.1f}s")
            if not cast.confirmed:
                flags += ", unconfirmed"
            print(f"  {cast.at:7.1f}s  {name_of(row):<14} {cast.drop:5.1%}  "
                  f"{cast.resource_before:.2f} -> {cast.resource_after:.2f}  ({flags})")
        print()

    return 1 if report(rows, casts, Path(args.timeline).name) else 0


if __name__ == "__main__":
    raise SystemExit(main())
