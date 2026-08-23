"""Derive the event log from a recorded timeline, and print it.

    python tools/derive_events.py session.jsonl
    python tools/derive_events.py session.jsonl --json
    python tools/derive_events.py session.jsonl --kind cast,death

This is the offline half of the events design: `EventDeriver` is a pure
function of the rows, so a timeline extracted once can answer "what happened"
without re-running any vision -- and the events it prints here are, by
construction, the ones a live run over the same footage published. That makes
this tool double as the measurement: run it over a clip with known deaths and
casts and count.

`--json` prints the wire form, one event per line, exactly as a live consumer
would receive them on the feed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from spectral_sight.events import EventDeriver
from spectral_sight.feed import read_frames


def clock(game_time: int | None, video_time: float) -> str:
    """Game time when the clip has one, video time marked as such when not."""
    if game_time is None:
        return f"v{video_time:.1f}s"
    return f"{game_time // 60:d}:{game_time % 60:02d}"


def describe(event) -> str:
    who = event.champion or (
        f"track {event.track_id}" if event.track_id is not None else ""
    )
    side = "" if event.team is None else f" ({event.team.value})"
    detail = event.detail
    match event.kind:
        case "cast":
            hedge = "" if detail.get("continuous") else " over a gap"
            firm = "" if detail.get("confirmed") else ", unconfirmed"
            return (f"{who}{side} cast: {detail['drop']:.1%} of pool"
                    f"{hedge}{firm}")
        case "death":
            return f"{who}{side} died"
        case "respawn":
            down = detail.get("down_for")
            for_ = "" if down is None else f" after {down:.1f}s down"
            return f"{who}{side} respawned{for_}"
        case "vanished":
            where = ""
            if "world_x" in detail:
                where = f" at ({detail['world_x']:.0f}, {detail['world_y']:.0f})"
            return f"{who}{side} entered fog{where}"
        case "reappeared":
            gone = detail.get("gone_for")
            for_ = "" if gone is None else f" after {gone:.1f}s"
            return f"{who}{side} reappeared{for_}"
        case "level_up":
            return f"{who}{side} reached level {detail['level']}"
        case "identified":
            tag = " (the local player)" if detail.get("is_self") else ""
            was = detail.get("replaces")
            correction = "" if was is None else f", correcting {was}"
            return f"{who}{side} identified{tag}{correction}"
        case "roster":
            return f"{event.team.value} roster: {', '.join(detail['champions'])}"
    return f"{event.kind} {who}{side} {detail}"  # a kind added after this tool


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("timeline", help="a JSONL timeline written by --export")
    parser.add_argument("--json", action="store_true",
                        help="print the wire form instead of prose")
    parser.add_argument("--kind",
                        help="only these kinds, comma-separated (e.g. cast,death)")
    args = parser.parse_args()

    wanted = None if args.kind is None else set(args.kind.split(","))
    deriver = EventDeriver()
    counts: Counter[str] = Counter()
    frames = 0

    try:
        for state in read_frames(args.timeline):
            frames += 1
            for event in deriver.update(state):
                counts[event.kind] += 1
                if wanted is not None and event.kind not in wanted:
                    continue
                if args.json:
                    print(json.dumps(event.to_dict()))
                else:
                    print(f"{clock(event.game_time, event.video_time):>7}  "
                          f"{event.kind:<10}  {describe(event)}")
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    total = sum(counts.values())
    summary = ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items()))
    print(f"\n{total} events over {frames} frames: {summary or 'none'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
