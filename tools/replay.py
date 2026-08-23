"""Serve a recorded timeline as a live feed, paced like the game it was.

    # the sample clip, at real time, on the usual port
    python tools/replay.py session.jsonl

    # four times as fast, on another port
    python tools/replay.py session.jsonl --serve 8724 --speed 4

    # jump to just before the teamfight
    python tools/replay.py session.jsonl --from 260

    # as fast as it will go, and keep serving /state afterwards
    python tools/replay.py session.jsonl --fast --hold

To anything listening this is `watch.py --serve`: the same endpoints, the
same messages, the same events. The difference is what it costs -- no League
client, no capture window, no vision -- which makes it the way the
downstream tool gets built: against a clip whose deaths and casts are known,
repeatably, at whatever speed the work needs. See `spectral_sight/replay.py`
for why a consumer cannot tell the difference.
"""

from __future__ import annotations

import argparse
import sys

from spectral_sight.export import read_meta
from spectral_sight.replay import replay
from spectral_sight.serve import DEFAULT_PORT, FeedServer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("timeline", help="a JSONL timeline written by --export")
    parser.add_argument("--serve", type=int, default=DEFAULT_PORT,
                        metavar="PORT",
                        help=f"port to serve on (default {DEFAULT_PORT})")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback rate; 1.0 is the recording's own pace")
    parser.add_argument("--fast", action="store_true",
                        help="no pacing: publish as fast as consumers accept")
    parser.add_argument("--from", dest="start", type=float, default=0.0,
                        metavar="SECONDS",
                        help="seek to this video time before publishing; the "
                             "skipped past becomes state, not events")
    parser.add_argument("--hold", action="store_true",
                        help="keep serving /state after the timeline ends, "
                             "until Ctrl+C")
    args = parser.parse_args()

    try:
        meta = read_meta(args.timeline)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    speed = None if args.fast else args.speed
    pace = "unpaced" if speed is None else f"{speed:g}x"
    with FeedServer(meta, port=args.serve) as server:
        print(f"replaying {meta.source} ({pace}) at {server.url}/stream")
        try:
            frames, events = replay(
                args.timeline, server, speed=speed, start=args.start,
            )
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"replayed {frames} frames, {events} events")
        if args.hold:
            # The stream is over but the last frame is still the state of
            # the game; a late consumer reading /state gets it until the
            # server is told to stop.
            print("holding; Ctrl+C to stop serving")
            try:
                import threading
                threading.Event().wait()
            except KeyboardInterrupt:
                print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
