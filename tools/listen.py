"""Watch a spectral-sight feed from the outside, with nothing but stdlib.

    python tools/listen.py
    python tools/listen.py --events
    python tools/listen.py http://127.0.0.1:8724 --limit 20

This is the reference consumer: everything a downstream tool needs to speak
the feed is on this page -- open the stream, split records on blank lines,
parse `data:` as JSON, discriminate on `t`. It is also the quickest answer to
"is the feed up", which is why `--limit` exists: two messages prove the whole
path.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Iterator


def messages(url: str) -> Iterator[dict]:
    """SSE records from `url`, each yielded as its parsed `data:` JSON.

    The whole protocol, as a consumer sees it: lines starting with `:` are
    heartbeats, a blank line ends a record, and the server closing the
    stream is the run ending, not an error.
    """
    with urllib.request.urlopen(url) as stream:
        fields: dict[str, str] = {}
        while True:
            raw = stream.readline()
            if raw == b"":
                return
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith(":"):
                continue
            if line == "":
                if "data" in fields:
                    yield json.loads(fields["data"])
                fields = {}
                continue
            key, _, value = line.partition(": ")
            fields[key] = value


def clock(message: dict) -> str:
    seconds = message.get("game_time")
    if seconds is None:
        return f"v{message.get('video_time', 0):.1f}s"
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def describe(message: dict) -> str:
    match message["t"]:
        case "frame":
            champions = message["champions"]
            visible = sum(1 for c in champions if c["visible"])
            lag = message.get("lag")
            health = f"  fps={message['fps'] or '-'}"
            if lag is not None:
                health += f" lag={lag * 1000:.0f}ms"
            if message.get("dropped"):
                health += f" dropped={message['dropped']}"
            return (f"{clock(message):>7}  seq={message['seq']:<6d} "
                    f"tracked={len(champions)} visible={visible}{health}")
        case "event":
            who = message.get("champion") or f"track {message.get('track_id')}"
            return f"{clock(message):>7}  {message['kind']:<10} {who}"
        case "gap":
            return f"         gap: missed messages {message['from']}..{message['to']}"
    return json.dumps(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8723",
                        help="feed base URL (default http://127.0.0.1:8723)")
    parser.add_argument("--events", action="store_true",
                        help="subscribe to /events instead of /stream")
    parser.add_argument("--limit", type=int,
                        help="stop after this many messages")
    args = parser.parse_args()

    url = args.url.rstrip("/") + ("/events" if args.events else "/stream")
    try:
        for count, message in enumerate(messages(url), start=1):
            print(describe(message))
            if args.limit and count >= args.limit:
                break
    except OSError as exc:
        print(f"cannot reach {url}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
