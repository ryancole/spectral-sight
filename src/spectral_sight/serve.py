"""The feed over HTTP: any language, any process, `curl` as the debugger.

This is the consumer boundary the whole output design was aiming at. The file
answers questions after the run; stdout reaches exactly one program that had
to be started as a pipe; this reaches anything that can open a socket, joins
mid-game, and survives its consumers crashing. Server-Sent Events rather than
WebSockets because the stream is strictly one-way and SSE is plain HTTP --
no dependency on either end, resume built into the protocol, and
`curl localhost:8723/stream` shows the live feed with no tooling at all.

Four endpoints, one schema:

- `GET /meta`   -- the capability header. A consumer must be able to tell "no
                   nameplate calibration" from "a quiet game", or it will read
                   a missing `resource` as an enemy who never cast.
- `GET /state`  -- the latest frame envelope, plus the meta. What a late
                   joiner reads first, and what a poller reads instead of
                   streaming.
- `GET /stream` -- every message as it happens: `frame` and `event`, each an
                   SSE record whose `data:` is exactly the wire form the
                   stdout sink writes. One format to learn, three transports.
- `GET /events` -- the same stream with the frames filtered out, for a
                   consumer that wants ten messages a minute instead of ten a
                   second.

**The pipeline thread never touches a socket.** `publish` serialises, appends
to a shared ring, notifies, returns; per-client handler threads do all the
waiting and writing. This is the `Mailbox` rule applied at the other end of
the process: a frame that waits for a slow consumer is a frame describing a
fight that has already resolved, so the vision loop must not be able to feel
the network at all -- not a stalled client, not a dead one, not fifty of them.

**One ring, one backpressure rule.** Every message gets a monotonic id -- the
SSE `id:`, distinct from a frame's `seq` because events share their frame's
seq and a resume key must be unique. A client consumes at its own cursor; one
that falls off the back of the ring gets `{"t": "gap", "from": N, "to": M}`
and resumes at the *newest* message, not the oldest kept. Latest-wins, again:
a consumer that cannot keep up gets a bounded-staleness feed and an honest
account of what it missed, rather than a backlog it will fall further behind
on, and `GET /state` is always there to resync from. A client that asks for
recent history it can still have -- `Last-Event-ID` on reconnect, or
`?since=<id>` -- gets it replayed in full, no gap, because catch-up over a
few seconds is what the ring is *for*.

Bound to 127.0.0.1 explicitly. This is a localhost tool; a remote consumer is
a different design with authentication in it, and pretending otherwise adds
surface for no user.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from spectral_sight.export import TimelineMeta

if TYPE_CHECKING:
    from spectral_sight.events import Event
    from spectral_sight.feed import FrameState

DEFAULT_PORT = 8723
HEARTBEAT = 15.0
"""Seconds of silence before a comment line goes down the stream, so a dead
TCP connection is discovered in seconds rather than at the OS's leisure."""

CAPACITY = 1024
"""Messages the ring keeps -- roughly a minute of frames and their events at
10 Hz. Enough for a reconnect to catch up over, small enough that a consumer
handed the whole backlog is seconds behind, not minutes."""


@dataclass(frozen=True, slots=True)
class Message:
    """One SSE record: the id is the resume key, the name is `frame` or
    `event`, the data is the same JSON the stdout sink writes."""

    id: int
    name: str
    data: str


class Ring:
    """The shared backlog between one publisher and any number of readers.

    Append never waits and never fails; reading waits. That asymmetry is the
    whole point -- see the module docstring -- and it is why the ring is
    bounded: an unbounded queue would let a slow reader cost memory instead
    of costing itself staleness, which is the same bad trade the capture
    `Mailbox` exists to refuse.
    """

    def __init__(self, capacity: int = CAPACITY) -> None:
        self._messages: deque[Message] = deque(maxlen=capacity)
        self._next_id = 0
        self._condition = threading.Condition()
        self._closed = False

    @property
    def head(self) -> int:
        """Id of the newest message, -1 while empty. A fresh subscriber
        starts here: live from now, backlog only on request."""
        with self._condition:
            return self._next_id - 1

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def append(self, name: str, data: str) -> int:
        with self._condition:
            message = Message(id=self._next_id, name=name, data=data)
            self._next_id += 1
            self._messages.append(message)
            self._condition.notify_all()
            return message.id

    def close(self) -> None:
        """End every subscriber's wait. Readers drain what is buffered and
        then see the stream end, mirroring how the capture mailbox lets the
        last frame out after the window shuts."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def since(
        self, cursor: int, timeout: float
    ) -> tuple[list[Message], tuple[int, int] | None]:
        """Messages after `cursor`, waiting up to `timeout` for one to exist.

        Returns `(messages, lost)`. An empty list means the wait timed out
        (the caller heartbeats) or the ring closed (the caller checks). When
        the cursor has fallen off the back, `lost` carries the inclusive id
        range that is gone and `messages` is just the newest -- the jump the
        module docstring promises. The copy happens under the lock and the
        caller's socket writes do not, so a reader that stalls mid-write
        stalls only itself.
        """
        with self._condition:
            if not self._closed and self._next_id - 1 <= cursor:
                self._condition.wait(timeout)
            if self._next_id - 1 <= cursor:
                return [], None
            oldest = self._messages[0].id
            if cursor + 1 < oldest:
                newest = self._messages[-1]
                return [newest], (cursor + 1, newest.id - 1)
            return [m for m in self._messages if m.id > cursor], None


class FeedServer:
    """The HTTP sink: a context manager over a daemon-threaded server.

    Implements the same `Sink` protocol as the file and stdout, so `watch.py`
    fans out to all three without knowing which is which. Unlike those two it
    swallows nothing and raises nothing on publish -- there is no failure a
    publisher could see, because publishing is an append to memory and every
    fallible part happens on a handler thread that only hurts itself.
    """

    def __init__(
        self,
        meta: TimelineMeta,
        *,
        port: int = DEFAULT_PORT,
        capacity: int = CAPACITY,
    ) -> None:
        self.meta = meta.stamped()
        self.port = port
        self._ring = Ring(capacity)
        self._latest: dict[str, object] | None = None
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> FeedServer:
        self._server = _Server(("127.0.0.1", self.port), _Handler, feed=self)
        # Port 0 asks the OS to pick; report what it picked.
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="feed-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._ring.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def publish(self, state: FrameState) -> None:
        envelope = state.to_dict()
        # The reference swap is atomic; handlers only read the reference and
        # nobody mutates a published dict, so /state needs no lock.
        self._latest = envelope
        self._ring.append("frame", json.dumps(envelope))

    def publish_event(self, event: Event) -> None:
        self._ring.append("event", json.dumps(event.to_dict()))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    """A handler mid-stream must not hold process exit hostage."""

    def __init__(self, address, handler, *, feed: FeedServer) -> None:
        self.feed = feed
        super().__init__(address, handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Silent. Per-request lines on stderr are noise for a feed whose
        clients reconnect as a matter of course."""

    @property
    def feed(self) -> FeedServer:
        return self.server.feed  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        url = urlparse(self.path)
        match url.path:
            case "/meta":
                self._json(self.feed.meta.to_dict())
            case "/state":
                self._json({
                    "meta": self.feed.meta.to_dict(),
                    "frame": self.feed._latest,
                })
            case "/stream":
                self._stream(url, events_only=False)
            case "/events":
                self._stream(url, events_only=True)
            case "/":
                self._dashboard()
            case _:
                self._json({"error": f"no such endpoint: {url.path}"},
                           status=404)

    def _dashboard(self) -> None:
        """The bundled page -- see `dashboard.html` for why it is one file.

        Read per request rather than cached: requests for it are rare, the
        file is small, and editing it against a running server showing live
        data is exactly how it gets worked on.
        """
        payload = (
            Path(__file__).with_name("dashboard.html").read_bytes()
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, body: dict[str, object], status: int = 200) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _cursor(self, url) -> int | None:
        """Where this client wants to start: `?since` wins, then the
        `Last-Event-ID` a reconnecting EventSource sends unasked, then live
        from now. None means a bad request already answered."""
        query = parse_qs(url.query)
        raw = query.get("since", [None])[0] or self.headers.get("Last-Event-ID")
        if raw is None:
            return self.feed._ring.head
        try:
            return int(raw)
        except ValueError:
            self._json({"error": f"since must be an integer, got {raw!r}"},
                       status=400)
            return None

    def _stream(self, url, *, events_only: bool) -> None:
        cursor = self._cursor(url)
        if cursor is None:
            return
        ring = self.feed._ring

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        # The stream is delimited by the connection ending, so it must end.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        try:
            while True:
                messages, lost = ring.since(cursor, timeout=HEARTBEAT)
                if not messages:
                    if ring.closed:
                        return
                    # A comment per SSE spec: ignored by consumers, but it
                    # forces a write, and the write is what discovers that
                    # the far end quietly went away.
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                if lost is not None:
                    gap = json.dumps(
                        {"t": "gap", "from": lost[0], "to": lost[1]}
                    )
                    self.wfile.write(f"event: gap\ndata: {gap}\n\n".encode())
                for message in messages:
                    cursor = message.id
                    # /events still advances past frames, or the ring would
                    # hand them back on every wait; they are skipped, not
                    # unseen.
                    if events_only and message.name != "event":
                        continue
                    self.wfile.write(
                        f"id: {message.id}\nevent: {message.name}\n"
                        f"data: {message.data}\n\n".encode()
                    )
                self.wfile.flush()
        except OSError:
            # The client went away, which is its right and its problem.
            return
