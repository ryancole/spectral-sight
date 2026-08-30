"""The HTTP feed: endpoints, the stream, resume, and the backpressure rule.

Exercised over real sockets on an ephemeral port rather than by calling
handler methods, because the properties worth pinning live in the plumbing:
that a subscriber starts live, that a reconnect resumes, that falling off the
ring is reported rather than silently absorbed -- and above all that a stalled
client costs the publisher nothing, which is the one promise the vision loop
is owed.
"""

from __future__ import annotations

import http.client
import json
import time

import pytest

from spectral_sight.events import Event
from spectral_sight.export import TimelineMeta
from spectral_sight.feed import FrameState
from spectral_sight.serve import FeedServer

META = TimelineMeta(
    source="clip.mp4", width=420, height=400, stride=3,
    created="2026-08-19T00:00:00+00:00",
)


def state(seq: int) -> FrameState:
    return FrameState(
        seq=seq, video_time=seq / 10, captured_at=None, game_time=None,
        game_time_observed=False, allies_dead=None, champions=[], fps=None,
        dropped=0, lag=None,
    )


def event(seq: int) -> Event:
    return Event(
        kind="cast", seq=seq, video_time=seq / 10, game_time=None,
        team=None, champion="Xerath", track_id=1, detail={"drop": 0.1},
    )


@pytest.fixture
def server():
    """A served feed on an OS-assigned port, torn down after the test."""
    with FeedServer(META, port=0) as served:
        yield served


def get(server: FeedServer, path: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return {
            "status": response.status,
            "body": json.loads(response.read()),
        }
    finally:
        connection.close()


class Stream:
    """One SSE subscription, reading records on demand.

    `getresponse()` does not return until the handler has sent its headers,
    and the handler picks its cursor before that -- so once construction
    finishes, everything published afterwards is guaranteed delivered here.
    """

    def __init__(
        self, server: FeedServer, path: str = "/stream",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", server.port, timeout=5
        )
        self.connection.request("GET", path, headers=headers or {})
        self.response = self.connection.getresponse()
        assert self.response.status == 200

    def read(self, count: int) -> list[tuple[int | None, str, dict]]:
        """The next `count` records as (id, event name, parsed data),
        skipping heartbeat comments."""
        records: list[tuple[int | None, str, dict]] = []
        fields: dict[str, str] = {}
        while len(records) < count:
            line = self.response.fp.readline().decode("utf-8").rstrip("\n")
            if line.startswith(":"):
                continue
            if line == "":
                if "data" in fields:
                    records.append((
                        None if "id" not in fields else int(fields["id"]),
                        fields.get("event", "message"),
                        json.loads(fields["data"]),
                    ))
                fields = {}
                continue
            key, _, value = line.partition(": ")
            fields[key] = value
        return records

    def at_end(self) -> bool:
        """True when the server has closed the stream."""
        return self.response.fp.readline() == b""

    def close(self) -> None:
        self.connection.close()


class TestEndpoints:
    def test_meta_is_the_capability_header(self, server: FeedServer) -> None:
        reply = get(server, "/meta")
        assert reply["status"] == 200
        assert reply["body"] == server.meta.to_dict()

    def test_state_is_honest_about_having_seen_nothing(
        self, server: FeedServer
    ) -> None:
        assert get(server, "/state")["body"]["frame"] is None

    def test_state_is_the_latest_frame_plus_the_meta(
        self, server: FeedServer
    ) -> None:
        server.publish(state(0))
        server.publish(state(1))
        body = get(server, "/state")["body"]
        assert body["frame"]["seq"] == 1
        assert body["meta"] == server.meta.to_dict()

    def test_the_root_serves_the_dashboard(self, server: FeedServer) -> None:
        """One self-contained page speaking the same four endpoints as any
        other consumer -- the browser is the proof of the cross-language
        claim, so it must arrive with no build step and no network."""
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.port, timeout=5
        )
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            page = response.read().decode("utf-8")
        finally:
            connection.close()
        assert response.status == 200
        assert "text/html" in response.getheader("Content-Type")
        # It consumes the public endpoints, not some private channel.
        assert "/stream" in page and "/meta" in page
        # And it carries no external references to break offline. Loopback is
        # not external: the coach fallback names another process on the same
        # machine, and a URL that never leaves it cannot need the network.
        external = page.replace("http://localhost:", "").replace(
            "http://127.0.0.1:", ""
        )
        assert "http://" not in external and "https://" not in external

    def test_an_unknown_path_is_a_404(self, server: FeedServer) -> None:
        assert get(server, "/nope")["status"] == 404

    def test_a_malformed_since_is_a_400(self, server: FeedServer) -> None:
        assert get(server, "/stream?since=then")["status"] == 400


class TestStream:
    def test_messages_arrive_in_order_with_their_ids(
        self, server: FeedServer
    ) -> None:
        stream = Stream(server)
        server.publish(state(0))
        server.publish_event(event(0))
        server.publish(state(1))
        records = stream.read(3)
        assert [(id, name) for id, name, _ in records] == [
            (0, "frame"), (1, "event"), (2, "frame"),
        ]
        assert records[0][2]["t"] == "frame"
        assert records[1][2] == event(0).to_dict()
        stream.close()

    def test_a_fresh_subscriber_starts_live_not_in_the_past(
        self, server: FeedServer
    ) -> None:
        server.publish(state(0))
        server.publish(state(1))
        stream = Stream(server)
        server.publish(state(2))
        id, name, data = stream.read(1)[0]
        assert (id, data["seq"]) == (2, 2)
        stream.close()

    def test_since_replays_what_the_ring_still_holds(
        self, server: FeedServer
    ) -> None:
        for seq in range(4):
            server.publish(state(seq))
        stream = Stream(server, "/stream?since=1")
        records = stream.read(2)
        assert [id for id, _, _ in records] == [2, 3]
        stream.close()

    def test_a_reconnecting_client_resumes_by_last_event_id(
        self, server: FeedServer
    ) -> None:
        """What a browser EventSource sends unasked on reconnect, so protocol-
        level resume costs a consumer nothing."""
        for seq in range(3):
            server.publish(state(seq))
        stream = Stream(server, headers={"Last-Event-ID": "0"})
        records = stream.read(2)
        assert [id for id, _, _ in records] == [1, 2]
        stream.close()

    def test_falling_off_the_ring_is_a_gap_then_the_newest(self) -> None:
        """Latest-wins with an honest account: the consumer is told exactly
        which ids it lost, and the next thing it holds is now -- not a backlog
        it will fall further behind on."""
        with FeedServer(META, port=0, capacity=4) as server:
            for seq in range(10):
                server.publish(state(seq))
            stream = Stream(server, "/stream?since=0")
            gap, newest = stream.read(2)
            assert gap[1] == "gap"
            assert gap[2] == {"t": "gap", "from": 1, "to": 8}
            assert newest[0] == 9
            stream.close()

    def test_events_endpoint_carries_only_the_changes(
        self, server: FeedServer
    ) -> None:
        stream = Stream(server, "/events")
        server.publish(state(0))
        server.publish_event(event(0))
        server.publish(state(1))
        server.publish_event(event(1))
        records = stream.read(2)
        assert [name for _, name, _ in records] == ["event", "event"]
        assert [data["seq"] for _, _, data in records] == [0, 1]
        stream.close()


class TestBackpressure:
    def test_a_stalled_client_costs_the_publisher_nothing(
        self, server: FeedServer
    ) -> None:
        """The promise the vision loop is owed: publishing is an append to
        memory, so a consumer that stops reading -- or fifty of them -- can
        only go stale, never push back."""
        stalled = Stream(server)  # connected, and never reads
        started = time.perf_counter()
        for seq in range(5000):
            server.publish(state(seq))
        elapsed = time.perf_counter() - started
        assert elapsed < 2.0
        stalled.close()


class TestListen:
    def test_the_reference_consumer_speaks_the_protocol(
        self, server: FeedServer
    ) -> None:
        """`tools/listen.py` is the copyable example, so it is held to the
        real wire: a subprocess, a real socket, and both message shapes."""
        import subprocess
        import sys
        from pathlib import Path

        tool = Path(__file__).resolve().parents[1] / "tools" / "listen.py"
        listener = subprocess.Popen(
            [sys.executable, str(tool), server.url, "--limit", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            # No handshake to wait on: ?since is not used, but the ring holds
            # these two messages far longer than the subprocess takes to
            # connect... except a fresh subscriber starts live, so publish
            # until it has had time to attach.
            import time
            for seq in range(50):
                server.publish(state(seq))
                server.publish_event(event(seq))
                time.sleep(0.1)
                if listener.poll() is not None:
                    break
            out, err = listener.communicate(timeout=10)
        finally:
            listener.kill()
        assert listener.returncode == 0, err
        lines = out.splitlines()
        assert len(lines) == 2
        assert any("seq=" in line for line in lines)


class TestLifecycle:
    def test_shutdown_ends_the_stream_rather_than_hanging_it(self) -> None:
        with FeedServer(META, port=0) as server:
            stream = Stream(server)
            server.publish(state(0))
            assert stream.read(1)[0][0] == 0
        assert stream.at_end()
        stream.close()

    def test_the_port_the_os_picked_is_reported(self) -> None:
        with FeedServer(META, port=0) as server:
            assert server.port != 0
            assert str(server.port) in server.url
