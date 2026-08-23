"""Replay: a timeline served back as the live feed it once was.

What is pinned: the pacing schedule (exactly, via injected clocks), the two
fields a replay is allowed to invent (`captured_at`, `lag`) and the rows it
must not touch, and the seek behaviour -- the skipped past warms the event
deriver's memory without being replayed at the consumer, so what follows the
seek is right *because* of what was skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectral_sight.export import Observation, TimelineMeta, TimelineWriter
from spectral_sight.feed import FrameState
from spectral_sight.replay import replay
from spectral_sight.serve import FeedServer
from spectral_sight.types import Team

META = TimelineMeta(
    source="clip.mp4", width=420, height=400, stride=3,
    created="2026-08-19T00:00:00+00:00",
)


def row(video_time: float, **overrides: object) -> Observation:
    fields: dict = dict(
        video_time=video_time, track_id=1, team=Team.BLUE,
        x=100.0, y=120.0, visible=True, seconds_since_seen=0.0,
        champion="Zilean",
    )
    fields.update(overrides)
    return Observation(**fields)


def timeline(path: Path, frames: list[list[Observation]]) -> Path:
    with TimelineWriter(path, META) as writer:
        for batch in frames:
            writer.write(batch)
    return path


class RecordingSink:
    def __init__(self) -> None:
        self.frames: list[FrameState] = []
        self.events: list[object] = []

    def publish(self, state: FrameState) -> None:
        self.frames.append(state)

    def publish_event(self, event: object) -> None:
        self.events.append(event)


class TestReplay:
    def test_the_game_passes_through_and_the_feed_is_fresh(
        self, tmp_path: Path
    ) -> None:
        """Rows untouched; `captured_at` and `lag` are the replay's own,
        because they describe the feed and the feed is happening now."""
        path = timeline(tmp_path / "run.jsonl", [
            [row(10.0)], [row(10.1)],
        ])
        sink = RecordingSink()
        frames, events = replay(path, sink, speed=None, stamp=lambda: 5000.0)
        assert (frames, events) == (2, 1)  # the identification is the event
        assert [f.video_time for f in sink.frames] == [10.0, 10.1]
        assert all(f.captured_at == 5000.0 for f in sink.frames)
        assert all(f.lag == 0.0 for f in sink.frames)
        assert sink.frames[0].champions == [row(10.0)]

    def test_pacing_follows_the_recordings_own_clock(
        self, tmp_path: Path
    ) -> None:
        """Scheduled against a fixed anchor -- each frame due at
        anchor + elapsed_video / speed -- so jitter cannot accumulate."""
        path = timeline(tmp_path / "run.jsonl", [
            [row(10.0)], [row(10.1)], [row(10.3)],
        ])
        slept: list[float] = []
        replay(path, RecordingSink(), speed=1.0,
               clock=lambda: 0.0, sleep=slept.append)
        assert slept == [pytest.approx(0.1), pytest.approx(0.3)]

    def test_speed_scales_the_schedule(self, tmp_path: Path) -> None:
        path = timeline(tmp_path / "run.jsonl", [
            [row(10.0)], [row(10.1)], [row(10.3)],
        ])
        slept: list[float] = []
        replay(path, RecordingSink(), speed=2.0,
               clock=lambda: 0.0, sleep=slept.append)
        assert slept == [pytest.approx(0.05), pytest.approx(0.15)]

    def test_a_replay_running_late_does_not_sleep(self, tmp_path: Path) -> None:
        """A slow sink eats its own delay; the schedule does not add to it."""
        path = timeline(tmp_path / "run.jsonl", [
            [row(10.0)], [row(10.1)],
        ])
        slept: list[float] = []
        replay(path, RecordingSink(), speed=1.0,
               clock=iter([0.0, 99.0]).__next__, sleep=slept.append)
        assert slept == []

    def test_a_nonsense_speed_is_refused(self, tmp_path: Path) -> None:
        path = timeline(tmp_path / "run.jsonl", [[row(10.0)]])
        with pytest.raises(ValueError, match="speed"):
            replay(path, RecordingSink(), speed=0.0)

    def test_seeking_warms_the_memory_without_replaying_the_past(
        self, tmp_path: Path
    ) -> None:
        """The consumer joining at the seek gets state, not four minutes of
        stale events -- but the events after the seek are right because the
        deriver lived through what was skipped: the respawn knows how long
        the death it never published lasted."""
        path = timeline(tmp_path / "run.jsonl", [
            [row(4.0, alive=True)],
            [row(5.0, alive=False, visible=False)],
            [row(12.0, alive=True)],
        ])
        sink = RecordingSink()
        frames, events = replay(path, sink, speed=None, start=10.0)
        assert frames == 1
        assert [f.video_time for f in sink.frames] == [12.0]
        assert [e.kind for e in sink.events] == ["respawn"]
        assert sink.events[0].detail == {"down_for": 7.0}


class TestServedReplay:
    def test_a_consumer_cannot_tell_it_from_a_live_run(
        self, tmp_path: Path
    ) -> None:
        """Unpaced replay into the real server; a client replaying the ring
        sees every message, frames carrying fresh wall-clock stamps."""
        from tests.test_serve import Stream

        path = timeline(tmp_path / "run.jsonl", [
            [row(10.0)], [row(10.1)], [row(10.2)],
        ])
        with FeedServer(META, port=0) as server:
            frames, events = replay(path, server, speed=None)
            stream = Stream(server, "/stream?since=-1")
            records = stream.read(frames + events)
            stream.close()
        names = [name for _, name, _ in records]
        assert names.count("frame") == 3
        assert names.count("event") == 1
        received = [data for _, name, data in records if name == "frame"]
        assert all(f["captured_at"] is not None for f in received)
        assert all(f["lag"] == 0.0 for f in received)
