"""The live output seam: the frame envelope and the sinks it fans out to.

Two properties carry the design and both are pinned here. The envelope is
self-contained -- everything a consumer needs about one instant, including the
health of the feed itself, in one message. And the rows inside it are the
timeline's rows byte for byte: `JsonlSink` fed through the seam must produce
the same file `TimelineWriter` produced before the seam existed, or every
recorded clip quietly stops being a test fixture for the live path.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from spectral_sight.export import Observation, TimelineMeta, TimelineWriter
from spectral_sight.feed import (
    FanOut,
    FrameState,
    JsonlSink,
    RateMeter,
    StdoutSink,
    read_frames,
)
from spectral_sight.perception.hud.alive import Liveness, SlotState
from spectral_sight.perception.hud.clock import GameClock
from spectral_sight.pipeline import PipelineResult
from spectral_sight.types import Frame, Team

META = TimelineMeta(
    source="clip.mp4", width=420, height=400, stride=3,
    created="2026-08-19T00:00:00+00:00",
)
"""`created` is pinned so two writers given this meta emit identical bytes."""


def observation(track_id: int = 1, **overrides: object) -> Observation:
    fields: dict = dict(
        video_time=12.3, track_id=track_id, team=Team.BLUE,
        x=100.0, y=120.0, visible=True, seconds_since_seen=0.0,
        game_time=200, game_time_observed=True, champion="Zilean",
    )
    fields.update(overrides)
    return Observation(**fields)


def liveness(dead: int) -> Liveness:
    states = tuple(
        SlotState(slot=f"ally{i}", saturation=10.0 if i < dead else 90.0,
                  baseline=100.0, alive=i >= dead)
        for i in range(5)
    )
    return Liveness(slots=states)


def frame(timestamp: float = 12.3, captured_at: float | None = None) -> Frame:
    import numpy as np
    return Frame(image=np.zeros((4, 6, 3), np.uint8), index=0,
                 timestamp=timestamp, captured_at=captured_at)


class TestFrameState:
    def test_lifts_the_frame_level_facts_out_of_the_rows(self) -> None:
        """A consumer should not have to read a champion to learn the clock."""
        result = PipelineResult(
            observations=[observation(1), observation(2)],
            clock=GameClock(total_seconds=200, confidence=0.9, observed=True),
            liveness=liveness(dead=1),
        )
        state = FrameState.of(result, frame(), seq=7, fps=9.8, dropped=3)
        assert state.seq == 7
        assert state.game_time == 200
        assert state.game_time_observed is True
        assert state.allies_dead == 1
        assert state.champions == result.observations
        assert state.dropped == 3

    def test_a_bare_result_answers_none_rather_than_inventing(self) -> None:
        state = FrameState.of(PipelineResult(), frame(), seq=0)
        assert state.game_time is None
        assert state.game_time_observed is False
        assert state.allies_dead is None
        assert state.fps is None
        assert state.lag is None

    def test_a_carried_clock_is_not_reported_as_observed(self) -> None:
        result = PipelineResult(
            clock=GameClock(total_seconds=201, confidence=0.5, observed=False),
        )
        state = FrameState.of(result, frame(), seq=0)
        assert state.game_time == 201
        assert state.game_time_observed is False

    def test_lag_is_measured_from_arrival(self) -> None:
        state = FrameState.of(
            PipelineResult(), frame(captured_at=1000.0), seq=0, now=1000.25,
        )
        assert state.lag == pytest.approx(0.25)

    def test_lag_never_goes_negative_across_skewed_clocks(self) -> None:
        """Wall clocks step; a consumer must never see time flow backwards."""
        state = FrameState.of(
            PipelineResult(), frame(captured_at=1000.0), seq=0, now=999.9,
        )
        assert state.lag == 0.0

    def test_a_recorded_clip_has_no_lag_to_measure(self) -> None:
        state = FrameState.of(PipelineResult(), frame(captured_at=None), seq=0)
        assert state.captured_at is None
        assert state.lag is None

    def test_the_envelope_wraps_the_rows_unmodified(self) -> None:
        """The one-schema promise: the rows on the wire are the file's rows."""
        rows = [observation(1), observation(2, champion=None, visible=False,
                                            seconds_since_seen=2.4)]
        state = FrameState.of(
            PipelineResult(observations=rows), frame(), seq=3,
        )
        envelope = state.to_dict()
        assert envelope["t"] == "frame"
        assert envelope["champions"] == [row.to_dict() for row in rows]

    def test_the_envelope_rounds_what_it_reports(self) -> None:
        state = FrameState.of(
            PipelineResult(),
            frame(timestamp=1.23456, captured_at=1000.111111),
            seq=0, fps=9.87654, now=1000.998877,
        )
        envelope = state.to_dict()
        assert envelope["video_time"] == 1.235
        assert envelope["captured_at"] == 1000.111
        assert envelope["fps"] == 9.9
        assert envelope["lag"] == 0.888


class TestRateMeter:
    def test_one_sample_is_not_a_rate(self) -> None:
        assert RateMeter().tick(now=100.0) is None

    def test_a_steady_feed_reads_its_rate(self) -> None:
        meter = RateMeter(window=5.0)
        rate = None
        for i in range(11):
            rate = meter.tick(now=100.0 + i * 0.1)
        assert rate == pytest.approx(10.0)

    def test_the_window_forgets_a_faster_past(self) -> None:
        """The whole-run average is the number a wedged pipeline hides behind."""
        meter = RateMeter(window=5.0)
        for i in range(50):
            meter.tick(now=100.0 + i * 0.1)  # 10 fps for 5 seconds
        rate = meter.tick(now=115.0)  # then one frame ten seconds later
        # Every earlier tick has left the window; one sample is not a rate.
        assert rate is None


class TestStdoutSink:
    def build(self) -> tuple[StdoutSink, io.StringIO]:
        stream = io.StringIO()
        return StdoutSink(META, stream), stream

    def test_every_line_is_discriminated_json(self) -> None:
        """A reader needs `t` on every line; the file header's bare form is
        pinned by compatibility, but this channel is new and can do better."""
        sink, stream = self.build()
        with sink:
            sink.publish(FrameState.of(
                PipelineResult(observations=[observation()]), frame(), seq=0,
            ))
        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert [line["t"] for line in lines] == ["meta", "frame"]
        assert lines[0]["schema"] == META.schema
        assert lines[1]["champions"][0]["champion"] == "Zilean"

    def test_stamps_an_unstamped_meta(self) -> None:
        stream = io.StringIO()
        unstamped = TimelineMeta(source="x.mp4", width=1, height=1, stride=1)
        with StdoutSink(unstamped, stream):
            pass
        header = json.loads(stream.getvalue().splitlines()[0])
        assert header["created"]


class TestJsonlSink:
    def test_writes_the_file_the_writer_wrote(self, tmp_path: Path) -> None:
        """Byte for byte. The seam must be invisible in the artefact, or the
        live path and the offline path fork into two formats."""
        rows = [
            [observation(1), observation(2)],
            [observation(1, video_time=12.4), observation(2, video_time=12.4)],
        ]
        direct = tmp_path / "direct.jsonl"
        with TimelineWriter(direct, META) as writer:
            for batch in rows:
                writer.write(batch)

        seamed = tmp_path / "seamed.jsonl"
        with JsonlSink(seamed, META) as sink:
            for seq, batch in enumerate(rows):
                sink.publish(FrameState.of(
                    PipelineResult(observations=batch), frame(), seq=seq,
                ))

        assert seamed.read_bytes() == direct.read_bytes()
        assert sink.rows == 4


class TestReadFrames:
    def test_rows_regroup_into_the_frames_that_wrote_them(
        self, tmp_path: Path
    ) -> None:
        """The file is rows; the feed was frames. Rows sharing a video_time
        were one frame, and the frame-level facts lift back off any row."""
        path = tmp_path / "run.jsonl"
        with TimelineWriter(path, META) as writer:
            writer.write([observation(1), observation(2)])
            writer.write([
                observation(1, video_time=12.4, game_time=201),
                observation(2, video_time=12.4, game_time=201),
                observation(3, video_time=12.4, game_time=201),
            ])

        frames = list(read_frames(path))
        assert [f.seq for f in frames] == [0, 1]
        assert [len(f.champions) for f in frames] == [2, 3]
        assert [f.video_time for f in frames] == [12.3, 12.4]
        assert [f.game_time for f in frames] == [200, 201]
        # A file read at leisure has no feed health to report.
        assert frames[0].captured_at is None
        assert frames[0].fps is None
        assert frames[0].lag is None


class RecordingSink:
    """Remembers what happened to it, for the fan-out tests."""

    def __init__(self, fail_on_enter: bool = False) -> None:
        self.fail_on_enter = fail_on_enter
        self.events: list[object] = []

    def __enter__(self) -> RecordingSink:
        if self.fail_on_enter:
            raise RuntimeError("would not open")
        self.events.append("open")
        return self

    def __exit__(self, *exc: object) -> None:
        self.events.append("close")

    def publish(self, state: FrameState) -> None:
        self.events.append(state.seq)

    def publish_event(self, event: object) -> None:
        self.events.append(("event", event))


class TestFanOut:
    def state(self, seq: int = 0) -> FrameState:
        return FrameState.of(PipelineResult(), frame(), seq=seq)

    def test_every_sink_sees_every_frame(self) -> None:
        first, second = RecordingSink(), RecordingSink()
        with FanOut([first, second]) as feed:
            feed.publish(self.state(0))
            feed.publish(self.state(1))
        assert first.events == ["open", 0, 1, "close"]
        assert second.events == ["open", 0, 1, "close"]

    def test_a_sink_that_cannot_open_closes_the_ones_that_did(self) -> None:
        opened = RecordingSink()
        with pytest.raises(RuntimeError, match="would not open"):
            with FanOut([opened, RecordingSink(fail_on_enter=True)]):
                pass  # pragma: no cover - never reached
        assert opened.events == ["open", "close"]

    def test_reports_how_many_are_listening(self) -> None:
        assert len(FanOut([])) == 0
        assert len(FanOut([RecordingSink()])) == 1
