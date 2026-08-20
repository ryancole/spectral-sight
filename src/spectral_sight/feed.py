"""The live output seam: one self-contained envelope per frame, fanned out.

`export.py` answers "what happened", after the fact, to one reader at a time.
This module answers "what is happening" to whoever is listening right now, and
the difference dictates the shape. A timeline row can leave frame-level facts
implicit in its neighbours; a message on a wire cannot, because the consumer
may have missed the neighbours. So the unit here is the *frame envelope*: one
message carrying everything the pipeline concluded about one instant, useful
on its own, and losing one costs a tenth of a second rather than a desync.

**The rows inside the envelope are the timeline's rows, byte for byte.** The
envelope wraps `Observation.to_dict()` output unmodified, so a consumer needs
one row schema whether it is tailing a file or a socket, and a recorded
timeline can be replayed through the live channel and produce what the live
run would have. The moment the two formats drift, every downstream tool needs
two readers and the recorded clips stop being test fixtures for the live path.

**Frame-level facts are lifted to the envelope**, not removed from the rows:
the clock, the death count, and the health of the feed itself. A consumer
should not have to read a champion to learn the time, and `fps`, `dropped` and
`lag` exist because a reactor must be able to tell "no enemies visible" from
"the vision process is wedged" -- the same distinction the timeline spends the
`has_*` flags on, applied to liveness instead of calibration.

**`seq` is the identity of a frame, not `video_time`.** Consumers resume,
detect gaps and deduplicate by it. Float equality across a process boundary is
a bug waiting for a slow afternoon, and every resume mechanism worth having
wants an integer.

A `Sink` is anything that accepts the stream: the JSONL file (`JsonlSink`,
which *is* the existing `TimelineWriter` behind the seam -- the file written
with `--export` is unchanged to the byte), stdout (`StdoutSink`, so
`watch.py --export - | your-tool` works with no server at all), and later an
HTTP server. `FanOut` lets a run feed several at once, because keeping the
file while serving the socket is the normal case, not a special one.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Protocol

from spectral_sight.export import Observation, TimelineMeta, TimelineWriter

if TYPE_CHECKING:
    from pathlib import Path

    from spectral_sight.pipeline import PipelineResult
    from spectral_sight.types import Frame


@dataclass(frozen=True, slots=True)
class FrameState:
    """Everything the pipeline concluded about one instant, in one message.

    Idempotent by construction: no field means "as before", so a consumer that
    joins late or misses a message is behind, never wrong. That property is
    what makes a lossy transport acceptable, and it is cheaper to keep than to
    retrofit.
    """

    seq: int
    """Monotonic from zero within a run. The resume key: a consumer that saw
    `seq` N and receives N+2 knows it missed exactly one frame."""

    video_time: float
    """Seconds since the start of the source, as on every timeline row."""

    captured_at: float | None
    """Wall-clock arrival of the frame at the capture layer, epoch seconds.
    None for a recorded clip. This is the only time in the envelope another
    process can compare against its own clock, which makes it the key that
    joins this feed to an OBS recording, a second capture, or a latency
    measurement -- `video_time` joins to nothing outside this run."""

    game_time: int | None
    game_time_observed: bool
    """Match time, as on the rows, lifted so a consumer need not read a
    champion to learn the clock."""

    allies_dead: int | None
    """The HUD's death count, lifted for the same reason -- and because it is
    frame-level truth that happens to be stored on rows: it stays meaningful
    on a frame whose rows cannot name the casualty."""

    champions: list[Observation]
    """One row per confirmed track, exactly as the timeline records them."""

    fps: float | None
    """Processed frames per second, measured over the last few seconds.
    None until there is enough history to measure."""

    dropped: int
    """Frames the source produced that the pipeline never saw, cumulative.
    Zero for a file source, which waits. A rising value is the feed saying it
    is describing moments the game has already moved past."""

    lag: float | None
    """Seconds between the frame arriving and this envelope being built.
    None when there is no `captured_at` to measure from. This is the feed's
    own contribution to staleness; a consumer adds its transport on top."""

    @classmethod
    def of(
        cls,
        result: PipelineResult,
        frame: Frame,
        *,
        seq: int,
        fps: float | None = None,
        dropped: int = 0,
        now: float | None = None,
    ) -> FrameState:
        """Build the envelope for one processed frame.

        `now` is the wall clock and exists to be injected by tests; production
        callers leave it to default to the moment of construction, which is as
        close to publication as makes no difference.
        """
        clock = result.clock
        liveness = result.liveness
        captured_at = frame.captured_at
        lag = None
        if captured_at is not None:
            lag = max(0.0, (time.time() if now is None else now) - captured_at)
        return cls(
            seq=seq,
            video_time=frame.timestamp,
            captured_at=captured_at,
            game_time=None if clock is None else clock.total_seconds,
            game_time_observed=clock is not None and clock.observed,
            allies_dead=None if liveness is None else liveness.dead_count,
            champions=list(result.observations),
            fps=fps,
            dropped=dropped,
            lag=lag,
        )

    def to_dict(self) -> dict[str, object]:
        """The wire form, discriminated by `t` like every message on the feed.

        Rounded to what the numbers carry, as the rows are. The rows keep
        their own frame-level copies of `game_time` and `allies_dead` rather
        than having them stripped: deduplicating would fork the row schema,
        and the whole point is that there is one.
        """
        return {
            "t": "frame",
            "seq": self.seq,
            "video_time": round(float(self.video_time), 3),
            "captured_at": (
                None if self.captured_at is None
                else round(float(self.captured_at), 3)
            ),
            "game_time": self.game_time,
            "game_time_observed": self.game_time_observed,
            "allies_dead": self.allies_dead,
            "fps": None if self.fps is None else round(float(self.fps), 1),
            "dropped": int(self.dropped),
            "lag": None if self.lag is None else round(float(self.lag), 3),
            "champions": [row.to_dict() for row in self.champions],
        }


class RateMeter:
    """Frames per second over a sliding window, for the envelope's `fps`.

    A whole-run average would answer a different question: a pipeline that ran
    at 15 fps for four minutes and is wedged now reads healthy on the average
    and dead on the window, and the window is the one a live consumer needs.
    """

    def __init__(self, window: float = 5.0) -> None:
        self.window = window
        self._ticks: deque[float] = deque()

    def tick(self, now: float | None = None) -> float | None:
        """Record one processed frame; report the current rate.

        None until two ticks span a measurable interval, because a rate
        invented from one sample would be exactly the kind of confident noise
        the rest of this project exists to avoid.
        """
        if now is None:
            now = time.perf_counter()
        self._ticks.append(now)
        floor = now - self.window
        while self._ticks and self._ticks[0] < floor:
            self._ticks.popleft()
        if len(self._ticks) < 2 or self._ticks[-1] <= self._ticks[0]:
            return None
        return (len(self._ticks) - 1) / (self._ticks[-1] - self._ticks[0])


class Sink(Protocol):
    """Anything that accepts the live stream.

    A context manager plus `publish`, and nothing else: a sink that needs the
    pipeline's internals is a sink that will break when they change, which is
    the coupling the envelope exists to prevent.
    """

    def __enter__(self) -> Sink: ...

    def __exit__(self, *exc: object) -> None: ...

    def publish(self, state: FrameState) -> None: ...


class JsonlSink:
    """The timeline file, fed from the envelope stream.

    A thin adapter over `TimelineWriter`, deliberately: the file this writes
    with a given meta and set of rows is byte-identical to what `--export`
    wrote before the seam existed. The envelope's frame-level fields are not
    recorded here because the rows already carry the durable ones -- `fps` and
    `lag` describe the *run*, not the game, and a timeline is about the game.
    """

    def __init__(self, path: str | Path, meta: TimelineMeta) -> None:
        self._writer = TimelineWriter(path, meta)

    @property
    def path(self) -> Path:
        return self._writer.path

    @property
    def rows(self) -> int:
        return self._writer.rows

    def __enter__(self) -> JsonlSink:
        self._writer.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._writer.__exit__(*exc)

    def publish(self, state: FrameState) -> None:
        self._writer.write(state.champions)


class StdoutSink:
    """Envelopes as JSON lines on stdout, for `--export - | your-tool`.

    The cheapest possible consumer boundary: no port, no dependency, works
    from any language that can read lines. The first line is the meta header
    wrapped as `{"t": "meta", ...}` so a reader can discriminate every line on
    `t` alone, unlike the file format, whose bare header predates the feed and
    is pinned by compatibility.
    """

    def __init__(self, meta: TimelineMeta, stream: IO[str] | None = None) -> None:
        self.meta = meta.stamped()
        self._stream = sys.stdout if stream is None else stream

    def __enter__(self) -> StdoutSink:
        self._write({"t": "meta", **self.meta.to_dict()})
        return self

    def __exit__(self, *exc: object) -> None:
        """Nothing to release: the stream belongs to the caller."""

    def publish(self, state: FrameState) -> None:
        self._write(state.to_dict())

    def _write(self, message: dict[str, object]) -> None:
        # Flush per message for the same reason TimelineWriter flushes per
        # frame: a consumer on the other end of a pipe is reacting in real
        # time, and a message sitting in a buffer is a message that has not
        # been sent, whatever this process thinks.
        self._stream.write(json.dumps(message) + "\n")
        self._stream.flush()


class FanOut:
    """Several sinks behind one, so a run can keep the file while serving.

    Publish order is construction order and failures are not isolated: a sink
    that raises stops the frame reaching later sinks and ends the run. That is
    the right default for the current pair -- a file that cannot be written or
    a broken stdout pipe both mean the run has lost its audience -- and sinks
    that should degrade instead of stop (a network one) can catch their own.
    """

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = list(sinks)

    def __enter__(self) -> FanOut:
        opened: list[Sink] = []
        try:
            for sink in self.sinks:
                sink.__enter__()
                opened.append(sink)
        except BaseException:
            for sink in reversed(opened):
                sink.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        # Every sink gets its close, whatever the earlier ones did with
        # theirs; the last error is the one that propagates.
        error: BaseException | None = None
        for sink in reversed(self.sinks):
            try:
                sink.__exit__(*exc)
            except BaseException as failure:
                error = failure
        if error is not None:
            raise error

    def publish(self, state: FrameState) -> None:
        for sink in self.sinks:
            sink.publish(state)

    def __len__(self) -> int:
        return len(self.sinks)
