"""A recorded timeline, played back as if it were happening now.

To a consumer, a replay is a live feed: the same envelopes, the same events,
the same pacing, on the same server. That is the point. The downstream tool
gets developed and regression-tested against the clips in `data/` -- with
their known deaths, known casts and known rosters -- and nothing it can
observe distinguishes that from a game in progress, so nothing it learns
against a replay is unlearned against the real thing. No League client, no
capture, no vision running: a five-minute clip replays on a laptop that is
busy doing other work.

The determinism the events module bought is what makes this trustworthy
rather than merely convenient: events are re-derived from the rows on the
way out, and Phase 2 measured that derivation byte-identical to what the
live run published. A replay is not a recording of the feed; it is the feed,
recomputed from the same facts.

Two fields are the replay's own, and they are exactly the ones that describe
the feed rather than the game. `captured_at` is stamped fresh at publish,
because the replay *is* happening now and a consumer's latency arithmetic
should work unchanged against it. `lag` is zero for the same reason. The
rows -- the game -- pass through untouched.

Seeking (`start`) fast-forwards the event deriver through the skipped rows
*silently*: its memory arrives warm, so the first published frames carry
correct current knowledge -- who is identified, who is dead -- without four
minutes of stale events being replayed at the consumer. What was true
before the seek is state, readable from `/state`; only what changes after
it is news.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from spectral_sight.events import EventDeriver
from spectral_sight.feed import read_frames

if TYPE_CHECKING:
    from spectral_sight.feed import Sink


def replay(
    path: str | Path,
    sink: Sink,
    *,
    speed: float | None = 1.0,
    start: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    stamp: Callable[[], float] = time.time,
) -> tuple[int, int]:
    """Publish a timeline through an already-opened sink. Blocks until done.

    `speed` scales the recording's own pace -- 1.0 is real time, 4.0 is four
    times it, None is as fast as the sink accepts. `start` seeks to a
    video_time in seconds. Returns (frames, events) published.

    Pacing is scheduled against a fixed anchor rather than frame-to-frame --
    each frame is due at `anchor + elapsed_video / speed` -- so sleep jitter
    does not accumulate: a five-minute replay ends five minutes after it
    began, not five minutes plus three hundred rounding errors.

    The clocks are injectable so tests assert exact schedules instead of
    bounding real ones, the same trade `ScriptedMailbox` makes.
    """
    if speed is not None and speed <= 0:
        raise ValueError(f"speed must be positive or None, got {speed}")

    deriver = EventDeriver()
    frames = events = 0
    anchor: tuple[float, float] | None = None
    """(wall, video_time) of the first published frame, the pacing origin."""

    for state in read_frames(path):
        if state.video_time < start:
            # Warm the deriver's memory without publishing: the skipped past
            # is state, not news.
            deriver.update(state)
            continue

        if speed is not None:
            if anchor is None:
                anchor = (clock(), state.video_time)
            else:
                due = anchor[0] + (state.video_time - anchor[1]) / speed
                delay = due - clock()
                if delay > 0:
                    sleep(delay)

        sink.publish(replace(state, captured_at=stamp(), lag=0.0))
        frames += 1
        for event in deriver.update(state):
            sink.publish_event(event)
            events += 1

    return frames, events
