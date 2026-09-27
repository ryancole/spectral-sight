"""Where a frame's time goes, stage by stage.

A live run drops the frames it cannot keep up with, and the drop count says
*that* the pipeline is too slow but not *where*. The answer decides what to do
about it: a stage whose time is spent inside OpenCV gets faster by doing less
image work, and one whose time is spent in Python gets faster by moving it out
of Python. Guessing wrong costs a rewrite that buys nothing.

The timer is lap-based: `Pipeline.process` calls `lap(stage)` after each stage,
and the time since the previous lap is charged to it. Stages run in a fixed
order within a frame, so a lap needs no nesting and costs one `perf_counter`
call. A pipeline without a timer skips even that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageStats:
    total: float = 0.0
    """Seconds, summed over every frame the stage ran on."""
    calls: int = 0
    worst: float = 0.0
    """The slowest single run, in seconds. A mean hides the spikes that
    actually cause drops."""


@dataclass
class StageTimer:
    stages: dict[str, StageStats] = field(default_factory=dict)
    frames: int = 0
    """Calls to `process`, sampled or not."""
    sampled: int = 0
    """Frames that ran the minimap stages as well."""
    frame_total: float = 0.0
    frame_worst: float = 0.0
    _start: float = 0.0
    _last: float = 0.0

    def start(self) -> None:
        self._start = self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        spent = now - self._last
        self._last = now
        stats = self.stages.get(stage)
        if stats is None:
            stats = self.stages[stage] = StageStats()
        stats.total += spent
        stats.calls += 1
        if spent > stats.worst:
            stats.worst = spent

    def finish(self, *, sampled: bool) -> None:
        spent = time.perf_counter() - self._start
        self.frames += 1
        self.sampled += sampled
        self.frame_total += spent
        if spent > self.frame_worst:
            self.frame_worst = spent

    def report(self) -> str:
        """A table of stages, most expensive first, in milliseconds.

        `per frame` divides by every frame processed, so the column sums to
        the mean frame time and a stage that runs only on sampled frames is
        weighed by what it costs the run, not by what one run of it costs.
        """
        if not self.frames:
            return "no frames timed"
        rows = sorted(self.stages.items(), key=lambda kv: kv[1].total, reverse=True)
        width = max(len(name) for name in self.stages) if self.stages else 5
        lines = [
            f"{self.frames} frames ({self.sampled} sampled), "
            f"mean {1000 * self.frame_total / self.frames:.1f} ms, "
            f"worst {1000 * self.frame_worst:.1f} ms",
            f"{'stage':<{width}}  {'per frame':>9}  {'per call':>9}  "
            f"{'worst':>8}  {'share':>6}",
        ]
        for name, stats in rows:
            lines.append(
                f"{name:<{width}}  "
                f"{1000 * stats.total / self.frames:8.2f}ms  "
                f"{1000 * stats.total / max(stats.calls, 1):8.2f}ms  "
                f"{1000 * stats.worst:6.1f}ms  "
                f"{stats.total / self.frame_total:6.1%}"
            )
        return "\n".join(lines)
