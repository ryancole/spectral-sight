"""Holding a champion's level steady across misreads.

The level glyphs are the clock's font at a smaller size, matched after
rescaling, and that costs accuracy. Measured over a coop-vs-AI clip, roughly one
reading in ten comes back '1' when the champion is on 3, 4 or 5. The bad
readings are not short fragments a size check would catch: they are full height
-- 10.1px against 10.2px for a correct digit -- and barely narrower, because a
small glyph rescaled onto a larger canvas genuinely correlates with the wrong
template.

What rejects them is not a better look at the pixels but the observation
`ClockFilter` already rests on: the next value is largely known in advance. A
champion's level starts at 1, never falls, and rises by exactly one at a time. A
'1' arriving in the middle of a run of '3's contradicts that and is dropped.

Measured on 516 transitions from associated plate series, this takes the rate of
readings that decrease or skip from 20.9% to zero -- and it recovers signal
rather than merely suppressing noise, resolving one track's series into a clean
level 4 to level 5 transition that 37 spurious '1's had buried.

As with the clock, sustained disagreement is treated as the world having
changed rather than as noise, so a genuine level-up is adopted once confirmed --
otherwise a single bad frame could pin the level forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LevelFilter:
    """One champion's level, filtered across frames."""

    confirm: int = 2
    """Consecutive agreeing readings needed before a change is accepted."""

    level: int | None = None
    _pending: int | None = None
    _count: int = 0

    def update(self, reading: int | None) -> int | None:
        """Fold one frame's reading in and return the level to trust."""
        if reading is None:
            return self.level

        if self.level is None:
            # Nothing to contradict yet, but still require confirmation so a
            # series cannot start on a misread and then reject the truth.
            if reading == self._pending:
                self._count += 1
                if self._count >= self.confirm:
                    self.level, self._pending, self._count = reading, None, 0
            else:
                self._pending, self._count = reading, 1
            return self.level

        if reading == self.level:
            self._pending, self._count = None, 0
            return self.level

        if reading == self.level + 1:
            # A level-up is the one change that needs no arguing with, but it
            # is still confirmed: a misread landing exactly one above the truth
            # would otherwise advance the level for good.
            self._count = self._count + 1 if self._pending == reading else 1
            self._pending = reading
            if self._count >= self.confirm:
                self.level, self._pending, self._count = reading, None, 0
            return self.level

        # A decrease, or a jump of more than one. Almost always a misread, but
        # tracked anyway so that a level-up missed while the champion was off
        # screen can still be picked up rather than deadlocking the filter.
        self._count = self._count + 1 if self._pending == reading else 1
        self._pending = reading
        if self._count >= self.confirm * 3 and reading > self.level:
            self.level, self._pending, self._count = reading, None, 0
        return self.level


@dataclass(slots=True)
class LevelBook:
    """A `LevelFilter` per track, since levels belong to champions not plates.

    Keyed by track id rather than by plate, because a plate is a per-frame
    observation with no memory -- the whole point of filtering is to accumulate
    across frames, and the track is what persists.
    """

    confirm: int = 2
    filters: dict[int, LevelFilter] = field(default_factory=dict)

    def update(self, track_id: int, reading: int | None) -> int | None:
        state = self.filters.get(track_id)
        if state is None:
            state = self.filters[track_id] = LevelFilter(confirm=self.confirm)
        return state.update(reading)

    def level(self, track_id: int) -> int | None:
        state = self.filters.get(track_id)
        return None if state is None else state.level

    def forget(self, track_id: int) -> None:
        """Drop a track's level, for when the tracker discards the track."""
        self.filters.pop(track_id, None)
