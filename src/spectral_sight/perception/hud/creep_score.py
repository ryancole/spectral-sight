"""The local player's creep score: the minion count in the top-right score bar.

It is the ground truth for last-hitting. The client credits a kill to the
player exactly when their hit is the one that killed it, and this number is
the only place that fact is displayed -- a minion's bar vanishing only says it
died, not who killed it. `screen/last_hits.py` joins the two.

**No calibration of its own.** The score bar is one right-aligned line: kills,
K/D/A, creep score, clock. The clock is already calibrated, so the creep score
is found at a fixed offset to its left -- measured on the 2116 and 2117 wide
captures, where the clock sits at the same place.

**The clock's digits read it.** Same face, same size; only the colour differs
(the count is yellow where the clock is white), so the glyph set the clock
taught itself is reused with the clock's saturation ceiling lifted. The gold
minion icon to the left is outside the box.

Measured over 15 minutes of the 2026-08-30 session at 2 Hz: 92% of samples
read, and the only errors were two readings where a 1 came back as a 4 --
"14" between an 11 and a 12. `CreepScoreFilter` is what makes that harmless:
the count only ever rises, by a few at a time, and a new value is adopted only
once two readings agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from spectral_sight.perception.hud.clock import (
    ClockConfig,
    ClockReader,
    GlyphSet,
    segment_glyphs,
)

BOX_FROM_CLOCK = (-106, -3, 60, 6)
"""(dx, dy, width, extra height) of the creep score box relative to the clock
region's top-left and height. Wide enough for three digits."""


@dataclass(frozen=True, slots=True)
class CreepScoreReader:
    """Frame in, creep score out -- or None when the box does not resolve."""

    x: int
    y: int
    width: int
    height: int
    glyphs: GlyphSet
    config: ClockConfig = replace(ClockConfig(), max_saturation=255)

    @classmethod
    def beside(cls, clock: ClockReader) -> CreepScoreReader:
        """The reader for the score bar the clock is part of."""
        dx, dy, width, extra = BOX_FROM_CLOCK
        region = clock.region
        return cls(
            x=region.x + dx,
            y=region.y + dy,
            width=width,
            height=region.height + extra,
            glyphs=clock.glyphs,
        )

    def read(self, frame: np.ndarray) -> int | None:
        strip = frame[self.y : self.y + self.height, self.x : self.x + self.width]
        if strip.size == 0:
            return None
        cfg = self.config
        digits = []
        for canvas in segment_glyphs(strip, self.glyphs.size, cfg):
            label, score, margin = self.glyphs.match(canvas)
            if score < cfg.min_score or margin < cfg.min_margin:
                return None
            if not label.isdigit():
                return None
            digits.append(label)
        if not digits or len(digits) > 3:
            return None
        return int("".join(digits))


class CreepScoreFilter:
    """Turns raw readings into a count that only moves when it really did.

    A reading one above the count is adopted once the next one agrees with it;
    a bigger rise, up to `max_step` (a wave cleared at once is six or seven),
    needs four agreeing readings, because a misread digit can repeat. Anything else is held against the count
    until `resync` agreeing readings say the count itself was wrong (a
    reconnect, a seek, a first reading that was a misread).
    """

    def __init__(self, max_step: int = 8, resync: int = 6) -> None:
        self.max_step = max_step
        self.resync = resync
        self.value: int | None = None
        self._candidate: int | None = None
        self._agree = 0

    def update(self, reading: int | None) -> int | None:
        if reading is None:
            return self.value
        if reading == self._candidate:
            self._agree += 1
        else:
            self._candidate, self._agree = reading, 1
        if reading == self.value:
            return self.value
        plausible = self.value is None or (
            self.value < reading <= self.value + self.max_step
        )
        # A step of one is the common case and two readings settle it. A
        # bigger step is either a wave cleared at once or a misread digit, and
        # a misread can hold for two readings -- measured, "11" read as "14"
        # twice running -- so it has to hold for longer.
        needed = 2 if self.value is None or reading == self.value + 1 else 4
        if (plausible and self._agree >= needed) or self._agree >= self.resync:
            self.value = reading
        return self.value

    def reset(self) -> None:
        self.value = None
        self._candidate = None
        self._agree = 0
