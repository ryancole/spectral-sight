"""The level-up chevrons: whether the local player has a skill point to spend.

While a skill point is unspent the client draws a row of chevron buttons
directly above the ability slots -- one per ability the point could go into,
gold on a teal panel -- and removes the row the moment it is spent. Reading
that row answers two things a coach wants and the nameplate level cannot
give: that a point is *waiting*, and which abilities could take it. R lights
only at 6, 11 and 16; an ability at its cap for the level never lights; so
the lit set is exactly the choice the player has.

**The signal is the chevron's shape, gated by its colour.** The row is
transparent when nothing is lit -- the world shows through the bar -- so the
plain "is anything bright there" reading fires on whatever walks past: a
fireball measured 0.22 gold fraction in an unlit box, and the player's own
buff icons overlap the Q box's left edge and read gold for seconds at a
stretch. The chevron itself, though, is a fixed drawing: correlating each
box against a captured lit chevron (`etc/abilities/chevron.png`, grey,
rescaled to the box) measured 0.60-0.97 on every lit box across sixteen
level-ups of the 2026-08-30 session and at most 0.42 on the world showing
through -- a gap with nothing in it. The one thing that shares the shape is
the chevron's own *unavailable* state, drawn dark grey where an ability
cannot be levelled (R before 6), which scored up to 0.57 at the start of the
game. It has no gold in it at all -- 0.000 measured -- where every lit box
has at least 0.064 of its pixels in the gold band, so the gold floor tells
the two apart with a factor of two to spare either way.

**A slot changes on two agreeing readings; the row changes after half a
second.** Spending a point flashes the chosen slot white for a frame, which
reads as neither state, so a slot adopts a new state only once two
consecutive readings agree. The row itself slides in left to right and out
right to left over a few readings at 10 Hz -- measured, a level-up arrived
as `Q`, then `QW`, then `QWE` on successive readings, once with `Q` alone
holding for two, and left as `QE` then nothing -- so the *set* is reported
only once it has held unchanged for `settle` seconds, which no intermediate
set does. Half a second of latency against a point that sits unspent for
one to thirteen seconds, and `held_for` is measured arrival to spend so the
two lags cancel.

Only the local player has this row. It is read alongside the cooldown veil
and gated the same way: not on a frame that is not the game, not while
dead.

Measured on the 2026-08-30 session (a human Ezreal): every level-up from 1
through 16 produced a lit window, each was spent within 1-13 seconds, and
nothing outside those windows read lit. The sets matched the rank rules --
`WER` at 6 and 11 with Q capped, `W` alone at 14 and 15 with Q and E capped,
`WR` at 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.hud.abilities import (
    ABILITY_SLOTS,
    LAYOUT_DIR,
    AbilityLayout,
)

TEMPLATE_PATH = LAYOUT_DIR / "chevron.png"
"""One lit chevron box, grey, cropped from the reference resolution. Shared
by every resolution: the box is rescaled to fit, the way the clock's glyphs
are."""


@dataclass(frozen=True, slots=True)
class SkillPointConfig:
    """Thresholds for reading a chevron box."""

    lit_above: float = 0.55
    unlit_below: float = 0.45
    """Correlation with the chevron template at which a box reads lit, and
    below which it reads unlit; between the two a reading abstains. Lit
    boxes measured 0.60 and up, the world showing through 0.42 and under,
    so the band sits in the gap -- and it holds the dimmed-out chevron
    (0.31-0.57) only because the gold floor below removes it first."""

    gold_hue: tuple[int, int] = (15, 35)
    gold_saturation: int = 90
    gold_value: int = 150
    gold_min: float = 0.03
    """The chevron's gold, and the share of the box that must be in it for
    a lit reading. Lit boxes measured 0.064 at their dimmest (the glow
    pulses, and the trough is what counts); the unavailable chevron, drawn
    grey, measured 0.000 on every frame it was visible."""

    confirm: int = 2
    """Consecutive readings that must agree before a slot's state changes.
    Two clears the spend flash; see the module docstring."""

    settle: float = 0.5
    """Seconds a changed set of slots must hold before it is reported. The
    slide-in and slide-out measured up to 0.3s of intermediate sets; the
    row then stays for seconds, so the hold costs a real change half a
    second and an intermediate set is never reported at all."""


@dataclass(slots=True)
class _SlotState:
    lit: bool | None = None
    """Adopted state; None until the first confirmed reading."""
    candidate: bool | None = None
    votes: int = 0
    """The state the last readings have been arguing for, and how many in
    a row have."""


def load_chevron_template(path: str | Path = TEMPLATE_PATH) -> np.ndarray:
    """The reference lit chevron, as a grey image."""
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"no chevron template at {path}")
    return template


class SkillPointReader:
    """Frame in, the set of slots with a lit chevron out. Stateful per slot;
    feed it frames in order."""

    def __init__(
        self,
        layout: AbilityLayout,
        template: np.ndarray,
        config: SkillPointConfig | None = None,
    ) -> None:
        boxes = layout.point_boxes()
        if boxes is None:
            raise ValueError("ability layout has no level-up row calibrated")
        self.layout = layout
        self.config = config or SkillPointConfig()
        self._boxes = boxes
        self._slots = {name: _SlotState() for name in ABILITY_SLOTS}
        self._templates: dict[tuple[int, int], np.ndarray] = {}
        self._template = template
        self._reported: tuple[str, ...] | None = None
        self._settling: tuple[str, ...] | None = None
        self._settling_since = 0.0

    @property
    def learnable(self) -> tuple[str, ...] | None:
        """Slots whose chevron is lit, or None until the row has settled."""
        return self._reported

    def _adopted(self) -> tuple[str, ...] | None:
        """What the slots say right now, or None while any is unsettled."""
        if any(state.lit is None for state in self._slots.values()):
            return None
        return tuple(name for name in ABILITY_SLOTS if self._slots[name].lit)

    def read(self, frame: np.ndarray, timestamp: float) -> tuple[str, ...] | None:
        """Fold one frame in, returning the confirmed lit set afterwards."""
        self._read_slots(frame)
        current = self._adopted()
        if current is None or current == self._reported:
            self._settling = None
        elif current != self._settling:
            self._settling, self._settling_since = current, timestamp
        elif timestamp - self._settling_since >= self.config.settle:
            self._reported = current
            self._settling = None
        return self._reported

    def _read_slots(self, frame: np.ndarray) -> None:
        for name in ABILITY_SLOTS:
            x, y, width, height = self._boxes[name]
            crop = frame[y : y + height, x : x + width]
            if crop.shape[0] != height or crop.shape[1] != width:
                continue
            verdict = self._classify(crop)
            state = self._slots[name]
            if verdict is None:
                # An abstention neither advances nor breaks a candidacy: a
                # box mid-transition is still headed where it was headed.
                continue
            if verdict == state.lit:
                state.candidate, state.votes = None, 0
                continue
            if state.candidate == verdict:
                state.votes += 1
            else:
                state.candidate, state.votes = verdict, 1
            if state.votes >= self.config.confirm:
                state.lit = verdict
                state.candidate, state.votes = None, 0

    def reset(self) -> None:
        """Forget every slot, for a torn stream or a death: the next
        `learnable` waits for fresh confirmed readings."""
        for state in self._slots.values():
            state.lit = None
            state.candidate = None
            state.votes = 0
        self._reported = None
        self._settling = None

    def _classify(self, crop: np.ndarray) -> bool | None:
        """True lit, False unlit, None when the reading abstains."""
        cfg = self.config
        score = self._score(crop)
        if score < cfg.unlit_below:
            return False
        if score < cfg.lit_above:
            return None
        return self._gold_fraction(crop) >= cfg.gold_min

    def _score(self, crop: np.ndarray) -> float:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(grey.std()) < 1.0:
            return 0.0  # a flat box correlates with nothing
        template = self._template_for(crop.shape[1], crop.shape[0])
        return float(
            cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        )

    def _template_for(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._templates.get(key)
        if cached is None:
            cached = self._template
            if cached.shape[:2] != (height, width):
                cached = cv2.resize(
                    cached, (width, height), interpolation=cv2.INTER_AREA
                )
            self._templates[key] = cached
        return cached

    def _gold_fraction(self, crop: np.ndarray) -> float:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        low, high = self.config.gold_hue
        return float(np.mean(
            (hue >= low) & (hue <= high)
            & (saturation >= self.config.gold_saturation)
            & (value >= self.config.gold_value)
        ))


def load_skill_point_reader(layout: AbilityLayout) -> SkillPointReader | None:
    """The reader for a layout, or None if its level-up row or the chevron
    template is not there -- either way nothing looks, and the rows say so
    by leaving `learnable` unset."""
    if layout.point_boxes() is None or not TEMPLATE_PATH.exists():
        return None
    return SkillPointReader(layout, load_chevron_template())
