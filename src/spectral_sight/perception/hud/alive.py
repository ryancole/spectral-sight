"""Who on your team is alive, read off the HUD portraits.

This is the fact the timeline could not previously state. An ally is never
hidden by fog, so an ally missing from the minimap is dead -- but nothing
checked that, and a dead ally read as a champion nobody had seen for twenty-four
seconds, which is the same row a fogged enemy produces and means something
entirely different.

**Death is read from the portrait, not from the health bar.** The health bar is
the obvious place to look and it is a trap. When a teammate dies their whole
slot chrome is removed -- portrait ring, bar box, borders -- and the game world
shows through where the box was. So the bar rectangle does not go empty, it
stops existing, and whatever terrain sits behind it is what gets measured.
Measured on real footage: an ally dead for eleven seconds read as *full health*
in the frames where grass happened to be behind the missing box. A reader built
on bar fill would call that champion alive at exactly the moment they were not.

The two slot kinds also fail differently, which rules out fixing it with a
better bar reader. A dead teammate's box vanishes; the local player's box stays
put and reads `0 / 746`. The portrait is the one signal that means the same
thing in both places: a dead champion's portrait is drawn desaturated.

**The threshold is relative to the slot's own history, not absolute.** Portrait
saturation is a property of the champion's art. Measured across the sample
recordings, living slots sit anywhere from 52 to 161 while dead slots read 1 to
29 -- so an absolute floor clearing a dead portrait would sit uncomfortably
close to a legitimately drab champion, and picking it would be fitting a
constant to the five champions that happened to be in these clips. Each slot
therefore carries its own baseline, learned from the footage: the same move as
the clock bootstrapping its digits, and for the same reason. The game is already
telling us what this portrait looks like alive.

The baseline is a running maximum. Death only ever lowers saturation, and a
living slot stays close to its own ceiling -- across three recordings no living
slot fell below 0.78 of its maximum -- so the maximum is a tight upper bound
rather than a high-water mark that drifts away from the living reading.

**Unknown is a reportable answer.** Until a slot has a baseline worth trusting
it reports None rather than guessing, so a clip that opens on a dead champion,
or on a loading screen with no HUD at all, produces no answer instead of a
confident wrong one.

**The baseline is only as good as the frames it was learned from, and the
reader cannot tell those frames apart on its own.** MIN_BASELINE guards the
too-drab direction -- a baseline learned from a corpse -- but a recording that
starts before the game fails in the other one: queue and loading screens put
splash art where the portraits will be, and splash art out-saturates any
living portrait. Measured on a real session, baselines learned in queue sat so
far above the real HUD that every living champion read dead from the first
in-game frame -- permanently, because a running maximum has no way back down.
Whether the in-game HUD is actually on screen is something only the caller can
know (the pipeline proves it by the match timer resolving), so `read` takes
`learn`: an untrusted frame is still judged, but teaches nothing. `reset`
handles the footage tearing wholesale -- a seek, or a different game spliced
into the same capture -- by starting the evidence over.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spectral_sight.perception.hud.portraits import PortraitLayout

DEAD_FRACTION = 0.5
"""Share of its own baseline a slot must fall below to count as dead.

Measured over 4,900 frames of real footage, the tightest living slot sat at 0.78
of its baseline and the loosest dead one at 0.24. Half is the middle of that
gap rather than a boundary fitted to either edge of it."""

MIN_BASELINE = 40.0
"""Saturation a slot's baseline must reach before it will answer at all.

Guards the case where the baseline itself was learned from a dead portrait,
which would otherwise anchor the slot at a tenth of its true value and read
alive forever after."""


@dataclass(frozen=True, slots=True)
class SlotState:
    """What one HUD portrait says about its champion."""

    slot: str
    """Layout slot name: `ally1`..`ally4`, or `self`."""

    saturation: float
    """Mean saturation inside the portrait disc, this frame."""

    baseline: float
    """The highest saturation this slot has shown, which is what it looks like
    alive."""

    alive: bool | None
    """None while the baseline is not yet trustworthy -- see MIN_BASELINE."""


@dataclass(frozen=True, slots=True)
class Liveness:
    """Every friendly portrait, read on one frame."""

    slots: tuple[SlotState, ...]

    @property
    def dead(self) -> tuple[SlotState, ...]:
        return tuple(s for s in self.slots if s.alive is False)

    @property
    def dead_count(self) -> int | None:
        """How many teammates are dead, or None if any slot could not be read.

        All or nothing deliberately. This count is compared against how many
        ally tracks are missing from the minimap, and a count assembled from
        four readable slots and one unreadable one would be quietly too low --
        which is exactly the way to turn a missing reading into a false death.
        """
        if any(s.alive is None for s in self.slots):
            return None
        return sum(1 for s in self.slots if not s.alive)

    def slot(self, name: str) -> SlotState | None:
        for state in self.slots:
            if state.slot == name:
                return state
        return None


class AliveReader:
    """Reads the friendly portraits frame by frame, learning as it goes.

    Stateful, because the baseline each slot is judged against comes from the
    frames already seen. Feed it frames in order.
    """

    def __init__(
        self,
        layout: PortraitLayout,
        *,
        dead_fraction: float = DEAD_FRACTION,
        min_baseline: float = MIN_BASELINE,
    ) -> None:
        self.layout = layout
        self.dead_fraction = dead_fraction
        self.min_baseline = min_baseline
        self._baselines: dict[str, float] = {}
        self._discs: dict[tuple[int, int], np.ndarray] = {}

    def read(self, frame: np.ndarray, *, learn: bool = True) -> Liveness:
        """Read every friendly portrait on this frame.

        `learn` says whether this frame is allowed to raise the baselines.
        Pass False on a frame not known to show the in-game HUD: the slots
        still answer from what they have already learned, without taking
        whatever is on screen as evidence of what a living portrait looks
        like -- see the module docstring for the session that made this
        necessary.
        """
        states = []
        for name, crop in self.layout.all_crops(frame).items():
            states.append(self._read_slot(name, crop, learn))
        return Liveness(slots=tuple(states))

    def reset(self) -> None:
        """Forget every baseline, for when the footage tears wholesale.

        A seek, or a different game spliced into the same capture, puts a
        different set of portraits in the same boxes, and a running maximum
        learned from art that is no longer on screen can only be wrong in one
        direction: reading living champions as dead."""
        self._baselines.clear()

    @property
    def baselines(self) -> dict[str, float]:
        """What each slot has learned it looks like alive. For inspection."""
        return dict(self._baselines)

    def _read_slot(self, name: str, crop: np.ndarray, learn: bool) -> SlotState:
        saturation = self._disc_saturation(crop)
        baseline = self._baselines.get(name, 0.0)
        if learn and saturation > baseline:
            baseline = saturation
            self._baselines[name] = baseline

        alive: bool | None = None
        if baseline >= self.min_baseline:
            alive = saturation >= baseline * self.dead_fraction
        return SlotState(
            slot=name, saturation=saturation, baseline=baseline, alive=alive
        )

    def _disc_saturation(self, crop: np.ndarray) -> float:
        """Median saturation inside the portrait, excluding its frame.

        A disc rather than the square crop: the corners are whatever the HUD is
        drawn over, which on a player-perspective capture is moving terrain.

        Median rather than mean, because a dead slot is not left blank -- the
        respawn countdown is drawn across it in saturated red. On the local
        player's larger portrait those digits dragged the mean of a grey
        portrait up to within 0.02 of the living threshold, and the reading
        flickered mid-death as the number changed width. They are a small
        minority of the disc, so the median does not see them at all.
        """
        if crop.size == 0:
            return 0.0
        saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]
        return float(np.median(saturation[self._disc(saturation.shape)]))

    def _disc(self, shape: tuple[int, int]) -> np.ndarray:
        """Cached circular mask inset from the crop edge, past the gold ring."""
        cached = self._discs.get(shape)
        if cached is None:
            height, width = shape
            cy, cx = (height - 1) / 2, (width - 1) / 2
            radius = max(min(cy, cx) - RING_INSET, 1.0)
            yy, xx = np.mgrid[:height, :width]
            cached = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius**2
            self._discs[shape] = cached
        return cached


RING_INSET = 4
"""Pixels trimmed off the portrait radius before measuring.

The slot's gold ring is bright and saturated and is drawn at a fixed colour for
every champion, so including it would dilute exactly the signal being measured
-- and it disappears along with the rest of the chrome when the champion dies,
which would make the reading depend on two things at once."""
