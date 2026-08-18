"""The local player's own health and mana, as the numbers the client prints.

Every other reading in this project is a *fraction*, because a bar's fill is all
an enemy gives you. The local player's own bars carry the figures as text --
`391 / 664` and `488 / 488` -- and reading them is the only place this project
gets an absolute quantity rather than a ratio.

That is worth having for itself, but it is here for what it settles. The cast
detector infers ability usage from steps in a *fraction* with no denominator and
no ground truth, so its precision could be argued from the shape of the errors
and its recall could not be measured at all. These numbers are the denominator
and the ground truth at once: an exact resource series for one champion on every
frame, against which a fraction-derived cast can be confirmed or found missing.

**Read with the clock's glyph set, rescaled.** The HUD numbers are the timer's
face at a smaller size -- roughly 10px against 15 -- exactly as the champion
level box is, so they are grown to a fixed stroke height and matched against the
templates that already exist rather than needing a set of their own. The stroke
was fitted, not guessed: swept from 10 to 16 over 50 frames, mean match score
peaks sharply at 13 (0.739, against 0.46 at 10 and 0.61 at 16).

**The separator is found by its gaps, not by its shape.** `/` matches a digit
template about as well as a digit does, so it cannot be rejected on score. It
does not have to be: measured across frames, the gaps flanking it run 5-8px
against 1-4px between digits, without overlap. The glyph with the widest pair of
neighbouring gaps is the separator, everything left of it is the current value
and everything right is the maximum -- which works whatever the digit counts,
where splitting at a fixed position would not.

**Checking it needs no labels.** The maximum only changes when the champion
levels or buys an item, so it is near-constant over any short window, and the
two halves agree exactly whenever the player is at full. Both are free
consistency checks over thousands of frames -- the same move the clock makes
with `clock - video_time`.

Only the local player has these. Nothing about this route generalises to an
enemy, whose numbers the client never draws.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.hud.clock import (
    ClockConfig,
    GlyphSet,
    centred,
    glyph_boxes,
    lit_mask,
)

LAYOUT_DIR = Path(__file__).resolve().parents[4] / "etc" / "resources"


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """Thresholds for pulling the numbers off the bars.

    Distinct from `ClockConfig` because the ground is different in kind: the
    timer is grey text on a flat dark panel, while these are white text over a
    bright, saturated bar that is itself half green or blue and half black. What
    survives both is brightness with low saturation, but the cutoffs are not the
    clock's.
    """

    min_value: int = 200
    max_saturation: int = 60
    """White text over a saturated fill. The clock's 110/45 was tuned against a
    dark panel and here it lets the lit part of the bar through, which bridges
    adjacent glyphs into one run."""

    stroke: int = 13
    """Height every glyph is grown to before matching.

    Fitted by sweeping 10-16 over 50 frames and taking the peak in mean match
    score. The digits themselves stand about 10-11px."""

    min_glyph_width: int = 1
    min_glyph_pixels: int = 4
    min_glyph_height: int = 4
    """Looser than the clock's, because these glyphs are smaller. The
    consistency checks downstream are what catch a bad segmentation, not this."""

    min_score: float = 0.45
    """Floor on the best correlation. Deliberately below the clock's 0.55: a
    rescaled glyph correlates worse than a native one, the same allowance the
    level reader makes."""

    min_margin: float = 0.04
    """How far the best match must beat the runner-up.

    The load-bearing threshold, and leaving it out was this reader's second bug.
    Rescaled to 13px a `9` and a `4` correlate almost identically -- measured on
    one frame the true `9` scored 0.56 and the `4` it was misread as scored
    0.54, so no floor on score separates them. The margins do: 0.045 against
    0.008. Without this the tens digit flickers between the two, and since the
    digits differ by five that invents a fifty-unit fall and a fifty-unit
    recovery on alternating frames -- which reads exactly like a cast."""

    max_digits: int = 5
    """Longest plausible number. Health passes 10,000 late in a game; five
    digits is clear of anything real and rejects a run of noise."""


@dataclass(frozen=True, slots=True)
class ResourceLayout:
    """Where the two lines of text sit, for one resolution.

    Hand-calibrated like every other HUD geometry here, because the numbers are
    fixed for a resolution and not derivable from it.
    """

    health: tuple[int, int, int, int]
    mana: tuple[int, int, int, int]
    """(x, y, width, height) of each line's strip, in frame pixels. Generous
    boxes are fine and preferred -- the reader finds the glyphs within them, and
    a tight box that clips a digit at a higher resolution is the failure that
    actually happens."""

    def to_dict(self) -> dict[str, list[int]]:
        return {"health": list(self.health), "mana": list(self.mana)}

    @classmethod
    def from_dict(cls, data: dict) -> ResourceLayout:
        return cls(
            health=tuple(int(v) for v in data["health"]),  # type: ignore[arg-type]
            mana=tuple(int(v) for v in data["mana"]),  # type: ignore[arg-type]
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ResourceLayout:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(cls, width: int, height: int) -> ResourceLayout | None:
        path = LAYOUT_DIR / f"{width}x{height}.json"
        return cls.load(path) if path.exists() else None


@dataclass(frozen=True, slots=True)
class Reading:
    """One `current / maximum` pair off one bar."""

    current: int
    maximum: int

    @property
    def fraction(self) -> float | None:
        """The same quantity a nameplate gives, for comparing the two.

        None when the maximum is zero, which is a misread rather than a state
        the game can be in."""
        return None if self.maximum <= 0 else self.current / self.maximum

    @property
    def plausible(self) -> bool:
        """Current within maximum, and the maximum not absurd.

        The cheapest of the consistency checks and the only one available from a
        single frame. It rejects a transposed digit often enough to be worth the
        line -- a misread `488` as `4881` fails it against a max of 488."""
        return 0 <= self.current <= self.maximum and 0 < self.maximum < 100_000

    def __str__(self) -> str:
        return f"{self.current}/{self.maximum}"


@dataclass(frozen=True, slots=True)
class PlayerResources:
    """What the player's own bars said on one frame."""

    health: Reading | None
    mana: Reading | None
    """None for a line that could not be read. The two are independent: a spell
    effect drawn over one does not touch the other, and a champion with no mana
    has no second line at all."""


class ResourceReader:
    """Frame in, the player's own numbers out."""

    def __init__(
        self,
        layout: ResourceLayout,
        glyphs: GlyphSet,
        config: ResourceConfig | None = None,
    ) -> None:
        self.layout = layout
        self.glyphs = glyphs
        self.config = config or ResourceConfig()
        self._clock_config = ClockConfig(
            min_value=self.config.min_value,
            max_saturation=self.config.max_saturation,
            min_glyph_width=self.config.min_glyph_width,
            min_glyph_pixels=self.config.min_glyph_pixels,
        )

    def read(self, frame: np.ndarray) -> PlayerResources:
        return PlayerResources(
            health=self.read_line(frame, self.layout.health),
            mana=self.read_line(frame, self.layout.mana),
        )

    def read_line(
        self, frame: np.ndarray, box: tuple[int, int, int, int]
    ) -> Reading | None:
        """One `current / maximum` pair, or None if the line is unreadable."""
        x, y, width, height = box
        strip = frame[y : y + height, x : x + width]
        if strip.size == 0:
            return None

        glyphs, boxes = self._glyphs(strip)
        if len(glyphs) < 3:
            # Two digits and a separator is the shortest real line.
            return None

        # The separator is located before anything is judged on its score,
        # because `/` is not a digit and has no business being held to a digit
        # template. Scoring it and rejecting the line was this reader's first
        # bug: mana passed only because `/` happened to correlate with a 7.
        split = self._separator(boxes)
        if split is None:
            return None

        left, right = glyphs[:split], glyphs[split + 1 :]
        if not left or not right:
            return None
        if len(left) > self.config.max_digits or len(right) > self.config.max_digits:
            return None
        if any(score < self.config.min_score for _, score in left + right):
            return None
        if not all(label.isdigit() for label, _ in left + right):
            return None

        reading = Reading(
            current=int("".join(label for label, _ in left)),
            maximum=int("".join(label for label, _ in right)),
        )
        return reading if reading.plausible else None

    def _glyphs(
        self, strip: np.ndarray
    ) -> tuple[list[tuple[str, float]], list[tuple[int, int, int, int]]]:
        """Every glyph in the strip, grown to a common stroke and matched.

        Scores come back with the labels rather than being applied here, so the
        caller can hold the digits to a threshold without holding the separator
        to one. Nothing is dropped: a number with a character silently missing
        from the middle is wrong in a way that still looks like a number.
        """
        cfg = self.config
        mask = lit_mask(strip, self._clock_config)
        grey = np.where(mask, cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), 0)
        grey = grey.astype(np.uint8)

        found: list[tuple[str, float]] = []
        kept: list[tuple[int, int, int, int]] = []
        for gx, gy, gw, gh in glyph_boxes(strip, self._clock_config):
            if gh < cfg.min_glyph_height:
                continue
            scale = cfg.stroke / gh
            grown = cv2.resize(
                grey[gy : gy + gh, gx : gx + gw],
                (max(1, round(gw * scale)), cfg.stroke),
                interpolation=cv2.INTER_CUBIC,
            )
            label, score, margin = self.glyphs.match(centred(grown, self.glyphs.size))
            # A glyph the templates cannot separate sinks the whole line. A
            # number with one digit quietly wrong still looks like a number,
            # and downstream there is nothing to catch it with.
            if margin < cfg.min_margin and label.isdigit():
                return [], []
            found.append((label, score))
            kept.append((gx, gy, gw, gh))
        return found, kept

    @staticmethod
    def _separator(boxes: list[tuple[int, int, int, int]]) -> int | None:
        """Index of the `/`, found by the gaps either side of it.

        It cannot be found by matching, because a diagonal stroke correlates
        with a digit template about as well as a digit does. It does not need
        to be: the gaps flanking it measure 5-8px against 1-4px between digits,
        and taking the widest pair works for any pair of digit counts.
        """
        if len(boxes) < 3:
            return None
        gaps = [
            boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])
            for i in range(len(boxes) - 1)
        ]
        best, best_index = -1, None
        for index in range(1, len(boxes) - 1):
            total = gaps[index - 1] + gaps[index]
            if total > best:
                best, best_index = total, index
        return best_index


def load_resource_reader(
    width: int, height: int, glyphs: GlyphSet | None
) -> ResourceReader | None:
    """The reader for a resolution, or None if it is not calibrated.

    Needs the clock's glyph set as well as its own geometry, so a run with no
    clock calibration gets no player numbers either -- there is nothing to match
    against. That is the same dependency the champion level box has.
    """
    layout = ResourceLayout.for_resolution(width, height)
    if layout is None or glyphs is None:
        return None
    return ResourceReader(layout, glyphs)


@dataclass(slots=True)
class MaximumFilter:
    """A pool's maximum, held steady across misreads.

    The maximum is the most constrained quantity on the HUD: it moves only when
    the champion levels or buys, so across any short window it is a constant
    being read repeatedly. That makes it the easiest thing here to get right and
    the most damaging to get wrong, because it is the denominator -- and because
    anything comparing two readings has to decide whether a change between them
    was real.

    Leaving it unfiltered cost a real cast. Validating the detector, a genuine
    fall of 48 mana was discarded because the maximum beside it read 502 on one
    frame and 501 on the next, and the check that excludes level-ups saw a
    maximum that had moved. One digit of flicker, and the ground truth quietly
    dropped an event it existed to catch.

    So a change is adopted only once it has been seen `confirm` times running,
    the same argument `LevelFilter` makes about a champion's level and
    `ClockFilter` about the timer.
    """

    confirm: int = 3
    maximum: int | None = None
    _pending: int | None = None
    _count: int = 0

    def update(self, reading: int | None) -> int | None:
        """Fold one frame's maximum in and return the value to trust."""
        if reading is None:
            return self.maximum
        if reading == self.maximum:
            self._pending, self._count = None, 0
            return self.maximum
        self._count = self._count + 1 if reading == self._pending else 1
        self._pending = reading
        if self._count >= self.confirm:
            self.maximum, self._pending, self._count = reading, None, 0
        return self.maximum
