"""The local player's gold: the number beside the coin under the inventory.

It is what says whether the player can buy anything. Nothing else on screen
carries it -- the shop greys items out, but only while it is open.

**No calibration of its own.** The gold box is part of the bottom-centre
panel, so it moves with the HUD scale the way the ability slots do (see
`scale.py`). It is found at a fixed offset from the first summoner slot,
measured on the 2116 wide captures, where that slot's top-left is
(1090, 1247) and 40px square: the digits stand 15px tall on rows 1328-1342,
start at x 1235, and are left-aligned -- the first glyph starts at the same
column whether the number has two digits or four. The coin ends at x 1223,
outside the box. The offset is kept in units of the slot's size, so a scaled
or stretched layout carries the box with it.

**The clock's digits read it, rescaled.** Same face, drawn larger (15px against
the clock's 13) and pale yellow rather than white. Each glyph is shrunk to the
clock's height before matching: at native size the templates score 0.5-0.65,
at 13px 0.85-0.94. The saturation ceiling is lifted to 120: at the clock's 45
only a scrap of one glyph survives, at 80 the strokes break up, and from 120
to 255 the glyphs come out whole and the scores stop changing.

**Stray pixels are dropped per glyph.** The box sometimes lights up -- a teal
fill and a bright frame -- and compression then leaves one to three lit pixels
a row or two under the digits, which made a glyph 16-17px tall and threw the
rescale off. Only parts at least `min_part_height` tall count toward a glyph.

**Touching digits are split.** On a lit box neighbouring digits can join on
one row, and a run wider than `max_aspect` times its height is cut in two.
That was most of what went unread: splitting took the read rate on the
footage below from 94.7% to 98.9%.

Measured at 10 Hz over the three 2026-09-25 live captures (9 minutes of
game) and 16 minutes of the 2026-08-30 session, 14,921 in-game frames: 98.9%
read (99.4% on the session, 97.3-99.0% on the captures), and no reading was
wrong. The rest are frames where a glyph falls under the score or margin
floor, and they come back as None, not as a number. Checked with a synthetic
1.25x and 1.5x panel (a real frame scaled about the bottom centre, as the
client does): the scaled box reads the same figures. See `GoldFilter` for
the errors that appear with the floors removed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from spectral_sight.perception.hud.abilities import AbilityLayout
from spectral_sight.perception.hud.clock import (
    ClockConfig,
    GlyphSet,
    centred,
    lit_mask,
)

REFERENCE_SLOT = 40.0
"""The summoner slot's width and height, in pixels, on the captures the box
was measured on. The offsets below are in those pixels."""

BOX_FROM_SUMMONER = (138.0, 79.0, 76.0, 19.0)
"""(dx, dy, width, height) of the gold box from the first summoner slot's
top-left, in reference pixels. Wide enough for five digits, and a few rows clear of
the box's frame above and below."""


@dataclass(frozen=True, slots=True)
class GoldConfig:
    """Thresholds for pulling the number out of the box."""

    lit: ClockConfig = replace(ClockConfig(), max_saturation=120)
    """Brightness floor as the clock's; saturation ceiling lifted for the
    yellow -- see the module docstring for why 120."""

    stroke: int | None = None
    """Height each glyph is grown or shrunk to before matching. None takes
    the digit height in the glyph set -- 13px for the clock's, on every
    resolution calibrated so far -- so the rescaled glyph lands on the
    templates' scale. Also what makes the reader indifferent to the HUD
    scale."""

    min_part_height: int = 5
    """A connected part of a glyph shorter than this is a stray, not a
    stroke. Every stroke of every digit is at least 9px tall at 15px; the
    strays measured were one to three rows."""

    min_score: float = 0.6
    min_margin: float = 0.06
    """Per glyph. On the 14,921 frames measured, genuine digits mostly score
    0.8-0.95 with margins of 0.12 and up. At 0.5 / 0.04 nothing was misread
    either; at 0.4 / 0.02 the first wrong reads appear. These sit clear of
    that at a cost of 0.2% of frames against 0.5 / 0.04."""

    max_digits: int = 5

    max_aspect: float = 1.1
    """A glyph wider than this times its height is two digits touching. The
    widest single digit, a 4, stands 12 wide by 15 tall (0.8); two of the
    narrowest, a 1 at 7 wide, span 19 with the gap between them (1.27)."""

    pitch: float = 0.82
    """Digit advance over digit height: 12.3px at 15px."""


@dataclass(frozen=True, slots=True)
class GoldReader:
    """Frame in, the player's gold out -- or None when the box does not
    resolve to a number."""

    x: int
    y: int
    width: int
    height: int
    glyphs: GlyphSet
    config: GoldConfig = GoldConfig()

    @classmethod
    def beside(cls, layout: AbilityLayout, glyphs: GlyphSet) -> GoldReader:
        """The reader for the panel the ability layout describes. Pass the
        layout at the current HUD scale."""
        dx, dy, width, height = BOX_FROM_SUMMONER
        kx = layout.summoner_width / REFERENCE_SLOT
        ky = layout.summoner_height / REFERENCE_SLOT
        return cls(
            x=round(layout.summoner_first_x + dx * kx),
            y=round(layout.summoner_y + dy * ky),
            width=round(width * kx),
            height=round(height * ky),
            glyphs=glyphs,
        )

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def read(self, frame: np.ndarray) -> int | None:
        strip = frame[self.y : self.y + self.height, self.x : self.x + self.width]
        if strip.size == 0:
            return None
        cfg = self.config
        digits = []
        for glyph in self._glyphs(strip):
            label, score, margin = self.glyphs.match(glyph)
            if score < cfg.min_score or margin < cfg.min_margin:
                return None
            if not label.isdigit():
                return None
            digits.append(label)
        if not digits or len(digits) > cfg.max_digits:
            return None
        if len(digits) > 1 and digits[0] == "0":
            return None
        return int("".join(digits))

    def _glyphs(self, strip: np.ndarray) -> list[np.ndarray]:
        """Each glyph, left to right, stray parts dropped and rescaled to
        the templates' height."""
        cfg = self.config
        mask = lit_mask(strip, cfg.lit)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_HEIGHT] >= cfg.min_part_height
        mask = keep[labels]
        grey = np.where(mask, cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), 0)
        grey = grey.astype(np.uint8)

        stroke = cfg.stroke or digit_height(self.glyphs)
        out = []
        for x0, x1 in self._split(mask, _runs(mask.any(axis=0))):
            if x1 - x0 < cfg.lit.min_glyph_width:
                continue
            rows = np.nonzero(mask[:, x0:x1].any(axis=1))[0]
            if rows.size == 0:
                continue
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            patch = grey[y0:y1, x0:x1]
            scale = stroke / (y1 - y0)
            patch = cv2.resize(
                patch,
                (max(1, round((x1 - x0) * scale)), stroke),
                interpolation=cv2.INTER_AREA,
            )
            out.append(centred(patch, self.glyphs.size))
        return out

    def _split(
        self, mask: np.ndarray, runs: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Cut runs too wide to be one digit into as many as they hold.

        Neighbouring digits can touch: on a lit box the crossbars of "44"
        join on one row, and the pair then reads as one glyph scoring 0.2-0.3
        -- most of the unread frames before this existed. Each cut goes at
        the emptiest column near where the pitch puts it.
        """
        cfg = self.config
        counts = mask.sum(axis=0)
        out = []
        for x0, x1 in runs:
            rows = np.nonzero(mask[:, x0:x1].any(axis=1))[0]
            height = int(rows[-1] - rows[0] + 1) if rows.size else 0
            width = x1 - x0
            if height == 0 or width <= cfg.max_aspect * height:
                out.append((x0, x1))
                continue
            pitch = cfg.pitch * height
            parts = max(2, round(width / pitch))
            cuts = [x0]
            for k in range(1, parts):
                guess = x0 + round(k * width / parts)
                lo, hi = max(cuts[-1] + 1, guess - 2), min(x1 - 1, guess + 3)
                if lo >= hi:
                    continue
                cuts.append(lo + int(np.argmin(counts[lo:hi])))
            cuts.append(x1)
            out.extend(zip(cuts[:-1], cuts[1:]))
        return out


def digit_height(glyphs: GlyphSet) -> int:
    """How tall the digits stand in a glyph set's canvases."""
    heights = [
        int(rows[-1] - rows[0] + 1)
        for label, glyph in glyphs.glyphs.items()
        if label.isdigit()
        for rows in [np.nonzero((glyph > 0.05).any(axis=1))[0]]
        if rows.size
    ]
    return max(heights) if heights else glyphs.size[1]


def _runs(lit: np.ndarray) -> list[tuple[int, int]]:
    """Half-open spans of True, left to right."""
    runs = []
    start = None
    for index, value in enumerate(lit):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(lit)))
    return runs


class GoldFilter:
    """Turns raw readings into the figure the row carries.

    Gold is not the creep score: it falls whenever the player buys and climbs
    a couple a second between buys, so nothing is ruled out by direction or
    size. What the filter does rule on is which way an error costs more. The
    figure gates an offer to buy, so reading high -- "you can afford it" when
    they cannot -- is the bad direction, and reading low only delays the
    offer.

    So a rise is adopted once `rise` readings in a row agree, and a fall on
    `fall`. Measured with the reader's thresholds at 10 Hz over the three
    2026-09-25 live captures (9 minutes of game) and 16 minutes of the
    2026-08-30 session -- 14,921 in-game frames -- 98.9% of frames read and
    no reading was wrong: each flagged jump was checked against its
    neighbours and was a purchase, a kill or a bounty. With the score and
    margin floors removed there were 7 wrong reads, every one a single frame,
    and the ones that were not a one-gold flicker at a tick read *low* -- a
    digit lost or misread down (2760 as 260, 661 as 261, 928 as 829). Under
    two-agree-both-ways the pre-purchase figure outlived the purchase by one
    or two frames on each of 11 purchases; taking a fall on one reading
    removes that, and costs at worst a reading of a too-low figure.
    """

    def __init__(self, rise: int = 2, fall: int = 1) -> None:
        self.rise = rise
        self.fall = fall
        self.value: int | None = None
        self._candidate: int | None = None
        self._count = 0

    def update(self, reading: int | None) -> int | None:
        if reading is None:
            return self.value
        if reading == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = reading, 1
        falling = self.value is not None and reading < self.value
        if self._count >= (self.fall if falling else self.rise):
            self.value = reading
        return self.value

    def reset(self) -> None:
        self.value = None
        self._candidate = None
        self._count = 0
