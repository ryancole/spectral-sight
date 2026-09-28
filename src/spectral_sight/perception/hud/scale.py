"""The in-game HUD scale: how large the client draws the player's own panel.

Every calibration under `etc/` is a set of rectangles for one frame size, and
that is enough for everything the client places relative to the screen -- the
minimap, the clock, the ally portraits. It is not enough for the panel at the
bottom centre (portrait, ability slots, level-up chevrons, health and mana),
because the client has a HUD scale setting that grows or shrinks that panel
independently of the window. A VOD recorded at a different setting from the
calibration puts every one of those readers on the wrong pixels while the clock
keeps reading perfectly -- measured, a HUD drawn 1.5x the calibrated size left
the chevron reader scoring 0.3 on three lit chevrons and the health line
unreadable, with nothing anywhere saying so.

**The panel scales about the bottom centre of the frame.** Measured on that
VOD, every slot, chevron and text line landed where the calibrated one would
under `x' = W/2 + (x - W/2) * s`, `y' = H + (y - H) * s` at s = 1.50, and the
fit is sharp: at 1.48 or 1.52 the chevrons are already lost. So the scale has
to be measured, not guessed from a coarse signal.

**Measured from the slot borders.** Each ability and summoner slot is a square
with a drawn frame, whatever champion is behind it; the frame is where the
panel has its strongest straight edges. For a candidate scale the six predicted
squares are laid on the frame's gradient, and the score is how strongly their
sides sit on edges, vertical and horizontal both (see `_border_score` for why
both). The true scale is a narrow, tall peak: on the
calibrated recordings it fell at 1.00 and on the VOD at 1.50, where the other
obvious signal -- does the health line read -- read at every scale from 0.82 to
1.76, because the text box is deliberately generous.

Read against a layout at scale 1.0, which is what the per-resolution files are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from spectral_sight.perception.hud.abilities import AbilityLayout
from spectral_sight.perception.hud.portraits import PortraitLayout
from spectral_sight.perception.hud.resources import ResourceLayout


@dataclass(frozen=True, slots=True)
class HudScaleConfig:
    """How the scale is searched for and when a reading is believed."""

    low: float = 0.6
    high: float = 2.0
    """The range searched, relative to the calibration."""

    coarse_step: float = 0.01
    fine_step: float = 0.0025
    fine_span: float = 0.015
    """A coarse sweep over the whole range, then a fine one around its best.
    The peak is a few hundredths wide, so 0.01 cannot step over it."""

    band: int = 1
    """Pixels either side of a predicted border searched for its edge, so
    sub-pixel rounding of a box does not throw away a true fit."""

    min_contrast: float = 2.0
    """Best score over the median of the sweep. Game frames measured 2.2
    and up across every recording here, the loading screen 1.4-1.7."""

    min_margin: float = 1.15
    """Best score over the best at least `peak_width` away from it. A second
    peak of nearly the same height is two plausible answers, which is none.
    Game frames measured 1.25 and up, the loading screen 1.01-1.03."""

    peak_width: float = 0.04


@dataclass(frozen=True, slots=True)
class HudScale:
    """The transform from the calibrated panel onto this frame's panel."""

    scale: float
    anchor_x: float
    anchor_y: float

    @classmethod
    def identity(cls, width: int, height: int) -> HudScale:
        return cls(1.0, width / 2, float(height))

    @classmethod
    def for_frame(cls, scale: float, width: int, height: int) -> HudScale:
        return cls(scale, width / 2, float(height))

    def x(self, value: float) -> float:
        return self.anchor_x + (value - self.anchor_x) * self.scale

    def y(self, value: float) -> float:
        return self.anchor_y + (value - self.anchor_y) * self.scale

    def length(self, value: float) -> float:
        return value * self.scale

    def box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = box
        return (round(self.x(x)), round(self.y(y)),
                round(self.length(width)), round(self.length(height)))


def scale_abilities(layout: AbilityLayout, hud: HudScale) -> AbilityLayout:
    return replace(
        layout,
        ability_first_x=hud.x(layout.ability_first_x),
        ability_y=hud.y(layout.ability_y),
        ability_width=hud.length(layout.ability_width),
        ability_height=hud.length(layout.ability_height),
        ability_spacing=hud.length(layout.ability_spacing),
        summoner_first_x=hud.x(layout.summoner_first_x),
        summoner_y=hud.y(layout.summoner_y),
        summoner_width=hud.length(layout.summoner_width),
        summoner_height=hud.length(layout.summoner_height),
        summoner_spacing=hud.length(layout.summoner_spacing),
        point_y=None if layout.point_y is None else hud.y(layout.point_y),
        point_height=(
            None if layout.point_height is None
            else hud.length(layout.point_height)
        ),
    )


def scale_resources(layout: ResourceLayout, hud: HudScale) -> ResourceLayout:
    return replace(layout, health=hud.box(layout.health),
                   mana=hud.box(layout.mana))


def scale_portraits(layout: PortraitLayout, hud: HudScale) -> PortraitLayout:
    """The self portrait only: the ally portraits sit at the screen's right
    edge and do not follow the panel."""
    return replace(
        layout,
        self_center_x=hud.x(layout.self_center_x),
        self_center_y=hud.y(layout.self_center_y),
        self_radius=max(1, round(hud.length(layout.self_radius))),
    )


@dataclass(frozen=True, slots=True)
class ScaleMeasurement:
    scale: float
    score: float
    contrast: float
    """Best over the sweep's median."""
    margin: float
    """Best over the best rival peak."""

    def confident(self, config: HudScaleConfig) -> bool:
        return (self.contrast >= config.min_contrast
                and self.margin >= config.min_margin)


def _slot_squares(
    layout: AbilityLayout, hud: HudScale
) -> list[tuple[float, float, float, float]]:
    scaled = scale_abilities(layout, hud)
    squares = []
    for i in range(4):
        squares.append((scaled.ability_first_x + i * scaled.ability_spacing,
                        scaled.ability_y, scaled.ability_width,
                        scaled.ability_height))
    for i in range(2):
        squares.append((scaled.summoner_first_x + i * scaled.summoner_spacing,
                        scaled.summoner_y, scaled.summoner_width,
                        scaled.summoner_height))
    return squares


def _border_score(
    gx: np.ndarray, gy: np.ndarray,
    squares: list[tuple[float, float, float, float]], band: int,
) -> float:
    """How squarely the squares sit on edges; 0 if any leaves the frame,
    since a scale that puts the panel off screen is not the answer.

    Each square scores the geometric mean of its vertical and horizontal
    sides. A plain mean of all sides let the health and mana bars win: at
    0.6 on the 1.5x VOD the shrunken squares laid their tops and bottoms on
    the bars' long horizontal edges and scored within 15% of the true fit.
    A bar has no matching verticals, and the product asks for both."""
    height, width = gx.shape
    total = 0.0
    for x, y, w, h in squares:
        left, right = round(x), round(x + w)
        top, bottom = round(y), round(y + h)
        if (left - band < 0 or top - band < 0
                or right + band >= width or bottom + band >= height):
            return 0.0
        vertical = sum(
            float(gx[top:bottom, column - band:column + band + 1]
                  .max(axis=1).mean())
            for column in (left, right)
        ) / 2
        horizontal = sum(
            float(gy[row - band:row + band + 1, left:right].max(axis=0).mean())
            for row in (top, bottom)
        ) / 2
        total += float(np.sqrt(vertical * horizontal))
    return total / len(squares) if squares else 0.0


def measure_hud_scale(
    frame: np.ndarray,
    layout: AbilityLayout,
    config: HudScaleConfig | None = None,
) -> ScaleMeasurement | None:
    """The panel's scale on this frame, against a scale-1.0 layout. None
    when nothing scored at all -- a flat frame, or a layout off screen."""
    cfg = config or HudScaleConfig()
    height, width = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))

    def score(scale: float) -> float:
        hud = HudScale.for_frame(scale, width, height)
        return _border_score(gx, gy, _slot_squares(layout, hud), cfg.band)

    coarse = np.arange(cfg.low, cfg.high + cfg.coarse_step / 2, cfg.coarse_step)
    scores = np.array([score(float(s)) for s in coarse])
    if not np.any(scores > 0):
        return None
    best = int(np.argmax(scores))
    fine = np.arange(coarse[best] - cfg.fine_span,
                     coarse[best] + cfg.fine_span + cfg.fine_step / 2,
                     cfg.fine_step)
    fine_scores = [score(float(s)) for s in fine]
    at = int(np.argmax(fine_scores))
    top = max(fine_scores[at], float(scores[best]))
    found = float(fine[at]) if fine_scores[at] >= scores[best] else float(coarse[best])

    median = float(np.median(scores[scores > 0]))
    far = scores[np.abs(coarse - found) >= cfg.peak_width]
    rival = float(far.max()) if far.size and far.max() > 0 else 0.0
    return ScaleMeasurement(
        scale=round(found, 4),
        score=top,
        contrast=top / median if median > 0 else float("inf"),
        margin=top / rival if rival > 0 else float("inf"),
    )


@dataclass(frozen=True, slots=True)
class HudScaleWatchConfig:
    """When to look, and when a look changes anything."""

    unsettled_interval: float = 0.5
    settled_interval: float = 10.0
    """Seconds between measurements: often while the scale is unknown -- a
    run just started, or the clock says the footage changed -- and rarely
    once it is known, which is what notices a setting changed mid-game
    without paying 40ms a frame for it. Until the first window fills, the
    panel is read at the last scale adopted, so the unsettled interval is
    the delay before a wrong one is corrected: 0.5s gives 2.5s."""

    window: int = 5
    """Confident readings the decision is taken over, as their median.
    Single frames scatter by about 1% around the truth (0.998-1.008 on
    one recording at 1.0); the median of five does not."""

    tolerance: float = 0.02
    """How far the median must sit from the current scale to replace it.
    Below the chevrons' own tolerance (lit boxes survive about 1.2% either
    side) only just, and above the scatter of a correct scale's readings,
    so a correct scale is never nudged by noise."""


class HudScaleWatch:
    """Decides the HUD scale over a run: measures on a cadence, adopts a
    new scale when a window of confident readings agrees on one."""

    def __init__(
        self,
        layout: AbilityLayout,
        config: HudScaleWatchConfig | None = None,
        measure: HudScaleConfig | None = None,
    ) -> None:
        self.layout = layout
        self.config = config or HudScaleWatchConfig()
        self.measure_config = measure or HudScaleConfig()
        self.scale = 1.0
        self.settled = False
        """Whether a full window has been seen since the last `unsettle`."""
        self._readings: list[float] = []
        self._last: float | None = None

    def unsettle(self) -> None:
        """Look again soon, and decide from fresh readings only: the footage
        may be a different game, recorded at a different setting."""
        self.settled = False
        self._readings.clear()
        self._last = None

    def update(self, frame: np.ndarray, timestamp: float) -> float | None:
        """Fold one trusted frame in. Returns the new scale when it changes,
        None otherwise (including on frames it did not look at)."""
        interval = (self.config.settled_interval if self.settled
                    else self.config.unsettled_interval)
        if self._last is not None and timestamp - self._last < interval:
            return None
        self._last = timestamp
        reading = measure_hud_scale(frame, self.layout, self.measure_config)
        if reading is None or not reading.confident(self.measure_config):
            return None
        self._readings.append(reading.scale)
        del self._readings[:-self.config.window]
        if len(self._readings) < self.config.window:
            return None
        self.settled = True
        median = float(np.median(self._readings))
        if abs(median - self.scale) < self.config.tolerance:
            return None
        self.scale = median
        return median
