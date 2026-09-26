"""Minion dots on the minimap: where every wave is, including off screen.

The world view's health bars (`nameplates/minions.py`) say where the minions
near the player are and how hurt they are. This is the other half: the minimap
draws every minion on the map as a small team-coloured dot, so it answers where
the *waves* are -- which lane is pushing, whether a wave is crashing into a
tower -- for the parts of the map the camera is not on.

It needs the minimap turned up. At the 325px panel the earlier footage was
recorded at, a dot is two or three pixels and indistinguishable from the map's
own decoration; at 486px (set on 2026-09-25) it is a disc with a lit core about
six pixels across inside a dark outline.

**A dot is found by being a bright disc of team colour with no team colour
around it.** Three things on the minimap pass a colour test and are not
minions, and each fails one of the other two:

- *Structures and objective icons* -- turret shields, inhibitors, the nexus
  towers, the crab and dragon icons -- are drawn in the team colours at lower
  brightness. Measured on one frame of the 2026-09-25 footage, every true dot
  had a core brightness (HSV V) of 203-237 and every false candidate 184 or
  less, so the core-brightness floor is what removes them. Run over a minute
  of that footage at 2 Hz, no position was lit in more than 33 of 116 samples,
  so no static icon was getting through.
- *Base shading and the river* are team-coloured over large areas; the
  surround test rejects anything whose neighbourhood is coloured too.
- *Champion markers* have team-coloured rings and portraits. Candidates
  inside a detected champion marker are dropped: whatever is drawn there
  belongs to the champion.

The faint, desaturated teal dots that line the lanes on this footage are not
minions (what they are is not established); they sit far under the saturation
floor.

A dot under a champion marker is not seen, and dots overlapping in a clump
count as fewer than there are. Counts are therefore a floor, and positions are
the useful output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from spectral_sight.types import Team

REFERENCE_MINIMAP_WIDTH = 486
"""Panel width the defaults were measured at -- not the blip detector's 325,
because at that size there are no dots worth reading."""


@dataclass(frozen=True, slots=True)
class MinionDotConfig:
    """Thresholds for `MinionDotDetector`. Radii are in minimap-crop pixels at
    `REFERENCE_MINIMAP_WIDTH`; see `scaled_dot_config`."""

    core_radius: float = 3.0
    """The lit core of a dot. Measured components were 6x6 to 7x7, area ~30."""

    blue_hue: tuple[int, int] = (95, 112)
    red_hue: tuple[int, int] = (170, 6)
    """Wraps past 179, like every red in this project."""

    min_saturation: int = 110
    min_value: int = 140
    """Colour mask floors. The faint teal lane dots sit at S 70-90."""

    min_fill: float = 0.7
    """Share of the core disc that must be team-coloured."""

    max_surround: float = 0.35
    """Share of the ring just outside the dot's outline that may be
    team-coloured. Neighbouring dots in a chain put some colour there; base
    shading and the river put a lot."""

    min_core_value: int = 195
    """Median brightness of the core. The separator between dots and icons --
    see the module docstring."""

    marker_margin: float = 3.0
    """How far outside a champion marker's radius a candidate is still taken to
    be part of the marker."""


def scaled_dot_config(
    minimap_width: int, config: MinionDotConfig | None = None
) -> MinionDotConfig:
    """Retarget the radii to a minimap of a different on-screen size."""
    config = config or MinionDotConfig()
    factor = minimap_width / REFERENCE_MINIMAP_WIDTH
    return replace(
        config,
        core_radius=config.core_radius * factor,
        marker_margin=config.marker_margin * factor,
    )


@dataclass(frozen=True, slots=True)
class MinionDot:
    """One minion dot, in minimap-crop pixels."""

    x: int
    y: int
    team: Team


def _disc(radius: float) -> np.ndarray:
    size = 2 * int(np.ceil(radius)) + 1
    centre = size // 2
    yy, xx = np.mgrid[:size, :size]
    kernel = (np.hypot(xx - centre, yy - centre) <= radius).astype(np.float32)
    return kernel / kernel.sum()


def _ring(inner: float, outer: float) -> np.ndarray:
    size = 2 * int(np.ceil(outer)) + 1
    centre = size // 2
    yy, xx = np.mgrid[:size, :size]
    distance = np.hypot(xx - centre, yy - centre)
    kernel = ((distance >= inner) & (distance <= outer)).astype(np.float32)
    return kernel / kernel.sum()


class MinionDotDetector:
    """Minimap crop in, minion dots out."""

    def __init__(self, config: MinionDotConfig | None = None) -> None:
        self.config = config or MinionDotConfig()
        r = self.config.core_radius
        self._core = _disc(r)
        # Outside the dark outline, which sits about one core radius out.
        self._surround = _ring(r + 2.0 * r / 3.0, r + 3.5 * r / 3.0)
        self._peak = np.ones((5, 5), np.uint8)
        self._spacing = max(2, round(4 * r / 3.0))

    def _masks(self, hsv: np.ndarray) -> dict[Team, np.ndarray]:
        cfg = self.config
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (s >= cfg.min_saturation) & (v >= cfg.min_value)
        lo, hi = cfg.blue_hue
        blue = (h >= lo) & (h <= hi) & lit
        lo, hi = cfg.red_hue
        red = ((h >= lo) | (h <= hi)) & lit
        return {Team.BLUE: blue.astype(np.float32), Team.RED: red.astype(np.float32)}

    def detect(
        self,
        minimap: np.ndarray,
        markers: Iterable[tuple[float, float, float]] = (),
    ) -> list[MinionDot]:
        """Every minion dot on the minimap.

        `markers` are champion markers as (x, y, radius); dots inside them are
        not reported.
        """
        if minimap.ndim != 3 or minimap.shape[2] != 3:
            raise ValueError(f"expected a BGR image, got shape {minimap.shape}")
        cfg = self.config
        markers = list(markers)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        value = hsv[..., 2]
        height, width = value.shape
        core = max(1, round(2 * cfg.core_radius / 3.0))

        dots: list[MinionDot] = []
        for team, mask in self._masks(hsv).items():
            fill = cv2.filter2D(mask, -1, self._core, borderType=cv2.BORDER_CONSTANT)
            around = cv2.filter2D(
                mask, -1, self._surround, borderType=cv2.BORDER_CONSTANT
            )
            peaks = (
                (fill == cv2.dilate(fill, self._peak))
                & (fill >= cfg.min_fill)
                & (around <= cfg.max_surround)
            )
            for y, x in zip(*np.nonzero(peaks)):
                x, y = int(x), int(y)
                patch = value[
                    max(0, y - core) : min(height, y + core + 1),
                    max(0, x - core) : min(width, x + core + 1),
                ]
                if np.median(patch) < cfg.min_core_value:
                    continue
                if any(
                    np.hypot(x - mx, y - my) < mr + cfg.marker_margin
                    for mx, my, mr in markers
                ):
                    continue
                # A flat-topped peak reports every pixel of its plateau.
                if any(
                    d.team is team
                    and abs(d.x - x) < self._spacing
                    and abs(d.y - y) < self._spacing
                    for d in dots
                ):
                    continue
                dots.append(MinionDot(x=x, y=y, team=team))
        return dots
