"""The camera viewport rectangle drawn on the minimap.

The minimap outlines the region of the map currently on screen. On footage
recorded with the camera locked to the player -- which is a stated assumption of
this project, see the README -- the player sits at screen centre, so the centre
of that rectangle *is* the local player's position.

That makes identifying your own marker geometric rather than visual, which
matters because the appearance route does not work: the HUD portrait and the
minimap marker for the local player are different art. Measured across frames,
they share almost no similarity, while the same comparison succeeds for every
teammate. Skins are the likely cause, and it does not matter -- the rectangle
sidesteps the question entirely.

The separation is not marginal. The nearest marker to the rectangle centre sits
5-7px away while the runner-up is 38-88px away, so this is effectively exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Viewport:
    """The on-screen region, as drawn on the minimap."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass(frozen=True, slots=True)
class ViewportConfig:
    """Thresholds for finding the rectangle.

    It is drawn as a thin, near-white outline, so it is picked out by low
    saturation and high value rather than by any hue.
    """

    max_saturation: int = 45
    min_value: int = 190
    min_width: int = 30
    min_height: int = 20
    aspect_range: tuple[float, float] = (1.1, 2.6)
    """Width over height. The viewport tracks the display aspect ratio, but the
    rectangle is clipped where it meets the edge of the map, so the observed
    range is wider than the display's own."""

    close_kernel: int = 3


def find_viewport(
    minimap: np.ndarray, config: ViewportConfig | None = None
) -> Viewport | None:
    """Locate the camera rectangle, or None if it cannot be found.

    Returns the largest candidate: the rectangle is bigger than the incidental
    white marks on the minimap (ward pips, text, ping flashes) that survive the
    same threshold.
    """
    if minimap.ndim != 3 or minimap.shape[2] != 3:
        raise ValueError(f"expected a BGR image, got shape {minimap.shape}")

    cfg = config or ViewportConfig()
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, cfg.min_value], np.uint8),
        np.array([179, cfg.max_saturation, 255], np.uint8),
    )
    if cfg.close_kernel > 1:
        kernel = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    best: Viewport | None = None
    best_area = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < cfg.min_width or height < cfg.min_height:
            continue
        aspect = width / max(height, 1)
        if not cfg.aspect_range[0] <= aspect <= cfg.aspect_range[1]:
            continue
        area = width * height
        if area > best_area:
            best_area = area
            best = Viewport(x=x, y=y, width=width, height=height)
    return best
