"""The camera viewport rectangle drawn on the minimap.

The minimap outlines the region of the map currently on screen. On footage
recorded with the camera locked to the player -- which is a stated assumption of
this project, see the README -- the player sits at screen centre, so the centre
of that rectangle *is* the local player's position.

That makes identifying your own marker geometric rather than visual, which
matters because matching the local player against their *HUD portrait* does not
work at all. Minimap icons are stock champion art while HUD portraits are
skin-specific, so the two disagree completely whenever the player is using a
skin -- as they did throughout the sample footage, scoring near zero while every
teammate matched. The rectangle sidesteps the question entirely, and keeps
working regardless of skins, gallery coverage, or fog.

The separation is not marginal. The nearest marker to the rectangle centre sits
5-7px away while the runner-up is 38-88px away, so this is effectively exact.

What is not guaranteed is that the marker is *there to find*. On a real game
the player's icon spends much of its time covered -- an enemy chasing them, a
support standing on them -- and a covered ring does not fill, so stage 1 drops
it. Measured on the 2026-08-30 session a blue marker sat within 12px of the
centre on only 24% of frames, while the centre itself agreed with the
player's own nameplate projected onto the minimap to within 2px on every frame
checked. The rectangle knows where the player is even when nothing is drawn
there; the pipeline's `_find_self` acts on that.

**The rectangle is a fixed size, and that rescues it when it is not whole.**
The camera's zoom is locked, so at one minimap scale the outline is always the
same box -- 78x48 on the 325px panel of the 2026-08-30 session, found at that
size on 619 of 1,135 frames sampled across the match. Two things break it up.
Near a map edge the box is drawn only where it overlaps the map, so its bounding
rectangle is the visible *part*, and the centre of that part is not the camera:
on the same session 393 frames were clipped that way, displaced by up to 14px,
which is 700 world units. And an icon drawn across the outline -- a base
structure, a champion -- splits it into pieces too thin to pass as a rectangle
at all; on footage recorded at a 486px minimap with the player in the fountain,
every frame for a minute read as no viewport.

So when the outline is missing or smaller than the box, a box of the known size
is slid over the white mask instead, scoring only the part of its perimeter that
lies on the map, and placed where the most of that part is lit. On whole boxes
it agrees with the contour to a pixel. The fountain minute resolves to one
position on every frame, 7px from the player's own marker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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

    box_width: float | None = None
    box_height: float | None = None
    """The outline's full size in minimap pixels, when known -- see the module
    docstring. None disables the fitted fallback, leaving the contour alone.
    `scaled_viewport_config` supplies it for a given minimap width."""

    fit_min_lit: float = 0.45
    """Share of the on-map perimeter that must be lit for a fitted box to be
    believed. Whole boxes score 0.58-0.74 against the outline's own one-pixel
    stroke; the fountain corner, half hidden by icons, 0.53-0.62."""

    fit_min_on_map: float = 0.3
    """Share of the box's perimeter that must lie on the map. A box pushed so
    far off the edge that less than this is drawn is too little to place."""


REFERENCE_MINIMAP_WIDTH = 325
REFERENCE_BOX = (78.0, 48.0)
"""The outline at the reference panel width. The most common sizes over the
2026-08-30 session were 78x49 and 78x48."""


def scaled_viewport_config(minimap_width: int) -> ViewportConfig:
    """A config carrying the outline's size for a minimap this wide.

    The box scales with the panel like everything drawn on it, and the minimum
    sizes scale with it so a larger panel does not let smaller marks through.
    """
    factor = minimap_width / REFERENCE_MINIMAP_WIDTH
    base = ViewportConfig()
    return replace(
        base,
        min_width=round(base.min_width * factor),
        min_height=round(base.min_height * factor),
        box_width=REFERENCE_BOX[0] * factor,
        box_height=REFERENCE_BOX[1] * factor,
    )


def find_viewport(
    minimap: np.ndarray,
    config: ViewportConfig | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> Viewport | None:
    """Locate the camera rectangle, or None if it cannot be found.

    Returns the largest candidate: the rectangle is bigger than the incidental
    white marks on the minimap (ward pips, text, ping flashes) that survive the
    same threshold. When the config knows the box's size and that candidate is
    missing or smaller than it, the box is fitted instead -- and may then extend
    past the map, which is where the camera really is.

    `bounds` is the map square within the crop, (x, y, width, height), where the
    outline can be drawn. Defaults to the whole crop.
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

    if cfg.box_width is None or cfg.box_height is None:
        return best
    whole = best is not None and (
        best.width >= cfg.box_width - 3 and best.height >= cfg.box_height - 3
    )
    if whole:
        return best
    fitted = _fit_box(mask > 0, cfg, bounds)
    return fitted if fitted is not None else best


def _fit_box(
    lit: np.ndarray,
    cfg: ViewportConfig,
    bounds: tuple[float, float, float, float] | None,
) -> Viewport | None:
    """The box of the known size best covered by lit pixels, on the map."""
    height, width = lit.shape
    box_w, box_h = round(cfg.box_width), round(cfg.box_height)
    on_map = np.zeros((height, width), np.float32)
    bx, by, bw, bh = bounds if bounds is not None else (0, 0, width, height)
    on_map[max(0, round(by)) : round(by + bh), max(0, round(bx)) : round(bx + bw)] = 1

    # The outline as a kernel, two pixels thick so a stroke antialiased across
    # two rows still lands on it.
    kernel = np.zeros((box_h + 1, box_w + 1), np.float32)
    kernel[:2, :] = kernel[-2:, :] = kernel[:, :2] = kernel[:, -2:] = 1
    pad = ((box_h, box_h), (box_w, box_w))
    hits = cv2.matchTemplate(
        np.pad(lit.astype(np.float32) * on_map, pad), kernel, cv2.TM_CCORR
    )
    room = cv2.matchTemplate(np.pad(on_map, pad), kernel, cv2.TM_CCORR)
    enough = room >= cfg.fit_min_on_map * kernel.sum()
    share = np.where(enough, hits / np.maximum(room, 1.0), 0.0)
    # More lit pixels breaks ties between placements equally well covered,
    # which is what keeps a short run of white from matching a corner.
    score = share * np.sqrt(np.maximum(hits, 0.0))
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    if share[y, x] < cfg.fit_min_lit:
        return None
    return Viewport(x=int(x) - box_w, y=int(y) - box_h, width=box_w, height=box_h)
