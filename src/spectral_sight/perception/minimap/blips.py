"""Stage 1: class-agnostic champion marker detection on the minimap.

A champion marker is a portrait disc inside a saturated team-coloured ring. The
detector finds the ring's *circular edge*, then confirms the ring's colour.

Two earlier approaches failed on real footage, and the reasons are worth keeping
because they constrain anything built here later:

1. **Match the ring as a coloured shape.** Ring brightness varies a lot around
   the circumference, so any value floor high enough to exclude the background
   also chops the ring into arcs.

2. **Find the hole the ring encloses.** This survives a champion standing in its
   own base -- where the ring merges into team-coloured shading -- but dies on a
   different case: champion portrait art frequently contains the team hue. A red
   champion with warm art fills the red mask solid, so there is no hole. Measured
   on real frames, two of seven markers had a perfect ring (fill 0.96 and 1.00)
   and were still missed for exactly this reason.

What is actually invariant is the *circular edge* at the ring, which exists
whether or not the interior matches, and whether or not the surroundings do. So:
Hough over the gradient to propose circles, then a colour test on the annulus to
confirm and assign a team. Hough tolerates broken arcs -- a partial ring still
votes for the same centre -- which neutralises failure 1, and it never looks
inside the disc, which neutralises failure 2.

Hough's radius estimate is coarse, so each candidate's radius is refit by
maximising ring colour fill. That matters: an unrefined radius scored a real
marker at 0.43 against a 0.45 threshold purely because Hough returned 11.8 where
the ring was at 14.0.

Output is deliberately identity-free. Deciding *which* champion a marker is
belongs to stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from spectral_sight.types import Blip, Team

REFERENCE_MINIMAP_WIDTH = 325
"""Minimap width the default radii were measured at. See `scaled_config`."""


@dataclass(frozen=True, slots=True)
class HsvBand:
    """An inclusive HSV threshold window, in OpenCV ranges (H 0-179, S/V 0-255)."""

    hue: tuple[int, int]
    saturation: tuple[int, int] = (60, 255)
    value: tuple[int, int] = (40, 255)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array([self.hue[0], self.saturation[0], self.value[0]], np.uint8)
        upper = np.array([self.hue[1], self.saturation[1], self.value[1]], np.uint8)
        return cv2.inRange(hsv, lower, upper)


# Fitted against real 1080p-class footage by sampling ring annuli at hand-marked
# champion positions. Two things surprised the initial guess and are worth
# stating: the enemy ring is magenta (H~168-178), not red, and ring brightness
# drops to V~70, so the value floor has to sit far lower than it looks like it
# should. The narrow (0, 8) red band is the hue wraparound, not a second colour.
BLUE_BANDS: tuple[HsvBand, ...] = (HsvBand(hue=(88, 115), saturation=(60, 255)),)
RED_BANDS: tuple[HsvBand, ...] = (
    HsvBand(hue=(160, 179), saturation=(80, 255)),
    HsvBand(hue=(0, 8), saturation=(80, 255)),
)


@dataclass(frozen=True, slots=True)
class BlipDetectorConfig:
    """Tunable thresholds for `BlipDetector`.

    Radii are in minimap-crop pixels at `REFERENCE_MINIMAP_WIDTH`; use
    `scaled_config` to retarget them to another minimap scale.
    """

    blue_bands: tuple[HsvBand, ...] = BLUE_BANDS
    red_bands: tuple[HsvBand, ...] = RED_BANDS

    min_radius: float = 11.0
    max_radius: float = 16.5
    """Observed marker radii spanned 11.5-15.0; this brackets that."""

    hough_dp: float = 1.0
    hough_param1: float = 90.0
    """Upper Canny threshold used inside HoughCircles."""

    hough_param2: float = 18.0
    """Accumulator threshold. Deliberately permissive -- the colour test is the
    real filter, so this is tuned for recall and left to over-propose. Raising it
    to 30 dropped recall from 7/7 to 4/7 while only removing candidates the
    colour test rejects for free."""

    hough_min_dist: float = 14.0

    proposal_channels: tuple[str, ...] = ("gray", "saturation")
    """Channels to run Hough over. See `BlipDetector._propose` for why both."""

    refine_step: float = 0.5
    ring_half_width: float = 1.5
    """Annulus sampled as radius +/- this when measuring colour fill."""

    min_ring_fill: float = 0.55
    """Fraction of the annulus that must match one team's colour."""

    blur_sigma: float = 0.7
    edge_blur_sigma: float = 1.0

    nms_overlap: float = 0.75
    max_blips: int = 10
    """A Summoner's Rift game has ten champions. Never return more.

    Note this is a cap, not an expectation: from a player's perspective fog of
    war hides most enemies most of the time, so a typical frame yields far fewer.
    Only a spectator or replay feed with full vision approaches ten.
    """


@dataclass(slots=True)
class BlipDebug:
    """Intermediate artifacts, for tuning. Not produced on the hot path."""

    masks: dict[Team, np.ndarray] = field(default_factory=dict)
    candidates: int = 0
    rejected: list[tuple[Blip, str]] = field(default_factory=list)


class BlipDetector:
    """Finds champion markers on a minimap crop."""

    def __init__(self, config: BlipDetectorConfig | None = None) -> None:
        self.config = config or BlipDetectorConfig()

    def detect(self, minimap: np.ndarray) -> list[Blip]:
        """Return champion markers, strongest first."""
        return self._detect(minimap, debug=None)

    def detect_with_debug(self, minimap: np.ndarray) -> tuple[list[Blip], BlipDebug]:
        """Same as `detect`, but also returns masks and rejection reasons."""
        debug = BlipDebug()
        return self._detect(minimap, debug=debug), debug

    # -- internals --------------------------------------------------------

    def _detect(self, minimap: np.ndarray, debug: BlipDebug | None) -> list[Blip]:
        if minimap.ndim != 3 or minimap.shape[2] != 3:
            raise ValueError(f"expected a BGR image, got shape {minimap.shape}")

        cfg = self.config
        masks = self._team_masks(minimap)
        if debug is not None:
            debug.masks = dict(masks)

        circles = self._propose(minimap)
        if debug is not None:
            debug.candidates = len(circles)

        found: list[Blip] = []
        for x, y, radius in circles:
            fill, fitted, team = self._best_ring(masks, x, y)
            if team is None or fill < cfg.min_ring_fill:
                self._reject(debug, x, y, radius, "ring_fill")
                continue
            found.append(
                Blip(x=float(x), y=float(y), radius=float(fitted), team=team,
                     score=float(min(fill, 1.0)))
            )

        found.sort(key=lambda b: b.score, reverse=True)
        return self._suppress(found)[: cfg.max_blips]

    def _team_masks(self, minimap: np.ndarray) -> dict[Team, np.ndarray]:
        cfg = self.config
        source = minimap
        if cfg.blur_sigma > 0:
            source = cv2.GaussianBlur(source, (0, 0), cfg.blur_sigma)
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

        masks: dict[Team, np.ndarray] = {}
        for team, bands in ((Team.BLUE, cfg.blue_bands), (Team.RED, cfg.red_bands)):
            mask = bands[0].mask(hsv)
            for band in bands[1:]:
                mask = cv2.bitwise_or(mask, band.mask(hsv))
            masks[team] = mask
        return masks

    def _propose(self, minimap: np.ndarray) -> list[tuple[float, float, float]]:
        """Circle candidates from the gradient, ignoring colour entirely.

        Proposals are pooled from more than one channel because neither alone is
        sufficient. Luminance misses a marker whose portrait happens to match the
        brightness of the team-coloured shading it stands on -- the ring is still
        there, but there is no intensity step to differentiate. Saturation
        catches that case, since the ring stroke is far more saturated than both
        the portrait and the shading.

        Over-proposing is cheap here: the colour test downstream is the real
        filter, and duplicate proposals collapse in suppression.
        """
        cfg = self.config
        channels = {
            "gray": cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY),
            "saturation": cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)[:, :, 1],
        }

        proposals: list[tuple[float, float, float]] = []
        for name in cfg.proposal_channels:
            channel = channels[name]
            if cfg.edge_blur_sigma > 0:
                channel = cv2.GaussianBlur(channel, (0, 0), cfg.edge_blur_sigma)
            circles = cv2.HoughCircles(
                channel,
                cv2.HOUGH_GRADIENT,
                dp=cfg.hough_dp,
                minDist=cfg.hough_min_dist,
                param1=cfg.hough_param1,
                param2=cfg.hough_param2,
                minRadius=int(cfg.min_radius),
                maxRadius=int(np.ceil(cfg.max_radius)),
            )
            if circles is not None:
                proposals.extend(
                    (float(c[0]), float(c[1]), float(c[2])) for c in circles[0]
                )
        return proposals

    def _best_ring(
        self, masks: dict[Team, np.ndarray], x: float, y: float
    ) -> tuple[float, float, Team | None]:
        """Refit radius by maximising ring colour fill; return (fill, radius, team).

        Hough's radius is only accurate to a couple of pixels, which is enough to
        push a real marker's annulus off its ring and under the threshold.
        """
        cfg = self.config
        any_mask = next(iter(masks.values()))
        height, width = any_mask.shape[:2]

        pad = int(np.ceil(cfg.max_radius + cfg.ring_half_width)) + 1
        x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
        x1, y1 = min(width, int(x) + pad + 1), min(height, int(y) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0, cfg.min_radius, None

        ys, xs = np.ogrid[y0:y1, x0:x1]
        distance = np.hypot(xs - x, ys - y)
        windows = {team: mask[y0:y1, x0:x1] > 0 for team, mask in masks.items()}

        best = (0.0, cfg.min_radius, None)
        radius = cfg.min_radius
        while radius <= cfg.max_radius:
            annulus = (distance >= radius - cfg.ring_half_width) & (
                distance <= radius + cfg.ring_half_width
            )
            if annulus.any():
                for team, window in windows.items():
                    fill = float(window[annulus].mean())
                    # >= so ties resolve to the largest radius. A champion whose
                    # portrait matches its own team hue fills every annulus
                    # inside the ring equally; the outermost is the real edge.
                    if fill > 0.0 and fill >= best[0]:
                        best = (fill, radius, team)
            radius += cfg.refine_step
        return best

    def _suppress(self, blips: list[Blip]) -> list[Blip]:
        """Greedy non-max suppression by centre distance, across teams as well as
        within one -- Hough happily proposes the same ring twice."""
        kept: list[Blip] = []
        for blip in blips:
            if any(
                blip.distance_to(other)
                < self.config.nms_overlap * (blip.radius + other.radius)
                for other in kept
            ):
                continue
            kept.append(blip)
        return kept

    @staticmethod
    def _reject(
        debug: BlipDebug | None, x: float, y: float, radius: float, reason: str
    ) -> None:
        if debug is None:
            return
        debug.rejected.append(
            (Blip(x=x, y=y, radius=radius, team=Team.UNKNOWN, score=0.0), reason)
        )


def scaled_config(
    config: BlipDetectorConfig,
    minimap_width: int,
    reference_width: int = REFERENCE_MINIMAP_WIDTH,
) -> BlipDetectorConfig:
    """Retarget a config to a minimap of a different on-screen size.

    Marker radius scales linearly with the minimap panel, so a config fitted at
    one scale-slider setting transfers to any other with just this adjustment.
    The colour bands are scale-invariant and carry over untouched.
    """
    factor = minimap_width / reference_width
    return replace(
        config,
        min_radius=config.min_radius * factor,
        max_radius=config.max_radius * factor,
        hough_min_dist=config.hough_min_dist * factor,
    )
