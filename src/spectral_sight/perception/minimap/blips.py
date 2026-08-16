"""Stage 1: class-agnostic champion marker detection on the minimap.

Champion markers are the only thing on the minimap drawn as a *portrait core
enclosed by a saturated team-coloured ring*. Everything else the minimap draws
in team colours is solid: turret, inhibitor and nexus glyphs, base and
turret-range shading, minion dots.

The naive reading of that is "find team-coloured rings". It does not survive
contact with the game: a champion standing in its own base has its ring merged
into the surrounding shading, and the outer boundary of that blob is the base,
not the marker. Champions sit in their base constantly, so this is not an edge
case.

So we detect the *hole* instead. Take the team-colour mask and look for enclosed
regions of non-team colour. A champion punches one out of whatever it is
standing on -- bare terrain, base shading, a turret's range circle -- because
the portrait interrupts the colour and the ring closes around it. A solid glyph
never does. That single inversion turns the hardest case into the easy one, and
it means the discriminating filters reduce to two: is the hole the right size,
and is it round.

Output is deliberately identity-free. Deciding *which* champion a marker is
belongs to stage 2, which matches the core crop against a reference gallery --
and the core crop is exactly the hole this stage already found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from spectral_sight.types import Blip, Team


@dataclass(frozen=True, slots=True)
class HsvBand:
    """An inclusive HSV threshold window, in OpenCV ranges (H 0-179, S/V 0-255)."""

    hue: tuple[int, int]
    saturation: tuple[int, int] = (90, 255)
    value: tuple[int, int] = (70, 255)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array([self.hue[0], self.saturation[0], self.value[0]], np.uint8)
        upper = np.array([self.hue[1], self.saturation[1], self.value[1]], np.uint8)
        return cv2.inRange(hsv, lower, upper)


# Red straddles the hue wraparound, so it needs two bands where blue needs one.
# These are starting values; they have to be fitted to real footage, since the
# exact ring colours shift with the client's colour-blind setting.
BLUE_BANDS: tuple[HsvBand, ...] = (HsvBand(hue=(92, 122)),)
RED_BANDS: tuple[HsvBand, ...] = (
    HsvBand(hue=(0, 10), saturation=(110, 255)),
    HsvBand(hue=(168, 179), saturation=(110, 255)),
)


@dataclass(frozen=True, slots=True)
class BlipDetectorConfig:
    """Tunable thresholds for `BlipDetector`.

    Radii are in minimap-crop pixels and describe the *core* -- the portrait
    hole -- not the full marker. At a 280px-wide minimap the core runs about
    6-7px; the default band is generous around that so a mis-set minimap scale
    degrades recall rather than zeroing it. Use `scaled_config` to retarget a
    config to a different minimap size.
    """

    blue_bands: tuple[HsvBand, ...] = BLUE_BANDS
    red_bands: tuple[HsvBand, ...] = RED_BANDS

    min_core_radius: float = 3.0
    max_core_radius: float = 13.0

    min_circularity: float = 0.65
    """Isoperimetric ratio of the hole. A clean portrait core scores near 1.0."""

    min_solidity: float = 0.65
    """Hole area over its enclosing disc. Rejects crescents and slivers."""

    outer_scale: float = 1.55
    """Core radius to full marker radius, including the ring stroke."""

    blur_sigma: float = 0.8
    close_kernel: int = 3
    """Morphological closing seals antialiasing gaps so thin rings stay closed.

    This is the knob that matters most on compressed footage: a ring broken by
    encoder artifacts encloses nothing, and an unclosed ring is a missed
    champion. Widen it before touching anything else if recall drops on real
    video.
    """

    nms_overlap: float = 0.75
    """Suppress a weaker blip within this fraction of the summed radii."""

    max_blips: int = 10
    """A Summoner's Rift game has exactly ten champions. Never return more."""


@dataclass(slots=True)
class BlipDebug:
    """Intermediate artifacts, for the tuning tool. Not produced on the hot path."""

    masks: dict[Team, np.ndarray] = field(default_factory=dict)
    rejected: list[tuple[Blip, str]] = field(default_factory=list)


class BlipDetector:
    """Finds champion markers on a minimap crop.

    Stateless and cheap: two `inRange` passes and a contour walk over a ~300x300
    crop, comfortably under a millisecond. Reuse one instance across frames so
    the morphology kernel is only built once.
    """

    def __init__(self, config: BlipDetectorConfig | None = None) -> None:
        self.config = config or BlipDetectorConfig()
        size = self.config.close_kernel
        self._kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            if size > 1
            else None
        )

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
        source = minimap
        if cfg.blur_sigma > 0:
            source = cv2.GaussianBlur(source, (0, 0), cfg.blur_sigma)
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

        candidates: list[Blip] = []
        for team, bands in (
            (Team.BLUE, cfg.blue_bands),
            (Team.RED, cfg.red_bands),
        ):
            mask = self._team_mask(hsv, bands)
            if debug is not None:
                debug.masks[team] = mask
            candidates.extend(self._candidates(mask, team, debug))

        candidates.sort(key=lambda b: b.score, reverse=True)
        return self._suppress(candidates)[: cfg.max_blips]

    def _team_mask(self, hsv: np.ndarray, bands: tuple[HsvBand, ...]) -> np.ndarray:
        mask = bands[0].mask(hsv)
        for band in bands[1:]:
            mask = cv2.bitwise_or(mask, band.mask(hsv))
        if self._kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def _candidates(
        self, mask: np.ndarray, team: Team, debug: BlipDebug | None
    ) -> list[Blip]:
        cfg = self.config

        # RETR_CCOMP gives a two-level hierarchy: outer boundaries at the top
        # level, enclosed holes beneath them. We only care about the holes.
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            return []

        found: list[Blip] = []
        for contour, meta in zip(contours, hierarchy[0]):
            has_parent = meta[3] >= 0
            if not has_parent:
                continue

            (cx, cy), core_radius = cv2.minEnclosingCircle(contour)
            radius = core_radius * cfg.outer_scale
            if not cfg.min_core_radius <= core_radius <= cfg.max_core_radius:
                self._reject(debug, cx, cy, radius, team, "core_radius")
                continue

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0 or area <= 0:
                continue

            circularity = min(4.0 * math.pi * area / (perimeter * perimeter), 1.0)
            if circularity < cfg.min_circularity:
                self._reject(debug, cx, cy, radius, team, "circularity")
                continue

            solidity = min(area / (math.pi * core_radius * core_radius), 1.0)
            if solidity < cfg.min_solidity:
                self._reject(debug, cx, cy, radius, team, "solidity")
                continue

            found.append(
                Blip(
                    x=float(cx),
                    y=float(cy),
                    radius=float(radius),
                    team=team,
                    score=float(np.clip(0.5 * circularity + 0.5 * solidity, 0.0, 1.0)),
                )
            )
        return found

    def _suppress(self, blips: list[Blip]) -> list[Blip]:
        """Greedy non-max suppression by centre distance.

        Runs across teams as well as within one, because the ambiguous case is
        exactly the cross-team one: a champion standing on an enemy marker can
        punch a hole in both masks.
        """
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
        debug: BlipDebug | None,
        cx: float,
        cy: float,
        radius: float,
        team: Team,
        reason: str,
    ) -> None:
        if debug is None:
            return
        debug.rejected.append(
            (Blip(x=cx, y=cy, radius=radius, team=team, score=0.0), reason)
        )


def scaled_config(
    config: BlipDetectorConfig, minimap_width: int, reference_width: int = 280
) -> BlipDetectorConfig:
    """Retarget a config to a minimap of a different on-screen size.

    Marker radius scales linearly with the minimap panel, so a config fitted at
    one scale-slider setting transfers to any other with just this adjustment.
    The colour bands are scale-invariant and carry over untouched.
    """
    factor = minimap_width / reference_width
    return replace(
        config,
        min_core_radius=config.min_core_radius * factor,
        max_core_radius=config.max_core_radius * factor,
    )
