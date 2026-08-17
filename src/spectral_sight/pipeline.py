"""The whole perception pipeline, wired together.

One object so callers do not have to rebuild the wiring, and so the ordering
constraints live in one place:

1. detect markers on the minimap crop (stage 1)
2. locate the camera viewport, which identifies the local player geometrically
3. match the remaining markers against the champion gallery (stage 2)
4. fold everything into the tracker, which carries identity across frames

Step 2 comes before step 3 deliberately. The local player's minimap marker is
stock art while their HUD portrait is skin-specific, so matching them by
appearance fails -- and leaving their marker in the gallery pass also lets it
steal a teammate's identity. Resolving it geometrically first removes both
problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spectral_sight.perception.identity import Gallery, Match, load_icon_gallery
from spectral_sight.perception.minimap import (
    BlipDetector,
    BlipDetectorConfig,
    MinimapRegion,
    Viewport,
    find_viewport,
)
from spectral_sight.perception.minimap.blips import scaled_config
from spectral_sight.tracking import Track, Tracker, TrackerConfig
from spectral_sight.types import Blip, Team

SELF_RADIUS = 12.0
"""How close to the viewport centre a marker must sit to be the local player.
The nearest marker measures 5-7px on real footage with the runner-up 38-88px
away, so this threshold sits comfortably inside a very wide gap."""


@dataclass(slots=True)
class PipelineResult:
    """Everything one frame produced."""

    blips: list[Blip] = field(default_factory=list)
    matches: list[Match | None] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    viewport: Viewport | None = None
    self_blip: Blip | None = None
    self_track: Track | None = None

    def named(self) -> dict[str, Track]:
        """Confirmed tracks that have settled on a champion."""
        return {t.identity: t for t in self.tracks if t.identity is not None}


class Pipeline:
    """Minimap frame in, tracked champions out."""

    def __init__(
        self,
        region: MinimapRegion,
        gallery: Gallery,
        *,
        detector: BlipDetector | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.region = region
        self.gallery = gallery
        self.detector = detector or BlipDetector(
            scaled_config(BlipDetectorConfig(), minimap_width=region.width)
        )
        self.tracker = tracker or Tracker(TrackerConfig())

    @classmethod
    def for_resolution(
        cls, width: int, height: int, icons: str | Path
    ) -> Pipeline:
        """Build from the calibrated region for a resolution plus an icon set."""
        return cls(
            region=MinimapRegion.for_resolution(width, height),
            gallery=load_icon_gallery(icons),
        )

    def process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """Run one frame. `timestamp` is in seconds and must increase."""
        minimap = self.region.crop(frame)
        blips = self.detector.detect(minimap)

        viewport = find_viewport(minimap)
        self_blip = self._find_self(blips, viewport)

        others = [b for b in blips if b is not self_blip]
        matches = self.gallery.assign_regions(
            minimap, [(b.x, b.y, b.radius) for b in others]
        )

        # Re-align matches with the full blip list; the local player is
        # identified by position, so it deliberately carries no gallery match.
        aligned: list[Match | None] = []
        iterator = iter(matches)
        for blip in blips:
            aligned.append(None if blip is self_blip else next(iterator))

        tracks = self.tracker.update(blips, timestamp, aligned)
        self_track = None
        if self_blip is not None:
            self_track = min(
                (t for t in tracks if t.team is self_blip.team),
                key=lambda t: t.distance_to(self_blip.x, self_blip.y),
                default=None,
            )

        return PipelineResult(
            blips=blips,
            matches=aligned,
            tracks=tracks,
            viewport=viewport,
            self_blip=self_blip,
            self_track=self_track,
        )

    @staticmethod
    def _find_self(blips: list[Blip], viewport: Viewport | None) -> Blip | None:
        if viewport is None:
            return None
        cx, cy = viewport.center
        candidates = [b for b in blips if b.team is Team.BLUE]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda b: np.hypot(b.x - cx, b.y - cy))
        if np.hypot(nearest.x - cx, nearest.y - cy) > SELF_RADIUS:
            return None
        return nearest
