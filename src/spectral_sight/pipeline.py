"""The whole perception pipeline, wired together.

One object so callers do not have to rebuild the wiring, and so the ordering
constraints live in one place:

1. read the match timer, so the frame has a game time and not just a video one
2. detect markers on the minimap crop (stage 1)
3. locate the camera viewport, which identifies the local player geometrically
4. match markers against the champion gallery, per team (stage 2)
5. accumulate roster evidence, and lock the gallery down once it settles
6. fold everything into the tracker, which carries identity across frames

Matching is per team rather than global. Before the roster locks that changes
little; after, it is what lets a blue marker be compared only against blue
champions.

The local player is identified by the viewport, not by appearance -- their
minimap art is stock while their HUD portrait is skin-specific. They are still
put through the gallery, because the stock icon set contains their champion and
naming them is useful; the viewport's job is to say *which* track is theirs,
which no amount of matching can establish.

The clock and the world transform are both optional and both need a calibration
step of their own, so `for_resolution` loads them if they are there and carries
on without them if they are not. Everything that worked before they existed
still works; a caller that wants them checks whether they arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spectral_sight.perception.hud.clock import (
    ClockFilter,
    ClockReader,
    GameClock,
    load_clock_reader,
)
from spectral_sight.perception.identity import Gallery, Match, load_icon_gallery
from spectral_sight.perception.identity.roster import Roster
from spectral_sight.perception.minimap import (
    BlipDetector,
    BlipDetectorConfig,
    MinimapRegion,
    Viewport,
    WorldTransform,
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
    clock: GameClock | None = None
    """Match time, when the clock is calibrated and readable."""

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
        roster: Roster | None = None,
        clock: ClockReader | None = None,
        world: WorldTransform | None = None,
    ) -> None:
        self.region = region
        self.gallery = gallery
        self.detector = detector or BlipDetector(
            scaled_config(BlipDetectorConfig(), minimap_width=region.width)
        )
        self.tracker = tracker or Tracker(TrackerConfig())
        self.roster = roster or Roster()
        self.clock = clock
        self.world = world
        self._clock_filter = ClockFilter()
        self._restricted: dict[Team, Gallery] = {}

    def world_position(self, x: float, y: float) -> tuple[float, float] | None:
        """Minimap-crop coordinate to world units, if the world is calibrated.

        Tracks and blips both report crop pixels, so this is the one place that
        needs to know the crop's offset within the frame.
        """
        if self.world is None:
            return None
        return self.world.from_minimap(self.region, x, y)

    def _gallery_for(self, team: Team) -> Gallery:
        """The full gallery, or just this team's champions once locked."""
        names = self.roster.locked(team)
        if names is None:
            return self.gallery
        cached = self._restricted.get(team)
        if cached is None:
            cached = Gallery(mask=self.gallery.mask)
            for name in sorted(names):
                cached.add_descriptor(name, self.gallery.entries[name])
            self._restricted[team] = cached
        return cached

    def _apply_roster(self) -> None:
        for team, names in self.roster.names().items():
            self.tracker.enforce_roster(team, names, self.roster.team_size)

    @classmethod
    def for_resolution(
        cls, width: int, height: int, icons: str | Path
    ) -> Pipeline:
        """Build from the calibrated region for a resolution plus an icon set.

        The minimap region and the icons are required. The clock and the world
        transform are picked up if they have been calibrated and skipped
        quietly if not, so adding them to an existing setup is opt-in.
        """
        try:
            clock = load_clock_reader(width, height)
        except FileNotFoundError:
            clock = None
        try:
            world = WorldTransform.for_resolution(width, height)
        except FileNotFoundError:
            world = None
        return cls(
            region=MinimapRegion.for_resolution(width, height),
            gallery=load_icon_gallery(icons),
            clock=clock,
            world=world,
        )

    def process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """Run one frame. `timestamp` is in seconds and must increase."""
        clock = None
        if self.clock is not None:
            clock = self._clock_filter.update(self.clock.read(frame), timestamp)

        minimap = self.region.crop(frame)
        blips = self.detector.detect(minimap)

        viewport = find_viewport(minimap)
        self_blip = self._find_self(blips, viewport)

        matches: list[Match | None] = [None] * len(blips)
        for team in (Team.BLUE, Team.RED):
            indices = [i for i, b in enumerate(blips) if b.team is team]
            if not indices:
                continue
            gallery = self._gallery_for(team)
            regions = [(blips[i].x, blips[i].y, blips[i].radius) for i in indices]
            for index, match in zip(indices, gallery.assign_regions(minimap, regions)):
                matches[index] = match
                if match is not None and match.confident:
                    self.roster.observe(team, match.name, match.margin)

        self.tracker.update(blips, timestamp, matches)
        # Enforcement can drop tracks, so read the surviving set afterwards
        # rather than trusting the snapshot update() returned.
        self._apply_roster()
        tracks = self.tracker.confirmed

        self_track = None
        if self_blip is not None:
            self_track = min(
                (t for t in tracks if t.team is self_blip.team),
                key=lambda t: t.distance_to(self_blip.x, self_blip.y),
                default=None,
            )

        return PipelineResult(
            blips=blips,
            matches=matches,
            tracks=tracks,
            viewport=viewport,
            self_blip=self_blip,
            self_track=self_track,
            clock=clock,
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
