"""The whole perception pipeline, wired together.

One object so callers do not have to rebuild the wiring, and so the ordering
constraints live in one place:

1. read the match timer, so the frame has a game time and not just a video one
2. read the friendly HUD portraits, which say how many teammates are dead
3. detect markers on the minimap crop (stage 1)
4. locate the camera viewport, which identifies the local player geometrically
5. match markers against the champion gallery, per team (stage 2)
6. accumulate roster evidence, and lock the gallery down once it settles
7. fold everything into the tracker, which carries identity across frames
8. read the nameplates of champions on screen, and attach them to those tracks
9. flatten the tracked state into observations, which is the output proper

Step 8 runs after the tracker rather than before because it has nothing to
attach to until the tracks exist. It is the only step that reads the 3D view
rather than the minimap or a HUD panel, and the only one whose coverage is set
by where the camera happens to be pointing -- an enemy is inside the camera view
in under a third of frames. What it adds is health, resource and level, and the
resource bar is the sole evidence this project can gather that an enemy used an
ability, because the client never displays an enemy's cooldowns.

Matching is per team rather than global. Before the roster locks that changes
little; after, it is what lets a blue marker be compared only against blue
champions.

The local player is identified by the viewport, not by appearance -- their
minimap art is stock while their HUD portrait is skin-specific. They are still
put through the gallery, because the stock icon set contains their champion and
naming them is useful; the viewport's job is to say *which* track is theirs,
which no amount of matching can establish.

Steps 2 and 7 are read from opposite ends of the screen and are joined in step
8. The HUD knows which portrait slots are dead but not which champions they
are, since portrait art is skin-specific; the minimap knows exactly which
champions are missing but not why. Neither is sufficient alone. Together they
settle it twice over: the local player's slot is named by the camera viewport,
and the ally slots are named by `SlotNaming`, which learns the fixed
slot-to-champion order from the deaths themselves -- see `_attribute_deaths`.

The clock, the world transform, the HUD portraits and the nameplate layout are
all optional and each needs a calibration step of its own, so `for_resolution`
loads them if they are there and carries on without them if they are not.
Everything that worked before they existed still works; a caller that wants them
checks whether they arrived.

Step 8 is why they exist. Game time and world units are the two keys that join
this footage to anything outside it, and a caller that has to remember to apply
them itself will sometimes not -- so `process` produces observations already
converted rather than leaving `world_position` as a method to be discovered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.export import (
    AbilityUse,
    LastHit,
    MinionSighting,
    Observation,
    Skillshot,
    Threat,
    TimelineMeta,
    TurretStatus,
)
from spectral_sight.perception.hud.abilities import AbilityLayout, AbilityReader
from spectral_sight.perception.hud.creep_score import (
    CreepScoreFilter,
    CreepScoreReader,
)
from spectral_sight.perception.hud.gold import GoldFilter, GoldReader
from spectral_sight.perception.hud.skill_points import load_skill_point_reader
from spectral_sight.perception.hud.resources import ResourceReader, load_resource_reader
from spectral_sight.perception.hud.scale import (
    HudScale,
    HudScaleWatch,
    scale_abilities,
    scale_portraits,
    scale_resources,
)
from spectral_sight.perception.nameplates.plates import Side
from spectral_sight.perception.screen.last_hits import LastHitDetector
from spectral_sight.perception.screen import (
    AimDetector,
    EnemyPlate,
    ProjectileTracker,
    ThreatDetector,
    WorldView,
)
from spectral_sight.perception.hud.alive import AliveReader, Liveness
from spectral_sight.perception.hud.clock import (
    ClockFilter,
    ClockReader,
    GameClock,
    load_clock_reader,
)
from spectral_sight.perception.hud.naming import SELF_SLOT, SlotNaming
from spectral_sight.perception.hud.self_champion import (
    SelfChampionReader,
    SpellGallery,
)
from spectral_sight.perception.hud.portraits import PortraitLayout
from spectral_sight.perception.identity import Gallery, Match, load_icon_gallery
from spectral_sight.perception.identity.roster import Roster
from spectral_sight.perception.minimap import (
    BlipDetector,
    BlipDetectorConfig,
    MinimapRegion,
    Viewport,
    WorldTransform,
    find_viewport,
    scaled_viewport_config,
)
from spectral_sight.perception.minimap.blips import scaled_config
from spectral_sight.perception.minimap.minions import (
    MinionDotDetector,
    scaled_dot_config,
)
from spectral_sight.perception.minimap.turrets import (
    TurretReader,
    TurretState,
    project_anchors,
)
from spectral_sight.perception.nameplates import (
    Cast,
    CastBook,
    LevelBook,
    MinionReader,
    Nameplate,
    NameplateLayout,
    NameplateReader,
    ScreenProjection,
    associate,
)
from spectral_sight.profiling import StageTimer
from spectral_sight.tracking import Track, Tracker, TrackerConfig, TrackState
from spectral_sight.types import Blip, Team

SELF_RADIUS = 12.0
"""How close to the viewport centre a marker must sit to be the local player.
The nearest marker measures 5-7px on real footage with the runner-up 38-88px
away, so this threshold sits comfortably inside a very wide gap."""

SELF_CLEARANCE = 20.0
"""How far the nearest blue marker must be from the viewport centre before the
player's marker is taken to be hidden rather than merely off-centre, and one is
placed at the centre in its stead. Wider than SELF_RADIUS so a marker that is
there but a few pixels out is used rather than doubled."""

PLATE_ABOVE_MODEL = 95.0
"""Pixels from a nameplate's bar down to the champion's model, measured on the
2026-08-30 session. The model is what a bolt comes from and goes to."""

MINION_ABOVE_MODEL = 55.0
"""Pixels from a minion's health bar down to its body, measured on the
2026-08-30 session. Minions carry their bars lower than champions do, which is
the difference `_read_minions` corrects for before borrowing the champion
plates' screen-to-minimap fit."""

MIN_DOT_MINIMAP = 400
"""Narrowest minimap panel, in pixels, the minion dots are read on. At 325px a
dot is two or three pixels and indistinguishable from the map's decoration; at
486px it is legible. Nothing between has been looked at."""

TURRET_COVER_MEMORY = 0.35
"""Seconds a track still counts as covering a turret after its marker was
last detected -- a few readings, for the frames the detector drops a marker
that has not moved."""

MINIMAP_INTERVAL = 0.1
"""Seconds between runs of the minimap stages: 10 Hz, whatever rate the window
delivers frames at.

Different signals have different natural rates. Minimap positions do not need
more than 10 Hz -- champions have a speed cap -- and the gallery pass that
identifies them cannot afford more. The world view does: a bolt is gone in a
few tenths of a second, so anything reading it has to see every frame. Feeding
every frame and sampling the slow stages on a clock, rather than decimating at
the source, is what lets both run in one pass.

A clock rather than every Nth frame, because a live capture has no fixed frame
rate to count in: frames are dropped whenever the pipeline falls behind, and a
count tied to `--fps` silently ran the minimap at 3 Hz at the default 10. The
enemy-plate gates in `AimConfig` assume this rate."""

MINIMAP_SLACK = 0.25
"""Share of `MINIMAP_INTERVAL` a frame may arrive early and still be sampled.
Live timestamps jitter by a few milliseconds, and at 10 fps a strict
threshold would skip every frame that landed at 99 ms -- halving the rate the
constant promises. A quarter keeps 30 fps at every third frame."""

GALLERY_RECHECK = 1.0
"""Seconds a named track's marker may go without a gallery read that agrees
with its name. Between reads the marker is carried by position alone -- see
`_needs_reading` -- and the read that comes due is what feeds swap repair and
proves the champion alive to the slot naming."""

GALLERY_CARRY_RADIUS = 8.0
GALLERY_CLEAR_RADIUS = 24.0
"""Minimap pixels. A marker is carried without a read only when it sits within
the carry radius of where its track was predicted, and no other marker or
same-team track is within the clear radius -- anything closer is a tangle, and
a tangle is where a name can jump markers, so it is read every sample."""

PROVEN_SELF_WEIGHT = 1.0
"""Identity evidence the camera-centre marker's track gains per frame for the
champion the ability slots proved -- a fully decisive gallery read's worth,
so a track that the gallery had misnamed (the clipped fountain marker read as
Samira) comes round within seconds."""


@dataclass(frozen=True, slots=True)
class PlayerStatus:
    """Whether this frame has a player row, and if not, why not.

    Published on the feed so a consumer that finds no `is_self` row does not
    have to guess which of several things went wrong.
    """

    champion: str | None
    """The player's champion as far as the pipeline knows it, row or not."""
    source: str | None
    """How the champion was learned: "abilities" (the HUD's spell icons, the
    only thing that identifies the player), or None while unknown. A string
    rather than a flag so a later source would not change the wire format."""
    reason: str | None
    """None when there is a player row. Otherwise one of:
    "no_game" -- the frame does not show the in-game HUD;
    "unidentified" -- the player's champion is not known yet;
    "dead" -- the player is dead and their track was not held;
    "not_on_map" -- the champion is known but their marker is not on the
    minimap: no track is theirs, or theirs has not been seen for longer than
    the tracker's `lost_after`, and the camera centre found no marker."""
    detail: str | None = None
    """Free text elaborating on `reason`, for a person reading the feed."""

    def to_dict(self) -> dict[str, object]:
        return {
            "champion": self.champion,
            "source": self.source,
            "reason": self.reason,
            "detail": self.detail,
        }


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

    game: int = 0
    """Which game in this run the frame belongs to, from zero. Advances when
    the clock goes back to the start of a match -- see `ClockFilter.new_game`
    -- and every piece of per-game state is dropped at that moment."""

    liveness: Liveness | None = None
    """What the HUD portraits say about which teammates are alive, when they
    are calibrated. Slot-indexed, so on its own it says how many are dead but
    not which champions they are -- see `_attribute_deaths` for the two routes
    that close the gap."""

    plates: list[Nameplate] = field(default_factory=list)
    """Champion nameplates read from the world view this frame, if calibrated.
    Both teams, and including any the reader blanked for occlusion."""

    plate_tracks: dict[int, int] = field(default_factory=dict)
    """Index into `plates` to track id, for the plates geometry could place.
    Sparse by design: a plate the gate could not resolve is left out rather
    than attached to a guess, since a plate on the wrong track corrupts that
    champion's whole series while an unmatched one costs a single frame."""

    observations: list[Observation] = field(default_factory=list)
    """One flat, serialisable row per confirmed track -- the same content as
    `tracks`, converted to game time and world units and stripped of the
    tracker's internals. This is what gets written out."""

    sampled: bool = True
    """Whether the minimap stages ran on this frame. False on the frames
    between samples when the pipeline is fed faster than it samples -- see
    `MINIMAP_INTERVAL` -- and then `tracks` and `observations` are empty
    because nothing looked, not because nothing was there. A caller
    publishing rows skips those frames."""

    player: PlayerStatus | None = None
    """Whether there is a player row this frame and why not -- see
    `PlayerStatus`. None on frames between samples."""

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
        portraits: PortraitLayout | None = None,
        nameplates: NameplateLayout | None = None,
        abilities: AbilityLayout | None = None,
        resolution: tuple[int, int] | None = None,
        place_self: bool = True,
        spells: SpellGallery | None = None,
    ) -> None:
        self.region = region
        self.gallery = gallery
        self._last_sample: float | None = None
        """When the minimap stages last ran -- see `MINIMAP_INTERVAL`. The HUD
        stages and the world view run on every call; rows are produced on
        sampled frames only, so the timeline stays at 10 Hz however fast
        frames are fed."""
        self.timer: StageTimer | None = None
        """Set to time each stage of `process` -- see `profiling`. None costs
        nothing, which is the default for every run that did not ask."""
        self.place_self = place_self
        """Whether to place the player's marker at the viewport centre when
        stage 1 cannot see it -- see `_find_self`. A switch so the two can be
        measured against each other."""
        # Frame size this pipeline was calibrated for. Not derivable from the
        # region, which knows only the crop rectangle, and needed only to
        # describe a run in a timeline header.
        self.resolution = resolution
        self.detector = detector or BlipDetector(
            scaled_config(BlipDetectorConfig(), minimap_width=region.width)
        )
        self._viewport_config = scaled_viewport_config(region.width)
        self._map_bounds = None if world is None else (
            world.x - region.x, world.y - region.y, world.width, world.height
        )
        """Where the camera outline can be drawn: the map square inside the
        crop. A box clipped at its edge is fitted against this, not the panel's
        ornate frame."""
        self.tracker = tracker or Tracker(TrackerConfig())
        self.roster = roster or Roster()
        self.clock = clock
        self.world = world
        self.portraits = portraits
        self.liveness = None if portraits is None else AliveReader(portraits)
        self.naming = None if portraits is None else SlotNaming()
        """Learns which champion each ally portrait slot belongs to, so their
        deaths can be attributed by name -- see `_attribute_deaths`."""
        self.nameplates = nameplates
        self.projection = (
            None if nameplates is None
            else ScreenProjection.from_layout(nameplates)
        )
        self.plate_reader = (
            None if nameplates is None
            else NameplateReader(
                nameplates, None if clock is None else clock.glyphs
            )
        )
        """Levels ride on the clock's glyph set, so a run with no clock
        calibration reads plates without them rather than not at all."""

        self.minion_reader = (
            MinionReader(nameplates)
            if nameplates is not None
            and nameplates.minion_width is not None
            and nameplates.minion_height is not None
            else None
        )
        """Minion health bars on the world view: the player's own lane."""
        self.dot_detector = (
            MinionDotDetector(scaled_dot_config(region.width))
            if region.width >= MIN_DOT_MINIMAP
            else None
        )
        """Minion dots on the minimap: every wave on the map."""
        self.turret_reader = (
            TurretReader(
                project_anchors(
                    lambda wx, wy: _frame_to_crop(region, *world.to_frame(wx, wy))
                ),
                region.width,
            )
            if world is not None and region.width >= MIN_DOT_MINIMAP
            else None
        )
        """Which turrets are standing, from their minimap icons. Needs the
        world calibration, which is what places each icon, and the enlarged
        panel the icons were captured at."""
        self._turrets: tuple[TurretState, ...] | None = None
        self.creep_score = (
            CreepScoreReader.beside(clock)
            if clock is not None and self.minion_reader is not None
            else None
        )
        """The player's creep score, which is what says a dying minion was
        theirs. Read only alongside the minion bars, since that is its use."""
        self._cs_filter = CreepScoreFilter()
        self._cs: int | None = None
        self.last_hits: LastHitDetector | None = None
        """Built on the first frame, once the world view's size is known."""
        self._pending_last_hits: list[LastHit] = []

        self.ability_reader = (
            None if abilities is None
            else AbilityReader(
                abilities, None if clock is None else clock.glyphs
            )
        )
        """The local player's own ability slots. Countdown digits ride on the
        clock's glyph set, so a run with no clock calibration still gets the
        casts, just without the seconds printed on them."""
        self._pending_abilities: list[AbilityUse] = []
        """Casts waiting for a self row to ride out on. A cast settles on
        whatever frame confirmed it, and the self track is occasionally
        unresolved on exactly that frame -- holding the cast until the next
        self row loses nothing, where dropping it loses the cast."""
        self.self_reader = (
            None if abilities is None or spells is None
            else SelfChampionReader(spells)
        )
        """Names the player from the icons in their ability slots -- see
        `perception.hud.self_champion`. Reads only until it has settled, and
        is reset only by a new game: the player cannot change champion."""
        self._ability_boxes = None if abilities is None else abilities.boxes()
        """The ability slots at the current HUD scale, for `self_reader`."""
        self.skill_points = (
            None if abilities is None else load_skill_point_reader(abilities)
        )
        """The level-up chevrons above the ability slots: whether a skill
        point is waiting and which abilities could take it. Rides the same
        calibration as the slots and the same gates as the cooldown veil."""
        self._learnable: tuple[str, ...] | None = None
        """The last confirmed chevron reading, carried onto the self row.
        State rather than a queue, unlike the casts: a point waiting is true
        for as long as it waits, and the row reports what is true."""
        self.gold = (
            None if abilities is None or clock is None
            else GoldReader.beside(abilities, clock.glyphs)
        )
        """The player's gold, under the inventory. Part of the panel, so it
        is placed from the ability layout and rebuilt with it on a rescale;
        read with the clock's digits."""
        self._gold_filter = GoldFilter()
        self._gold: int | None = None
        """The filtered gold, set only on a frame whose box read -- see
        `_read_gold`."""

        # The world-view stage: projectiles at every frame, threats to the
        # player resolved against their printed health. It wants every frame
        # it can get -- see `MINIMAP_INTERVAL` -- and needs a nameplate
        # calibration, since the player's own plate is the anchor a bolt is
        # judged against.
        self.projectiles: ProjectileTracker | None = None
        self.threats: ThreatDetector | None = None
        self.aim: AimDetector | None = None
        self.resources: ResourceReader | None = None
        if nameplates is not None:
            self.projectiles = ProjectileTracker()
            self.threats = ThreatDetector()
            # The other end of the same bolts: what the player threw. Needs
            # the ability HUD as well, since a skillshot begins with a cast
            # and an unattributed bolt beside the player is not one.
            self.aim = None if abilities is None else AimDetector()
            if resolution is not None and clock is not None:
                self.resources = load_resource_reader(
                    resolution[0], resolution[1], clock.glyphs
                )
        self.hud_scale = None if abilities is None else HudScaleWatch(abilities)
        """The player panel's size relative to the calibration -- the
        client's HUD scale setting, which the per-resolution files cannot
        know. Measured on a cadence, and on a change every reader of the
        panel is rebuilt on the scaled geometry; see `_apply_hud_scale`."""
        self._panel_sources = (
            abilities,
            portraits,
            None if self.resources is None else self.resources.layout,
        )
        """The calibrated panel geometry, at scale 1.0. Every rescale starts
        from these rather than from the last scaled copy, so rounding never
        accumulates."""
        self._view = WorldView()
        self._anchor: tuple[float, float] | None = None
        self._enemies: list[tuple[float, float]] = []
        """The player's model and the enemy models on the world view, from
        the plates read on the last sampled frame and held until the next.
        Plates move a few pixels between samples; a bolt moves a hundred."""
        self._pending_threats: list[Threat] = []
        self._pending_skillshots: list[Skillshot] = []
        """Resolved shots waiting for a self row, held for the same reason
        the abilities are: the row is the carrier, not the clock."""

        self.levels = LevelBook()
        self.casts = CastBook()
        self._clock_filter = ClockFilter()
        self.game = 0
        """Games seen this run before the current one -- see
        `PipelineResult.game`."""
        self._restricted: dict[Team, Gallery] = {}
        self._gallery_read: dict[int, float] = {}
        """Track id to when the gallery last read its marker as the track's own
        name -- see `_needs_reading`."""

    @property
    def self_champion(self) -> str | None:
        """The champion the local player is on, or None until the ability
        slots have settled it -- the only thing that names the player.

        The minimap used to vote for this: the marker at the camera centre
        named whoever it matched, and the most-voted champion won. That broke
        on an AFK player whose marker, clipped in the fountain corner, read as
        a champion not in the game -- and a vote can only ever be as good as
        the camera being on the player. The minimap now only says *where* the
        proven champion is.
        """
        if self.self_reader is None:
            return None
        return self.self_reader.champion

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

    def _needs_reading(self, blips: list[Blip], timestamp: float) -> list[bool]:
        """Which markers the gallery should read this sample.

        Reading every marker every sample was the most expensive stage in the
        pipeline, and nearly all of it re-confirmed names the tracker already
        held. A marker is carried on position alone when all of these hold:

        - its team's roster has locked, so evidence for the lock is not
          being starved
        - exactly one same-team track is near it, confirmed, named, not
          held, and predicted within `GALLERY_CARRY_RADIUS`
        - no other marker is within `GALLERY_CLEAR_RADIUS`
        - that track's name was confirmed by a read within `GALLERY_RECHECK`

        Everything else is read, so a tangle, a new arrival or a doubtful
        track gets the gallery every sample, as before.
        """
        held = self.tracker.held
        predicted = [
            (t, *t.predict(timestamp - t.last_seen)) for t in self.tracker.tracks
        ]
        reading = [True] * len(blips)
        for i, blip in enumerate(blips):
            if self.roster.locked(blip.team) is None:
                continue
            crowded = any(
                j != i and math.hypot(o.x - blip.x, o.y - blip.y)
                < GALLERY_CLEAR_RADIUS
                for j, o in enumerate(blips)
            )
            if crowded:
                continue
            near = [
                (track, math.hypot(x - blip.x, y - blip.y))
                for track, x, y in predicted
                if track.team is blip.team
                and math.hypot(x - blip.x, y - blip.y) < GALLERY_CLEAR_RADIUS
            ]
            if len(near) != 1:
                continue
            track, distance = near[0]
            last = self._gallery_read.get(track.id)
            reading[i] = not (
                distance < GALLERY_CARRY_RADIUS
                and track.state is TrackState.CONFIRMED
                and track.identity is not None
                and track.id not in held
                and last is not None
                and timestamp - last < GALLERY_RECHECK
            )
        return reading

    def _apply_roster(self) -> None:
        for team, names in self.roster.names().items():
            self.tracker.enforce_roster(team, names, self.roster.team_size)

    @classmethod
    def for_resolution(
        cls, width: int, height: int, icons: str | Path
    ) -> Pipeline:
        """Build from the calibrated region for a resolution plus an icon set.

        The minimap region and the icons are required. The clock, the world
        transform and the HUD portraits are picked up if they have been
        calibrated and skipped quietly if not, so adding any of them to an
        existing setup is opt-in.
        """
        try:
            clock = load_clock_reader(width, height)
        except FileNotFoundError:
            clock = None
        try:
            world = WorldTransform.for_resolution(width, height)
        except FileNotFoundError:
            world = None
        try:
            portraits = PortraitLayout.for_resolution(width, height)
        except FileNotFoundError:
            portraits = None
        try:
            nameplates = NameplateLayout.for_resolution(width, height)
        except FileNotFoundError:
            nameplates = None
        return cls(
            region=MinimapRegion.for_resolution(width, height),
            gallery=load_icon_gallery(icons),
            clock=clock,
            world=world,
            portraits=portraits,
            nameplates=nameplates,
            abilities=AbilityLayout.for_resolution(width, height),
            resolution=(width, height),
            spells=SpellGallery.load(icons),
        )

    def process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """Run one frame. `timestamp` is in seconds and must increase."""
        if self.timer is None:
            return self._process(frame, timestamp)
        self.timer.start()
        result = self._process(frame, timestamp)
        self.timer.finish(sampled=result.sampled)
        return result

    def _lap(self, stage: str) -> None:
        if self.timer is not None:
            self.timer.lap(stage)

    def _apply_hud_scale(self, hud: HudScale) -> None:
        """Rebuild every reader of the player panel on geometry scaled from
        the calibration. Each starts clean: what they had accumulated was
        read off the wrong pixels, which is why the scale was measured."""
        abilities, portraits, resources = self._panel_sources
        glyphs = None if self.clock is None else self.clock.glyphs
        if abilities is not None:
            scaled = scale_abilities(abilities, hud)
            self.ability_reader = AbilityReader(scaled, glyphs)
            self._ability_boxes = scaled.boxes()
            if self.self_reader is not None and self.self_reader.champion is None:
                # Readings so far were of the wrong pixels; a settled answer
                # was not, or it could not have settled.
                self.self_reader.reset()
            self._pending_abilities.clear()
            self.skill_points = load_skill_point_reader(scaled)
            self._learnable = None
            if self.gold is not None:
                self.gold = GoldReader.beside(scaled, self.gold.glyphs)
                self._gold_filter.reset()
                self._gold = None
            if self.aim is not None:
                self.aim.reset()
        if portraits is not None and self.liveness is not None:
            self.portraits = scale_portraits(portraits, hud)
            self.liveness.relayout(self.portraits)
        if resources is not None and self.resources is not None:
            self.resources = ResourceReader(
                scale_resources(resources, hud), self.resources.glyphs
            )

    def _new_game(self) -> None:
        """Forget everything that was about the last game.

        Broader than the resync reset, which keeps the roster, the tracks and
        the player's identity because a seek within one match leaves them
        true. A new match makes all of it evidence about other champions: a
        locked roster would make this game's names unrepresentable, and a
        track carried over would hand its old name to whoever turns up near
        it. What stays is calibration -- geometry, glyphs, the gallery -- and
        the clock filter, which has just adopted the new game's time.

        Track ids keep counting rather than restart, so an id names one track
        across the whole run and a consumer keyed on it cannot mistake a new
        game's champion for an old one's.
        """
        self.game += 1
        ids = self.tracker._ids
        self.tracker = Tracker(self.tracker.config)
        self.tracker._ids = ids
        self.roster = Roster(
            team_size=self.roster.team_size,
            min_evidence=self.roster.min_evidence,
            lock_margin=self.roster.lock_margin,
        )
        self._restricted.clear()
        self._gallery_read.clear()
        if self.self_reader is not None:
            self.self_reader.reset()
        self.levels = LevelBook(confirm=self.levels.confirm)
        self.casts = CastBook(config=self.casts.config)
        if self.liveness is not None:
            self.liveness.reset()
            self.naming.reset()
        if self.ability_reader is not None:
            self.ability_reader.reset()
        self._pending_abilities.clear()
        if self.skill_points is not None:
            self.skill_points.reset()
        self._learnable = None
        self._gold_filter.reset()
        self._gold = None
        if self.turret_reader is not None:
            self.turret_reader.reset()
        self._turrets = None
        if self.hud_scale is not None:
            self.hud_scale.unsettle()
        self._cs_filter.reset()
        self._cs = None
        if self.last_hits is not None:
            self.last_hits.reset()
        self._pending_last_hits.clear()
        for stage in (self.projectiles, self.threats, self.aim):
            if stage is not None:
                stage.reset()
        self._anchor = None
        self._enemies = []
        self._pending_threats.clear()
        self._pending_skillshots.clear()

    def _process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        sampled = (
            self._last_sample is None
            or timestamp - self._last_sample
            >= MINIMAP_INTERVAL * (1 - MINIMAP_SLACK)
        )
        if sampled:
            self._last_sample = timestamp
        clock = None
        if self.clock is not None:
            clock = self._clock_filter.update(self.clock.read(frame), timestamp)
        self._lap("clock")

        # Whether this frame provably shows the in-game HUD: the timer is the
        # one element that can prove it, so with a clock calibrated, trust
        # follows the clock resolving. With none there is no proof to wait
        # for, and every frame is taken at its word -- the behaviour that
        # predates the clock.
        trusted = self.clock is None or (clock is not None and clock.observed)

        if self._clock_filter.new_game:
            self._new_game()
        elif self._clock_filter.resynced:
            # The clock just contradicted its own prediction: a seek, or a
            # different game spliced into the same capture. What every HUD
            # reader has accumulated is evidence about footage that ended --
            # the portrait boxes may hold different champions now, and an
            # ability slot's last clear reading describes pixels from another
            # game.
            if self.liveness is not None:
                self.liveness.reset()
                self.naming.reset()
            if self.ability_reader is not None:
                self.ability_reader.reset()
                self._pending_abilities.clear()
            if self.skill_points is not None:
                self.skill_points.reset()
                self._learnable = None
            self._gold_filter.reset()
            self._gold = None
            if self.turret_reader is not None:
                self.turret_reader.reset()
                self._turrets = None
            if self.hud_scale is not None:
                # A different game may be at a different HUD setting.
                self.hud_scale.unsettle()

        if self.hud_scale is not None and trusted:
            # Before every panel reader, so the frame that settles a new
            # scale is also the first one read at it.
            scale = self.hud_scale.update(frame, timestamp)
            if scale is not None:
                height, width = frame.shape[:2]
                self._apply_hud_scale(HudScale.for_frame(scale, width, height))
        self._lap("hud scale")

        liveness = None
        if self.liveness is not None:
            # Baselines learn only from trusted frames. A recording that
            # starts in queue otherwise teaches the portraits splash art,
            # which out-saturates any living champion -- and a running
            # maximum never comes back down, so the real HUD would read dead
            # from its first frame onward.
            liveness = self.liveness.read(frame, learn=trusted)
        self._lap("portraits")

        if self.ability_reader is not None:
            # Two gates, both structural. An untrusted frame is not read at all
            # -- this reader has no baseline to fall back on, only transitions,
            # and the one thing that matters is never deriving a transition
            # from a screen that is not the game. And a dead player casts
            # nothing: death greys every slot at once, which without this reads
            # as a burst of simultaneous casts as each ready ability veils. The
            # self portrait is the death signal the pipeline already trusts, so
            # ability reading stops the moment it reads dead and resumes -- from
            # a clean slate, since the on-screen cooldowns are stale -- when it
            # does not. A frame where the portrait is merely unreadable (None)
            # is not proof of death, so it is still read.
            portrait = None if liveness is None else liveness.slot(SELF_SLOT)
            dead = portrait is not None and portrait.alive is False
            if trusted and not dead:
                for cast in self.ability_reader.read(frame, timestamp):
                    self._pending_abilities.append(AbilityUse(
                        slot=cast.slot,
                        at=cast.at,
                        countdown=cast.countdown,
                        confirmed=cast.confirmed,
                    ))
                    if self.aim is not None:
                        self.aim.observe_cast(cast.slot, cast.at)
            elif dead:
                self.ability_reader.reset()
                if self.aim is not None:
                    # Death veils every slot; whatever the aim stage was
                    # holding was cast in a life that has ended.
                    self.aim.reset()
        self._lap("abilities")

        if (self.self_reader is not None and self.self_reader.champion is None
                and sampled and trusted and self._ability_boxes is not None):
            # Only until it settles, and not on the death screen, which greys
            # every slot into something no kit looks like.
            portrait = None if liveness is None else liveness.slot(SELF_SLOT)
            if not (portrait is not None and portrait.alive is False):
                self.self_reader.read(frame, self._ability_boxes)
        self._lap("self champion")

        if self.skill_points is not None:
            # The same two gates, for the same reasons: a screen that is not
            # the game has no chevrons to read, and the death screen is not
            # evidence about them either way. Dead, the reading is dropped
            # rather than carried -- the row says nothing looked, and a point
            # spent from the grey screen is reported on respawn, late rather
            # than wrong.
            portrait = None if liveness is None else liveness.slot(SELF_SLOT)
            dead = portrait is not None and portrait.alive is False
            if trusted and not dead:
                self._learnable = self.skill_points.read(frame, timestamp)
            elif dead:
                self.skill_points.reset()
                self._learnable = None
        self._lap("skill points")

        self._read_gold(frame, trusted)
        self._lap("gold")

        if self.projectiles is not None and self.threats is not None:
            self._watch_world(frame, timestamp, trusted)
        self._lap("world view")

        if not sampled:
            return PipelineResult(
                clock=clock, game=self.game, liveness=liveness, sampled=False
            )

        minimap = self.region.crop(frame)
        blips = self.detector.detect(minimap)
        self._lap("minimap markers")

        viewport = find_viewport(minimap, self._viewport_config, self._map_bounds)
        me = None if liveness is None else liveness.slot(SELF_SLOT)
        self_dead = me is not None and me.alive is False
        self_blip, placed = self._find_self(
            blips, viewport,
            # Placing a marker asserts the camera is on a living player on
            # an in-game frame; each of those is something the pipeline can
            # check, so each is required rather than assumed.
            place=self.place_self and trusted and not self_dead,
        )
        if self_dead:
            # A dead player has no marker. Whatever sits at the camera centre
            # is a teammate the death-cam is watching or one standing by the
            # corpse, and calling it the player hands them the self row.
            self_blip = None
        if placed:
            blips = [*blips, self_blip]
        corpse = self._hold_self(self_dead, self_blip, timestamp)
        self._lap("viewport + self")

        matches: list[Match | None] = [None] * len(blips)
        reading = self._needs_reading(blips, timestamp)
        proven = (None if self.self_reader is None
                  else self.self_reader.champion)
        for team in (Team.BLUE, Team.RED):
            # A placed marker is kept away from the gallery: whatever is
            # drawn at the centre is the thing covering the player's icon,
            # most often an enemy's, and a confident match there would hand
            # an enemy's name to the blue roster. So is the player's own
            # marker once the ability slots have proved who they are: it
            # carries that name instead (below), and a read could only be
            # worse -- the AFK Annie's marker, clipped in the fountain
            # corner, read as Samira on a third of the frames.
            indices = [i for i, b in enumerate(blips)
                       if b.team is team and reading[i]
                       and not ((placed or proven is not None)
                                and b is self_blip)]
            if not indices:
                continue
            gallery = self._gallery_for(team)
            regions = [(blips[i].x, blips[i].y, blips[i].radius) for i in indices]
            for index, match in zip(indices, gallery.assign_regions(minimap, regions)):
                matches[index] = match
                if match is not None and match.confident:
                    self.roster.observe(team, match.name, match.margin)
        self._lap("gallery match")

        self.tracker.update(blips, timestamp, matches)
        if proven is not None and self_blip is not None and not placed:
            # A detected marker only. One placed at the centre is the camera's
            # guess at where the player is, and at the fountain -- camera
            # clamped in the corner -- it feeds phantoms; giving them the
            # name would win it off the player's real track.
            index = next(i for i, b in enumerate(blips) if b is self_blip)
            track = self.tracker.assignment.get(index)
            holder = self.tracker.identified().get(proven)
            if track is not None and holder is not None and holder is not track:
                # The player already has a track, so the one at the centre is
                # someone else's -- a teammate standing on them, Fiddlesticks
                # back in the fountain beside the AFK Annie. Only when no
                # track carries the name is the centre track given it: that
                # is the clipped fountain marker the gallery misread (as
                # Samira), which then has nothing else to name it.
                track = None
            if track is not None:
                # The centre marker is where the camera says the player is;
                # the slots have said who, so the marker's track takes the
                # name rather than the gallery's read of a clipped icon.
                track.observe_identity(proven, PROVEN_SELF_WEIGHT)
                self.roster.observe(Team.BLUE, proven, PROVEN_SELF_WEIGHT)
        carried: set[str] = set()
        for index, track in self.tracker.assignment.items():
            match = matches[index]
            if not reading[index]:
                if track.identity is not None and track.team is Team.BLUE:
                    carried.add(track.identity)
            elif (match is not None and match.confident
                  and match.name == track.identity):
                self._gallery_read[track.id] = timestamp
        # Enforcement can drop tracks, so read the surviving set afterwards
        # rather than trusting the snapshot update() returned.
        self._apply_roster()
        live = {t.id for t in self.tracker.tracks}
        self._gallery_read = {
            k: v for k, v in self._gallery_read.items() if k in live
        }
        tracks = self.tracker.confirmed
        self._lap("tracker")

        # The player's track is the one the tracker fed with the player's
        # marker -- not the nearest track to it, which with no gate at all
        # took whichever teammate's track sat closest to the fountain on a
        # respawn and named the player after them for the rest of the game.
        # A marker that only started a tentative track resolves nobody yet.
        fed = None
        if self_blip is not None:
            index = next(i for i, b in enumerate(blips) if b is self_blip)
            fed = self.tracker.assignment.get(index)
            if fed is not None and not any(t is fed for t in tracks):
                fed = None
        # Nobody is the player until the ability slots say who they are; the
        # minimap only places them.
        self_track = None
        if proven is None:
            pass
        elif corpse is not None and any(t is corpse for t in tracks):
            # Dead, the self row is the corpse: where the player fell, under
            # their name, reading dead for the whole timer.
            self_track = corpse
        else:
            # The track under the proven name, wherever it is: the camera
            # centre is not always the player -- clamped in the fountain
            # corner, or a replay's free camera -- and a proven name needs no
            # camera box to vouch for it. Failing that, the centre marker's
            # track, which carries the proven name from this frame on, unless
            # it is already someone else's: a teammate standing on the player,
            # or a misread not yet overwritten. No row beats the wrong row.
            self_track = self._named_self(tracks, timestamp)
            if self_track is None and fed is not None and fed.identity in (None, proven):
                self_track = fed

        if self.naming is not None and liveness is not None:
            # An ally is drawn on the minimap exactly while alive, so the
            # champions the gallery matched with confidence this frame are
            # proven living -- the negative space is what names a dead slot.
            # A marker carried without a read is its track's champion by
            # the same standard, so it counts as seen too.
            seen = {
                match.name
                for blip, match in zip(blips, matches)
                if match is not None and match.confident
                and blip.team is Team.BLUE
            } | carried
            self.naming.update(
                liveness, seen, self.roster.locked(Team.BLUE),
                self.self_champion, timestamp, trusted=trusted,
            )
        self._lap("self + naming")

        # One colour conversion of the whole frame, shared by both readers of
        # the world view's bars -- it is the largest single cost either has.
        hsv = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.plate_reader is not None or self.minion_reader is not None
            else None
        )
        self._lap("frame to HSV")
        plates, pairing, casts = self._read_plates(
            frame, tracks, viewport, timestamp, hsv, self_track
        )
        self._lap("nameplates")
        minions = self._read_minions(frame, viewport, trusted, hsv)
        self._lap("minions")
        self._judge_last_hits(
            frame, timestamp, trusted, minions,
            dead=me is not None and me.alive is False,
        )
        self._lap("last hits")
        minion_dots = self._read_minion_dots(minimap, blips, trusted)
        self._lap("minion dots")
        self._read_turrets(minimap, blips, timestamp, trusted, self_blip)
        self._lap("turrets")

        result = PipelineResult(
            game=self.game,
            blips=blips,
            matches=matches,
            tracks=tracks,
            viewport=viewport,
            self_blip=self_blip,
            self_track=self_track,
            player=self._player_status(self_track, trusted, self_dead, timestamp,
                                       fed),
            clock=clock,
            liveness=liveness,
            plates=plates,
            plate_tracks=pairing,
            observations=self._observe(
                tracks, timestamp, clock, self_track, liveness, plates,
                pairing, casts, self._take_abilities(self_track),
                self._take_threats(self_track),
                self._take_skillshots(self_track),
                self._learnable,
                minions=minions,
                minion_dots=minion_dots,
                turrets=self._turrets,
                cs=self._cs,
                gold=self._gold,
                last_hits=self._take_last_hits(self_track),
            ),
        )
        self._lap("observations")
        return result

    def _judge_last_hits(
        self,
        frame: np.ndarray,
        timestamp: float,
        trusted: bool,
        minions: tuple[MinionSighting, ...] | None,
        *,
        dead: bool,
    ) -> None:
        """Read the creep score and follow enemy minion bars to their deaths."""
        if self.creep_score is None or not trusted:
            return
        self._cs = self._cs_filter.update(self.creep_score.read(frame))
        if self.last_hits is None:
            height, width = frame.shape[:2]
            _, _, view_w, view_h = self._view.box(width, height)
            # Minion positions are world-view pixels, so the view is the box
            # at the origin.
            self.last_hits = LastHitDetector((0, 0, view_w, view_h))
        if dead or minions is None:
            # A dead player's camera is on their corpse or roaming; their
            # bars are not the lane's.
            self.last_hits.reset()
        else:
            self.last_hits.observe(
                timestamp,
                [(m.x, m.y, m.health) for m in minions if m.team == "red"],
                self._cs,
            )
        self._pending_last_hits.extend(self.last_hits.resolve(timestamp))

    def _read_gold(self, frame: np.ndarray, trusted: bool) -> None:
        """Read the gold on every trusted frame, dead or alive -- the shop
        is open to a dead player, and the number is drawn either way.

        `_gold` is the filtered figure on a frame whose box read, and None
        on one where it did not: a consumer gating a purchase on it should
        see "not looked at", not the last figure carried past a read that
        failed."""
        self._gold = None
        if self.gold is None or not trusted:
            return
        reading = self.gold.read(frame)
        value = self._gold_filter.update(reading)
        if reading is not None:
            self._gold = value

    def _take_last_hits(self, self_track: Track | None) -> tuple[LastHit, ...]:
        if self_track is None or not self._pending_last_hits:
            return ()
        taken = tuple(self._pending_last_hits)
        self._pending_last_hits.clear()
        return taken

    def _read_minions(
        self,
        frame: np.ndarray,
        viewport: Viewport | None,
        trusted: bool,
        hsv: np.ndarray | None = None,
    ) -> tuple[MinionSighting, ...] | None:
        """Minions on the world view, or None when nothing looked."""
        if self.minion_reader is None or not trusted:
            return None
        height, width = frame.shape[:2]
        vx, vy, _, _ = self._view.box(width, height)
        placed = (
            self.projection is not None
            and viewport is not None
            and self.world is not None
        )
        sightings = []
        for minion in self.minion_reader.read(frame, hsv):
            cx, top = minion.center
            body = top + MINION_ABOVE_MODEL
            world = None
            if placed:
                # Where a champion's bar would float over the same body.
                mx, my = self.projection.point_to_minimap(
                    cx, body - PLATE_ABOVE_MODEL, viewport, (width, height)
                )
                world = self.world_position(mx, my)
            sightings.append(MinionSighting(
                team=minion.team.value,
                x=cx - vx,
                y=body - vy,
                health=minion.health,
                world_x=None if world is None else world[0],
                world_y=None if world is None else world[1],
            ))
        return tuple(sightings)

    def _read_minion_dots(
        self, minimap: np.ndarray, blips: list[Blip], trusted: bool
    ) -> tuple[MinionSighting, ...] | None:
        """Minions on the minimap, or None when nothing looked."""
        if self.dot_detector is None or not trusted:
            return None
        sightings = []
        for dot in self.dot_detector.detect(
            minimap, [(b.x, b.y, b.radius) for b in blips]
        ):
            world = self.world_position(dot.x, dot.y)
            sightings.append(MinionSighting(
                team=dot.team.value,
                x=float(dot.x),
                y=float(dot.y),
                world_x=None if world is None else world[0],
                world_y=None if world is None else world[1],
            ))
        return tuple(sightings)

    def _read_turrets(
        self,
        minimap: np.ndarray,
        blips: list[Blip],
        timestamp: float,
        trusted: bool,
        self_blip: Blip | None,
    ) -> None:
        """Update the turret verdicts from this frame's minimap.

        Cover is every marker the detector found this frame, whether or not
        the tracker has confirmed it -- on the live receiver the markers
        crossing a turret were mostly on tentative tracks -- plus any track
        seen in the last `TURRET_COVER_MEMORY` seconds, for the frame the
        detector misses a marker that is still there. The detector also
        fires on the turret-and-inhibitor clusters themselves (20-58% of
        frames at the busiest spots), which costs nothing: a turret whose
        icon is clearly seen reads standing before cover is asked about, and
        a destroyed one only needs its uncovered readings to add up.
        """
        if self.turret_reader is None or not trusted:
            return
        radius = float(np.median([b.radius for b in blips])) if blips else 13.0
        markers = [(b.x, b.y, b.radius) for b in blips]
        markers.extend(
            (t.x, t.y, radius) for t in self.tracker.tracks
            if t.age(timestamp) <= TURRET_COVER_MEMORY
        )
        if self_blip is not None:
            markers.append((self_blip.x, self_blip.y, self_blip.radius))
        self._turrets = self.turret_reader.read(minimap, timestamp, markers)

    def _watch_world(
        self, frame: np.ndarray, timestamp: float, trusted: bool
    ) -> None:
        """The every-frame stage: bolts, and what became of the ones aimed
        at the player. Runs before the sampled stages so a frame between
        samples still advances it."""
        assert self.projectiles is not None and self.threats is not None
        if self.resources is not None:
            reading = self.resources.read_line(frame, self.resources.layout.health)
            if reading is not None and reading.plausible:
                self.threats.observe_health(timestamp, reading.current)
        # Only bolt-shaped tracks are judged; the tracker finishes every
        # mover it followed, and a walking minion is not a threat however
        # straight it walks.
        candidates = [
            track for track in self.projectiles.update(frame, timestamp)
            if track.is_projectile(self.projectiles.config)
        ]
        motion = self.projectiles.last_motion
        if motion is not None:
            self.threats.observe_motion(timestamp, motion)
            if self.aim is not None:
                self.aim.observe_motion(timestamp, motion)
        if trusted:
            self.threats.consider(candidates, self._anchor, self._enemies)
            if self.aim is not None:
                self.aim.consider(candidates, self._anchor)
        self._pending_threats.extend(self.threats.resolve(timestamp))
        if self.aim is not None:
            self._pending_skillshots.extend(self.aim.resolve(timestamp))

    def _hold_models(
        self, plates: list[Nameplate], frame_size: tuple[int, int],
        timestamp: float,
    ) -> None:
        """Remember where the player and the enemies stand on the world view,
        from this sampled frame's plates, for the frames until the next."""
        vx, vy, _, _ = self._view.box(*frame_size)

        def model(plate: Nameplate) -> tuple[float, float]:
            cx, cy = plate.center
            return cx - vx, cy - vy + PLATE_ABOVE_MODEL

        mine = [p for p in plates if p.side is Side.SELF]
        self._anchor = model(mine[0]) if len(mine) == 1 else None
        hostile = [p for p in plates if p.hostile]
        self._enemies = [model(p) for p in hostile]
        if self.aim is not None:
            # The aim stage needs the bar as well as the position: an enemy
            # is a target to hit, not just a place a bolt came from.
            self.aim.observe_enemies(timestamp, [
                EnemyPlate(*model(p), health=p.health) for p in hostile
            ])

    def _take_threats(self, self_track: Track | None) -> tuple[Threat, ...]:
        if self_track is None or not self._pending_threats:
            return ()
        taken = tuple(self._pending_threats)
        self._pending_threats.clear()
        return taken

    def _take_skillshots(self, self_track: Track | None) -> tuple[Skillshot, ...]:
        if self_track is None or not self._pending_skillshots:
            return ()
        taken = tuple(self._pending_skillshots)
        self._pending_skillshots.clear()
        return taken

    def _take_abilities(
        self, self_track: Track | None
    ) -> tuple[AbilityUse, ...]:
        """Casts to hang on this frame's self row, or an empty tuple.

        Abilities belong to the local player, so they ride the self row. When
        no self track resolved this frame the casts are held rather than
        dropped -- the reader confirms a cast a frame after it happened, and
        the self track occasionally blinks out on exactly that frame. Held,
        they land on the next self row, seconds later at worst; dropped, the
        cast is gone.
        """
        if self_track is None or not self._pending_abilities:
            return ()
        taken = tuple(self._pending_abilities)
        self._pending_abilities.clear()
        return taken

    def _read_plates(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        viewport: Viewport | None,
        timestamp: float,
        hsv: np.ndarray | None = None,
        self_track: Track | None = None,
    ) -> tuple[list[Nameplate], dict[int, int], dict[int, Cast]]:
        """Read nameplates, attach them to tracks, and call any casts.

        Plates are matched against *both* teams' tracks rather than just the
        enemy's. An ally plate is less interesting -- the HUD already says how
        many teammates are alive -- but it is the same read for free, and an
        ally's resource is the one case where a cast can be checked against
        something else that was observed.

        The player's own plate is the exception to matching by distance: its
        green bar names it outright, so it goes to the self track and nowhere
        else, and no teammate's plate goes to the player. Matched by distance
        alone, a teammate's bar could land on the player's track the moment
        the green one vanished in a death -- the likeliest source of a live
        reading that jumped from 23% to 90% health, with a level-up, in the
        instant the player died.
        """
        if self.plate_reader is None:
            return [], {}, {}

        plates = self.plate_reader.read(frame, hsv)
        height, width = frame.shape[:2]
        if self.threats is not None:
            self._hold_models(plates, (width, height), timestamp)
        pairing: dict[int, int] = {}
        for team, hostile in ((Team.RED, True), (Team.BLUE, False)):
            indices = [i for i, p in enumerate(plates) if p.hostile is hostile]
            if not indices:
                continue
            side = [t for t in tracks if t.team is team]
            if team is Team.BLUE:
                mine = [i for i in indices if plates[i].side is Side.SELF]
                if self_track is not None and len(mine) == 1:
                    pairing[mine[0]] = self_track.id
                indices = [i for i in indices if plates[i].side is not Side.SELF]
                player = self.self_champion
                side = [
                    t for t in side
                    if t is not self_track
                    and (player is None or t.identity != player)
                ]
                if not indices:
                    continue
            local = associate(
                [plates[i] for i in indices], side, viewport, self.projection,
                (width, height),
            )
            for local_index, track_id in local.items():
                pairing[indices[local_index]] = track_id

        # Levels and resource series belong to champions, so they accumulate
        # against the track and have to be dropped when the tracker drops it --
        # otherwise a reused id would inherit a stranger's level, or read a
        # stranger's mana as one enormous step the first time it sees a plate.
        live = {t.id for t in self.tracker.tracks}
        for track_id in [i for i in self.levels.filters if i not in live]:
            self.levels.forget(track_id)
        for track_id in [i for i in self.casts.detectors if i not in live]:
            # Whatever candidate it was holding dies with it. That is at most
            # one unconfirmed cast on a track the tracker has already given up
            # on, which is not worth a row with no champion attached to it.
            self.casts.forget(track_id)

        # Levels first: a cast records the level it was cast at, and reading it
        # before this frame's digit is folded in would stamp a stale one.
        casts: dict[int, Cast] = {}
        for index, track_id in pairing.items():
            self.levels.update(track_id, plates[index].level)
        for index, track_id in pairing.items():
            cast = self.casts.update(
                track_id, timestamp, plates[index].resource,
                plates[index].health, self.levels.level(track_id),
            )
            if cast is not None:
                casts[track_id] = cast

        return plates, pairing, casts

    def _player_status(
        self, self_track: Track | None, trusted: bool, dead: bool,
        timestamp: float, centre: Track | None = None,
    ) -> PlayerStatus:
        """Who the player is, how that is known, and why there is no row if
        there is none -- the answer to "coach: no player row" in one place."""
        name = self.self_champion
        reader = self.self_reader
        source = None if name is None else "abilities"
        if self_track is not None:
            return PlayerStatus(name, source, None)
        if not trusted:
            return PlayerStatus(name, source, "no_game",
                                "the in-game HUD is not on screen")
        if name is None:
            if reader is None:
                detail = ("no spell icons or no ability calibration, so the "
                          "player cannot be identified: run "
                          "tools/fetch_icons.py")
            elif reader.last is None:
                detail = "ability slots not read yet"
            else:
                last = reader.last
                detail = (f"ability slots look most like {last.champion} "
                          f"(score {last.score:.2f}, lead {last.margin:.2f}), "
                          "not yet settled")
            return PlayerStatus(None, None, "unidentified", detail)
        if dead:
            return PlayerStatus(name, source, "dead",
                                f"{name} is dead and no track was held")
        track = self.tracker.identified().get(name)
        if track is not None:
            detail = (f"{name}'s marker has not been seen for "
                      f"{track.age(timestamp):.1f}s")
            if centre is not None and centre.identity not in (None, name):
                detail += f"; the camera centre is on {centre.identity}"
        else:
            detail = (f"no track on the minimap is {name}'s and the camera "
                      "centre found no marker")
        return PlayerStatus(name, source, "not_on_map", detail)

    def _named_self(
        self, tracks: list[Track], timestamp: float
    ) -> Track | None:
        """The recently seen track under the player's proven name."""
        name = self.self_champion
        if name is None:
            return None
        lost_after = self.tracker.config.lost_after
        for track in tracks:
            if track.identity == name and track.age(timestamp) < lost_after:
                return track
        return None
        proven = self.self_reader is not None and self.self_reader.champion == name
        if viewport is None and not proven:
            return None
        lost_after = self.tracker.config.lost_after
        for track in tracks:
            if track.identity != name or track.age(timestamp) >= lost_after:
                continue
            if proven or (
                viewport.x <= track.x <= viewport.x + viewport.width
                and viewport.y <= track.y <= viewport.y + viewport.height
            ):
                return track
        return None

    def _hold_self(
        self, dead: bool, self_blip: Blip | None, timestamp: float
    ) -> Track | None:
        """Keep the player's track where they fell until they respawn.

        While the self portrait reads dead, the track carrying the player's
        name is held out of association (see `Tracker.hold`) and returned, so
        it can stand as the self row. Once the portrait reads alive and the
        player's marker is found again -- at the fountain -- the track is
        released onto that marker, and the name comes back with the player
        instead of being left on whatever the tracker would have paired.

        Only the player is held this way: theirs is the one death the pipeline
        can pin on a track, and theirs is the one marker it can find again
        without the gallery.
        """
        held = self.tracker.held
        if dead:
            corpse = next((t for t in self.tracker.tracks if t.id in held), None)
            if corpse is not None:
                return corpse
            name = self.self_champion
            track = None if name is None else self.tracker.identified().get(name)
            if track is not None:
                self.tracker.hold(track.id)
            return track
        if self_blip is not None:
            for track_id in held:
                self.tracker.release(track_id, self_blip.x, self_blip.y, timestamp)
        return None

    def _attribute_deaths(
        self,
        tracks: list[Track],
        liveness: Liveness | None,
    ) -> dict[int, bool | None]:
        """Work out which *tracks* the HUD's dead slots refer to.

        The HUD counts dead teammates but does not name them: a portrait slot is
        skin-specific art, which is the one gallery this project has already
        established cannot be trusted to identify a champion.

        **The obvious way to close that gap does not work, and it is worth
        recording why.** An ally is drawn on the minimap only while alive, so
        one dead portrait ought to mean one ally track missing, and matching the
        counts ought to name the casualty. Measured, it names the wrong one. The
        marker really does disappear -- blue detections fall from a mean of 5.84
        per frame to 5.24 when a teammate dies -- but stage 1 over-produces by
        about one marker per frame by design, so five candidates remain and the
        tracker, capped at five per team, keeps feeding all five tracks. No
        track ever goes quiet. What the counts then match on is ordinary
        frame-to-frame blinking, which hands the dead champion's name to
        whichever living ally the detector happened to drop that instant. On the
        sample clip that turned one twelve-second death into a dozen fragments
        spread across three champions who were never dead at all.

        So deaths are only attributed to champions that can be named outright,
        and there are exactly two such routes:

        - **The local player**, who is resolved from the camera viewport
          rather than by appearance. Their own portrait is a known slot, so
          when it greys out, the champion at the centre of the camera is the
          one who died -- no counting involved. It has to be `self_champion`,
          the accumulated answer, rather than this frame's: the viewport finds
          the player by the marker at the camera centre, and a dead player has
          no marker, so `self_track` resolves in none of the frames in which
          the self portrait reads dead.
        - **An ally slot `SlotNaming` has locked onto a champion.** The
          counting that fails per frame works integrated over a whole death:
          across the seconds a slot stays dead, the living allies keep being
          confidently matched on the minimap and the casualty does not, and
          the slot-to-champion pairing that survives that evidence is locked
          for the match -- portrait order is fixed. See `naming` for the
          voting; here a locked slot simply names its dead champion the way
          the self slot always has.

        Every slot -- the local player's included -- is judged on
        `SlotNaming`'s *debounced* state rather than this frame's raw
        reading, which is what keeps the HUD warm-up flicker and the
        post-game flapping -- both sub-second -- from ever reaching the
        timeline as deaths. Measured, the post-game screen flapped the self
        portrait into five sub-two-second "deaths" that the old raw reading
        emitted as real; the cost of the hold is each event landing a second
        late, symmetrically, so `down_for` is unchanged.

        A champion whose own slot is known answers for themselves: their
        verdict is that slot's state, whatever the rest of the frame looks
        like. Everyone else is cleared collectively, and only when the frame
        is resolved end to end -- every slot has a definite state, every dead
        slot has a name, and every named casualty is among the ally tracks'
        identities (otherwise a track about to be called alive could be the
        corpse under a name that has not settled yet). An unresolved remainder
        reports None rather than guessing, and `allies_dead` still carries the
        raw count, so a consumer knows someone was down even when nobody can
        say who.

        Enemies are always None. Fog means their absence says nothing, and no
        HUD panel names them.
        """
        verdicts: dict[int, bool | None] = {t.id: None for t in tracks}
        if liveness is None:
            return verdicts

        states: dict[str, bool | None] = {}
        names: dict[str, str | None] = {}
        for slot in liveness.slots:
            states[slot.slot] = (
                None if self.naming is None else self.naming.state(slot.slot)
            )
            if slot.slot == SELF_SLOT:
                names[slot.slot] = self.self_champion or None
            else:
                names[slot.slot] = (
                    None if self.naming is None else self.naming.name(slot.slot)
                )

        by_name = {
            name: states[slot]
            for slot, name in names.items()
            if name is not None
        }
        named_dead = {name for name, state in by_name.items() if state is False}

        allies = [t for t in tracks if t.team is Team.BLUE]
        identities = {t.identity for t in allies}
        resolved = (
            all(state is not None for state in states.values())
            and all(
                names[slot] is not None
                for slot, state in states.items() if state is False
            )
            and named_dead <= identities
        )

        for track in allies:
            verdict = (
                None if track.identity is None
                else by_name.get(track.identity)
            )
            if verdict is None and resolved:
                verdict = True
            verdicts[track.id] = verdict
        return verdicts

    def _observe(
        self,
        tracks: list[Track],
        timestamp: float,
        clock: GameClock | None,
        self_track: Track | None,
        liveness: Liveness | None = None,
        plates: list[Nameplate] | None = None,
        pairing: dict[int, int] | None = None,
        casts: dict[int, Cast] | None = None,
        abilities: tuple[AbilityUse, ...] = (),
        threats: tuple[Threat, ...] = (),
        skillshots: tuple[Skillshot, ...] = (),
        learnable: tuple[str, ...] | None = None,
        *,
        minions: tuple[MinionSighting, ...] | None = None,
        minion_dots: tuple[MinionSighting, ...] | None = None,
        turrets: tuple[TurretState, ...] | None = None,
        cs: int | None = None,
        gold: int | None = None,
        last_hits: tuple[LastHit, ...] = (),
    ) -> list[Observation]:
        """Flatten this frame's tracks into rows.

        Ordered by track id so two runs over the same clip produce the same
        file, which the tracker's own list does not guarantee.
        """
        lost_after = self.tracker.config.lost_after
        alive = self._attribute_deaths(tracks, liveness)
        by_track = {
            track_id: plates[index]
            for index, track_id in (pairing or {}).items()
        }
        rows = []
        for track in sorted(tracks, key=lambda t: t.id):
            plate = by_track.get(track.id)
            cast = (casts or {}).get(track.id)
            world = self.world_position(track.x, track.y)
            age = track.age(timestamp)
            rows.append(
                Observation(
                    video_time=timestamp,
                    track_id=track.id,
                    team=track.team,
                    x=track.x,
                    y=track.y,
                    visible=age < lost_after,
                    seconds_since_seen=age,
                    game_time=None if clock is None else clock.total_seconds,
                    game_time_observed=clock is not None and clock.observed,
                    game=self.game,
                    champion=track.identity,
                    world_x=None if world is None else world[0],
                    world_y=None if world is None else world[1],
                    is_self=self_track is not None and track.id == self_track.id,
                    health=None if plate is None else plate.health,
                    resource=None if plate is None else plate.resource,
                    level=self.levels.level(track.id),
                    cast_drop=None if cast is None else cast.drop,
                    cast_at=None if cast is None else cast.at,
                    cast_span=None if cast is None else cast.span,
                    cast_continuous=None if cast is None else cast.continuous,
                    cast_confirmed=None if cast is None else cast.confirmed,
                    alive=alive.get(track.id),
                    abilities=(
                        abilities
                        if abilities
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    threats=(
                        threats
                        if threats
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    skillshots=(
                        skillshots
                        if skillshots
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    learnable=(
                        learnable
                        if self_track is not None and track.id == self_track.id
                        else None
                    ),
                    minions=(
                        minions
                        if self_track is not None and track.id == self_track.id
                        else None
                    ),
                    minion_dots=(
                        minion_dots
                        if self_track is not None and track.id == self_track.id
                        else None
                    ),
                    turrets=(
                        tuple(
                            TurretStatus(
                                team=t.team.value,
                                lane=t.turret.lane,
                                tier=t.turret.tier.value,
                                standing=t.standing,
                                side=t.turret.side,
                            )
                            for t in turrets
                        )
                        if turrets is not None
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    map_side=(
                        self.turret_reader.side.value
                        if self.turret_reader is not None
                        and self.turret_reader.side is not None
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    cs=(
                        cs
                        if self_track is not None and track.id == self_track.id
                        else None
                    ),
                    gold=(
                        gold
                        if self_track is not None and track.id == self_track.id
                        else None
                    ),
                    last_hits=(
                        last_hits
                        if last_hits
                        and self_track is not None
                        and track.id == self_track.id
                        else None
                    ),
                    allies_dead=None if liveness is None else liveness.dead_count,
                )
            )
        return rows

    def timeline_meta(
        self,
        source: str | Path,
        size: tuple[int, int] | None = None,
    ) -> TimelineMeta:
        """Header describing what this pipeline is configured to produce.

        Built here rather than in `TimelineMeta` because the calibration state
        it records is the pipeline's, and a header assembled by hand at the call
        site is a header that can describe a run that did not happen.
        """
        resolution = size or self.resolution
        if resolution is None:
            raise ValueError(
                "pipeline has no resolution; pass size=(width, height) or build "
                "it with Pipeline.for_resolution"
            )
        bounds = None if self.world is None else self.world.bounds.to_dict()
        scale = (
            None
            if self.world is None
            else [float(u) for u in self.world.units_per_pixel]
        )
        return TimelineMeta(
            source=Path(source).name,
            width=resolution[0],
            height=resolution[1],
            has_game_time=self.clock is not None,
            has_liveness=self.liveness is not None,
            has_nameplates=self.plate_reader is not None
            and self.projection is not None,
            has_abilities=self.ability_reader is not None,
            has_threats=self.threats is not None,
            has_skillshots=self.aim is not None,
            has_minions=self.minion_reader is not None,
            has_minion_dots=self.dot_detector is not None,
            has_turrets=self.turret_reader is not None,
            has_last_hits=self.creep_score is not None,
            has_gold=self.gold is not None,
            has_self_abilities=self.self_reader is not None,
            world_bounds=bounds,
            world_units_per_pixel=scale,
        )

    @staticmethod
    def _find_self(
        blips: list[Blip], viewport: Viewport | None, place: bool = False
    ) -> tuple[Blip | None, bool]:
        """The player's marker, and whether it had to be placed.

        The marker nearest the viewport centre is the player's -- on the
        locked-camera footage this was built on it sits 5-7px away with the
        runner-up 38-88px off. But on a real game the player's marker is
        *covered* much of the time: an enemy chasing them, or a support
        standing on them, draws over the icon and its ring no longer fills.
        Measured on the 2026-08-30 session, stage 1 had a blue marker within
        12px of the centre on 24% of frames, and projecting the player's own
        nameplate onto the minimap agreed with the centre to within 2px on
        every frame checked -- so the position was never in doubt, only the
        marker. When `place` is set and no blue marker is within
        `SELF_CLEARANCE`, one is put at the centre with no score, so the
        tracker keeps the player's track fed through the cover. It is the
        viewport doing the detecting -- position only: who the player is
        comes from the ability slots.
        """
        if viewport is None:
            return None, False
        cx, cy = viewport.center
        candidates = [b for b in blips if b.team is Team.BLUE]
        nearest = min(candidates, key=lambda b: np.hypot(b.x - cx, b.y - cy),
                      default=None)
        distance = (float("inf") if nearest is None
                    else float(np.hypot(nearest.x - cx, nearest.y - cy)))
        if distance <= SELF_RADIUS:
            return nearest, False
        if place and distance > SELF_CLEARANCE:
            radius = (float(np.median([b.radius for b in blips]))
                      if blips else 13.0)
            return Blip(x=cx, y=cy, radius=radius, team=Team.BLUE, score=0.0), True
        return None, False


def _frame_to_crop(
    region: MinimapRegion, x: float, y: float
) -> tuple[float, float]:
    return x - region.x, y - region.y
