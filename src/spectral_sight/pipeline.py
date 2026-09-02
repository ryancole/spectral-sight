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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spectral_sight.export import AbilityUse, Observation, TimelineMeta
from spectral_sight.perception.hud.abilities import AbilityLayout, AbilityReader
from spectral_sight.perception.hud.alive import AliveReader, Liveness
from spectral_sight.perception.hud.clock import (
    ClockFilter,
    ClockReader,
    GameClock,
    load_clock_reader,
)
from spectral_sight.perception.hud.naming import SELF_SLOT, SlotNaming
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
)
from spectral_sight.perception.minimap.blips import scaled_config
from spectral_sight.perception.nameplates import (
    Cast,
    CastBook,
    LevelBook,
    Nameplate,
    NameplateLayout,
    NameplateReader,
    ScreenProjection,
    associate,
)
from spectral_sight.tracking import Track, Tracker, TrackerConfig
from spectral_sight.types import Blip, Team

SELF_RADIUS = 12.0
"""How close to the viewport centre a marker must sit to be the local player.
The nearest marker measures 5-7px on real footage with the runner-up 38-88px
away, so this threshold sits comfortably inside a very wide gap."""

SELF_MIN_SIGHTINGS = 10
SELF_MIN_LEAD = 2.0
"""How much the viewport must favour one champion before it is called the local
player: this many resolutions, and this many times the runner-up.

Measured over a 5.3-minute clip the viewport named the player in 2,276 frames
and got Zilean 84.6% of the time, with the next candidate on 5.8%. The lead is
enormous, so these are loose thresholds that only exclude the opening frames and
genuine confusion -- not a fit to that ratio."""


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
    ) -> None:
        self.region = region
        self.gallery = gallery
        # Frame size this pipeline was calibrated for. Not derivable from the
        # region, which knows only the crop rectangle, and needed only to
        # describe a run in a timeline header.
        self.resolution = resolution
        self.detector = detector or BlipDetector(
            scaled_config(BlipDetectorConfig(), minimap_width=region.width)
        )
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

        self.levels = LevelBook()
        self.casts = CastBook()
        self._clock_filter = ClockFilter()
        self._restricted: dict[Team, Gallery] = {}
        self._self_evidence: dict[str, int] = {}
        """How often the viewport has named each champion as the local player.

        Accumulated rather than read fresh, for two reasons. The viewport finds
        the player by the marker at the camera centre, and a dead player has no
        marker -- so it is unavailable in exactly the frames where a death needs
        attributing. And the latest answer is not reliable enough to build on:
        over a 5.3-minute clip it named the right champion 85% of the time but
        drifted for seconds at a stretch when the camera sat on a teammate, and
        taking the most recent value attributed the player's second death to two
        champions who were alive throughout.

        The player is one champion for the whole game, so this is the same move
        the tracker makes for a marker's identity: let the evidence pile up and
        a handful of bad frames cannot outvote it.

        Frames vote only while the self portrait reads alive -- see the gate in
        `process`. The camera is assumed locked on the player, and death is the
        one departure from that the pipeline can prove: the death-cam watches a
        corpse or a teammate for the whole respawn timer, which is minutes of
        wrong votes over a session, not a handful."""

    @property
    def self_champion(self) -> str | None:
        """The champion the local player is on, or None while it is unsettled.

        Requires a clear lead rather than a bare majority, since the cost of
        being wrong is naming the wrong casualty -- and a champion who is merely
        standing where the camera is pointing can win a handful of frames.
        """
        if not self._self_evidence:
            return None
        ranked = sorted(self._self_evidence.items(), key=lambda kv: kv[1],
                        reverse=True)
        name, sightings = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if sightings < SELF_MIN_SIGHTINGS:
            return None
        if sightings < runner_up * SELF_MIN_LEAD:
            return None
        return name

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
        )

    def process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        """Run one frame. `timestamp` is in seconds and must increase."""
        clock = None
        if self.clock is not None:
            clock = self._clock_filter.update(self.clock.read(frame), timestamp)

        # Whether this frame provably shows the in-game HUD: the timer is the
        # one element that can prove it, so with a clock calibrated, trust
        # follows the clock resolving. With none there is no proof to wait
        # for, and every frame is taken at its word -- the behaviour that
        # predates the clock.
        trusted = self.clock is None or (clock is not None and clock.observed)

        if self._clock_filter.resynced:
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

        liveness = None
        if self.liveness is not None:
            # Baselines learn only from trusted frames. A recording that
            # starts in queue otherwise teaches the portraits splash art,
            # which out-saturates any living champion -- and a running
            # maximum never comes back down, so the real HUD would read dead
            # from its first frame onward.
            liveness = self.liveness.read(frame, learn=trusted)

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
            me = None if liveness is None else liveness.slot(SELF_SLOT)
            dead = me is not None and me.alive is False
            if trusted and not dead:
                self._pending_abilities.extend(
                    AbilityUse(
                        slot=cast.slot,
                        at=cast.at,
                        countdown=cast.countdown,
                        confirmed=cast.confirmed,
                    )
                    for cast in self.ability_reader.read(frame, timestamp)
                )
            elif dead:
                self.ability_reader.reset()

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
        # The champion at the camera centre is only a witness to who the
        # player is while the camera is credibly locked on them, and the one
        # violation this pipeline can prove is death: a dead player's camera
        # sits on their corpse or roams teammates for the whole respawn timer,
        # and on a real session those frames poured up to 72 seconds of votes
        # at a stretch onto whoever was being watched. So the self portrait
        # must read alive for a frame to vote. An unproven portrait withholds
        # the vote too -- pre-game frames are not a game -- and a run with no
        # portrait calibration keeps the old behaviour, having no gate to
        # apply.
        me = None if liveness is None else liveness.slot(SELF_SLOT)
        witnessed = liveness is None or (me is not None and me.alive is True)
        if witnessed and self_track is not None and self_track.identity is not None:
            name = self_track.identity
            self._self_evidence[name] = self._self_evidence.get(name, 0) + 1

        if self.naming is not None and liveness is not None:
            # An ally is drawn on the minimap exactly while alive, so the
            # champions the gallery matched with confidence this frame are
            # proven living -- the negative space is what names a dead slot.
            seen = {
                match.name
                for blip, match in zip(blips, matches)
                if match is not None and match.confident
                and blip.team is Team.BLUE
            }
            self.naming.update(
                liveness, seen, self.roster.locked(Team.BLUE),
                self.self_champion, timestamp, trusted=trusted,
            )

        plates, pairing, casts = self._read_plates(
            frame, tracks, viewport, timestamp
        )

        return PipelineResult(
            blips=blips,
            matches=matches,
            tracks=tracks,
            viewport=viewport,
            self_blip=self_blip,
            self_track=self_track,
            clock=clock,
            liveness=liveness,
            plates=plates,
            plate_tracks=pairing,
            observations=self._observe(
                tracks, timestamp, clock, self_track, liveness, plates,
                pairing, casts, self._take_abilities(self_track),
            ),
        )

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
    ) -> tuple[list[Nameplate], dict[int, int], dict[int, Cast]]:
        """Read nameplates, attach them to tracks, and call any casts.

        Plates are matched against *both* teams' tracks rather than just the
        enemy's. An ally plate is less interesting -- the HUD already says how
        many teammates are alive -- but it is the same read for free, and an
        ally's resource is the one case where a cast can be checked against
        something else that was observed.
        """
        if self.plate_reader is None:
            return [], {}, {}

        plates = self.plate_reader.read(frame)
        height, width = frame.shape[:2]
        pairing: dict[int, int] = {}
        for team, hostile in ((Team.RED, True), (Team.BLUE, False)):
            indices = [i for i, p in enumerate(plates) if p.hostile is hostile]
            if not indices:
                continue
            side = [t for t in tracks if t.team is team]
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
                    allies_dead=None if liveness is None else liveness.dead_count,
                )
            )
        return rows

    def timeline_meta(
        self,
        source: str | Path,
        stride: int,
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
            stride=stride,
            has_game_time=self.clock is not None,
            has_liveness=self.liveness is not None,
            has_nameplates=self.plate_reader is not None
            and self.projection is not None,
            has_abilities=self.ability_reader is not None,
            world_bounds=bounds,
            world_units_per_pixel=scale,
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
