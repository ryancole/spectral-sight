"""The player's own skillshots, and whether they landed.

Where `threats` judges what came *at* the player, this judges what they threw.
The three signals it joins are each already built and each says something the
others cannot:

- **The cast** (`perception.hud.abilities`) says which button was pressed and
  when, to a frame. It cannot say where the ability went, or whether it went
  anywhere at all.
- **The bolt** (`projectiles`) is a fast, straight, brief track born beside the
  player's own model in the half-second after the cast. It says where the shot
  went and how fast, and it is the only thing that does.
- **The target's health** (`nameplates`) is the free ground truth, the same
  move the threat stage makes with the player's own printed health: a fall in
  the window when the bolt would reach an enemy is a hit.

**A skillshot is a cast that launched something.** There is deliberately no
per-champion table of which abilities are skillshots. A blink, a self-buff or
a targeted point-and-click launches no bolt from the player's model, so it
produces no `Skillshot` and excludes itself; a champion whose kit this project
has never seen needs no entry to work. What the roster does give -- the
champion -- rides on the row, so the slot names the ability downstream without
this module knowing any kit.

**What the outcome rests on, and what it does not.** The plan expected the
target's health to be the label, the same free ground truth the threat stage
uses on the player's own printed health. Measured, it is not one *on this
footage*, and the number is worth stating plainly: an enemy champion's bar
falls by more than a bar-pixel or two in **52% of all 0.65 s windows** they are
on screen at all (1,327 windows over 62 plate tracks, 150-700 s of the
2026-08-30 session). A lane trade against bots is continuous damage -- minions,
the turret, auto-attacks, the player's other abilities -- so "the bar moved
after your Q" is barely more than a coin toss, and tightening the window or the
threshold until the baseline is informative also drops the near-miss rate to
chance. There is no setting of it that separates a hit from a miss here.

So **the verdict is geometric**: a shot is a hit when the bolt's line passed
within `hit_radius` of the target's model, and a miss when it went wide. That
is measured from the stabilised residual and the nameplate, and it does not
consult the bar at all. The fall is still carried, as `fall`, because it is
real evidence a consumer may want to weigh. Gated (see `max_origin_miss`),
the sample is too small to read it against: of 15 aimed shots on 150-700 s,
14 passed within 130 px and the bar fell after 9 of them, and the one wide
shot was not followed by a fall. One wide shot is not a result.

**What would make the bar a label again is exactly the plan's Phase 0**: one
enemy, one skillshot, and nothing else on the map dealing damage. Then the
baseline is zero and the fall is the ground truth it was supposed to be -- for
this stage, and for the classifier the projectile stage is waiting on.

**Where the sample goes.** Over 150-700 s of the 2026-08-30 session: 111 casts
in ability slots, 38 of which launched a bolt this stage could find *and*
trace back to the player's model, and 15 of those with an enemy champion on
screen in front of it to have been aimed at. The attrition has two causes and
neither is a failure of the stage. An enemy is simply not on screen for most
of a lane phase -- only 30% of sampled frames hold an enemy plate at all, a
quarter of those occluded or clipped and reading `None` rather than a number
-- so `unknown` is the common outcome and is reported as such rather than
folded into a miss. And the bolt is seen in about a third of casts, for
reasons traced in `docs/aim-bolt-findings.md`: Ezreal's Q and W always launch
one, and where the stage did not credit it, the projectile stage had lost the
track at three points (a rival chance track or the ghost mask took its fourth
blob) or never had one, and the remedies measured for that all cost more in
strays than they returned.

**Which way the errors run.** Before `max_origin_miss` this stage credited 83
of those 111 casts, and most of those bolts were not the player's -- enemy
bolts arriving, allied bolts passing by, effects near the model -- and seven
of the eight wide shots it reported were among them. A consumer of the old
output was coached on strays. Now a cast reported with no bolt is silence,
biased toward under-counting the shots thrown and never toward inventing one,
and a credited bolt is one that left the player's model. What remains among
the credited bolts is a stray floor of about 7% -- a fake cast at a random
moment is credited a bolt that often -- which is where the player's own
auto-attacks would sit if they are being credited; the sample cannot say.

Coordinates are world-view pixels, like the rest of `perception.screen`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectral_sight.export import Skillshot
from spectral_sight.perception.screen.motion import CameraMotion
from spectral_sight.perception.screen.projectiles import ProjectileTrack


@dataclass(frozen=True, slots=True)
class EnemyPlate:
    """One enemy champion as this stage needs them: a model position on the
    world view and a health fill, or None where the bar could not be read."""

    x: float
    y: float
    health: float | None


@dataclass(frozen=True, slots=True)
class AimConfig:
    slots: tuple[str, ...] = ("Q", "W", "E", "R")
    """Slots that can be a skillshot. The summoner spells are excluded --
    none of them is a projectile the player aims -- and the four ability
    slots are all admitted, because whether the cast launched a bolt is
    decided by observation rather than by a table."""

    launch_before: float = 0.1
    launch_after: float = 0.5
    """Window around the cast in which the bolt must have been first seen. A
    bolt does not appear the instant the veil does: there is the cast
    animation, and then the bolt has to separate from the player's own
    residual before it is a track at all. Measured over the 38 casts credited
    a bolt on 150-700 s of the 2026-08-30 session, the first sighting lands a
    median +0.12 s after the veil, p90 +0.43. That p90 is close enough to the gate to
    say the distribution is censored by it -- there are probably bolts past
    half a second -- but widening it buys them at the price of admitting more
    of the ~210 unrelated candidates a minute, and an unrelated bolt credited
    to a cast is worse than a cast credited with none. The small allowance
    before covers the veil being confirmed a frame late.

    Both edges are frame edges and are compared with `_EDGE` of slack: a
    bolt first seen exactly three frames before the veil is at -0.1 s, and
    two of them on the 2026-08-30 session were rejected because the
    difference of two frame timestamps came out a hair under -0.1 in
    floating point. The slack is a seventh of a frame, so it admits the
    frame on the edge and not the next one."""

    max_launch: float = 250.0
    """How near the player's model the bolt must be born. A bolt does not
    become a track at the model -- it has to separate from the champion's own
    residual first -- so measured it appears 60-250 px out. This is the
    mirror of `ThreatConfig.min_launch`: the same distance that disqualifies
    a bolt as a threat qualifies it as the player's shot."""

    max_origin_miss: float = 80.0
    """How far the bolt's line may pass from the player's model, px, with the
    model *behind* the bolt's first point. A shot the player fired starts at
    their model and flies straight, so its track, extended backwards, goes
    through where they stand; a bolt born beside them that does not trace
    back to them is something else -- an enemy's bolt arriving, an ally's
    passing by, an effect near the model.

    This is the gate that makes a credited bolt mean anything, and it was
    measured before it was written (`docs/aim-bolt-findings.md`). Without it,
    of 46 bolts credited to real casts on 200-470 s of the 2026-08-30
    session, only 13 traced back through the model even at this tolerance;
    frame crops showed a Q "hit" credited to a track 240 px above the player
    at the enemy's feet, and another credited to an enemy's crescent gliding
    *in*. With it, on 150-700 s: 83 credited bolts became 38, and the eight
    called misses became one -- nearly every wide shot the stage had reported
    was a stray. The rate at which a fake cast at a random moment is credited
    a bolt fell from 25% to 7%.

    80 rather than 45 because Ezreal's W is a large orb whose residual
    centroid sits well off the line the orb travels; at 45 px the gate loses
    it and at 80 it does not, for two more points of strays. The fitted origin
    of the real bolt lines sits within 30 px of the anchor, so the anchor's
    own error is inside this."""

    hit_radius: float = 130.0
    """Closest approach of the bolt's line to an enemy model, px, within which
    it is judged to have hit them. This is the verdict -- see the module for
    why the health bar is not -- and it is the least settled number here.

    What argued for 130 was a gap in the measured miss distances, 124 px to
    183 px, with the bar leaning the right way across it. That measurement
    was made before `max_origin_miss`, and the far side of the gap was
    strays: gated, the 15 aimed shots on 150-700 s miss by 5, 14, 14, 16, 20,
    26, 26, 39, 61, 71, 81, 92, 95, 111 and then 314. One shot past the
    radius is no basis for placing it, so 130 stands as the number that was
    there, not as one the footage chose.

    What argues for less is the scale. The viewport rectangle measures 78
    minimap pixels wide at 48.0 world units each, over a 2,116 px screen:
    **1.8 world units per screen pixel**, so 130 px is about 230 world units --
    wider than a champion's hitbox and a missile put together. The difference
    is the anchor's own error, which is unmeasured: the target is the plate's
    centre dropped a fixed 95 px, and the plate floats a champion-dependent
    height over a model of champion-dependent size. (That scale is also not
    reconciled with the threat stage, whose `ThreatConfig.radius` quotes 2.3
    units per pixel from an earlier derivation. Neither has been checked
    against the other; this one shows its working.)

    Phase 0 footage settles it, by making the bar a label again and letting
    this be swept against one. Until then it is a threshold chosen in a gap,
    and a consumer who wants to draw the line elsewhere has `miss` on every
    shot to do it with."""

    max_flight: float = 1.5
    """Longest flight to a target, s. Beyond this the bolt's line happens to
    pass through a champion who is nowhere near its range. Not a gate that
    does much work: the measured flights run 0.13-0.46 s.

    Where the bolt's track *ended* is deliberately not a gate, and that was
    measured rather than assumed. The chord of a bolt's track over the
    distance to its target ran a median 0.62 -- most tracks stop well short,
    because a track ends when the residual fades and not where the bolt stops
    -- and the split between near and wide shots was 0.75 against 0.52, no use
    at all. It is the same finding the threat stage records from the other end
    (`ThreatConfig.max_end`)."""

    hp_fall: float = 0.015
    """Smallest fall in the target's health *fraction* recorded as `fall`. The
    bar is 117 px wide, so a pixel is 0.009; this is two. Deliberately low,
    because `fall` is evidence a consumer weighs and not a verdict this module
    reaches -- a threshold high enough to mean "an ability hit them" would be
    this module deciding, on a signal it has measured cannot decide."""

    window_before: float = 0.15
    window_after: float = 0.5
    """The window around estimated arrival in which the target's health fall
    is this bolt's doing. Asymmetric for the same reason as the threat
    stage's: arrival is extrapolated from the track's start and speed, and a
    bolt that stops early stopped on something."""

    target_span: float = 0.4
    """How far back the target's own screen velocity is measured, s. Enemy
    plates are read at the sampled rate, so this is four points at a stride
    of three, against a champion moving a measured median 146 px/s once the
    camera is taken out of it."""

    min_target_speed: float = 60.0
    """Below this the target was standing still and `lead` is not reported:
    the sign of a miss relative to a direction that is mostly plate-centroid
    noise would be a coin flip with a decimal point on it."""


@dataclass(slots=True)
class _EnemyTrack:
    """One enemy plate followed across sampled frames."""

    points: list[tuple[float, float, float, float | None]]
    """(video_time, x, y, health)."""

    @property
    def last(self) -> tuple[float, float, float, float | None]:
        return self.points[-1]

    def at(self, t: float, tol: float) -> tuple[float, float] | None:
        best = min(self.points, key=lambda p: abs(p[0] - t))
        return None if abs(best[0] - t) > tol else (best[1], best[2])

    def fall(self, lo: float, hi: float) -> float | None:
        """Largest fall in health across the window, or None if the bar was
        not readable anywhere in it. Compared against the last reading before
        the window when there is one, so a fall that happened between two
        samples is still this window's."""
        inside = [(t, h) for t, _, _, h in self.points if lo <= t <= hi and h is not None]
        if not inside:
            return None
        before = [h for t, _, _, h in self.points if t < lo and h is not None]
        start = before[-1] if before else inside[0][1]
        return max(0.0, start - min(h for _, h in inside))


@dataclass(slots=True)
class _Pending:
    slot: str
    at: float
    offered: list[tuple[float, ProjectileTrack]] = field(default_factory=list)
    """(distance from the player's model when it was born, track). The
    distance is kept rather than the anchor, because the anchor in force is
    the one from the frame the bolt appeared on and not the one from
    whichever frame the cast happens to settle in."""

    bolt: ProjectileTrack | None = None
    settled: bool = False
    arrival: float = 0.0
    target: _EnemyTrack | None = None
    flight: float | None = None
    miss: float | None = None
    lead: float | None = None


@dataclass
class AimDetector:
    """Casts, bolts, enemy plates and camera motion in; skillshots out.

    Feed it in frame order. `observe_cast` takes the HUD reader's casts;
    `observe_enemies` and `observe_motion` are per frame (the first only on
    frames where plates were read); `consider` takes the candidate tracks
    that finished on a frame; `resolve` returns the skillshots whose windows
    have closed.
    """

    config: AimConfig = field(default_factory=AimConfig)
    _pending: list[_Pending] = field(default_factory=list)
    _enemies: list[_EnemyTrack] = field(default_factory=list)
    _camera: list[tuple[float, float, float]] = field(default_factory=list)
    """(video_time, accumulated dx, accumulated dy) -- where a point standing
    still in the world has been carried to by the camera."""

    _ax: float = 0.0
    _ay: float = 0.0
    _anchor: tuple[float, float] | None = None
    """The last anchor `consider` was given. A frame whose plate read did not
    resolve the player -- most often the reader finding two self-side plates
    rather than one -- hands over None, and without this the candidates that
    finished on that frame are offered to nobody. Measured on the 2026-08-30
    session, that silenced two of fifteen Q/W casts that had a bolt beside the
    player passing every other gate. The camera is locked, so the anchor is
    not a moving target: across the whole clip it sat within 6 px in x and
    8 px in y, and the last good one is as good as this frame's would have
    been. Cleared by `reset`, since death is the one thing that moves the
    player's model without moving the camera."""

    _claimed: set[int] = field(default_factory=set)
    """Track ids already taken by a settled cast. One bolt is one cast's:
    measured on the 2026-08-30 session, three pairs of casts fired within half
    a second of each other were each credited with the same bolt, and so with
    the same miss and the same outcome, one of them wrongly. The earlier cast
    takes it and the later reports no bolt, which under-counts by one rather
    than inventing a shot that was never on screen."""

    ASSOCIATE_GATE = 200.0
    """How far an enemy plate may move between sampled frames and still be
    the same champion. A champion walks a measured median 146 px/s and the
    camera can carry them as fast again; at a stride of three that is well
    under this, and two enemy plates are rarely within 200 px of each other
    without the reader marking both occluded."""

    STALE = 0.5
    """A plate track unseen this long is over: the champion left the view,
    died, or went into fog, and the next plate to appear near where they were
    starts a new track rather than continuing theirs. It is also how near a
    cast a plate must have been read for the target's position to be trusted
    at all."""

    def observe_cast(self, slot: str, at: float) -> None:
        if slot in self.config.slots:
            self._pending.append(_Pending(slot=slot, at=at))

    def observe_motion(self, t: float, motion: CameraMotion) -> None:
        if motion.repeat or not motion.estimated:
            return
        self._ax += motion.dx
        self._ay += motion.dy
        self._camera.append((t, self._ax, self._ay))

    def observe_enemies(self, t: float, plates: list[EnemyPlate]) -> None:
        """Fold one sampled frame's enemy plates into the plate tracks.

        Nearest-neighbour association in screen space, deliberately, rather
        than the pipeline's plate-to-minimap-track pairing: what this stage
        needs is that the bar it compares against itself belongs to one
        champion for the two seconds around a shot, and a projection that
        resolves the player on a fifth of frames is a worse answer to that
        than the fact that a champion does not teleport.
        """
        # Tracks the champion has walked out of are kept, not dropped: a cast
        # settles a second after it happened, and an enemy who left the view
        # in that second was still standing there when the bolt was aimed at
        # them. Only the *live* ones can take a new plate. Dropping them cost
        # a third of the aimed shots on the 2026-08-30 session.
        fresh = [e for e in self._enemies if t - e.last[0] <= self.STALE]
        taken: set[int] = set()
        for track in fresh:
            _, lx, ly, _ = track.last
            best, best_d = None, self.ASSOCIATE_GATE
            for i, plate in enumerate(plates):
                if i in taken:
                    continue
                d = float(np.hypot(plate.x - lx, plate.y - ly))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                taken.add(best)
                p = plates[best]
                track.points.append((t, p.x, p.y, p.health))
        for i, plate in enumerate(plates):
            if i not in taken:
                self._enemies.append(
                    _EnemyTrack([(t, plate.x, plate.y, plate.health)])
                )
        self._trim(t)

    def consider(
        self, tracks: list[ProjectileTrack], anchor: tuple[float, float] | None
    ) -> None:
        """Offer this frame's finished candidates to the casts still open.

        A track is offered to every cast whose launch window it falls in; the
        one it belongs to is chosen at settle time, when they have all been
        seen. `anchor` is the player's model on the world view, or None when
        this frame's plate read did not resolve it, in which case the last
        one that did stands in -- see `_anchor`.
        """
        if anchor is None:
            anchor = self._anchor
            if anchor is None:
                return
        else:
            self._anchor = anchor
        cfg = self.config
        for p in self._pending:
            if p.settled:
                continue
            for track in tracks:
                t0, x0, y0 = track.start
                if not (-cfg.launch_before - _EDGE
                        <= t0 - p.at
                        <= cfg.launch_after + _EDGE):
                    continue
                away = float(np.hypot(x0 - anchor[0], y0 - anchor[1]))
                if away > cfg.max_launch:
                    continue
                if not self._from_model(track, anchor):
                    continue
                p.offered.append((away, track))

    def _from_model(
        self, track: ProjectileTrack, anchor: tuple[float, float]
    ) -> bool:
        """Does the track's line, run backwards, pass through the player's
        model? See `AimConfig.max_origin_miss`."""
        ux, uy = track.heading
        if ux == 0.0 and uy == 0.0:
            return False
        _, x0, y0 = track.start
        rx, ry = x0 - anchor[0], y0 - anchor[1]
        behind = rx * ux + ry * uy > 0
        perp = abs(rx * uy - ry * ux)
        return behind and perp <= self.config.max_origin_miss

    def resolve(self, now: float) -> list[Skillshot]:
        """Skillshots whose windows have closed by `now`.

        Two windows in sequence. The launch window closes a track's length
        after the last moment a bolt could have been born, because a track is
        reported when it *finishes*; only then is the bolt chosen. The health
        window closes after the bolt would have reached its target.
        """
        cfg = self.config
        done: list[Skillshot] = []
        keep: list[_Pending] = []
        for p in self._pending:
            if not p.settled:
                if now < p.at + cfg.launch_after + _TRACK_GRACE:
                    keep.append(p)
                    continue
                self._settle(p)
            if p.bolt is not None and now < p.arrival + cfg.window_after:
                keep.append(p)
                continue
            done.append(self._resolve(p))
        self._pending = keep
        return done

    def flush(self) -> list[Skillshot]:
        out = []
        for p in self._pending:
            if not p.settled:
                self._settle(p)
            out.append(self._resolve(p))
        self._pending.clear()
        return out

    def reset(self) -> None:
        self._pending.clear()
        self._enemies.clear()
        self._camera.clear()
        self._claimed.clear()
        self._anchor = None
        self._ax = self._ay = 0.0

    def _settle(self, p: _Pending) -> None:
        """Choose the cast's bolt, and where it was going.

        The choice is made on the launch alone -- nearest to the player's
        model, earliest to break a tie -- and never on where the bolt ended
        up. Picking the candidate that passes nearest an enemy would make the
        miss distance a function of the choice, and the whole validation of
        this stage is that the miss distance and the health fall are measured
        independently.
        """
        p.settled = True
        offers = [(d, t) for d, t in p.offered if t.id not in self._claimed]
        if not offers:
            return
        p.bolt = min(offers, key=lambda o: (o[0], o[1].start[0]))[1]
        self._claimed.add(p.bolt.id)
        # The target is fixed here too, while the plate track that was on
        # screen at launch is certainly still held. Only the health reading
        # waits, because only the health reading is in the future.
        found = self._target(p.bolt)
        if found is not None:
            p.target, p.flight, p.miss = found
            p.arrival = p.bolt.start[0] + p.flight
            p.lead = self._lead(p.bolt, p.target)
        else:
            p.arrival = p.bolt.start[0]

    def _target(
        self, bolt: ProjectileTrack
    ) -> tuple[_EnemyTrack, float, float] | None:
        """The enemy the bolt was going at: (track, flight, miss).

        Of the enemies on screen when it launched, the one its line passes
        nearest -- ahead of it, and inside the flight it could have made.

        Measured against where the target stood *at the launch*, not where
        they stood when the bolt got there, and that is a choice with a
        measurement behind it -- one made before `max_origin_miss`, on a
        sample that included strays, so the numbers are indicative. A champion walks a median 146 px/s and a bolt
        flies for a median 0.28 s, so the two differ: on the 2026-08-30
        session, by a median 12 px and a p90 of 29. They disagreed about the
        verdict on 1 of 24 shots. What decides it is availability -- the
        target has a plate reading near the launch by construction, and only
        24 of 29 had one near arrival, the rest having walked out of view --
        so the launch position is measured on every shot where the arrival
        position is measured on five sixths of them, for a difference the
        footage says is smaller than the threshold's own uncertainty.
        """
        cfg = self.config
        ux, uy = bolt.heading
        if ux == 0.0 and uy == 0.0:
            return None
        t0, x0, y0 = bolt.start
        best = None
        for track in self._enemies:
            here = track.at(t0, tol=self.STALE)
            if here is None:
                continue
            ax, ay = here[0] - x0, here[1] - y0
            along = ax * ux + ay * uy
            if along <= 0 or bolt.speed <= 0:
                continue
            flight = along / bolt.speed
            if flight > cfg.max_flight:
                continue
            miss = abs(ax * uy - ay * ux)
            if best is None or miss < best[2]:
                best = (track, flight, miss)
        return best

    def _resolve(self, p: _Pending) -> Skillshot:
        cfg = self.config
        if p.bolt is None:
            # The cast launched nothing this stage could see. Not a
            # skillshot, as far as the footage is concerned -- and for a
            # blink or a self-buff, that is the right answer.
            return Skillshot(
                slot=p.slot, at=p.at, launched=None, speed=None, heading=None,
                miss=None, flight=None, outcome="unknown", fall=None, lead=None,
            )
        outcome, fall = "unknown", None
        if p.target is not None and p.miss is not None:
            # Geometry decides, not the bar. See the module docstring: on
            # this footage a fall in the window is not evidence either way,
            # so the verdict rests on the measurement that does discriminate
            # -- how near the bolt's line passed the target's model -- and
            # the bar rides along as `fall` for a consumer to weigh itself.
            outcome = "hit" if p.miss <= cfg.hit_radius else "missed"
            dropped = p.target.fall(
                p.arrival - cfg.window_before, p.arrival + cfg.window_after
            )
            if dropped is not None and dropped >= cfg.hp_fall:
                fall = dropped
        return Skillshot(
            slot=p.slot, at=p.at, launched=p.bolt.start[0], speed=p.bolt.speed,
            heading=p.bolt.heading, miss=p.miss, flight=p.flight,
            outcome=outcome, fall=fall, lead=p.lead,
        )

    def _lead(self, bolt: ProjectileTrack, target: _EnemyTrack) -> float | None:
        """Which side of the target the shot went past, signed: positive
        ahead of their movement, negative behind it.

        The target's own velocity is their plate's motion with the camera's
        taken out of it -- the camera carries every stationary thing on
        screen, so a plate that holds still while the view scrolls is a
        champion running. Measured over the span *ending at the launch*,
        because leading a target is a judgement made when the button is
        pressed and what it is judged against is how they were moving then.
        Reported only when they were actually moving; see
        `min_target_speed`.

        This is the one number here the footage has not validated: the
        clip's misses are too few to check the sign against anything, and it
        is emitted because the geometry is sound, not because it has been
        scored.
        """
        cfg = self.config
        vx, vy = self._velocity(target, bolt.start[0])
        if vx is None:
            return None
        speed = float(np.hypot(vx, vy))
        if speed < cfg.min_target_speed:
            return None
        here = target.at(bolt.start[0], tol=self.STALE)
        if here is None:
            return None
        ux, uy = bolt.heading
        t0, x0, y0 = bolt.start
        # Signed perpendicular offset of the target from the bolt's line,
        # then flipped so the sign is read against the target's heading
        # rather than the bolt's: the shot is "ahead" when it passed on the
        # side the target was walking toward.
        offset = (here[0] - x0) * uy - (here[1] - y0) * ux
        forward = (vx * uy - vy * ux) / speed
        return float(-offset * np.sign(forward)) if forward else None

    def _velocity(
        self, target: _EnemyTrack, when: float
    ) -> tuple[float, float] | tuple[None, None]:
        """The target's screen velocity, camera removed, over the span before
        `when`."""
        cfg = self.config
        span = [p for p in target.points if when - cfg.target_span <= p[0] <= when]
        if len(span) < 2:
            return None, None
        (t0, x0, _, _), (t1, x1, _, _) = span[0], span[-1]
        y0, y1 = span[0][2], span[-1][2]
        dt = t1 - t0
        if dt <= 0:
            return None, None
        cx0, cy0 = self._carried(t0)
        cx1, cy1 = self._carried(t1)
        return ((x1 - x0) - (cx1 - cx0)) / dt, ((y1 - y0) - (cy1 - cy0)) / dt

    def _carried(self, t: float) -> tuple[float, float]:
        """Accumulated camera displacement at `t` -- where a point standing
        still in the world has been carried to."""
        if not self._camera:
            return 0.0, 0.0
        best = min(self._camera, key=lambda c: abs(c[0] - t))
        return best[1], best[2]

    def _trim(self, now: float, keep: float = 6.0) -> None:
        """Enough history for the longest a cast stays open: the launch
        window, a track's length, a flight and the health window."""
        cut = now - keep
        for track in self._enemies:
            if track.points[0][0] < cut:
                track.points = [p for p in track.points if p[0] >= cut]
        self._enemies = [e for e in self._enemies if e.points]
        if self._camera and self._camera[0][0] < cut:
            self._camera = [c for c in self._camera if c[0] >= cut]


_EDGE = 0.005
"""Slack on the launch window's edges, s: a seventh of a frame at 30 fps.
See `AimConfig.launch_after`."""

_TRACK_GRACE = 0.5
"""How long after the launch window a bolt born inside it can still be
reported. A candidate is handed over when its track *finishes*, and a bolt
runs four to six distinct frames -- a quarter of a second at the recording's
effective rate, doubled for the frames the tracker waits before giving up."""
