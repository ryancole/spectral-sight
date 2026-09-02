"""Projectiles heading for the player, and what came of them.

A candidate track (see `projectiles`) becomes a *threat* when its line passes
within a threat radius of the player and it is moving toward them. Three
more things are then measured, each from a signal the pipeline already has:

- **Arrival**: where along the heading the player sits, over the speed. On
  the 2026-08-30 session a bolt is first seen 150-400 px out and arrives a
  median 0.26 s later. That is a fact about the footage worth stating
  plainly: a quarter of a second is the edge of human reaction, so a player
  cannot in general be coached to dodge *the bolt* -- they dodge the cast
  that launched it, which is the enemy-ability work of a later phase. What
  this stage can say is whether the bolt arrived, and whether the player
  was already moving out of its way.
- **Outcome**: the player's own printed health, read as text, is the free
  ground truth. A fall in a window around arrival is a hit; none is a dodge
  -- or a miss, which from the player's side is the same thing. The window
  is asymmetric because arrival is estimated from the track's start and a
  bolt that stops early was a hit, not a miss; the health text also reads on
  only half the frames at full rate, so an outcome can be *unknown* when no
  reading landed in the window, and says so rather than guessing.
- **Response**: the camera-motion track is the player's own velocity, at
  30 Hz and sub-pixel. The component of it perpendicular to the bolt's
  heading, over the window from onset to arrival, is how far the player
  moved *out of the line* -- the one number that distinguishes a dodge from
  standing still and not being hit anyway.

**What a threat is not, yet.** Lane traffic. Measured, 35 approaching
candidates a minute passed within 120 px of the player, and health fell in
the arrival window after 14% of them against an 8% baseline for any window
of that length -- most of them were minion bolts aimed at the minion beside
the player, or bolts the player was never in the way of. Two things the
frame already knows cut that down: where the bolt came *from* (an enemy
champion's nameplate is on screen and its model position is known) and
where the track *ended* (a hit ends at the player, a pass carries on). Both
are gates here, both measured, and the measurements are in the config
docstrings. What remains after them is still not a labelled truth, and the
classifier the plan describes is what turns it into one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectral_sight.export import Threat
from spectral_sight.perception.screen.motion import CameraMotion
from spectral_sight.perception.screen.projectiles import ProjectileTrack


@dataclass(frozen=True, slots=True)
class ThreatConfig:
    radius: float = 120.0
    """Closest approach of the track's line to the player's model, px, within
    which the bolt threatened them. About 280 world units at the calibrated
    scale: a champion's hitbox plus a wide skillshot's, with room for the
    anchor being the plate's projection rather than the feet."""

    min_launch: float = 150.0
    """A track starting nearer the player than this was launched *by* them --
    their own Q or W -- not at them."""

    max_origin: float | None = 160.0
    """How near an enemy champion's model the track must start to count as
    that champion's. None disables the gate. Measured on the 2026-08-30
    session over three minutes: of 105 approaching candidates, the 17 that
    started within 160 px of an enemy plate were followed by a health fall
    24% of the time, the other 88 12%, against an 8% baseline for any window
    of that length. The gate keeps a sixth of the candidates and triples the
    signal. A bolt with no enemy plate on screen to have come from is not a
    champion's, and is dropped when the gate is on."""

    max_end: float | None = None
    """How near the player the track must *end* for the bolt to have reached
    them. Off by default, because measured it does not help: tracks ending
    within 90 px of the player were followed by a health fall 12% of the
    time against 15% for tracks flying past. A bolt that hits ends inside
    the champion's own effects, where the residual merges and the track
    ends wherever the blob happened to be -- not at a clean point of
    impact. Kept as a switch for footage where that turns out otherwise."""

    hp_fall: int = 10
    """Smallest fall in printed health read as damage."""

    window_before: float = 0.15
    window_after: float = 0.5
    """The window around estimated arrival in which a health fall is this
    bolt's doing. Arrival is extrapolated from the track's start and speed,
    so the window is wider after than before."""

    response_before: float = 0.1
    """How long before onset the player's velocity is taken as the baseline
    the response is measured against."""


@dataclass(slots=True)
class _Pending:
    track: ProjectileTrack
    arrival: float
    closest: float
    origin: float | None


@dataclass
class ThreatDetector:
    """Tracks, health readings and camera motion in; resolved threats out.

    Feed it in frame order. `observe_health` and `observe_motion` are per
    frame; `consider` takes the tracks that finished on a frame; `resolve`
    returns the threats whose arrival window has closed.
    """

    config: ThreatConfig = field(default_factory=ThreatConfig)
    _health: list[tuple[float, int]] = field(default_factory=list)
    _motion: list[tuple[float, float, float, float]] = field(default_factory=list)
    _pending: list[_Pending] = field(default_factory=list)

    def observe_health(self, t: float, current: int) -> None:
        self._health.append((t, current))
        self._trim(t)

    def observe_motion(self, t: float, motion: CameraMotion) -> None:
        if motion.repeat or not motion.estimated:
            return
        self._motion.append((t, motion.dx, motion.dy, motion.dt))
        self._trim(t)

    def consider(
        self,
        tracks: list[ProjectileTrack],
        anchor: tuple[float, float] | None,
        enemies: list[tuple[float, float]] = (),
    ) -> None:
        """Judge finished tracks against the player's position.

        `anchor` is the player's model in world-view pixels; None means the
        player is not on screen and nothing can threaten them visibly.
        `enemies` are enemy champion models, for the origin gate.
        """
        cfg = self.config
        if anchor is None:
            return
        for track in tracks:
            (t0, x0, y0) = track.start
            if np.hypot(x0 - anchor[0], y0 - anchor[1]) < cfg.min_launch:
                continue
            closest = track.approaches(anchor, within=cfg.radius)
            if closest is None:
                continue
            if cfg.max_end is not None:
                (_, x1, y1) = track.end
                if np.hypot(x1 - anchor[0], y1 - anchor[1]) > cfg.max_end:
                    continue
            origin = None
            if enemies:
                origin = min(np.hypot(x0 - ex, y0 - ey) for ex, ey in enemies)
                if cfg.max_origin is not None and origin > cfg.max_origin:
                    continue
            elif cfg.max_origin is not None:
                # No enemy plate on screen and the gate is on: the bolt's
                # source is unknown, and an unknown source is not a champion.
                continue
            ux, uy = track.heading
            along = (anchor[0] - x0) * ux + (anchor[1] - y0) * uy
            arrival = t0 + along / track.speed if track.speed > 0 else t0
            self._pending.append(_Pending(track, arrival, closest, origin))

    def resolve(self, now: float) -> list[Threat]:
        """Threats whose window has closed by `now`, resolved."""
        cfg = self.config
        done, keep = [], []
        for p in self._pending:
            if now < p.arrival + cfg.window_after:
                keep.append(p)
                continue
            done.append(self._resolve(p))
        self._pending = keep
        return done

    def flush(self) -> list[Threat]:
        out = [self._resolve(p) for p in self._pending]
        self._pending.clear()
        return out

    def reset(self) -> None:
        self._health.clear()
        self._motion.clear()
        self._pending.clear()

    def _resolve(self, p: _Pending) -> Threat:
        cfg = self.config
        lo, hi = p.arrival - cfg.window_before, p.arrival + cfg.window_after
        inside = [(t, c) for t, c in self._health if lo <= t <= hi]
        before = [(t, c) for t, c in self._health if t < lo]
        outcome, damage = "unknown", None
        if inside:
            start = before[-1][1] if before else inside[0][1]
            fall = start - min(c for _, c in inside)
            if fall >= cfg.hp_fall:
                outcome, damage = "hit", int(fall)
            else:
                outcome = "dodged"
        moved = self._moved_across(p.track, p.arrival)
        return Threat(
            at=p.track.start[0], arrival=p.arrival, closest=p.closest,
            speed=p.track.speed, heading=p.track.heading, outcome=outcome,
            damage=damage, moved_across=moved, origin=p.origin,
        )

    def _moved_across(self, track: ProjectileTrack, arrival: float) -> float | None:
        """Player displacement perpendicular to the bolt between onset and
        arrival. The camera's shift is the terrain's; the player's own motion
        is its negative, but the sign is immaterial to a distance."""
        t0 = track.start[0]
        span = [(dx, dy) for t, dx, dy, _ in self._motion if t0 <= t <= arrival]
        if not span:
            return None
        ux, uy = track.heading
        across = sum(dx * uy - dy * ux for dx, dy in span)
        return abs(float(across))

    def _trim(self, now: float, keep: float = 5.0) -> None:
        cut = now - keep
        if self._health and self._health[0][0] < cut:
            self._health = [h for h in self._health if h[0] >= cut]
        if self._motion and self._motion[0][0] < cut:
            self._motion = [m for m in self._motion if m[0] >= cut]
