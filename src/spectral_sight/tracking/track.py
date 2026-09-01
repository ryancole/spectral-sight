"""One champion, followed across frames.

A track carries two things the per-frame detector cannot: where the champion is
expected to be next, and who it has decided the champion is.

The identity half matters more than it first appears. Matching a single frame's
crop against the icon set succeeds maybe half the time -- the marker is 26px,
sometimes half-occluded by a ping or another champion, sometimes just badly
framed. But a champion does not change identity between frames, so evidence
accumulates: a track that matched Swain confidently three times keeps that name
through the frames where the crop is unreadable. Identity becomes a property of
the track's whole life rather than a fresh guess every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from spectral_sight.types import Team


class TrackState(Enum):
    """Where a track is in its lifecycle."""

    TENTATIVE = "tentative"
    """Seen, but not yet enough times to be trusted. Not reported."""

    CONFIRMED = "confirmed"
    """Currently visible and believed real."""

    LOST = "lost"
    """Not currently visible. For an enemy this usually means fog; for an ally,
    who is never hidden by fog, it means death. The track is kept so the
    champion can be re-identified when it reappears."""


@dataclass
class Track:
    """A single champion followed over time."""

    id: int
    team: Team
    x: float
    y: float
    last_seen: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    state: TrackState = TrackState.TENTATIVE
    evidence: dict[str, float] = field(default_factory=dict)
    """Accumulated identification support, keyed by champion name."""

    contradictions: dict[str, int] = field(default_factory=dict)
    """Confident reads disagreeing with the settled identity since it last
    agreed, keyed by the name the gallery offered instead. This is the
    signature a swapped track leaves -- see `Tracker._repair_swaps`."""

    # -- motion -----------------------------------------------------------

    def predict(self, dt: float) -> tuple[float, float]:
        """Where this champion is expected to be `dt` seconds after last seen.

        Constant velocity. Champions do change direction constantly, so this is
        not a strong prediction -- its job is to keep association sane over the
        short gaps between frames, not to extrapolate across long absences.
        """
        return self.x + self.vx * dt, self.y + self.vy * dt

    def observe(self, x: float, y: float, timestamp: float, alpha: float,
                beta: float) -> None:
        """Fold in a new sighting with an alpha-beta filter.

        Alpha-beta rather than a Kalman filter: with one position sensor and no
        meaningful process model, the extra machinery would only add tuning
        surface for the same behaviour.
        """
        dt = max(timestamp - self.last_seen, 1e-3)
        px, py = self.predict(dt)
        rx, ry = x - px, y - py

        self.x = px + alpha * rx
        self.y = py + alpha * ry
        self.vx += beta * rx / dt
        self.vy += beta * ry / dt
        self.last_seen = timestamp
        self.hits += 1

    def coast(self) -> None:
        """Mark that this champion was not seen. Velocity decays toward rest.

        A champion last seen running into fog is not still running in a straight
        line a few seconds later, so extrapolation is deliberately damped.
        """
        self.vx *= 0.5
        self.vy *= 0.5

    def age(self, timestamp: float) -> float:
        """Seconds since this champion was last actually seen."""
        return timestamp - self.last_seen

    # -- identity ---------------------------------------------------------

    def observe_identity(self, name: str, weight: float) -> None:
        """Add support for `name` being this champion."""
        settled = self.identity
        if settled is not None:
            if name == settled:
                self.contradictions.clear()
            else:
                self.contradictions[name] = self.contradictions.get(name, 0) + 1
        self.evidence[name] = self.evidence.get(name, 0.0) + weight

    @property
    def identity(self) -> str | None:
        """Best-supported name, or None while the evidence is inconclusive."""
        if not self.evidence:
            return None
        ranked = sorted(self.evidence.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < IDENTITY_MIN_EVIDENCE:
            return None
        if best_score - runner_up < IDENTITY_MIN_MARGIN:
            return None
        return best_name

    @property
    def identity_support(self) -> float:
        return max(self.evidence.values(), default=0.0)

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


IDENTITY_MIN_EVIDENCE = 0.6
"""Roughly four decisive matches, since evidence is weighted by how decisively
each match won rather than by raw similarity. One good-looking frame should not
lock an identity in: a confident-looking match on a partly occluded marker is
exactly the error worth outvoting."""

IDENTITY_MIN_MARGIN = 0.3
"""Lead the runner-up must hold. Champions with similar icon palettes trade
frames, and forcing a decision between them is worse than reporting none."""
