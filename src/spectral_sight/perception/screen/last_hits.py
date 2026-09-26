"""Last hits and missed ones: enemy minions dying on screen, judged against
the creep score.

A minion's health bar vanishing says it died; the creep score rising says the
player got the kill. Neither alone says what coaching wants -- "you missed that
one" needs a death the counter did not credit -- so this follows enemy minion
bars from reading to reading, calls a death when a low bar disappears where it
could not simply have walked out of view, and then looks at the counter.

**What counts as a death.** A bar last seen at or below `low` health, away from
the edges of the world view, that is not seen again for `gone_after` seconds.
The health floor is what separates dying from leaving: a full-health minion
whose bar disappears has been covered, not killed, since a wave in a clump
draws its bars over one another. The edge margin is for the minion walking off
screen or under the HUD.

A bar that vanishes while still healthy is usually hidden, not dead -- but an
ability can kill a minion from half health, so such a disappearance is kept
as a candidate and credited as a last hit if, and only if, the creep score
rose for it. It is never reported as a miss.

**What counts as the player's.** A rise in the creep score in the window
around the death. Each point of the rise is claimed by one death at most,
earliest death first, so a double kill needs a rise of two. A death with no
rise in its window is a miss; a death the counter could not be read around is
`unknown`, and is reported rather than guessed.

The creep score also counts kills the world view never showed -- a minion
killed by an ability off screen -- which is why nothing here infers a death
from the counter.
"""

from __future__ import annotations

from dataclasses import dataclass

from spectral_sight.export import LastHit


@dataclass(frozen=True, slots=True)
class LastHitConfig:
    low: float = 0.35
    """Highest last-seen health a vanishing bar can have and be called a death.
    A caster minion takes about a third of its bar from one champion auto at
    this stage of a game, so a bar above this was more likely hidden than
    killed."""

    gone_after: float = 0.3
    """Seconds a bar must stay unseen before its minion is called dead -- three
    readings at 10 Hz, enough to ride out a bar covered for a frame."""

    min_sightings: int = 2
    """Readings a track needs before its disappearance means anything."""

    gate: float = 80.0
    """Largest move, in pixels, between two readings for a bar to be the same
    minion. The camera follows the player at up to ~35px a reading, and the
    minion moves too."""

    edge: float = 60.0
    """Margin inside the world view within which a vanishing bar is taken to
    have left the view rather than died."""

    revive_slack: float = 0.1
    """How much healthier a reappearing bar may read and still be the minion
    whose death was called -- fills are read to a pixel or two."""

    settle: float = 1.5
    """Seconds a called death is held open for its bar to come back before it
    is reported. Longer than the creep-score window because they measure
    different things: on the 2026-09-25 footage a bar under a champion's
    translucent cape blinked out for 1.1s and came back."""

    before: float = 0.4
    after: float = 1.0
    """The creep-score window around a death: the counter can tick a reading
    before the bar is last seen, and the reader confirms a new value a reading
    or two late."""


@dataclass(slots=True)
class _Track:
    x: float
    y: float
    health: float | None
    last_seen: float
    sightings: int = 1


@dataclass(slots=True)
class _Death:
    at: float
    health: float
    x: float
    y: float
    low: bool = True
    """Whether the bar was low enough to be called dead on its own. A healthy
    bar that vanishes is only a kill if the creep score says so."""


class LastHitDetector:
    """Feed each sampled frame's enemy minions and creep score; collect the
    deaths as their windows close."""

    def __init__(
        self,
        view: tuple[int, int, int, int],
        config: LastHitConfig | None = None,
    ) -> None:
        self.view = view
        """The world view's (x, y, w, h) in frame pixels."""
        self.config = config or LastHitConfig()
        self._tracks: list[_Track] = []
        self._deaths: list[_Death] = []
        self._score: int | None = None
        self._points: list[list] = []
        """One [time, claimed] per point the creep score rose, stamped when
        the rise was adopted. A death claims one point from inside its own
        window, so a kill made off screen earlier cannot be credited to a
        later miss."""
        self._read: list[float] = []
        """Times the creep score was read at all."""

    def observe(
        self,
        timestamp: float,
        minions: list[tuple[float, float, float | None]],
        score: int | None,
    ) -> None:
        """One reading: enemy minions as (x, y, health) -- health None when
        the bar was not legible -- and the filtered creep score, if known."""
        self._observe_score(timestamp, score)
        self._associate(timestamp, minions)

    def _observe_score(self, timestamp: float, score: int | None) -> None:
        if score is None:
            return
        self._read.append(timestamp)
        if self._score is not None and score > self._score:
            self._points.extend([timestamp, False] for _ in range(score - self._score))
        self._score = score

    def _associate(
        self, timestamp: float, minions: list[tuple[float, float, float | None]]
    ) -> None:
        cfg = self.config
        pairs = sorted(
            (
                ((t.x - x) ** 2 + (t.y - y) ** 2) ** 0.5, ti, mi
            )
            for ti, t in enumerate(self._tracks)
            for mi, (x, y, _) in enumerate(minions)
        )
        used_t: set[int] = set()
        used_m: set[int] = set()
        for distance, ti, mi in pairs:
            if distance > cfg.gate:
                break
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            track = self._tracks[ti]
            x, y, health = minions[mi]
            track.x, track.y, track.last_seen = x, y, timestamp
            track.sightings += 1
            if health is not None:
                track.health = health
        for mi, (x, y, health) in enumerate(minions):
            if mi in used_m:
                continue
            revived = self._revive(x, y, health, timestamp)
            self._tracks.append(
                revived or _Track(x, y, health, timestamp)
            )

        alive = []
        for track in self._tracks:
            if timestamp - track.last_seen < cfg.gone_after:
                alive.append(track)
                continue
            if (
                track.sightings >= cfg.min_sightings
                and track.health is not None
                and self._inside(track.x, track.y)
            ):
                self._deaths.append(_Death(
                    track.last_seen, track.health, track.x, track.y,
                    low=track.health <= cfg.low,
                ))
        self._tracks = alive

    def _revive(
        self, x: float, y: float, health: float | None, timestamp: float
    ) -> _Track | None:
        """A called death whose bar has come back.

        Bars go missing without the minion dying: a champion's cape or a spell
        effect drawn over one hides it from the reader for a few readings.
        Measured on the 2026-09-25 lane footage, that made one minion "die"
        twice in a second. A bar reappearing near the spot, no healthier than
        it was, is taken to be the same minion and the death is withdrawn.
        """
        cfg = self.config
        for index, death in enumerate(self._deaths):
            if timestamp - death.at > cfg.settle:
                continue
            if ((death.x - x) ** 2 + (death.y - y) ** 2) ** 0.5 > cfg.gate:
                continue
            if health is not None and health > death.health + cfg.revive_slack:
                continue
            del self._deaths[index]
            return _Track(x, y, health if health is not None else death.health,
                          timestamp, sightings=cfg.min_sightings)
        return None

    def _inside(self, x: float, y: float) -> bool:
        vx, vy, vw, vh = self.view
        m = self.config.edge
        return vx + m <= x <= vx + vw - m and vy + m <= y <= vy + vh - m

    def resolve(self, timestamp: float) -> list[LastHit]:
        """Deaths whose creep-score window has closed by `timestamp`."""
        cfg = self.config
        done: list[LastHit] = []
        waiting: list[_Death] = []
        for death in sorted(self._deaths, key=lambda d: d.at):
            if timestamp < death.at + max(cfg.after, cfg.settle):
                waiting.append(death)
                continue
            lo, hi = death.at - cfg.before, death.at + cfg.after
            point = next(
                (p for p in self._points if not p[1] and lo <= p[0] <= hi), None
            )
            if point is not None:
                point[1] = True
                outcome = "last_hit"
            elif not death.low:
                # Hidden, or walked out of view: nothing to report.
                continue
            elif any(lo <= t <= hi for t in self._read):
                outcome = "missed"
            else:
                outcome = "unknown"
            done.append(
                LastHit(death.at, outcome, death.health, death.x, death.y)
            )
        self._deaths = waiting
        # Points too old for any death still waiting can never be claimed.
        horizon = timestamp - max(cfg.after, cfg.settle) - cfg.before - 5.0
        self._points = [p for p in self._points if p[0] >= horizon]
        self._read = [t for t in self._read if t >= horizon]
        return done

    def reset(self) -> None:
        """Forget the tracks -- the player died, or the view is not the game.
        Credit already counted stays counted."""
        self._tracks.clear()
