"""Working out which ten champions are actually in this game.

Matching against all 173 champions is how identification starts, because nothing
in a player-perspective VOD states the enemy team. But it is not how it should
continue: a game contains ten champions, and the other 163 are only there to be
mistaken for them.

So the roster is *discovered* rather than configured. Evidence accumulates per
team across frames, and once a team's top five are clearly ahead of the sixth
the roster locks. From then on the gallery is restricted to those five per team,
which does three things at once:

- removes an entire class of error, since an out-of-roster name becomes
  unrepresentable rather than merely unlikely
- widens every match's margin, because the runner-up is now drawn from four
  alternatives instead of 172
- bounds the number of tracks a team can have, which per-frame detection has no
  way to constrain on its own

Locking is deliberately conservative. An incorrect lock is far more expensive
than a late one -- it makes the right answer permanently unreachable -- so the
fifth place must be both well supported and clearly ahead of the sixth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectral_sight.types import Team

TEAM_SIZE = 5


@dataclass
class Roster:
    """Accumulated evidence for who is playing, per team."""

    team_size: int = TEAM_SIZE

    min_evidence: float = 1.0
    """Support the last roster slot needs before a team can lock. Evidence is
    weighted by match margin, so this is several decisive sightings."""

    lock_margin: float = 0.4
    """Lead the last roster slot must hold over the first name outside it.
    Without this a team could lock while two champions were still trading
    fifth place, which is exactly when locking is most damaging."""

    evidence: dict[Team, dict[str, float]] = field(
        default_factory=lambda: {Team.BLUE: {}, Team.RED: {}}
    )
    _locked: dict[Team, frozenset[str]] = field(default_factory=dict)

    # -- accumulation -----------------------------------------------------

    def observe(self, team: Team, name: str, weight: float) -> None:
        """Record support for `name` playing on `team`."""
        if team not in self.evidence:
            return
        if team in self._locked:
            return
        if name in self._claimed_by_other(team):
            return
        team_evidence = self.evidence[team]
        team_evidence[name] = team_evidence.get(name, 0.0) + weight

    def _claimed_by_other(self, team: Team) -> frozenset[str]:
        """Champions the opposing team has already locked in.

        A champion plays for exactly one side, so anything the other team has
        settled on cannot be a candidate here. Leaving this out is not a
        rounding error: on real footage the enemy roster locked with an *ally*
        in it, which permanently displaced the champion who was really there.
        """
        other = Team.RED if team is Team.BLUE else Team.BLUE
        return self._locked.get(other, frozenset())

    def ranked(self, team: Team) -> list[tuple[str, float]]:
        """Candidate champions for a team, best supported first."""
        taken = self._claimed_by_other(team)
        return sorted(
            (
                (name, score)
                for name, score in self.evidence.get(team, {}).items()
                if name not in taken
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )

    # -- locking ----------------------------------------------------------

    def locked(self, team: Team) -> frozenset[str] | None:
        """The team's roster once settled, else None.

        Locking is sticky: once decided it does not revisit, because a roster
        that can change is not a constraint.
        """
        if team in self._locked:
            return self._locked[team]

        ranked = self.ranked(team)
        if len(ranked) < self.team_size:
            return None

        last_score = ranked[self.team_size - 1][1]
        if last_score < self.min_evidence:
            return None

        next_score = (
            ranked[self.team_size][1] if len(ranked) > self.team_size else 0.0
        )
        if last_score - next_score < self.lock_margin:
            return None

        roster = frozenset(name for name, _ in ranked[: self.team_size])
        self._locked[team] = roster
        # Whatever this team just claimed cannot belong to the other one, so
        # discard the support it had accrued there rather than letting it
        # compete for a slot it can no longer hold.
        other = Team.RED if team is Team.BLUE else Team.BLUE
        for name in roster:
            self.evidence.get(other, {}).pop(name, None)
        return roster

    @property
    def fully_locked(self) -> bool:
        return all(self.locked(team) is not None for team in (Team.BLUE, Team.RED))

    def names(self) -> dict[Team, frozenset[str]]:
        """Locked rosters, omitting teams that have not settled."""
        return {
            team: roster
            for team in (Team.BLUE, Team.RED)
            if (roster := self.locked(team)) is not None
        }
