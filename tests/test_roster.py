"""Roster discovery and locking."""

from __future__ import annotations

from spectral_sight.perception.identity import Roster
from spectral_sight.tracking import Tracker, TrackState
from spectral_sight.types import Blip, Team

BLUE_FIVE = ["Galio", "MasterYi", "Kaisa", "Swain", "Zilean"]


def feed(roster: Roster, team: Team, names, times: int = 10,
         weight: float = 0.2) -> None:
    for _ in range(times):
        for name in names:
            roster.observe(team, name, weight)


def test_a_team_locks_once_five_are_clear() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE)
    assert roster.locked(Team.BLUE) == frozenset(BLUE_FIVE)


def test_a_team_does_not_lock_with_too_few_candidates() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE[:4])
    assert roster.locked(Team.BLUE) is None


def test_a_team_does_not_lock_on_weak_evidence() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE, times=1)
    assert roster.locked(Team.BLUE) is None


def test_a_contested_fifth_place_blocks_locking() -> None:
    """Locking early is worse than locking late: a wrong roster is permanent."""
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE[:4])
    # Two *other* champions trading the last slot, neither pulling ahead.
    feed(roster, Team.BLUE, ["Yorick"], times=6)
    feed(roster, Team.BLUE, ["Sivir"], times=6)
    assert roster.locked(Team.BLUE) is None

    # Once one of them pulls clear, the team settles.
    feed(roster, Team.BLUE, ["Yorick"], times=4)
    assert roster.locked(Team.BLUE) == frozenset(BLUE_FIVE[:4] + ["Yorick"])


def test_noise_does_not_displace_well_supported_names() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE, times=12)
    for name in ("Zaahen", "RekSai", "Thresh"):
        roster.observe(Team.BLUE, name, 0.2)
    assert roster.locked(Team.BLUE) == frozenset(BLUE_FIVE)


def test_locking_is_sticky() -> None:
    """A roster that can change is not a constraint."""
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE)
    locked = roster.locked(Team.BLUE)

    feed(roster, Team.BLUE, ["Yorick"], times=50)
    assert roster.locked(Team.BLUE) == locked


def test_teams_lock_independently() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE)
    assert roster.locked(Team.BLUE) is not None
    assert roster.locked(Team.RED) is None
    assert not roster.fully_locked


def test_a_locked_champion_cannot_join_the_other_team() -> None:
    """A champion plays for one side. On real footage, omitting this locked an
    ally into the enemy roster and permanently displaced the champion who was
    really there."""
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE)
    assert roster.locked(Team.BLUE) is not None

    # Swain is an ally, but enemy markers keep being misread as him.
    feed(roster, Team.RED, ["Swain"], times=30)
    assert dict(roster.ranked(Team.RED)).get("Swain") is None

    feed(roster, Team.RED, ["Yorick", "Xerath", "Renata", "Sivir", "Malphite"])
    assert roster.locked(Team.RED) == frozenset(
        ["Yorick", "Xerath", "Renata", "Sivir", "Malphite"]
    )


def test_locking_clears_stale_support_on_the_other_team() -> None:
    roster = Roster()
    feed(roster, Team.RED, ["Swain"], times=30)  # accrued before blue settled
    feed(roster, Team.BLUE, BLUE_FIVE)

    assert roster.locked(Team.BLUE) is not None
    assert "Swain" not in dict(roster.ranked(Team.RED))


def test_names_reports_only_settled_teams() -> None:
    roster = Roster()
    feed(roster, Team.BLUE, BLUE_FIVE)
    assert set(roster.names()) == {Team.BLUE}


# -- enforcement on the tracker --------------------------------------------


def blip(x: float, y: float, team: Team = Team.BLUE) -> Blip:
    return Blip(x=x, y=y, radius=13.0, team=team, score=0.9)


def _confirmed_tracker(positions) -> Tracker:
    tracker = Tracker()
    for step in range(4):
        tracker.update([blip(x, y) for x, y in positions], step * 0.1)
    return tracker


def test_enforcing_a_roster_strips_foreign_identities() -> None:
    tracker = _confirmed_tracker([(60, 60)])
    track = tracker.confirmed[0]
    track.observe_identity("Zaahen", 5.0)
    assert track.identity == "Zaahen"

    tracker.enforce_roster(Team.BLUE, frozenset(BLUE_FIVE), 5)
    assert track.identity is None


def test_a_team_never_holds_more_than_five_tracks() -> None:
    """A team fields five champions whether or not their names are known, so
    this holds from the first frame rather than waiting for a roster."""
    positions = [(40 * i + 20, 40 * i + 20) for i in range(8)]
    tracker = _confirmed_tracker(positions)
    assert len(tracker.confirmed) == 5


def test_capping_keeps_identified_tracks_first() -> None:
    positions = [(40 * i + 20, 40 * i + 20) for i in range(5)]
    tracker = _confirmed_tracker(positions)
    keepers = tracker.confirmed[:2]
    for track, name in zip(keepers, BLUE_FIVE):
        track.observe_identity(name, 3.0)

    tracker.enforce_roster(Team.BLUE, frozenset(BLUE_FIVE), 2)

    survivors = {id(t) for t in tracker.confirmed}
    assert len(survivors) == 2
    assert all(id(t) in survivors for t in keepers)


def test_enforcement_leaves_the_other_team_alone() -> None:
    tracker = Tracker()
    for step in range(4):
        tracker.update([blip(60, 60, Team.RED)], step * 0.1)
    red = tracker.confirmed[0]
    red.observe_identity("Yorick", 5.0)

    tracker.enforce_roster(Team.BLUE, frozenset(BLUE_FIVE), 5)
    assert red.identity == "Yorick"


def test_tentative_tracks_are_not_counted_against_the_cap() -> None:
    tracker = _confirmed_tracker([(60, 60), (160, 160)])
    tracker.update([blip(60, 60), blip(160, 160), blip(250, 60)], 0.4)

    tracker.enforce_roster(Team.BLUE, frozenset(BLUE_FIVE), 5)
    states = [t.state for t in tracker.tracks]
    assert TrackState.TENTATIVE in states
