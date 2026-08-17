"""Tracking behaviour: association, lifecycle across fog, and identity memory."""

from __future__ import annotations

import pytest

from spectral_sight.perception.identity import Match
from spectral_sight.tracking import Track, Tracker, TrackerConfig, TrackState
from spectral_sight.types import Blip, Team

DT = 0.1


def blip(x: float, y: float, team: Team = Team.BLUE) -> Blip:
    return Blip(x=x, y=y, radius=13.0, team=team, score=0.9)


def match(name: str, similarity: float = 0.7, margin: float = 0.2) -> Match:
    return Match(name=name, similarity=similarity, margin=margin)


def run(tracker: Tracker, frames, start: float = 0.0) -> float:
    """Feed a sequence of (detections, matches) pairs; return the final time."""
    t = start
    for detections, matches in frames:
        tracker.update(detections, t, matches)
        t += DT
    return t


# -- confirmation -----------------------------------------------------------


def test_a_single_sighting_is_not_reported() -> None:
    """Stage 1 is tuned to over-produce; one-off blobs must not become tracks."""
    tracker = Tracker()
    tracker.update([blip(100, 100)], 0.0)
    assert tracker.confirmed == []


def test_a_repeated_sighting_is_confirmed() -> None:
    tracker = Tracker()
    run(tracker, [([blip(100 + i * 0.5, 100)], [None]) for i in range(4)])
    assert len(tracker.confirmed) == 1


def test_a_moving_champion_stays_one_track() -> None:
    tracker = Tracker()
    run(tracker, [([blip(100 + i * 1.5, 100 + i)], [None]) for i in range(10)])
    assert len(tracker.confirmed) == 1
    assert tracker.confirmed[0].hits == 10


def test_two_champions_stay_separate() -> None:
    tracker = Tracker()
    run(tracker, [
        ([blip(60, 60), blip(200, 200)], [None, None]) for _ in range(5)
    ])
    assert len(tracker.confirmed) == 2


def test_teams_never_associate() -> None:
    """A red marker must not be absorbed into a blue track at the same spot."""
    tracker = Tracker()
    run(tracker, [([blip(100, 100, Team.BLUE)], [None]) for _ in range(4)])
    run(tracker, [([blip(100, 100, Team.RED)], [None]) for _ in range(4)], start=0.4)

    teams = {t.team for t in tracker.confirmed}
    assert teams == {Team.BLUE, Team.RED}


# -- fog --------------------------------------------------------------------


def test_a_champion_going_into_fog_is_kept_not_dropped() -> None:
    tracker = Tracker()
    run(tracker, [([blip(100, 100)], [None]) for _ in range(4)])

    for i in range(20):  # two seconds with nothing detected
        tracker.update([], 0.4 + i * DT)

    assert len(tracker.confirmed) == 1
    assert tracker.confirmed[0].state is TrackState.LOST


def test_reappearing_nearby_resumes_the_same_track() -> None:
    """The whole point of keeping lost tracks: fog should not reset identity."""
    tracker = Tracker()
    run(tracker, [([blip(100, 100)], [match("Swain")]) for _ in range(4)])
    original = tracker.confirmed[0].id

    for i in range(15):
        tracker.update([], 0.4 + i * DT)
    tracker.update([blip(112, 104)], 1.9, [match("Swain")])

    assert len(tracker.confirmed) == 1
    assert tracker.confirmed[0].id == original


def test_the_gate_widens_with_time_spent_in_fog() -> None:
    """A long absence permits a larger jump; a momentary one does not."""
    near = Tracker()
    run(near, [([blip(100, 100)], [None]) for _ in range(4)])
    near.update([blip(160, 100)], 0.5)  # 60px after 0.1s -- impossible
    assert len(near.confirmed) == 1  # the far blip started its own tentative track

    far = Tracker()
    run(far, [([blip(100, 100)], [None]) for _ in range(4)])
    for i in range(50):
        far.update([], 0.4 + i * DT)
    far.update([blip(160, 100)], 5.4)  # same jump, five seconds later
    assert len(far.confirmed) == 1
    assert far.confirmed[0].hits == 5


def test_a_track_is_eventually_forgotten() -> None:
    tracker = Tracker(TrackerConfig(forget_after=1.0))
    run(tracker, [([blip(100, 100)], [None]) for _ in range(4)])
    tracker.update([], 5.0)
    assert tracker.confirmed == []


# -- identity ---------------------------------------------------------------


def test_identity_needs_more_than_one_good_frame() -> None:
    tracker = Tracker()
    run(tracker, [
        ([blip(100, 100)], [match("Swain")]),
        ([blip(100, 100)], [None]),
        ([blip(100, 100)], [None]),
    ])
    assert tracker.confirmed[0].identity is None


def test_identity_accumulates_across_frames() -> None:
    tracker = Tracker()
    run(tracker, [([blip(100, 100)], [match("Swain")]) for _ in range(4)])
    assert tracker.confirmed[0].identity == "Swain"


def test_identity_survives_frames_the_gallery_cannot_read() -> None:
    """The reason tracking helps: a 26px marker is often unreadable."""
    tracker = Tracker()
    run(tracker, [([blip(100, 100)], [match("Swain")]) for _ in range(4)])
    run(tracker, [([blip(100, 100)], [None]) for _ in range(20)], start=0.4)

    assert tracker.confirmed[0].identity == "Swain"


def test_a_contested_identity_reports_nothing() -> None:
    """Two champions trading frames is better reported as unknown than forced."""
    tracker = Tracker()
    run(tracker, [
        ([blip(100, 100)], [match("Swain" if i % 2 else "Galio")])
        for i in range(8)
    ])
    assert tracker.confirmed[0].identity is None


def test_a_wrong_identity_is_outvoted() -> None:
    tracker = Tracker()
    frames = [([blip(100, 100)], [match("Galio")])]
    frames += [([blip(100, 100)], [match("Swain")]) for _ in range(6)]
    run(tracker, frames)
    assert tracker.confirmed[0].identity == "Swain"


def test_identity_pulls_association_toward_the_right_track() -> None:
    """Two champions close together are separated by who they look like."""
    tracker = Tracker()
    run(tracker, [
        ([blip(100, 100), blip(118, 100)], [match("Swain"), match("Galio")])
        for _ in range(4)
    ])
    # Swapped positions on the next frame; identity should follow the names.
    tracker.update(
        [blip(118, 100), blip(100, 100)], 0.4, [match("Galio"), match("Swain")]
    )

    identified = tracker.identified()
    assert set(identified) == {"Swain", "Galio"}
    assert identified["Swain"].distance_to(100, 100) < 6
    assert identified["Galio"].distance_to(118, 100) < 6


def test_identified_reports_one_track_per_champion() -> None:
    """A champion is in exactly one place, however many tracks claim the name."""
    tracker = Tracker()
    run(tracker, [
        ([blip(60, 60), blip(200, 200)], [match("Swain"), match("Swain")])
        for _ in range(4)
    ])
    assert list(tracker.identified()) == ["Swain"]


# -- interface --------------------------------------------------------------


def test_mismatched_match_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        Tracker().update([blip(1, 1), blip(2, 2)], 0.0, [None])


def test_visible_excludes_champions_currently_in_fog() -> None:
    tracker = Tracker()
    run(tracker, [([blip(100, 100)], [None]) for _ in range(4)])
    for i in range(10):
        tracker.update([], 0.4 + i * DT)

    assert len(tracker.confirmed) == 1
    assert tracker.visible(1.4) == []


def test_prediction_is_used_for_association() -> None:
    track = Track(id=1, team=Team.BLUE, x=100.0, y=100.0, last_seen=0.0,
                  vx=10.0, vy=0.0)
    assert track.predict(0.5) == pytest.approx((105.0, 100.0))


def test_coasting_damps_velocity() -> None:
    track = Track(id=1, team=Team.BLUE, x=0.0, y=0.0, last_seen=0.0, vx=10.0)
    track.coast()
    assert track.vx < 10.0
