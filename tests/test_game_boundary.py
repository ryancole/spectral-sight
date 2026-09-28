"""One run, several games: telling them apart and starting each one clean.

A VOD can hold more than one match, and a live run can stay up across a queue.
The timer is what says a new one began -- it goes back to the start of a match
-- and everything accumulated about the last game has to go with it: the roster
lock, the tracks, the player's identity, the events' baselines.
"""

from __future__ import annotations

from spectral_sight.events import EventDeriver
from spectral_sight.export import Observation
from spectral_sight.feed import FrameState, KnownRoster
from spectral_sight.perception.hud.clock import ClockFilter, GameClock
from spectral_sight.types import Team
from tests.test_alive import ScriptedClock, build_pipeline, frame, run


def resync(clock: ClockFilter, seconds: int, at: float) -> GameClock | None:
    """Show the filter `seconds` long enough for it to give way."""
    clock.update(GameClock(seconds, 1.0), at)
    return clock.update(GameClock(seconds + 4, 1.0), at + 4.0)


# -- the filter -----------------------------------------------------------


def test_the_clock_going_back_to_the_start_is_a_new_game() -> None:
    clock = ClockFilter()
    clock.update(GameClock(1800, 1.0), 0.0)
    resync(clock, 5, 60.0)
    assert clock.resynced and clock.new_game


def test_a_new_game_is_announced_for_exactly_one_update() -> None:
    clock = ClockFilter()
    clock.update(GameClock(1800, 1.0), 0.0)
    resync(clock, 5, 60.0)
    clock.update(GameClock(10, 1.0), 65.0)
    assert not clock.new_game


def test_a_forward_seek_is_not_a_new_game() -> None:
    clock = ClockFilter()
    clock.update(GameClock(300, 1.0), 0.0)
    resync(clock, 1200, 1.0)
    assert clock.resynced and not clock.new_game


def test_a_seek_back_within_a_match_is_not_a_new_game() -> None:
    """Back, but not to the start: the same match, earlier on."""
    clock = ClockFilter()
    clock.update(GameClock(1500, 1.0), 0.0)
    resync(clock, 600, 1.0)
    assert clock.resynced and not clock.new_game


def test_an_early_pause_with_the_timer_hidden_is_not_a_new_game() -> None:
    """Paused at 0:40 for two minutes behind an overlay. The prediction kept
    counting to 2:40, but the timer resumes level with its last reading."""
    clock = ClockFilter()
    clock.update(GameClock(40, 1.0), 0.0)
    for t in range(1, 120):
        clock.update(None, float(t))
    resync(clock, 40, 120.0)
    assert clock.resynced and not clock.new_game


def test_the_first_reading_of_a_run_is_not_a_new_game() -> None:
    clock = ClockFilter()
    clock.update(GameClock(3, 1.0), 0.0)
    assert not clock.new_game


def test_the_clock_is_not_carried_between_games() -> None:
    """A ping hides the timer for a moment; the lobby hides it for minutes.
    Past `max_hold` there is no match left to count."""
    clock = ClockFilter(max_hold=20.0)
    clock.update(GameClock(1800, 1.0), 0.0)
    held = clock.update(None, 5.0)
    assert held is not None and held.total_seconds == 1805
    assert clock.update(None, 25.0) is None


def test_a_misread_after_the_lobby_does_not_start_a_game() -> None:
    """The state survives the hold, so a stray read in the lobby still has to
    outlast `resync_after` before it is believed."""
    clock = ClockFilter()
    clock.update(GameClock(1800, 1.0), 0.0)
    assert clock.update(GameClock(7, 1.0), 60.0) is None
    assert clock.update(None, 61.0) is None
    assert not clock.new_game


# -- the pipeline ---------------------------------------------------------


def test_a_new_game_starts_the_pipeline_over() -> None:
    clock = ScriptedClock()
    pipeline = build_pipeline()
    pipeline.clock = clock
    clock.reading = GameClock(1500, 1.0)
    first = run(pipeline, frames=5)
    assert first.game == 0 and first.observations
    old_ids = {row.track_id for row in first.observations}
    pipeline.roster.observe(Team.RED, "Zed", 5.0)
    pipeline._self_evidence["Ahri"] = 50

    clock.reading = GameClock(2, 1.0)
    pipeline.process(frame(), 100.0)
    result = run(pipeline, frames=5, start=104.0)

    assert result.game == 1 and pipeline.game == 1
    assert all(row.game == 1 for row in result.observations)
    assert pipeline.roster.evidence[Team.RED] == {}
    assert pipeline.self_champion is None and not pipeline._self_evidence
    new_ids = {row.track_id for row in result.observations}
    assert new_ids and not (new_ids & old_ids), (
        "track ids keep counting, so one id never names two games' tracks"
    )


def test_a_seek_keeps_the_game() -> None:
    clock = ScriptedClock()
    pipeline = build_pipeline()
    pipeline.clock = clock
    clock.reading = GameClock(300, 1.0)
    run(pipeline, frames=5)
    pipeline.roster.observe(Team.RED, "Zed", 5.0)

    clock.reading = GameClock(1200, 1.0)
    result = run(pipeline, frames=50, start=1.0)
    assert result.game == 0
    assert pipeline.roster.evidence[Team.RED] == {"Zed": 5.0}


# -- the feed -------------------------------------------------------------


def row(game: int, champion: str | None = "Ahri", **overrides: object) -> Observation:
    fields: dict = dict(
        video_time=10.0, track_id=1, team=Team.BLUE, x=100.0, y=120.0,
        visible=True, seconds_since_seen=0.0, champion=champion, game=game,
    )
    fields.update(overrides)
    return Observation(**fields)


def state(seq: int, rows: list[Observation], game: int) -> FrameState:
    return FrameState(
        seq=seq, video_time=rows[0].video_time if rows else float(seq),
        captured_at=None, game_time=None, game_time_observed=False,
        allies_dead=None, champions=rows, fps=None, dropped=0, lag=None,
        game=game,
    )


def test_the_row_carries_its_game_through_the_file() -> None:
    original = row(2)
    assert original.to_dict()["game"] == 2
    assert Observation.from_dict(original.to_dict()) == original


def test_a_row_written_before_games_were_counted_reads_as_the_first() -> None:
    data = row(0).to_dict()
    del data["game"]
    assert Observation.from_dict(data).game == 0


def test_the_known_roster_starts_over_with_a_new_game() -> None:
    known = KnownRoster()
    known.update([row(0, "Ahri")])
    assert known.update([], game=1) == {Team.BLUE: (), Team.RED: ()}, (
        "a new game with no rows yet publishes no one, not the last game's team"
    )
    assert known.update([row(1, "Garen")])[Team.BLUE] == ("Garen",)


def test_a_new_game_is_an_event_and_the_deriver_forgets() -> None:
    deriver = EventDeriver()
    deriver.update(state(0, [row(0, alive=True)], game=0))
    deriver.update(state(1, [row(0, alive=False, video_time=11.0)], game=0))

    assert deriver.update(state(2, [], game=1)) == [], (
        "a frame with no rows is not written, so a replay could not place "
        "the event on it"
    )
    events = deriver.update(state(3, [row(1, alive=True, video_time=20.0)], game=1))
    assert [e.kind for e in events] == ["new_game", "identified"]
    assert events[0].to_dict()["game"] == 1
    assert "respawn" not in [e.kind for e in events], (
        "last game's death is not a baseline for this one"
    )


def test_the_first_game_of_a_run_is_not_an_event() -> None:
    events = EventDeriver().update(state(0, [row(0)], game=0))
    assert "new_game" not in [e.kind for e in events]
