"""Events derived from the envelope stream, and the property that makes them
trustworthy: a recorded timeline replayed through the deriver produces exactly
what the live run published.

Most tests here pin a transition rule -- what counts as a death, when a vanish
is news -- because those rules are where a change-detector quietly goes wrong:
a None that re-triggers, a corpse's fog-exit reported as movement, a track id
that did not survive the death it was reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectral_sight.events import EventDeriver
from spectral_sight.export import (
    AbilityUse,
    Observation,
    Skillshot,
    Threat,
    TimelineMeta,
)
from spectral_sight.feed import FrameState, JsonlSink, read_frames
from spectral_sight.types import Team


def row(track_id: int = 1, video_time: float = 10.0, **overrides: object) -> Observation:
    fields: dict = dict(
        video_time=video_time, track_id=track_id, team=Team.BLUE,
        x=100.0, y=120.0, visible=True, seconds_since_seen=0.0,
    )
    fields.update(overrides)
    return Observation(**fields)


def state(seq: int, rows: list[Observation], **overrides: object) -> FrameState:
    fields: dict = dict(
        seq=seq,
        video_time=rows[0].video_time if rows else 0.0,
        captured_at=None,
        game_time=rows[0].game_time if rows else None,
        game_time_observed=False,
        allies_dead=rows[0].allies_dead if rows else None,
        champions=rows,
        fps=None,
        dropped=0,
        lag=None,
    )
    fields.update(overrides)
    return FrameState(**fields)


def derive(*frames: FrameState) -> list:
    deriver = EventDeriver()
    return [event for frame in frames for event in deriver.update(frame)]


def kinds(events: list) -> list[str]:
    return [event.kind for event in events]


class TestCast:
    def test_the_row_is_the_event(self) -> None:
        """The pipeline already settles a cast onto exactly one row, so the
        deriver wraps it rather than re-deciding it."""
        events = derive(
            state(0, [row(champion="Xerath")]),
            state(1, [row(video_time=10.1, champion="Xerath", cast_drop=0.133,
                          cast_at=10.0, cast_span=0.1, cast_continuous=True,
                          cast_confirmed=True)]),
            state(2, [row(video_time=10.2, champion="Xerath")]),
        )
        casts = [e for e in events if e.kind == "cast"]
        assert len(casts) == 1
        assert casts[0].champion == "Xerath"
        assert casts[0].detail == {
            "drop": 0.133, "at": 10.0, "span": 0.1,
            "continuous": True, "confirmed": True,
        }


class TestAbility:
    def test_each_ability_use_is_one_event(self) -> None:
        """An ability rides exactly the row it settled on and is never carried,
        so a Q and a W in one interval are two events, no diffing involved."""
        events = derive(state(0, [row(
            champion="Ezreal", is_self=True,
            abilities=(
                AbilityUse(slot="Q", at=9.9, countdown=5, confirmed=True),
                AbilityUse(slot="W", at=9.95, countdown=None, confirmed=True),
            ),
        )]))
        abilities = [e for e in events if e.kind == "ability"]
        assert [e.detail["slot"] for e in abilities] == ["Q", "W"]
        assert abilities[0].detail == {
            "slot": "Q", "at": 9.9, "countdown": 5, "confirmed": True,
        }
        # An unread countdown is carried as None, not dropped.
        assert abilities[1].detail["countdown"] is None

    def test_a_row_without_abilities_emits_none(self) -> None:
        events = derive(state(0, [row(champion="Ezreal", is_self=True)]))
        assert "ability" not in kinds(events)


class TestThreat:
    def test_each_resolved_threat_is_one_event(self) -> None:
        threat = Threat(at=9.9, arrival=10.2, closest=12.5, speed=1500.0,
                        heading=(1.0, 0.0), outcome="dodged", damage=None,
                        moved_across=31.0, origin=140.0)
        events = derive(state(0, [row(champion="Ezreal", is_self=True,
                                      threats=(threat,))]))
        threats = [e for e in events if e.kind == "threat"]
        assert len(threats) == 1
        assert threats[0].detail["outcome"] == "dodged"
        assert threats[0].detail["moved_across"] == 31.0
        assert "damage" not in threats[0].detail


class TestSkillshot:
    def test_each_resolved_skillshot_is_one_event(self) -> None:
        shot = Skillshot(slot="Q", at=9.9, launched=10.0, speed=1200.0,
                         heading=(1.0, 0.0), miss=42.0, flight=0.3,
                         outcome="hit", fall=0.12, lead=None)
        events = derive(state(0, [row(champion="Ezreal", is_self=True,
                                      skillshots=(shot,))]))
        shots = [e for e in events if e.kind == "skillshot"]
        assert len(shots) == 1
        assert shots[0].detail["slot"] == "Q"
        assert shots[0].detail["outcome"] == "hit"
        assert shots[0].detail["miss"] == 42.0
        # A target who was not moving carries no lead, rather than a zero.
        assert "lead" not in shots[0].detail

    def test_a_cast_that_launched_nothing_still_reports(self) -> None:
        shot = Skillshot(slot="R", at=9.9, launched=None, speed=None,
                         heading=None, miss=None, flight=None,
                         outcome="unknown", fall=None, lead=None)
        events = derive(state(0, [row(champion="Ezreal", is_self=True,
                                      skillshots=(shot,))]))
        detail = [e for e in events if e.kind == "skillshot"][0].detail
        assert detail["outcome"] == "unknown"
        assert "launched" not in detail


class TestLiveness:
    def test_a_death_is_a_transition_not_a_state(self) -> None:
        events = derive(
            state(0, [row(champion="Zilean", alive=True)]),
            state(1, [row(video_time=10.1, champion="Zilean", alive=False)]),
            state(2, [row(video_time=10.2, champion="Zilean", alive=False)]),
        )
        assert kinds(events) == ["identified", "death"]

    def test_a_none_between_two_falses_is_one_death(self) -> None:
        """None means the HUD and the minimap disagreed, and a disagreement
        must not resurrect anyone just to kill them again."""
        events = derive(
            state(0, [row(champion="Zilean", alive=True)]),
            state(1, [row(video_time=10.1, champion="Zilean", alive=False)]),
            state(2, [row(video_time=10.2, champion="Zilean", alive=None)]),
            state(3, [row(video_time=10.3, champion="Zilean", alive=False)]),
        )
        assert kinds(events).count("death") == 1

    def test_respawn_reports_the_downtime(self) -> None:
        events = derive(
            state(0, [row(champion="Zilean", alive=True)]),
            state(1, [row(video_time=10.0, champion="Zilean", alive=False,
                          visible=False)]),
            state(2, [row(video_time=22.0, champion="Zilean", alive=True)]),
        )
        respawn = [e for e in events if e.kind == "respawn"]
        assert len(respawn) == 1
        assert respawn[0].detail == {"down_for": 12.0}

    def test_death_survives_the_tracks_churn(self) -> None:
        """A corpse's track is often dropped before the respawn arrives on a
        fresh one; the name is the identity that survives being dead."""
        events = derive(
            state(0, [row(track_id=3, champion="Zilean", alive=True)]),
            state(1, [row(track_id=3, video_time=10.1, champion="Zilean",
                          alive=False, visible=False)]),
            # The tracker dropped track 3; the respawn arrives on track 9.
            state(2, [row(track_id=9, video_time=22.1, champion="Zilean",
                          alive=True)]),
        )
        assert kinds(events).count("death") == 1
        assert kinds(events).count("respawn") == 1

    def test_a_stream_joined_mid_corpse_still_reports_the_death(self) -> None:
        """Late knowledge of a real event is an event, not a non-event."""
        events = derive(
            state(0, [row(champion="Zilean", alive=False, visible=False)]),
        )
        assert kinds(events) == ["identified", "death"]


class TestFog:
    def test_fog_is_a_round_trip(self) -> None:
        events = derive(
            state(0, [row(champion="Ekko", team=Team.RED)]),
            state(1, [row(video_time=11.0, champion="Ekko", team=Team.RED,
                          visible=False, seconds_since_seen=0.6,
                          world_x=8000.0, world_y=9000.0)]),
            state(2, [row(video_time=14.5, champion="Ekko", team=Team.RED)]),
        )
        assert kinds(events) == ["identified", "vanished", "reappeared"]
        vanished, reappeared = events[1], events[2]
        assert vanished.detail["world_x"] == 8000.0
        # Gone since last *seen* (11.0 - 0.6), not since the debounced event:
        # the tracker waits `lost_after` before calling anyone unaccounted
        # for, and that wait was also time spent in fog.
        assert reappeared.detail == {"gone_for": 4.1}

    def test_first_sight_of_a_track_is_not_a_reappearance(self) -> None:
        assert kinds(derive(state(0, [row()]))) == []

    def test_a_death_explains_the_disappearance(self) -> None:
        """The corpse left the minimap because it is on a gray screen, and a
        consumer told 'vanished' would treat it as a champion in fog -- one
        who might walk out at them, which is the one thing a corpse cannot."""
        events = derive(
            state(0, [row(champion="Zilean", alive=True)]),
            state(1, [row(video_time=10.1, champion="Zilean", alive=False,
                          visible=False)]),
            state(2, [row(video_time=22.0, champion="Zilean", alive=True,
                          visible=True)]),
        )
        assert "vanished" not in kinds(events)
        assert "reappeared" not in kinds(events)
        assert kinds(events).count("death") == 1
        assert kinds(events).count("respawn") == 1


class TestLevels:
    def test_learning_a_level_is_state_reaching_one_is_change(self) -> None:
        events = derive(
            state(0, [row(level=3)]),
            state(1, [row(video_time=10.1, level=3)]),
            state(2, [row(video_time=10.2, level=4)]),
        )
        ups = [e for e in events if e.kind == "level_up"]
        assert len(ups) == 1
        assert ups[0].detail == {"level": 4}


class TestIdentity:
    def test_identified_fires_once_per_belief(self) -> None:
        events = derive(
            state(0, [row(champion="Zilean", is_self=True)]),
            state(1, [row(video_time=10.1, champion="Zilean", is_self=True)]),
        )
        assert kinds(events) == ["identified"]
        assert events[0].detail == {"is_self": True}

    def test_a_changed_belief_is_corrected_not_hidden(self) -> None:
        """Identity can shift before the roster locks; a consumer holding the
        old name must be told, or it holds a name the pipeline no longer
        believes."""
        events = derive(
            state(0, [row(champion="Galio")]),
            state(1, [row(video_time=10.1, champion="Zilean")]),
        )
        assert kinds(events) == ["identified", "identified"]
        assert events[1].detail == {"is_self": False, "replaces": "Galio"}


class TestRoster:
    def five(self, video_time: float, names: list[str]) -> FrameState:
        return state(0, [
            row(track_id=i, video_time=video_time, champion=name)
            for i, name in enumerate(names)
        ])

    def test_five_named_at_once_is_a_roster(self) -> None:
        names = ["A", "B", "C", "D", "E"]
        events = derive(self.five(10.0, names))
        rosters = [e for e in events if e.kind == "roster"]
        assert len(rosters) == 1
        assert rosters[0].team is Team.BLUE
        assert rosters[0].detail == {"champions": names}

    def test_four_named_is_not_a_roster_and_neither_is_a_repeat(self) -> None:
        events = derive(
            self.five(10.0, ["A", "B", "C", "D", "E"]),
            self.five(10.1, ["A", "B", "C", "D", "E"]),
            state(2, [row(track_id=i, video_time=10.2, champion=n)
                      for i, n in enumerate(["A", "B", "C", "D"])]),
        )
        assert kinds(events).count("roster") == 1

    def test_a_changed_set_is_re_announced(self) -> None:
        events = derive(
            self.five(10.0, ["A", "B", "C", "D", "E"]),
            self.five(10.1, ["A", "B", "C", "D", "F"]),
        )
        rosters = [e for e in events if e.kind == "roster"]
        assert [r.detail["champions"] for r in rosters] == [
            ["A", "B", "C", "D", "E"],
            ["A", "B", "C", "D", "F"],
        ]


class TestWireForm:
    def test_flat_and_discriminated(self) -> None:
        events = derive(
            state(0, [row(champion="Zilean", game_time=200, alive=True)]),
            state(1, [row(video_time=10.1, champion="Zilean", game_time=200,
                          alive=False, allies_dead=1)],
                  allies_dead=1, game_time=200),
        )
        death = [e for e in events if e.kind == "death"][0].to_dict()
        assert death == {
            "t": "event", "kind": "death", "seq": 1, "video_time": 10.1,
            "game_time": 200, "team": "blue", "champion": "Zilean",
            "track_id": 1, "allies_dead": 1,
        }


class TestDeterminism:
    def test_a_replay_produces_the_live_events(self, tmp_path: Path) -> None:
        """The property the whole design rests on: events are a pure function
        of the rows, so the timeline a run writes re-derives, byte for byte,
        the events that run published. Anything less means a downstream tool
        tested against replays was tested against a different game."""
        frames = [
            state(0, [row(champion="Zilean", alive=True, level=3,
                          world_x=4000.0, world_y=5000.0),
                      row(track_id=2, team=Team.RED, champion="Ekko")]),
            state(1, [row(video_time=10.1, champion="Zilean", alive=True,
                          level=4, cast_drop=0.133, cast_at=10.0,
                          cast_span=0.1, cast_continuous=True,
                          cast_confirmed=False),
                      row(track_id=2, video_time=10.1, team=Team.RED,
                          champion="Ekko", visible=False,
                          seconds_since_seen=0.6)]),
            state(2, [row(video_time=10.2, champion="Zilean", alive=False,
                          visible=False, allies_dead=1)], allies_dead=1),
            state(3, [row(video_time=22.0, champion="Zilean", alive=True),
                      row(track_id=2, video_time=22.0, team=Team.RED,
                          champion="Ekko")]),
        ]
        live = [event
                for deriver in [EventDeriver()]
                for frame in frames
                for event in deriver.update(frame)]

        meta = TimelineMeta(source="clip.mp4", width=420, height=400, stride=3)
        path = tmp_path / "run.jsonl"
        with JsonlSink(path, meta) as sink:
            for frame in frames:
                sink.publish(frame)

        deriver = EventDeriver()
        replayed = [event
                    for frame in read_frames(path)
                    for event in deriver.update(frame)]

        assert [e.to_dict() for e in replayed] == [e.to_dict() for e in live]
        # And the stream was worth comparing: it exercised most of the kinds.
        assert set(kinds(live)) >= {
            "identified", "level_up", "cast", "vanished", "death", "respawn",
            "reappeared",
        }
