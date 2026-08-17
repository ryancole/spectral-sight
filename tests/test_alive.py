"""Reading who is alive off the HUD portraits, and attributing it to champions.

The frames here are painted, not captured: a slot is a disc of flat colour, and
killing a champion means draining the colour out of their disc. What that pins
is the machinery -- that the judgement is made against the slot's own history
rather than a fixed number, that an unproven slot declines to answer, and that
a death only reaches the timeline attached to a champion the pipeline can
actually name. Whether real portraits desaturate far enough to clear the
threshold is a question about footage, and the answer measured over three
recordings is in the module docstring.

The attribution tests matter more than the reading ones. Reading a grey circle
is easy; the mistake that survived a working reader was joining the HUD's count
of dead teammates to the minimap's count of missing ones, which is wrong in a
way no synthetic frame reveals and only real footage exposed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.export import Observation
from spectral_sight.perception.hud.alive import (
    DEAD_FRACTION,
    MIN_BASELINE,
    AliveReader,
    Liveness,
    SlotState,
)
from spectral_sight.perception.hud.portraits import PortraitLayout
from spectral_sight.perception.identity import Gallery
from spectral_sight.perception.minimap import MinimapRegion
from spectral_sight.pipeline import SELF_SLOT, Pipeline
from spectral_sight.types import Team
from tests.synthetic import Marker, synthetic_minimap

FRAME_SIZE = (420, 400)
"""(width, height), matching tests/test_export.py so the two can share a frame."""

MINIMAP_SIZE = 280
REGION = MinimapRegion(x=60, y=80, width=MINIMAP_SIZE, height=MINIMAP_SIZE)

LAYOUT = PortraitLayout(
    ally_first_center_x=20.0,
    ally_center_y=370.0,
    ally_spacing=30.0,
    ally_radius=12,
    self_center_x=160.0,
    self_center_y=370.0,
    self_radius=14,
)

LIVING = (40, 200, 200)
"""BGR for a saturated portrait. Any strong hue does; only saturation is read."""

DEAD = (120, 120, 120)
"""Flat grey, which is how the game draws a dead champion's portrait."""


def frame(*, dead: tuple[str, ...] = (), markers: tuple[Marker, ...] = ()) -> np.ndarray:
    """A frame carrying five friendly portraits, and optionally a minimap."""
    canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    if markers:
        minimap, _ = synthetic_minimap(MINIMAP_SIZE, markers, with_distractors=False)
        canvas[REGION.y : REGION.y + REGION.height,
               REGION.x : REGION.x + REGION.width] = minimap

    for index in range(LAYOUT.ally_count):
        name = f"ally{index + 1}"
        cx, cy = LAYOUT.ally_center(index)
        cv2.circle(canvas, (int(cx), int(cy)), LAYOUT.ally_radius,
                   DEAD if name in dead else LIVING, -1)
    cv2.circle(canvas, (int(LAYOUT.self_center_x), int(LAYOUT.self_center_y)),
               LAYOUT.self_radius, DEAD if SELF_SLOT in dead else LIVING, -1)
    return canvas


def read(reader: AliveReader, *, dead: tuple[str, ...] = (), frames: int = 1):
    for _ in range(frames):
        liveness = reader.read(frame(dead=dead))
    return liveness


# -- reading a slot -------------------------------------------------------


def test_a_living_slot_reads_alive() -> None:
    assert all(s.alive for s in read(AliveReader(LAYOUT)).slots)


def test_every_friendly_slot_is_reported() -> None:
    slots = {s.slot for s in read(AliveReader(LAYOUT)).slots}
    assert slots == {"ally1", "ally2", "ally3", "ally4", SELF_SLOT}


def test_a_drained_portrait_reads_dead() -> None:
    reader = AliveReader(LAYOUT)
    read(reader, frames=3)
    liveness = read(reader, dead=("ally2",))
    assert liveness.slot("ally2").alive is False
    assert liveness.dead_count == 1
    assert [s.slot for s in liveness.dead] == ["ally2"]


def test_the_local_player_is_read_like_any_other_slot() -> None:
    reader = AliveReader(LAYOUT)
    read(reader, frames=3)
    assert read(reader, dead=(SELF_SLOT,)).slot(SELF_SLOT).alive is False


def test_a_slot_is_judged_against_its_own_baseline_not_a_constant() -> None:
    """The point of the whole design. A drab champion sitting below another
    champion's dead reading must still read alive, which no fixed floor placed
    between two absolute values can do."""
    drab = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    for index in range(LAYOUT.ally_count):
        cx, cy = LAYOUT.ally_center(index)
        # Saturation ~85: below a vivid champion's dead-portrait threshold
        # would be if that threshold were absolute.
        cv2.circle(drab, (int(cx), int(cy)), LAYOUT.ally_radius, (85, 128, 128), -1)
    cv2.circle(drab, (int(LAYOUT.self_center_x), int(LAYOUT.self_center_y)),
               LAYOUT.self_radius, LIVING, -1)

    reader = AliveReader(LAYOUT)
    for _ in range(3):
        liveness = reader.read(drab)

    vivid = reader.baselines[SELF_SLOT]
    assert reader.baselines["ally1"] < vivid * DEAD_FRACTION, (
        "this test is only meaningful if the drab champion sits below what the "
        "vivid one's threshold would be"
    )
    assert all(s.alive for s in liveness.slots)


def test_a_slot_that_has_never_looked_alive_says_nothing() -> None:
    """A clip opening on a dead champion, or on a loading screen. Anchoring the
    baseline to a grey portrait would read alive forever after."""
    blank = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    liveness = AliveReader(LAYOUT).read(blank)
    assert all(s.alive is None for s in liveness.slots)
    assert liveness.dead_count is None


def test_the_baseline_survives_a_death_and_the_slot_recovers() -> None:
    reader = AliveReader(LAYOUT)
    read(reader, frames=3)
    baseline = reader.baselines["ally1"]
    read(reader, dead=("ally1",), frames=5)
    assert reader.baselines["ally1"] == baseline
    assert read(reader).slot("ally1").alive is True


def test_the_respawn_countdown_does_not_revive_a_dead_slot() -> None:
    """A dead portrait is not blank -- the game draws a countdown across it in
    saturated red. Measured on real footage those digits pulled the *mean*
    saturation of the local player's grey portrait to within 0.02 of the living
    threshold, and flickered as the number changed width."""
    painted = frame(dead=(SELF_SLOT,))
    cv2.putText(painted, "8", (int(LAYOUT.self_center_x) - 6,
                               int(LAYOUT.self_center_y) + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2, cv2.LINE_AA)

    reader = AliveReader(LAYOUT)
    read(reader, frames=3)
    assert reader.read(painted).slot(SELF_SLOT).alive is False


def test_one_unreadable_slot_withholds_the_whole_count() -> None:
    """A count assembled from four readable slots and one unreadable one is
    quietly too low, which is how a missing reading becomes a false death."""
    liveness = Liveness(slots=(
        SlotState("ally1", 100.0, 100.0, True),
        SlotState("ally2", 10.0, 100.0, False),
        SlotState("ally3", 0.0, 0.0, None),
        SlotState("ally4", 100.0, 100.0, True),
        SlotState(SELF_SLOT, 100.0, 100.0, True),
    ))
    assert liveness.dead_count is None


def test_the_minimum_baseline_is_what_makes_unproven_slots_silent() -> None:
    assert MIN_BASELINE > 0
    reader = AliveReader(LAYOUT, min_baseline=0.0)
    assert all(s.alive is not None for s in reader.read(
        np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)).slots)


# -- attributing a death to a champion ------------------------------------


ALLIES = (
    Marker(60, 60, Team.BLUE),
    Marker(150, 170, Team.BLUE),
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        region=REGION,
        gallery=Gallery(),
        portraits=LAYOUT,
        resolution=FRAME_SIZE,
    )


def run(pipeline: Pipeline, *, dead: tuple[str, ...] = (),
        markers: tuple[Marker, ...] = ALLIES, frames: int = 5,
        start: float = 0.0, step: float = 0.1):
    result = None
    for i in range(frames):
        result = pipeline.process(frame(dead=dead, markers=markers),
                                  start + i * step)
    return result


def test_nobody_dead_clears_every_ally_including_the_ones_lost() -> None:
    """The case the HUD exists for. An ally the tracker dropped is not a
    casualty, and without the portraits nothing could tell the difference."""
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    result = run(pipeline, markers=(), frames=1, start=2.0)

    assert result.observations
    for row in result.observations:
        assert row.visible is False, "the markers were removed"
        assert row.alive is True, "but the HUD says nobody died"
    assert result.liveness.dead_count == 0


def test_a_dead_teammate_nobody_can_name_is_reported_as_unknown() -> None:
    """Counting is not enough: the HUD knows one teammate is down but not
    which, and guessing hands a dead champion's name to a living one."""
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    result = run(pipeline, dead=("ally1",), markers=(), frames=1, start=2.0)

    assert result.liveness.dead_count == 1
    assert all(row.alive is None for row in result.observations)
    assert all(row.allies_dead == 1 for row in result.observations)


def test_the_count_is_recorded_even_when_no_champion_can_be_named() -> None:
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    result = run(pipeline, dead=("ally1", "ally2"), frames=1, start=2.0)
    assert all(row.allies_dead == 2 for row in result.observations)
    assert all(row.alive is None for row in result.observations)


def name_tracks(pipeline: Pipeline, *names: str) -> None:
    """Force identities onto the confirmed tracks, in id order.

    The synthetic minimap carries no champion art, so the gallery can never
    name anything here. What is under test is what the pipeline does with names
    once it has them, not how it gets them.
    """
    tracks = sorted(pipeline.tracker.confirmed, key=lambda t: t.id)
    assert len(tracks) >= len(names), "not enough tracks to name"
    for track, name in zip(tracks, names):
        track.evidence[name] = 5.0


def test_the_local_players_death_is_attributed_by_name() -> None:
    """The one route that does not need counting. The viewport names the local
    player while they are alive; that name is what identifies the casualty when
    their own portrait greys out."""
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    name_tracks(pipeline, "Zilean", "Ryze")
    pipeline._self_evidence = {"Zilean": 50}

    result = run(pipeline, dead=(SELF_SLOT,), markers=(), frames=1, start=2.0)

    verdicts = {row.champion: row.alive for row in result.observations}
    assert verdicts["Zilean"] is False
    assert verdicts["Ryze"] is True, (
        "naming the only casualty clears everyone else"
    )


def test_a_few_stray_frames_cannot_change_who_the_player_is() -> None:
    """The camera sometimes sits on a teammate, and the viewport names them.
    Taking the most recent answer attributed a real death to two champions who
    were alive throughout; the player is one champion for the whole game, so the
    evidence decides it."""
    pipeline = build_pipeline()
    pipeline._self_evidence = {"Zilean": 200, "Lux": 12, "Galio": 9}
    assert pipeline.self_champion == "Zilean"


def test_a_contested_player_identity_is_not_claimed() -> None:
    pipeline = build_pipeline()
    pipeline._self_evidence = {"Zilean": 30, "Lux": 28}
    assert pipeline.self_champion is None


def test_too_few_sightings_is_not_enough_to_name_the_player() -> None:
    pipeline = build_pipeline()
    pipeline._self_evidence = {"Zilean": 3}
    assert pipeline.self_champion is None


def test_a_death_alongside_the_local_player_is_not_guessed() -> None:
    """Two down and only one nameable: the other could be any teammate."""
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    name_tracks(pipeline, "Zilean", "Ryze")
    pipeline._self_evidence = {"Zilean": 50}

    result = run(pipeline, dead=(SELF_SLOT, "ally1"), markers=(), frames=1,
                 start=2.0)
    assert all(row.alive is None for row in result.observations)
    assert all(row.allies_dead == 2 for row in result.observations)


def test_an_unnamed_local_player_cannot_attribute_their_own_death() -> None:
    """Without a name for the casualty, clearing the others would be a guess:
    one of them could be the dead champion under an unsettled identity."""
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    name_tracks(pipeline, "Zilean", "Ryze")
    pipeline._self_evidence = {}

    result = run(pipeline, dead=(SELF_SLOT,), markers=(), frames=1, start=2.0)
    assert all(row.alive is None for row in result.observations)


def test_a_named_casualty_missing_from_the_tracks_clears_nobody() -> None:
    pipeline = build_pipeline()
    run(pipeline, frames=5)
    name_tracks(pipeline, "Zilean", "Ryze")
    pipeline._self_evidence = {"Sejuani": 50}

    result = run(pipeline, dead=(SELF_SLOT,), markers=(), frames=1, start=2.0)
    assert all(row.alive is None for row in result.observations)


def test_enemies_never_carry_a_verdict() -> None:
    """No HUD panel names them, and fog means their absence says nothing."""
    pipeline = build_pipeline()
    result = run(pipeline, markers=(Marker(60, 60, Team.BLUE),
                                    Marker(200, 90, Team.RED)))
    reds = [row for row in result.observations if row.team is Team.RED]
    assert reds and all(row.alive is None for row in reds)


def test_without_calibrated_portraits_nothing_is_claimed() -> None:
    pipeline = Pipeline(region=REGION, gallery=Gallery(), resolution=FRAME_SIZE)
    result = run(pipeline)
    assert result.liveness is None
    assert all(row.alive is None and row.allies_dead is None
               for row in result.observations)
    assert pipeline.timeline_meta("c.mp4", 3).has_liveness is False


def test_the_header_records_that_liveness_was_read() -> None:
    assert build_pipeline().timeline_meta("c.mp4", 3).has_liveness is True


def test_a_verdict_round_trips_through_the_file_format() -> None:
    row = Observation(video_time=1.0, track_id=1, team=Team.BLUE, x=1.0, y=2.0,
                      visible=False, seconds_since_seen=3.0, alive=False,
                      allies_dead=2)
    assert Observation.from_dict(row.to_dict()) == row


@pytest.mark.parametrize("value", [True, False, None])
def test_alive_is_written_even_when_unknown(value: bool | None) -> None:
    """Null rather than absent: a consumer should not have to distinguish a
    missing key from an unknown verdict."""
    row = Observation(video_time=1.0, track_id=1, team=Team.BLUE, x=1.0, y=2.0,
                      visible=True, seconds_since_seen=0.0, alive=value)
    assert "alive" in row.to_dict()
    assert Observation.from_dict(row.to_dict()).alive is value
