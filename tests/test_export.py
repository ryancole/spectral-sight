"""The timeline format, and the pipeline's production of it.

Two things are being pinned here and they fail differently. The format tests
guard a file that outlives the process that wrote it, so they care about
round-tripping and about refusing input this build cannot read. The pipeline
tests care that a row says what happened -- that a champion in fog is recorded
as absent rather than dropped, and that world units appear when, and only when,
the calibration that makes them meaningful is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectral_sight.export import (
    Observation,
    SCHEMA,
    Skillshot,
    TimelineMeta,
    TimelineWriter,
    iter_timeline,
    read_timeline,
)
from spectral_sight.perception.identity import Gallery
from spectral_sight.perception.minimap import MinimapRegion, WorldTransform
from spectral_sight.pipeline import Pipeline
from spectral_sight.types import Team
from tests.synthetic import Marker, synthetic_minimap

MINIMAP_SIZE = 280
REGION = MinimapRegion(x=60, y=80, width=MINIMAP_SIZE, height=MINIMAP_SIZE)
FRAME_SIZE = (420, 400)
"""(width, height) of the synthetic full frame the minimap is pasted into."""


def frame_with(markers: tuple[Marker, ...]) -> np.ndarray:
    """A full frame carrying a synthetic minimap at `REGION`."""
    minimap, _ = synthetic_minimap(MINIMAP_SIZE, markers, with_distractors=False)
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
    frame[REGION.y : REGION.y + REGION.height,
          REGION.x : REGION.x + REGION.width] = minimap
    return frame


def build_pipeline(*, world: WorldTransform | None = None) -> Pipeline:
    """A pipeline with no gallery: identification is not what these test."""
    return Pipeline(
        region=REGION,
        gallery=Gallery(),
        world=world,
        resolution=FRAME_SIZE,
    )


def run(pipeline: Pipeline, markers: tuple[Marker, ...], frames: int,
        *, start: float = 0.0, step: float = 0.1):
    """Feed the same scene for `frames` frames, returning the last result."""
    result = None
    for i in range(frames):
        result = pipeline.process(frame_with(markers), start + i * step)
    return result


TWO_ALLIES = (Marker(60, 60, Team.BLUE), Marker(150, 170, Team.BLUE))


# -- the format -----------------------------------------------------------


def sample_observation(**overrides) -> Observation:
    fields = dict(
        video_time=12.5,
        track_id=3,
        team=Team.RED,
        x=101.25,
        y=88.5,
        visible=True,
        seconds_since_seen=0.0,
        game_time=754,
        game_time_observed=True,
        champion="Swain",
        world_x=4820.5,
        world_y=9110.2,
        is_self=False,
    )
    fields.update(overrides)
    return Observation(**fields)


def test_an_observation_round_trips() -> None:
    """Exactly, for values already at the precision the file stores. Anything
    finer is rounded on the way out and is not expected back -- see below."""
    original = sample_observation()
    assert Observation.from_dict(original.to_dict()) == original


def test_a_row_carrying_skillshots_round_trips() -> None:
    original = sample_observation(skillshots=(
        Skillshot(slot="Q", at=9.9, launched=10.0, speed=1200.0,
                  heading=(0.6, -0.8), miss=42.5, flight=0.3, outcome="hit",
                  fall=0.12, lead=-42.5),
        Skillshot(slot="W", at=10.0, launched=None, speed=None, heading=None,
                  miss=None, flight=None, outcome="unknown", fall=None,
                  lead=None),
    ))
    assert Observation.from_dict(original.to_dict()) == original


def test_a_row_with_no_skillshots_omits_the_key() -> None:
    assert "skillshots" not in sample_observation().to_dict()


def test_a_row_with_no_world_omits_the_keys_entirely() -> None:
    """Absent rather than null: a reader should not have to distinguish a
    position of None from a position at the origin."""
    row = sample_observation(world_x=None, world_y=None).to_dict()
    assert "world_x" not in row and "world_y" not in row
    assert Observation.from_dict(row).world_x is None


def test_positions_are_rounded_to_the_precision_they_carry() -> None:
    row = sample_observation(x=101.256789, world_x=4820.5678).to_dict()
    assert row["x"] == 101.26
    assert row["world_x"] == 4820.6


def test_numpy_scalars_survive_serialisation() -> None:
    """Track positions arrive as numpy floats, which json refuses natively.
    Getting this wrong fails only on real footage, never on hand-built rows."""
    row = sample_observation(x=np.float64(10.5), y=np.float32(20.25),
                             track_id=np.int64(4)).to_dict()
    assert json.loads(json.dumps(row))["x"] == 10.5


def test_an_unidentified_champion_is_written_as_null() -> None:
    row = sample_observation(champion=None).to_dict()
    assert row["champion"] is None
    assert Observation.from_dict(row).champion is None


def test_meta_round_trips() -> None:
    meta = TimelineMeta(
        source="clip.mp4", width=1920, height=1080, stride=3,
        created="2026-08-16T12:00:00+00:00", has_game_time=True,
        world_bounds={"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 1.0},
        world_units_per_pixel=[48.0, 48.3],
    )
    assert TimelineMeta.from_dict(meta.to_dict()) == meta


def test_the_writer_stamps_a_creation_time(tmp_path: Path) -> None:
    meta = TimelineMeta(source="clip.mp4", width=100, height=100, stride=3)
    with TimelineWriter(tmp_path / "out.jsonl", meta) as writer:
        pass
    assert writer.meta.created
    assert read_timeline(tmp_path / "out.jsonl")[0].created == writer.meta.created


def test_a_timeline_round_trips_through_a_file(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    meta = TimelineMeta(source="clip.mp4", width=100, height=100, stride=3)
    rows = [sample_observation(video_time=t / 10, track_id=t) for t in range(5)]
    with TimelineWriter(path, meta) as writer:
        writer.write(rows[:2])
        writer.write(rows[2:])

    loaded_meta, loaded = read_timeline(path)
    assert loaded_meta.source == "clip.mp4"
    assert loaded == rows
    assert writer.rows == 5


def test_the_header_lands_before_any_rows(tmp_path: Path) -> None:
    """A run killed part way through should leave a readable prefix, not a
    file whose header never got written."""
    path = tmp_path / "out.jsonl"
    with TimelineWriter(path, TimelineMeta("c.mp4", 100, 100, 3)) as writer:
        writer.write([sample_observation()])
        surviving = path.read_text(encoding="utf-8")

    truncated = tmp_path / "killed.jsonl"
    truncated.write_text(surviving, encoding="utf-8")
    meta, rows = read_timeline(truncated)
    assert meta.source == "c.mp4" and len(rows) == 1


def test_streaming_and_loading_agree(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    rows = [sample_observation(track_id=i) for i in range(4)]
    with TimelineWriter(path, TimelineMeta("c.mp4", 100, 100, 3)) as writer:
        writer.write(rows)
    assert list(iter_timeline(path)) == read_timeline(path)[1]


def test_a_timeline_from_the_future_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"schema": SCHEMA + 1, "source": "c.mp4",
                                "width": 1, "height": 1, "stride": 3}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_timeline(path)


def test_a_file_that_is_not_a_timeline_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_text('{"champion": "Swain"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="timeline"):
        read_timeline(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_timeline(path)


def test_writing_outside_a_context_is_an_error(tmp_path: Path) -> None:
    writer = TimelineWriter(tmp_path / "out.jsonl", TimelineMeta("c", 1, 1, 3))
    with pytest.raises(RuntimeError):
        writer.write([sample_observation()])


# -- what the pipeline produces -------------------------------------------


def test_every_confirmed_track_gets_a_row() -> None:
    pipeline = build_pipeline()
    result = run(pipeline, TWO_ALLIES, frames=5)
    assert len(result.observations) == len(result.tracks) > 0
    assert {o.track_id for o in result.observations} == {t.id for t in result.tracks}


def test_rows_are_ordered_by_track_id() -> None:
    """Two extractions of one clip should produce identical files, which the
    tracker's own ordering does not promise."""
    result = run(build_pipeline(), TWO_ALLIES, frames=5)
    ids = [o.track_id for o in result.observations]
    assert ids == sorted(ids)


def test_a_row_carries_crop_pixels_even_with_no_world() -> None:
    result = run(build_pipeline(), TWO_ALLIES, frames=5)
    for observation in result.observations:
        assert 0 <= observation.x <= MINIMAP_SIZE
        assert observation.world_x is None and observation.world_y is None


def test_world_units_appear_once_the_world_is_calibrated() -> None:
    world = WorldTransform.assuming_crop(REGION)
    result = run(build_pipeline(world=world), TWO_ALLIES, frames=5)
    assert result.observations
    for observation in result.observations:
        assert observation.world_x is not None
        assert world.bounds.min_x <= observation.world_x <= world.bounds.max_x
        assert world.bounds.min_y <= observation.world_y <= world.bounds.max_y


def test_world_y_is_flipped_relative_to_pixels() -> None:
    """A marker nearer the top of the minimap must have the *higher* world y.
    An axis flip is invisible in a round trip and wrong in every join."""
    world = WorldTransform.assuming_crop(REGION)
    high = Marker(140, 40, Team.BLUE)
    low = Marker(140, 240, Team.BLUE)
    result = run(build_pipeline(world=world), (high, low), frames=5)
    by_pixel_y = sorted(result.observations, key=lambda o: o.y)
    assert by_pixel_y[0].world_y > by_pixel_y[-1].world_y


def test_a_champion_in_fog_is_recorded_not_dropped() -> None:
    """The difference between 'elsewhere' and 'not looked at' is most of what
    player-perspective footage has to say."""
    pipeline = build_pipeline()
    run(pipeline, TWO_ALLIES, frames=5)
    result = run(pipeline, (), frames=1, start=2.0)

    assert result.observations, "fogged champions should still produce rows"
    for observation in result.observations:
        assert observation.visible is False
        assert observation.seconds_since_seen > pipeline.tracker.config.lost_after


def test_a_visible_champion_reports_no_gap() -> None:
    result = run(build_pipeline(), TWO_ALLIES, frames=5)
    assert all(o.visible for o in result.observations)
    assert all(o.seconds_since_seen == pytest.approx(0.0)
               for o in result.observations)


def test_no_game_time_without_a_clock() -> None:
    result = run(build_pipeline(), TWO_ALLIES, frames=5)
    for observation in result.observations:
        assert observation.game_time is None
        assert observation.game_time_observed is False


def test_observations_serialise_straight_out_of_the_pipeline() -> None:
    """The end-to-end shape of the thing: real tracker output, no hand-built
    values, through json.dumps."""
    result = run(build_pipeline(world=WorldTransform.assuming_crop(REGION)),
                 TWO_ALLIES, frames=5)
    encoded = [json.dumps(o.to_dict()) for o in result.observations]
    assert all(json.loads(line)["team"] == "blue" for line in encoded)


# -- the header the pipeline describes itself with ------------------------


def test_meta_records_a_missing_calibration_honestly() -> None:
    meta = build_pipeline().timeline_meta("data/clip.mp4", stride=3)
    assert meta.has_game_time is False
    assert meta.world_bounds is None and meta.world_units_per_pixel is None
    assert meta.schema == SCHEMA


def test_meta_records_the_world_scale_in_force() -> None:
    world = WorldTransform.assuming_crop(REGION)
    meta = build_pipeline(world=world).timeline_meta("data/clip.mp4", stride=3)
    assert meta.world_bounds == world.bounds.to_dict()
    assert meta.world_units_per_pixel == pytest.approx(
        list(world.units_per_pixel)
    )


def test_meta_records_the_source_name_not_its_path() -> None:
    """A timeline is shareable; someone's directory layout is not part of it."""
    meta = build_pipeline().timeline_meta(r"C:\Users\someone\clips\game.mp4", 3)
    assert meta.source == "game.mp4"


def test_meta_needs_a_resolution_from_somewhere() -> None:
    pipeline = Pipeline(region=REGION, gallery=Gallery())
    with pytest.raises(ValueError, match="resolution"):
        pipeline.timeline_meta("clip.mp4", stride=3)
    assert pipeline.timeline_meta("clip.mp4", 3, size=(1920, 1080)).width == 1920
