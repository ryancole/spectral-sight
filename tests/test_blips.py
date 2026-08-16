"""Stage 1 detector behaviour against synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_sight.perception.minimap import BlipDetector, BlipDetectorConfig
from spectral_sight.perception.minimap.blips import scaled_config
from spectral_sight.types import Team
from tests.synthetic import (
    DEFAULT_MARKERS,
    Marker,
    draw_base_shading,
    draw_champion,
    draw_minion,
    draw_turret,
    synthetic_minimap,
)


@pytest.fixture
def detector() -> BlipDetector:
    return BlipDetector()


def _match(blips, marker: Marker, tolerance: float = 3.0):
    """The detected blip nearest `marker`, or None if nothing is within tolerance."""
    for blip in blips:
        if np.hypot(blip.x - marker.x, blip.y - marker.y) <= tolerance:
            return blip
    return None


def test_finds_every_champion_and_nothing_else(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap()
    blips = detector.detect(image)

    assert len(blips) == len(markers), (
        f"expected {len(markers)} blips, got {len(blips)}"
    )
    for marker in markers:
        found = _match(blips, marker)
        assert found is not None, f"missed champion at {(marker.x, marker.y)}"
        assert found.team is marker.team


def test_positions_are_accurate_to_a_pixel(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap()
    blips = detector.detect(image)

    for marker in markers:
        found = _match(blips, marker)
        assert found is not None
        assert abs(found.x - marker.x) <= 1.5
        assert abs(found.y - marker.y) <= 1.5


def test_scores_are_normalised(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap()
    for blip in detector.detect(image):
        assert 0.0 <= blip.score <= 1.0
        assert blip.score > 0.5, "a clean synthetic ring should score well"


def test_results_are_sorted_by_score(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap()
    scores = [b.score for b in detector.detect(image)]
    assert scores == sorted(scores, reverse=True)


# -- rejection of the things that are team-coloured but are not champions ----


def _blank(size: int = 280) -> np.ndarray:
    image, _ = synthetic_minimap(size=size, markers=(), with_distractors=False)
    return image


def test_rejects_solid_turret_glyphs(detector: BlipDetector) -> None:
    image = _blank()
    draw_turret(image, 80, 80, Team.BLUE)
    draw_turret(image, 180, 180, Team.RED)
    assert detector.detect(image) == []


def test_rejects_minion_dots(detector: BlipDetector) -> None:
    image = _blank()
    for i in range(10):
        draw_minion(image, 40 + i * 15, 140, Team.BLUE)
    assert detector.detect(image) == []


def test_rejects_base_shading(detector: BlipDetector) -> None:
    image = _blank()
    draw_base_shading(image, 60, 60, Team.BLUE, radius=50)
    draw_base_shading(image, 220, 220, Team.RED, radius=50)
    assert detector.detect(image) == []


def test_rejects_empty_terrain(detector: BlipDetector) -> None:
    assert detector.detect(_blank()) == []


# -- structural behaviour ---------------------------------------------------


def test_never_returns_more_than_ten(detector: BlipDetector) -> None:
    extra = DEFAULT_MARKERS + (
        Marker(90, 40, Team.BLUE),
        Marker(255, 175, Team.RED),
        Marker(20, 110, Team.BLUE),
    )
    image, _ = synthetic_minimap(markers=extra)
    assert len(detector.detect(image)) == 10


def test_suppresses_overlapping_detections(detector: BlipDetector) -> None:
    image = _blank()
    draw_champion(image, 140, 140, Team.BLUE)
    draw_champion(image, 143, 141, Team.RED)
    assert len(detector.detect(image)) == 1


def test_teams_are_separated(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap()
    blips = detector.detect(image)
    blue = sum(1 for b in blips if b.team is Team.BLUE)
    red = sum(1 for b in blips if b.team is Team.RED)
    assert blue == sum(1 for m in markers if m.team is Team.BLUE)
    assert red == sum(1 for m in markers if m.team is Team.RED)


def test_rejects_non_bgr_input(detector: BlipDetector) -> None:
    with pytest.raises(ValueError):
        detector.detect(np.zeros((64, 64), np.uint8))


def test_debug_exposes_masks(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap()
    blips, debug = detector.detect_with_debug(image)

    assert len(blips) == 10
    assert set(debug.masks) == {Team.BLUE, Team.RED}
    for mask in debug.masks.values():
        assert mask.shape == image.shape[:2]


def test_debug_records_why_a_candidate_was_dropped(detector: BlipDetector) -> None:
    # An oversized ring still encloses a hole, so it reaches the size filter
    # rather than being skipped outright -- which is what we want to observe.
    image = _blank()
    draw_champion(image, 140, 140, Team.BLUE, radius=45)

    blips, debug = detector.detect_with_debug(image)

    assert blips == []
    assert [reason for _, reason in debug.rejected] == ["core_radius"]


# -- scaling ----------------------------------------------------------------


def test_scaled_config_tracks_minimap_size() -> None:
    base = BlipDetectorConfig()
    doubled = scaled_config(base, minimap_width=560, reference_width=280)
    assert doubled.min_core_radius == base.min_core_radius * 2
    assert doubled.max_core_radius == base.max_core_radius * 2
    assert doubled.blue_bands == base.blue_bands, "colour is scale-invariant"


def test_detects_at_a_larger_minimap_scale() -> None:
    size = 560
    markers = tuple(
        Marker(m.x * 2, m.y * 2, m.team) for m in DEFAULT_MARKERS
    )
    image, _ = synthetic_minimap(size=size, markers=(), with_distractors=False)
    for marker in markers:
        draw_champion(image, marker.x, marker.y, marker.team, radius=20)

    detector = BlipDetector(scaled_config(BlipDetectorConfig(), minimap_width=size))
    blips = detector.detect(image)

    assert len(blips) == len(markers)
    for marker in markers:
        assert _match(blips, marker, tolerance=4.0) is not None
