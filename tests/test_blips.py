"""Stage 1 detector behaviour against synthetic ground truth.

Synthetic frames cannot tell us whether the HSV bands are fitted correctly --
only real footage does that. What they can pin is the geometry and the structural
rules, so that a later failure on real video is diagnosable as a tuning problem
rather than a logic one.

Several tests here encode specific failures found on real footage. They are
marked as regressions and should not be relaxed without new measurements.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_sight.perception.minimap import BlipDetector, BlipDetectorConfig
from spectral_sight.perception.minimap.blips import (
    REFERENCE_MINIMAP_WIDTH,
    scaled_config,
)
from spectral_sight.types import Team
from tests.synthetic import (
    DEFAULT_MARKERS,
    TEAM_BGR,
    Marker,
    draw_base_shading,
    draw_champion,
    draw_minion,
    draw_turret,
    synthetic_minimap,
)

CANVAS = 280


def make_detector(width: int = CANVAS) -> BlipDetector:
    return BlipDetector(scaled_config(BlipDetectorConfig(), minimap_width=width))


@pytest.fixture
def detector() -> BlipDetector:
    return make_detector()


def _blank(size: int = CANVAS) -> np.ndarray:
    image, _ = synthetic_minimap(size=size, markers=(), with_distractors=False)
    return image


def _match(blips, x: float, y: float, tolerance: float = 4.0):
    for blip in blips:
        if np.hypot(blip.x - x, blip.y - y) <= tolerance:
            return blip
    return None


# -- finding champions ------------------------------------------------------


def test_finds_every_champion(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap(size=CANVAS)
    blips = detector.detect(image)

    for marker in markers:
        found = _match(blips, marker.x, marker.y)
        assert found is not None, f"missed champion at {(marker.x, marker.y)}"
        assert found.team is marker.team


def test_positions_are_accurate(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap(size=CANVAS)
    blips = detector.detect(image)

    for marker in markers:
        found = _match(blips, marker.x, marker.y)
        assert found is not None
        assert abs(found.x - marker.x) <= 2.5
        assert abs(found.y - marker.y) <= 2.5


def test_scores_are_normalised(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap(size=CANVAS)
    for blip in detector.detect(image):
        assert 0.0 <= blip.score <= 1.0
        assert blip.score >= detector.config.min_ring_fill


def test_results_are_sorted_by_score(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap(size=CANVAS)
    scores = [b.score for b in detector.detect(image)]
    assert scores == sorted(scores, reverse=True)


def test_teams_are_separated(detector: BlipDetector) -> None:
    image, markers = synthetic_minimap(size=CANVAS)
    blips = detector.detect(image)
    for team in (Team.BLUE, Team.RED):
        expected = sum(1 for m in markers if m.team is team)
        assert sum(1 for b in blips if b.team is team) == expected


# -- regressions from real footage ------------------------------------------


@pytest.mark.parametrize("team", [Team.BLUE, Team.RED])
def test_finds_champion_whose_portrait_matches_its_own_team(
    detector: BlipDetector, team: Team
) -> None:
    """Regression: this is what killed hole-based detection.

    Champion portrait art often contains the team hue, which fills the colour
    mask solid and leaves no enclosed region. On real frames two of seven
    markers had a flawless ring and were still missed for this reason. Detecting
    the circular edge instead of the hole is what fixes it, so this test is the
    load-bearing one for the whole approach.
    """
    image = _blank()
    draw_champion(image, 140, 140, team, core_bgr=TEAM_BGR[team])

    found = _match(detector.detect(image), 140, 140)
    assert found is not None, "a solid team-coloured marker must still be found"
    assert found.team is team


@pytest.mark.parametrize("team", [Team.BLUE, Team.RED])
def test_finds_champion_standing_inside_base_shading(
    detector: BlipDetector, team: Team
) -> None:
    """Regression: champions sit in their own base constantly.

    The ring merges into the surrounding team-coloured shading, so there is no
    outer boundary to trace. The circular edge survives regardless.
    """
    image = _blank()
    draw_base_shading(image, 140, 140, team, radius=55)
    draw_champion(image, 140, 140, team)

    assert _match(detector.detect(image), 140, 140) is not None


# -- rejecting things that are team-coloured but are not champions ----------


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


def test_rejects_bare_base_shading(detector: BlipDetector) -> None:
    image = _blank()
    draw_base_shading(image, 70, 70, Team.BLUE, radius=50)
    draw_base_shading(image, 210, 210, Team.RED, radius=50)
    assert detector.detect(image) == []


def test_rejects_empty_terrain(detector: BlipDetector) -> None:
    assert detector.detect(_blank()) == []


# -- structural behaviour ---------------------------------------------------


def test_never_returns_more_than_ten() -> None:
    extra = DEFAULT_MARKERS + (
        Marker(90, 40, Team.BLUE),
        Marker(255, 175, Team.RED),
        Marker(20, 110, Team.BLUE),
    )
    image, _ = synthetic_minimap(size=CANVAS, markers=extra)
    assert len(make_detector().detect(image)) <= 10


def test_suppresses_overlapping_detections(detector: BlipDetector) -> None:
    image = _blank()
    draw_champion(image, 140, 140, Team.BLUE)
    draw_champion(image, 144, 141, Team.RED)
    assert len(detector.detect(image)) == 1


def test_rejects_non_bgr_input(detector: BlipDetector) -> None:
    with pytest.raises(ValueError):
        detector.detect(np.zeros((64, 64), np.uint8))


def test_debug_exposes_masks_and_candidates(detector: BlipDetector) -> None:
    image, _ = synthetic_minimap(size=CANVAS)
    blips, debug = detector.detect_with_debug(image)

    assert set(debug.masks) == {Team.BLUE, Team.RED}
    for mask in debug.masks.values():
        assert mask.shape == image.shape[:2]
    # Hough is tuned to over-propose; the colour test is what filters.
    assert debug.candidates >= len(blips)


# -- scaling ----------------------------------------------------------------


def test_scaled_config_tracks_minimap_size() -> None:
    base = BlipDetectorConfig()
    doubled = scaled_config(
        base, minimap_width=REFERENCE_MINIMAP_WIDTH * 2,
        reference_width=REFERENCE_MINIMAP_WIDTH,
    )
    assert doubled.min_radius == base.min_radius * 2
    assert doubled.max_radius == base.max_radius * 2
    assert doubled.hough_min_dist == base.hough_min_dist * 2
    assert doubled.blue_bands == base.blue_bands, "colour is scale-invariant"


def test_detects_at_a_larger_minimap_scale() -> None:
    size = CANVAS * 2
    image, _ = synthetic_minimap(size=size, markers=(), with_distractors=False)
    markers = tuple(Marker(m.x * 2, m.y * 2, m.team) for m in DEFAULT_MARKERS)
    for marker in markers:
        draw_champion(image, marker.x, marker.y, marker.team, radius=26)

    blips = make_detector(width=size).detect(image)
    for marker in markers:
        assert _match(blips, marker.x, marker.y, tolerance=6.0) is not None
