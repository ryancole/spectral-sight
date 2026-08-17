"""Camera viewport detection on the minimap."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.minimap import find_viewport
from spectral_sight.types import Team
from tests.synthetic import draw_champion, synthetic_minimap

WHITE = (235, 235, 235)


def _blank(size: int = 280) -> np.ndarray:
    image, _ = synthetic_minimap(size=size, markers=(), with_distractors=False)
    return image


def test_finds_the_rectangle() -> None:
    image = _blank()
    cv2.rectangle(image, (90, 110), (170, 160), WHITE, 1)

    viewport = find_viewport(image)
    assert viewport is not None
    cx, cy = viewport.center
    assert abs(cx - 130) <= 2
    assert abs(cy - 135) <= 2


def test_returns_none_when_absent() -> None:
    assert find_viewport(_blank()) is None


def test_ignores_small_white_marks() -> None:
    """Ward pips and map text pass the same colour threshold."""
    image = _blank()
    for x in range(40, 240, 25):
        cv2.circle(image, (x, 60), 3, WHITE, -1)
    assert find_viewport(image) is None


def test_prefers_the_largest_candidate() -> None:
    image = _blank()
    cv2.rectangle(image, (30, 30), (70, 55), WHITE, 1)
    cv2.rectangle(image, (120, 140), (230, 210), WHITE, 1)

    viewport = find_viewport(image)
    assert viewport is not None
    assert viewport.width > 100


def test_rejects_wrong_aspect_ratio() -> None:
    """A tall rectangle is not a viewport; the display is wider than it is tall."""
    image = _blank()
    cv2.rectangle(image, (100, 60), (140, 200), WHITE, 1)
    assert find_viewport(image) is None


def test_survives_a_champion_marker_inside_it() -> None:
    """The player is drawn at the centre of their own viewport."""
    image = _blank()
    cv2.rectangle(image, (90, 110), (170, 160), WHITE, 1)
    draw_champion(image, 130, 135, Team.BLUE)

    viewport = find_viewport(image)
    assert viewport is not None
    cx, cy = viewport.center
    assert np.hypot(cx - 130, cy - 135) <= 4


def test_rejects_non_bgr_input() -> None:
    with pytest.raises(ValueError):
        find_viewport(np.zeros((64, 64), np.uint8))
