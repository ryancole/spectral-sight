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


# -- the fixed-size fit ----------------------------------------------------

from spectral_sight.perception.minimap import (  # noqa: E402
    ViewportConfig,
    scaled_viewport_config,
)

BOX = ViewportConfig(box_width=80, box_height=50)


def test_fit_leaves_a_whole_box_alone() -> None:
    image = _blank()
    cv2.rectangle(image, (90, 110), (170, 160), WHITE, 1)
    viewport = find_viewport(image, BOX)
    assert viewport is not None
    assert (viewport.x, viewport.y, viewport.width, viewport.height) == (90, 110, 81, 51)


def test_a_box_clipped_at_the_map_edge_is_placed_where_the_camera_is() -> None:
    """Only the part on the map is drawn. Its bounding box is the visible part,
    and the centre of that is not the camera."""
    image = _blank()
    # A box whose left 30px are off the map: top, bottom and right edges only.
    x0, y0 = -30, 100
    cv2.line(image, (0, y0), (x0 + 80, y0), WHITE, 1)
    cv2.line(image, (0, y0 + 50), (x0 + 80, y0 + 50), WHITE, 1)
    cv2.line(image, (x0 + 80, y0), (x0 + 80, y0 + 50), WHITE, 1)

    viewport = find_viewport(image, BOX)
    assert viewport is not None
    cx, cy = viewport.center
    assert abs(cx - (x0 + 40)) <= 2
    assert abs(cy - (y0 + 25)) <= 2


def test_a_box_split_by_an_icon_is_still_found() -> None:
    """An icon drawn across the outline breaks it into pieces too thin to pass
    as a rectangle on their own. Drawn two pixels thick, as the antialiased
    outline is on footage."""
    image = _blank()
    cv2.rectangle(image, (90, 110), (170, 160), WHITE, 2)
    for x, y in ((130, 110), (170, 135), (90, 135), (130, 160)):
        cv2.circle(image, (x, y), 9, (40, 60, 50), -1)
    assert find_viewport(image) is None
    viewport = find_viewport(image, BOX)
    assert viewport is not None
    cx, cy = viewport.center
    assert np.hypot(cx - 130, cy - 135) <= 2


def test_fit_declines_when_nothing_is_drawn() -> None:
    assert find_viewport(_blank(), BOX) is None


def test_bounds_limit_where_the_outline_can_be_drawn() -> None:
    """Outside the map square nothing is drawn, so a box running off it is
    judged only on the part that could have been lit."""
    image = _blank()
    cv2.line(image, (20, 100), (70, 100), WHITE, 1)
    cv2.line(image, (20, 150), (70, 150), WHITE, 1)
    cv2.line(image, (70, 100), (70, 150), WHITE, 1)
    viewport = find_viewport(image, BOX, bounds=(20, 0, 260, 280))
    assert viewport is not None
    cx, _ = viewport.center
    assert abs(cx - 30) <= 2


def test_box_size_scales_with_the_panel() -> None:
    small = scaled_viewport_config(325)
    large = scaled_viewport_config(486)
    assert small.box_width == pytest.approx(78.0)
    assert large.box_width == pytest.approx(78.0 * 486 / 325)
    assert large.box_height / large.box_width == pytest.approx(48.0 / 78.0)
