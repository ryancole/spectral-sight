"""The playfield the bar readers mask over.

The property that matters is that cropping changes no reading: every pixel a
reader may look at from a bar start outside the excluded regions is inside the
field, and the field is otherwise as small as that allows. Real footage is
checked by `tools/compare_crop.py`; these pin the geometry.
"""

from __future__ import annotations

import cv2
import numpy as np

from spectral_sight.perception.nameplates.playfield import Playfield

W, H = 200, 100


def covered(field: Playfield) -> np.ndarray:
    mask = np.zeros((field.height, field.width), dtype=np.uint8)
    for x0, y0, x1, y1 in field.rects:
        mask[y0:y1, x0:x1] += 1
    return mask


def test_no_exclusions_is_the_whole_frame() -> None:
    field = Playfield.of((), W, H, left=5, right=5, above=5, below=5)
    assert field.rects == ((0, 0, W, H),)
    assert field.rows == (0, H)
    assert field.coverage == 1.0


def test_rectangles_are_disjoint() -> None:
    field = Playfield.of(
        ((0.0, 0.0, 0.3, 0.4), (0.0, 0.8, 1.0, 1.0), (0.7, 0.5, 1.0, 1.0)),
        W, H, left=6, right=20, above=8, below=3,
    )
    assert covered(field).max() == 1


def test_an_excluded_region_is_skipped_beyond_the_reach() -> None:
    # The bottom fifth is HUD: bars start above it, and look 3px down.
    field = Playfield.of(((0.0, 0.8, 1.0, 1.0),), W, H,
                         left=0, right=0, above=0, below=3)
    mask = covered(field)
    # Row 80 is excluded (inclusive, as the readers test it); 79 is the last
    # row a bar can start on, and 82 the last it can look at.
    assert mask[:83].all()
    assert not mask[83:].any()
    assert field.rows == (0, 83)


def test_reach_extends_into_a_panel_only_from_where_bars_start() -> None:
    # A panel in the top-left corner: a bar starting just right of it looks
    # left into it, one starting just below looks up into it.
    field = Playfield.of(((0.0, 0.0, 0.5, 0.5),), W, H,
                         left=10, right=0, above=4, below=0)
    mask = covered(field)
    assert mask[0, 101 - 10]          # left of the first free column
    assert not mask[0, 101 - 11]
    assert mask[51 - 4, 0]            # above the first free row
    assert not mask[51 - 5, 0]


def test_everything_excluded_leaves_nothing() -> None:
    field = Playfield.of(((0.0, 0.0, 1.0, 1.0),), W, H,
                         left=5, right=5, above=5, below=5)
    assert field.rects == ()
    assert field.coverage == 0.0


def test_in_range_matches_whole_frame_inside_and_is_zero_outside() -> None:
    rng = np.random.default_rng(0)
    hsv = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    field = Playfield.of(((0.0, 0.0, 0.5, 1.0),), W, H,
                         left=0, right=0, above=0, below=0)
    ranges = (((0, 50, 50), (40, 255, 255)), ((150, 50, 50), (179, 255, 255)))
    mask = field.in_range(hsv, *ranges)
    whole = cv2.inRange(hsv, *ranges[0]) | cv2.inRange(hsv, *ranges[1])
    inside = covered(field).astype(bool)
    assert (mask[inside] == whole[inside]).all()
    assert not mask[~inside].any()
