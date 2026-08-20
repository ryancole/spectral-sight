"""Finding the minimap panel by recognising the map drawn on it.

These use a synthetic reference pasted into a synthetic frame rather than the
one in `etc/map/`. That is deliberate: what is being pinned here is the search
-- that it recovers position, scale and aspect, and that it declines when there
is nothing to find -- and none of that should be able to pass or fail because
of which season's map art happens to be checked in.

Whether the real reference matches real footage is a measurement, not a unit
test, and it lives in the module docstring: 80 frames over four clips, every
corner within a pixel, correlation 0.825 at worst.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.minimap.locate import (
    MIN_SCORE,
    PanelMatch,
    locate_panel,
)
from spectral_sight.perception.minimap.region import MinimapRegion


TOLERANCE = 8
"""Pixels of slack allowed against the pasted-in truth.

Looser than the one pixel measured on real footage, and the reason is the
texture rather than the search. Sixteen blocks of noise correlate almost as well
a few pixels off as dead on, so the peak is flat; the real map has lanes, walls
and structures that pin it. Tightening this would be tuning the test's fixture
rather than the thing under test -- accuracy is measured on footage, not here.
"""


def reference(size: int = 256) -> np.ndarray:
    """Blobby, high-contrast texture -- a stand-in for the map's layout.

    Blurred noise rather than raw noise: raw noise has no structure to survive
    being scaled down, which is exactly what the locator does to it.
    """
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    grown = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(grown, (5, 5), 0)


def frame_with(panel: MinimapRegion, ref: np.ndarray,
               size: tuple[int, int] = (800, 600)) -> np.ndarray:
    """A dim gradient background with the reference pasted in at `panel`."""
    width, height = size
    ramp = np.linspace(10, 70, width, dtype=np.uint8)
    image = np.repeat(ramp[None, :], height, axis=0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image[panel.y:panel.y + panel.height, panel.x:panel.x + panel.width] = (
        cv2.resize(ref, (panel.width, panel.height))
    )
    return image


@pytest.mark.parametrize(
    "panel",
    [
        MinimapRegion(560, 380, 180, 180),   # square
        MinimapRegion(600, 420, 160, 160),   # smaller, further into the corner
        MinimapRegion(520, 340, 220, 220),   # larger
    ],
)
def test_finds_a_square_panel(panel: MinimapRegion) -> None:
    ref = reference()
    found = locate_panel(frame_with(panel, ref), ref)
    assert found is not None and found.confident
    assert abs(found.region.x - panel.x) <= TOLERANCE
    assert abs(found.region.y - panel.y) <= TOLERANCE
    assert abs(found.region.width - panel.width) <= TOLERANCE


@pytest.mark.parametrize(
    "panel",
    [
        MinimapRegion(560, 380, 200, 160),   # stretched wide
        MinimapRegion(580, 340, 150, 200),   # stretched tall
    ],
)
def test_finds_a_stretched_panel(panel: MinimapRegion) -> None:
    """The receiver does not preserve aspect, so the panel is rarely square."""
    ref = reference()
    found = locate_panel(frame_with(panel, ref), ref)
    assert found is not None and found.confident
    assert abs(found.region.width - panel.width) <= TOLERANCE
    assert abs(found.region.height - panel.height) <= TOLERANCE
    # The point of this case: a square answer would miss by far more than that.
    assert abs(found.region.width - found.region.height) > TOLERANCE


def test_declines_when_there_is_no_panel() -> None:
    """The whole basis for trusting this: it can say it did not find one."""
    ref = reference()
    ramp = np.linspace(10, 70, 800, dtype=np.uint8)
    empty = cv2.cvtColor(np.repeat(ramp[None, :], 600, axis=0), cv2.COLOR_GRAY2BGR)
    found = locate_panel(empty, ref)
    assert found is None or not found.confident


def test_declines_on_noise() -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    found = locate_panel(noise, reference())
    assert found is None or not found.confident


def test_confidence_is_the_documented_threshold() -> None:
    region = MinimapRegion(0, 0, 10, 10)
    assert PanelMatch(region, MIN_SCORE).confident
    assert not PanelMatch(region, MIN_SCORE - 0.01).confident


def test_a_panel_outside_the_searched_corner_is_not_reported() -> None:
    """Only the bottom-right is searched; a hit elsewhere would be a false one."""
    ref = reference()
    top_left = frame_with(MinimapRegion(20, 20, 180, 180), ref)
    found = locate_panel(top_left, ref)
    assert found is None or not found.confident
