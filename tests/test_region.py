"""The minimap crop rectangle, and the sanity check on a fresh one.

`select` is not exercised -- it is a drag in a GUI window. What is worth pinning
is the check that runs on whatever it returns, because the shape it accepts has
to cover the real receiver geometry and still catch a misdrag. Those two pull in
opposite directions: the panel is square in the game but arrives stretched, so a
strict test would reject every genuine calibration this project has.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_sight.perception.minimap import MinimapRegion


def test_the_calibrated_receiver_panel_is_accepted() -> None:
    """325x322 is the real 2118x1354 calibration, stretch and all."""
    assert MinimapRegion(1787, 1020, 325, 322).looks_square


def test_an_exactly_square_panel_is_accepted() -> None:
    assert MinimapRegion(0, 0, 300, 300).looks_square


def test_a_misdrag_is_caught() -> None:
    assert not MinimapRegion(0, 0, 400, 100).looks_square


def test_the_tolerance_is_relative_not_absolute() -> None:
    """A 10px error is nothing on a big panel and a lot on a small one."""
    assert MinimapRegion(0, 0, 400, 390).looks_square
    assert not MinimapRegion(0, 0, 60, 50).looks_square


def test_a_region_must_have_extent() -> None:
    with pytest.raises(ValueError):
        MinimapRegion(0, 0, 0, 10)


def test_crop_refuses_to_run_off_the_frame() -> None:
    """A region calibrated for a bigger frame is a bug, not a smaller crop."""
    region = MinimapRegion(1787, 1020, 325, 322)
    with pytest.raises(ValueError, match="does not fit"):
        region.crop(np.zeros((600, 800, 3), np.uint8))
