"""Minimap pixels to world units.

The transform is four numbers and a flip, which is exactly the kind of thing
that works on the clip you tested and is silently mirrored everywhere else. The
corners are pinned here so an axis inversion cannot pass.
"""

from __future__ import annotations

import pytest

from spectral_sight.perception.minimap import (
    SUMMONERS_RIFT,
    MinimapRegion,
    WorldBounds,
    WorldTransform,
)

# A deliberately un-round map area, so an off-by-one in the offset shows up
# rather than cancelling against a tidy number.
AREA = WorldTransform(x=1791.0, y=1027.0, width=310.0, height=310.0)


def test_bottom_left_pixel_is_the_low_corner() -> None:
    """Blue base sits at low x and low y, and is drawn bottom-left."""
    x, y = AREA.to_world(1791.0, 1337.0)
    assert x == pytest.approx(SUMMONERS_RIFT.min_x)
    assert y == pytest.approx(SUMMONERS_RIFT.min_y)


def test_top_right_pixel_is_the_high_corner() -> None:
    x, y = AREA.to_world(2101.0, 1027.0)
    assert x == pytest.approx(SUMMONERS_RIFT.max_x)
    assert y == pytest.approx(SUMMONERS_RIFT.max_y)


def test_y_is_flipped_and_x_is_not() -> None:
    """The failure this guards against looks fine on a symmetric test case."""
    low_x, low_y = AREA.to_world(1800.0, 1300.0)
    high_x, high_y = AREA.to_world(2000.0, 1100.0)
    assert high_x > low_x
    assert high_y > low_y


def test_centre_maps_to_the_middle_of_the_map() -> None:
    x, y = AREA.to_world(1791.0 + 155.0, 1027.0 + 155.0)
    assert x == pytest.approx(SUMMONERS_RIFT.span_x / 2)
    assert y == pytest.approx(SUMMONERS_RIFT.span_y / 2)


def test_round_trips_through_world_space() -> None:
    for pixel in ((1800.0, 1100.0), (2050.0, 1300.0), (1900.5, 1222.25)):
        assert AREA.to_frame(*AREA.to_world(*pixel)) == pytest.approx(pixel)


def test_from_minimap_applies_the_crop_offset() -> None:
    """Blip coordinates are crop-relative; the transform is frame-anchored."""
    region = MinimapRegion(x=1787, y=1020, width=325, height=322)
    assert AREA.from_minimap(region, 4.0, 7.0) == pytest.approx(
        AREA.to_world(1791.0, 1027.0)
    )


def test_scale_is_reported_per_axis() -> None:
    ux, uy = AREA.units_per_pixel
    assert ux == pytest.approx(SUMMONERS_RIFT.span_x / 310.0)
    assert uy == pytest.approx(SUMMONERS_RIFT.span_y / 310.0)
    # Summoner's Rift is not quite square in units, so a square pixel area is
    # expected to land near 1.0 rather than on it.
    assert AREA.squareness == pytest.approx(14870.0 / 14980.0)


def test_squareness_catches_a_stretched_area() -> None:
    stretched = WorldTransform(x=0.0, y=0.0, width=400.0, height=200.0)
    assert stretched.squareness == pytest.approx(0.4964, abs=1e-3)


def test_assuming_crop_uses_the_whole_panel() -> None:
    region = MinimapRegion(x=1787, y=1020, width=325, height=322)
    naive = WorldTransform.assuming_crop(region)
    assert (naive.x, naive.y, naive.width, naive.height) == (1787.0, 1020.0, 325.0, 322.0)
    # Coarser than the calibrated area, which is the whole point of measuring.
    assert naive.units_per_pixel[0] < AREA.units_per_pixel[0]


def test_survives_a_save_and_load(tmp_path) -> None:
    path = tmp_path / "world.json"
    AREA.save(path)
    assert WorldTransform.load(path) == AREA


def test_custom_bounds_are_carried_through(tmp_path) -> None:
    bounds = WorldBounds(min_x=-120.0, min_y=-120.0, max_x=14870.0, max_y=14980.0)
    transform = WorldTransform(x=0.0, y=0.0, width=100.0, height=100.0, bounds=bounds)
    path = tmp_path / "world.json"
    transform.save(path)
    assert WorldTransform.load(path).bounds == bounds
    assert transform.to_world(0.0, 100.0)[0] == pytest.approx(-120.0)


def test_rejects_a_degenerate_area() -> None:
    with pytest.raises(ValueError):
        WorldTransform(x=0.0, y=0.0, width=0.0, height=10.0)


def test_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        WorldBounds(min_x=100.0, min_y=0.0, max_x=0.0, max_y=100.0)
