"""Matching a nameplate to the track it belongs to.

The projection's *accuracy* is a question about footage and is reported by
`tools/calibrate_nameplates.py --fit`. What is pinned here is that the fit
recovers coefficients it is given, that the gate is enforced, and that the
assignment is one-to-one -- the last of which is what stops one champion's plate
being handed to another champion's track, which corrupts a whole series rather
than one frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spectral_sight.perception.minimap.viewport import Viewport
from spectral_sight.perception.nameplates import (
    Side,
    Nameplate,
    NameplateLayout,
    ScreenProjection,
    associate,
    fit,
)

FRAME = (2000, 1000)
VIEWPORT = Viewport(x=100, y=50, width=80, height=50)

IDENTITY = ScreenProjection(x=(1.0, 0.0, 0.0), y=(0.0, 1.0, 0.0))
"""Screen position maps straight onto the viewport rectangle. Not what real
footage fits -- the camera is tilted -- but it makes the expected minimap
position calculable by hand."""


@dataclass
class FakeTrack:
    """Just enough of a Track: an id, and a distance to a point."""

    id: int
    x: float
    y: float

    def distance_to(self, x: float, y: float) -> float:
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5


def plate(x: int, y: int, width: int = 100) -> Nameplate:
    return Nameplate(x=x, y=y, width=width, health=1.0, resource=1.0, side=Side.HOSTILE)


# -- projection -----------------------------------------------------------


def test_maps_screen_centre_to_viewport_centre() -> None:
    """The plate's centre is its bar centre, not its left edge."""
    at = IDENTITY.to_minimap(plate(950, 500), VIEWPORT, FRAME)
    assert at == pytest.approx((140.0, 75.0))


def test_maps_the_top_left_corner() -> None:
    assert IDENTITY.to_minimap(plate(-50, 0), VIEWPORT, FRAME) == pytest.approx(
        (100.0, 50.0)
    )


def test_from_layout_returns_none_without_a_fit() -> None:
    layout = NameplateLayout(
        bar_width=117, bar_height=11, resource_dy=(13, 19),
        level_dx=(-28, -3), level_dy=(-19, 4),
    )
    assert ScreenProjection.from_layout(layout) is None


def test_from_layout_reads_a_fitted_projection() -> None:
    layout = NameplateLayout(
        bar_width=117, bar_height=11, resource_dy=(13, 19),
        level_dx=(-28, -3), level_dy=(-19, 4),
        projection_x=(0.5, 0.0, 0.25), projection_y=(0.25, 0.6, 0.15),
    )
    projection = ScreenProjection.from_layout(layout)
    assert projection is not None
    assert projection.x == (0.5, 0.0, 0.25)


# -- fitting --------------------------------------------------------------


def test_fit_recovers_known_coefficients() -> None:
    coef_x = (0.4, 0.1, 0.2)
    coef_y = (0.2, 0.6, 0.05)
    samples = []
    for i in range(20):
        u, v = (i % 5) / 4.0, (i // 5) / 4.0
        samples.append((
            u, v,
            coef_x[0] * u + coef_x[1] * v + coef_x[2],
            coef_y[0] * u + coef_y[1] * v + coef_y[2],
        ))
    got_x, got_y = fit(samples)
    assert got_x == pytest.approx(coef_x, abs=1e-6)
    assert got_y == pytest.approx(coef_y, abs=1e-6)


def test_fit_refuses_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        fit([(0.1, 0.2, 0.3, 0.4)])


# -- association ----------------------------------------------------------


def test_pairs_a_plate_with_the_nearest_track() -> None:
    tracks = [FakeTrack(7, 140.0, 75.0), FakeTrack(8, 105.0, 55.0)]
    pairing = associate([plate(950, 500)], tracks, VIEWPORT, IDENTITY, FRAME)
    assert pairing == {0: 7}


def test_leaves_a_plate_unassigned_beyond_the_gate() -> None:
    """An unmatched plate costs one frame; a wrongly matched one corrupts a
    champion's whole series."""
    tracks = [FakeTrack(7, 400.0, 400.0)]
    assert associate([plate(950, 500)], tracks, VIEWPORT, IDENTITY, FRAME) == {}


def test_one_track_serves_one_plate() -> None:
    """Two champions cannot be in the same place, so the second plate has to
    take its own track or none."""
    plates = [plate(950, 500), plate(960, 505)]
    tracks = [FakeTrack(7, 140.0, 75.0)]
    pairing = associate(plates, tracks, VIEWPORT, IDENTITY, FRAME)
    assert len(pairing) == 1
    assert set(pairing.values()) == {7}


def test_assigns_each_plate_to_its_own_track() -> None:
    plates = [plate(150, 100), plate(1550, 800)]
    tracks = [FakeTrack(7, 108.0, 55.0), FakeTrack(8, 172.0, 90.0)]
    pairing = associate(plates, tracks, VIEWPORT, IDENTITY, FRAME)
    assert pairing == {0: 7, 1: 8}


def test_no_viewport_means_no_association() -> None:
    tracks = [FakeTrack(7, 140.0, 75.0)]
    assert associate([plate(950, 500)], tracks, None, IDENTITY, FRAME) == {}


def test_no_projection_means_no_association() -> None:
    """An uncalibrated projection must not silently fall back to guessing."""
    tracks = [FakeTrack(7, 140.0, 75.0)]
    assert associate([plate(950, 500)], tracks, VIEWPORT, None, FRAME) == {}


def test_empty_inputs_are_not_an_error() -> None:
    assert associate([], [], VIEWPORT, IDENTITY, FRAME) == {}
