"""Reading the health bars drawn over minions.

Synthetic scenes, for the reason the champion plate tests use them: the colours
are exact by construction, so a failure here is a bug in the frame, tail or
overlap logic rather than an HSV band a few degrees out. Whether the bands and
the contrast floors suit League's own rendering is a question about footage,
which `tools/detect_minions.py --overlay` answers by eye.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.nameplates import NameplateLayout
from spectral_sight.perception.nameplates.minions import Minion, MinionReader
from spectral_sight.types import Team
from tests.synthetic import (
    PLATE_ALLY_BGR,
    PLATE_BOX_BGR,
    draw_minion_bar,
    draw_nameplate,
    plate_scene,
)

MINION_WIDTH = 66
MINION_HEIGHT = 4

LAYOUT = NameplateLayout(
    bar_width=100,
    bar_height=10,
    resource_dy=(11, 15),
    level_dx=(-26, -3),
    level_dy=(-18, 6),
    minion_width=MINION_WIDTH,
    minion_height=MINION_HEIGHT,
)

TOLERANCE = 2.0 / MINION_WIDTH


def reader() -> MinionReader:
    return MinionReader(LAYOUT)


def bar(canvas: np.ndarray, x: int, y: int, **kwargs) -> None:
    draw_minion_bar(
        canvas, x, y, width=MINION_WIDTH, height=MINION_HEIGHT, **kwargs
    )


def only(minions: list[Minion]) -> Minion:
    assert len(minions) == 1, minions
    return minions[0]


# -- reading --------------------------------------------------------------


@pytest.mark.parametrize("health", [1.0, 0.6, 0.25])
@pytest.mark.parametrize("hostile", [True, False])
def test_reads_team_and_health(health: float, hostile: bool) -> None:
    canvas = plate_scene()
    bar(canvas, 200, 150, hostile=hostile, health=health)
    minion = only(reader().read(canvas))
    assert minion.team is (Team.RED if hostile else Team.BLUE)
    assert (minion.x, minion.y) == (200, 150)
    assert minion.health == pytest.approx(health, abs=TOLERANCE)
    assert not minion.occluded and not minion.clipped


def test_finds_a_nearly_dead_minion() -> None:
    """Three pixels of colour is nothing on its own; three pixels followed by
    exactly the rest of a bar's empty frame is a minion about to die -- the
    reading last-hitting is made of."""
    canvas = plate_scene()
    bar(canvas, 200, 150, health=0.05)
    minion = only(reader().read(canvas))
    assert minion.health == pytest.approx(0.05, abs=TOLERANCE)


def test_reads_a_wave() -> None:
    canvas = plate_scene()
    spots = [(60, 60), (160, 90), (260, 60), (360, 200), (460, 300)]
    for index, (x, y) in enumerate(spots):
        bar(canvas, x, y, hostile=index % 2 == 0, health=0.2 + 0.15 * index)
    found = sorted(reader().read(canvas), key=lambda m: m.x)
    assert [(m.x, m.y) for m in found] == spots
    assert [m.team for m in found] == [
        Team.RED, Team.BLUE, Team.RED, Team.BLUE, Team.RED
    ]


# -- rejecting what is not a minion ---------------------------------------


def test_ignores_a_champion_nameplate() -> None:
    """A champion's resource bar is the same blue and the same height as an
    ally minion's bar. Its empty tail running on past a minion's width is what
    gives it away."""
    canvas = plate_scene()
    draw_nameplate(canvas, 200, 150, health=0.7, resource=0.4, level=5)
    draw_nameplate(canvas, 200, 280, health=0.9, resource=0.3, ally=True)
    assert reader().read(canvas) == []


def test_ignores_a_longer_bar() -> None:
    """A turret's bar: a minion-like fill in a frame that keeps going."""
    canvas = plate_scene()
    draw_minion_bar(canvas, 200, 150, width=180, health=0.2, height=4)
    assert reader().read(canvas) == []


def test_ignores_a_speck_on_dark_ground() -> None:
    """Dark ground beside a speck of team colour looks like an empty tail to a
    brightness floor. It is not darker than its surroundings, and it has no
    end, which is what a real frame has."""
    canvas = plate_scene()
    canvas[100:200, 100:400] = (30, 32, 28)
    cv2.rectangle(canvas, (150, 150), (152, 153), PLATE_ALLY_BGR, -1)
    assert reader().read(canvas) == []


def test_ignores_text_in_team_colours() -> None:
    """Chat names are drawn in team colours, a stroke at a time."""
    canvas = plate_scene()
    cv2.rectangle(canvas, (140, 140), (400, 170), PLATE_BOX_BGR, -1)
    cv2.putText(canvas, "PieLoveYou mentalyy", (145, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_ALLY_BGR, 1)
    assert reader().read(canvas) == []


# -- bars that cannot be read ---------------------------------------------


def test_a_covered_bar_keeps_its_place_but_not_its_health() -> None:
    """The bar in front hides the one behind from where it starts. The one
    behind would otherwise report its fill up to that column -- a confident,
    wrong, low health."""
    canvas = plate_scene()
    bar(canvas, 200, 150, health=0.3)
    bar(canvas, 240, 150, hostile=False, health=1.0)
    found = {m.x: m for m in reader().read(canvas)}
    assert set(found) == {200, 240}
    assert found[200].occluded and found[200].health is None
    assert found[240].health == pytest.approx(1.0, abs=TOLERANCE)


def test_a_bar_off_the_frame_is_clipped() -> None:
    canvas = plate_scene()
    width = canvas.shape[1]
    bar(canvas, width - 40, 150, health=0.9)
    minion = only(reader().read(canvas))
    assert minion.clipped and minion.health is None


# -- layout ---------------------------------------------------------------


def test_needs_minion_geometry() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        MinionReader(replace(LAYOUT, minion_width=None))


def test_layout_round_trips_minion_geometry() -> None:
    data = LAYOUT.to_dict()
    assert NameplateLayout.from_dict(data) == LAYOUT
    del data["minion_width"], data["minion_height"]
    older = NameplateLayout.from_dict(data)
    assert older.minion_width is None and older.minion_height is None
