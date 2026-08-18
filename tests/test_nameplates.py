"""Reading the bars drawn over champions.

Synthetic scenes rather than crops of the game, for the reason the minimap tests
use them: the colours here are exact by construction, so a failure is a bug in
the pairing, fill or occlusion logic rather than an HSV band being a few degrees
out. Whether the bands suit League's own rendering is a question about real
footage, which `tools/calibrate_nameplates.py --validate` answers over thousands
of frames.

The one thing deliberately tested against numbers taken from real footage is the
fill *denominator*, since a plate reader with a wrong `bar_width` returns
plausible fractions on every frame and never looks broken.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.clock import (
    GlyphSet,
    glyph_boxes,
    segment_glyphs,
)
from spectral_sight.perception.nameplates import (
    Nameplate,
    NameplateLayout,
    NameplateReader,
)
from tests.synthetic import (
    draw_minion_bar,
    draw_nameplate,
    plate_scene,
)

BAR_WIDTH = 100
BAR_HEIGHT = 10
RESOURCE_DY = 13
RESOURCE_HEIGHT = 4

LAYOUT = NameplateLayout(
    bar_width=BAR_WIDTH,
    bar_height=BAR_HEIGHT,
    resource_dy=(RESOURCE_DY - 2, RESOURCE_DY + 2),
    level_dx=(-26, -3),
    level_dy=(-(RESOURCE_DY + 5), RESOURCE_HEIGHT + 2),
    exclude=(),
)

TOLERANCE = 3.0 / BAR_WIDTH
"""Three pixels. Antialiasing and the tick marks move an edge by a pixel or two,
and the measurement is only ever claimed to be pixel-exact, not exact."""


@pytest.fixture(scope="module")
def glyphs() -> GlyphSet:
    """A digit set built the way calibration builds one, from rendered samples."""
    strips = {}
    for digit in "0123456789":
        strip = np.full((26, 20, 3), (24, 28, 26), np.uint8)
        cv2.putText(strip, digit, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (238, 238, 238), 2, cv2.LINE_AA)
        strips[digit] = strip
    width = height = 0
    for strip in strips.values():
        for _, _, w, h in glyph_boxes(strip):
            width, height = max(width, w), max(height, h)
    size = (width + 2, height + 2)
    return GlyphSet(
        glyphs={d: segment_glyphs(s, size)[0] for d, s in strips.items()},
        size=size,
    )


def reader(glyphs: GlyphSet | None = None) -> NameplateReader:
    return NameplateReader(LAYOUT, glyphs)


def scene(**plate) -> np.ndarray:
    canvas = plate_scene()
    draw_nameplate(canvas, bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
                   resource_dy=RESOURCE_DY, resource_height=RESOURCE_HEIGHT,
                   **plate)
    return canvas


# -- finding plates -------------------------------------------------------


def test_finds_a_champion_plate() -> None:
    plates = reader().read(scene(x=200, y=120, health=0.8, resource=0.6))
    assert len(plates) == 1
    assert plates[0].hostile


def test_reports_the_health_bars_top_left() -> None:
    plate = reader().read(scene(x=200, y=120, health=0.8, resource=0.6))[0]
    assert abs(plate.x - 200) <= 1
    assert abs(plate.y - 120) <= 1


@pytest.mark.parametrize(
    "health, resource",
    [(1.0, 1.0), (0.8, 0.6), (0.35, 0.9), (0.5, 0.2)],
)
def test_reads_the_fill_fractions(health: float, resource: float) -> None:
    plate = reader().read(
        scene(x=180, y=100, health=health, resource=resource)
    )[0]
    assert plate.health == pytest.approx(health, abs=TOLERANCE)
    assert plate.resource == pytest.approx(resource, abs=TOLERANCE)


def test_a_champion_on_empty_resource_is_not_found() -> None:
    """The blind spot the resource anchor buys: with no blue left there is
    nothing to anchor on, and it is the champion who just spent everything."""
    assert reader().read(scene(x=180, y=100, health=0.8, resource=0.0)) == []


def test_hops_the_tick_marks() -> None:
    """Without gap tolerance this reports the health at the first tick."""
    plate = reader().read(scene(x=180, y=100, health=0.9, resource=0.9))[0]
    assert plate.health == pytest.approx(0.9, abs=TOLERANCE)


def test_tells_an_ally_from_an_enemy() -> None:
    plates = reader().read(
        scene(x=180, y=100, health=0.9, resource=0.9, hostile=False)
    )
    assert len(plates) == 1
    assert not plates[0].hostile


# -- rejecting things that are not champions ------------------------------


def test_ignores_a_minion_bar() -> None:
    """A health bar with no resource bar beneath it is not a champion."""
    canvas = plate_scene()
    for offset in range(5):
        draw_minion_bar(canvas, 120 + offset * 60, 90 + offset * 30)
    assert reader().read(canvas) == []


def test_ignores_a_bar_with_no_level_box() -> None:
    """The level box is what separates a real bar start from a split run."""
    canvas = scene(x=200, y=120, health=0.8, resource=0.6)
    # Paint over the level box with the same lit colour a bar has.
    cv2.rectangle(canvas, (200 - 26, 120 - 3), (200 - 4, 120 + 19),
                  (40, 35, 190), -1)
    assert reader().read(canvas) == []


def test_a_split_resource_bar_stays_one_plate() -> None:
    """A model or an effect cutting the bar must not spawn a second plate,
    measured from the wrong left edge."""
    canvas = scene(x=200, y=120, health=0.9, resource=0.9)
    top = 120 + RESOURCE_DY
    cv2.rectangle(canvas, (240, top), (248, top + RESOURCE_HEIGHT),
                  (60, 68, 58), -1)
    plates = reader().read(canvas)
    assert len(plates) == 1
    assert abs(plates[0].x - 200) <= 1


def test_blanks_a_plate_clipped_by_the_frame_edge() -> None:
    """A clipped bar truncates *both* fills at the same column, so they come
    back equal and plausible -- a wounded champion low on mana, from a
    champion who is simply standing at the edge of the screen."""
    canvas = plate_scene(width=640)
    draw_nameplate(canvas, x=610, y=120, health=1.0, resource=1.0,
                   bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
                   resource_dy=RESOURCE_DY, resource_height=RESOURCE_HEIGHT)
    plates = reader().read(canvas)
    assert len(plates) == 1
    assert plates[0].clipped
    assert plates[0].health is None and plates[0].resource is None


def test_blanks_a_plate_running_under_a_hud_region() -> None:
    layout = NameplateLayout(
        bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
        resource_dy=LAYOUT.resource_dy, level_dx=LAYOUT.level_dx,
        level_dy=LAYOUT.level_dy,
        exclude=((0.55, 0.0, 1.0, 1.0),),
    )
    canvas = scene(x=300, y=120, health=1.0, resource=1.0)
    plates = NameplateReader(layout).read(canvas)
    assert len(plates) == 1
    assert plates[0].clipped
    assert plates[0].health is None


def test_a_plate_clear_of_everything_is_not_clipped() -> None:
    plate = reader().read(scene(x=200, y=120, health=0.8, resource=0.6))[0]
    assert not plate.clipped
    assert plate.health is not None


def test_honours_the_exclusion_regions() -> None:
    excluded = NameplateLayout(
        bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
        resource_dy=LAYOUT.resource_dy, level_dx=LAYOUT.level_dx,
        level_dy=LAYOUT.level_dy,
        exclude=((0.0, 0.0, 1.0, 1.0),),
    )
    canvas = scene(x=200, y=120, health=0.8, resource=0.6)
    assert NameplateReader(excluded).read(canvas) == []


# -- occlusion ------------------------------------------------------------


def test_blanks_the_plate_behind_an_overlapping_one() -> None:
    """The rear bar is truncated at whatever pixel the front one starts, which
    reads as a sharp drop to a plausible fill."""
    canvas = plate_scene()
    common = dict(bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
                  resource_dy=RESOURCE_DY, resource_height=RESOURCE_HEIGHT)
    draw_nameplate(canvas, x=200, y=120, health=0.9, resource=0.9, **common)
    draw_nameplate(canvas, x=250, y=128, health=0.5, resource=0.5, **common)

    plates = sorted(reader().read(canvas), key=lambda p: p.x)
    assert len(plates) == 2
    behind, front = plates
    assert behind.occluded
    assert behind.health is None and behind.resource is None
    assert not front.occluded
    assert front.health is not None


def test_separate_plates_are_both_read() -> None:
    """Two champions far enough apart do not occlude each other."""
    canvas = plate_scene()
    common = dict(bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
                  resource_dy=RESOURCE_DY, resource_height=RESOURCE_HEIGHT)
    draw_nameplate(canvas, x=120, y=100, health=0.9, resource=0.8, **common)
    draw_nameplate(canvas, x=400, y=250, health=0.4, resource=0.3, **common)

    plates = reader().read(canvas)
    assert len(plates) == 2
    assert not any(p.occluded for p in plates)


# -- levels ---------------------------------------------------------------


def test_reads_a_single_digit_level(glyphs: GlyphSet) -> None:
    plate = reader(glyphs).read(
        scene(x=200, y=120, health=0.8, resource=0.6, level=4)
    )[0]
    assert plate.level == 4


def test_reads_a_two_digit_level(glyphs: GlyphSet) -> None:
    plate = reader(glyphs).read(
        scene(x=200, y=120, health=0.8, resource=0.6, level=12)
    )[0]
    assert plate.level == 12


def test_level_is_none_without_a_glyph_set() -> None:
    """Levels are opt-in: the reader works without a clock calibration."""
    plate = reader().read(
        scene(x=200, y=120, health=0.8, resource=0.6, level=4)
    )[0]
    assert plate.level is None


def test_level_survives_occlusion(glyphs: GlyphSet) -> None:
    """The box sits left of the bar, so a plate covering the bar leaves it."""
    canvas = plate_scene()
    common = dict(bar_width=BAR_WIDTH, bar_height=BAR_HEIGHT,
                  resource_dy=RESOURCE_DY, resource_height=RESOURCE_HEIGHT)
    draw_nameplate(canvas, x=200, y=120, health=0.9, resource=0.9, level=7,
                   **common)
    draw_nameplate(canvas, x=250, y=128, health=0.5, resource=0.5, level=3,
                   **common)

    behind = min(reader(glyphs).read(canvas), key=lambda p: p.x)
    assert behind.occluded
    assert behind.health is None
    assert behind.level == 7


# -- the layout -----------------------------------------------------------


def test_layout_survives_a_save_and_load(tmp_path) -> None:
    path = tmp_path / "layout.json"
    LAYOUT.save(path)
    assert NameplateLayout.load(path) == LAYOUT


def test_layout_roundtrips_a_projection(tmp_path) -> None:
    with_fit = NameplateLayout(
        bar_width=117, bar_height=11, resource_dy=(13, 19),
        level_dx=(-28, -3), level_dy=(-19, 4),
        projection_x=(0.486, 0.017, 0.295),
        projection_y=(0.264, 0.589, 0.152),
    )
    path = tmp_path / "layout.json"
    with_fit.save(path)
    assert NameplateLayout.load(path) == with_fit


def test_missing_calibration_names_the_tool() -> None:
    with pytest.raises(FileNotFoundError, match="calibrate_nameplates"):
        NameplateLayout.for_resolution(1, 1)


def test_the_shipped_calibration_loads() -> None:
    layout = NameplateLayout.for_resolution(2118, 1354)
    assert layout.bar_width == 117
    assert layout.projection_x is not None


def test_plate_centre_uses_its_own_width() -> None:
    plate = Nameplate(x=100, y=50, width=117, health=1.0, resource=1.0,
                      hostile=True)
    assert plate.center == (158.5, 50.0)
