"""Reading which turrets stand, from their minimap icons.

The icons are the captured templates themselves, pasted onto flat ground, so
these pin the decision logic -- shape and colour together, cover, the hold
before a loss, the lane turret that never comes back and the nexus turret
that does -- rather than how well the templates suit a new game. That is a
question for footage.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.minimap.turrets import (
    ICON_OFFSET,
    TURRETS,
    Reading,
    Tier,
    Turret,
    TurretConfig,
    TurretReader,
    load_templates,
)
from spectral_sight.types import Team

GROUND_BGR = (40, 60, 60)
"""Dark olive, roughly the lane terrain."""

SIZE = (484, 486)

TEMPLATES = load_templates()


def anchors() -> dict[Turret, tuple[float, float]]:
    """A grid, well apart, standing in for the projected world positions."""
    return {
        turret: (40.0 + 60.0 * (i % 7), 40.0 + 70.0 * (i // 7))
        for i, turret in enumerate(TURRETS)
    }


ANCHORS = anchors()


def centre(turret: Turret) -> tuple[int, int]:
    dx, dy = ICON_OFFSET["nexus" if turret.tier is Tier.NEXUS else "shield"]
    x, y = ANCHORS[turret]
    return round(x + dx), round(y + dy)


def scene(
    standing: set[Turret] | None = None, grey: set[Turret] = frozenset()
) -> np.ndarray:
    """Every turret in `standing` drawn in colour (all of them by default);
    those in `grey` drawn as the destroyed nexus turret's grey outline."""
    standing = set(TURRETS) if standing is None else standing
    canvas = np.full((*SIZE, 3), GROUND_BGR, np.uint8)
    for turret in TURRETS:
        if turret not in standing and turret not in grey:
            continue
        icon = TEMPLATES[(turret.team, turret.kind)]
        if turret in grey:
            icon = cv2.cvtColor(cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        h, w = icon.shape[:2]
        cx, cy = centre(turret)
        canvas[cy - h // 2 : cy - h // 2 + h, cx - w // 2 : cx - w // 2 + w] = icon
    return canvas


def reader(**config: object) -> TurretReader:
    return TurretReader(ANCHORS, config=TurretConfig(**config))


def find(team: Team, lane: str, tier: Tier) -> Turret:
    return next(
        t for t in TURRETS if t.team is team and t.lane == lane and t.tier is tier
    )


BLUE_MID_INNER = find(Team.BLUE, "mid", Tier.INNER)
RED_BOT_OUTER = find(Team.RED, "bot", Tier.OUTER)
BLUE_NEXUS_TOP = next(
    t for t in TURRETS if t.team is Team.BLUE and t.tier is Tier.NEXUS and t.side == "top"
)


def test_the_set_is_the_twenty_two() -> None:
    assert len(TURRETS) == 22
    for team in (Team.BLUE, Team.RED):
        mine = [t for t in TURRETS if t.team is team]
        assert len(mine) == 11
        assert {t.side for t in mine if t.tier is Tier.NEXUS} == {"top", "bot"}
        assert all(t.lane == "base" for t in mine if t.tier is Tier.NEXUS)


# -- one reading ----------------------------------------------------------


def test_every_drawn_icon_reads_standing() -> None:
    readings = reader().read_once(scene())
    assert set(readings.values()) == {Reading.STANDING}


def test_a_missing_icon_reads_gone() -> None:
    readings = reader().read_once(scene(set(TURRETS) - {BLUE_MID_INNER}))
    assert readings[BLUE_MID_INNER] is Reading.GONE
    assert sum(r is Reading.STANDING for r in readings.values()) == 21


def test_a_grey_nexus_turret_is_the_shape_without_the_colour() -> None:
    """The client keeps a destroyed nexus turret on the map as a grey
    outline; the shape matches, the colour does not."""
    image = scene(set(TURRETS) - {BLUE_NEXUS_TOP}, grey={BLUE_NEXUS_TOP})
    assert reader().read_once(image)[BLUE_NEXUS_TOP] is Reading.GONE


def test_an_icon_of_the_other_team_is_not_this_turret() -> None:
    image = scene(set(TURRETS) - {RED_BOT_OUTER})
    blue = TEMPLATES[(Team.BLUE, "shield")]
    h, w = blue.shape[:2]
    cx, cy = centre(RED_BOT_OUTER)
    image[cy - h // 2 : cy - h // 2 + h, cx - w // 2 : cx - w // 2 + w] = blue
    assert reader().read_once(image)[RED_BOT_OUTER] is Reading.GONE


def test_a_champion_marker_over_the_spot_is_no_reading() -> None:
    x, y = ANCHORS[BLUE_MID_INNER]
    image = scene(set(TURRETS) - {BLUE_MID_INNER})
    cv2.circle(image, (round(x), round(y)), 20, (200, 180, 160), -1)
    readings = reader().read_once(image, markers=[(x, y, 20.0)])
    assert readings[BLUE_MID_INNER] is Reading.UNKNOWN


def test_anything_drawn_over_the_spot_is_no_reading() -> None:
    """The cursor, a ping, a stack of markers the detector missed: a missing
    icon only counts when the spot shows bare map."""
    image = scene(set(TURRETS) - {BLUE_MID_INNER})
    cx, cy = centre(BLUE_MID_INNER)
    cursor = np.array([[cx - 6, cy - 12], [cx + 8, cy + 4], [cx - 6, cy + 10]], np.int32)
    cv2.fillPoly(image, [cursor], (235, 235, 235))
    assert reader().read_once(image)[BLUE_MID_INNER] is Reading.UNKNOWN


def test_a_standing_icon_beside_a_marker_still_reads() -> None:
    """Cover only matters when the icon is not seen; a marker nearby does
    not take away a clear sighting."""
    x, y = ANCHORS[BLUE_MID_INNER]
    readings = reader().read_once(scene(), markers=[(x + 30, y, 12.0)])
    assert readings[BLUE_MID_INNER] is Reading.STANDING


# -- the verdicts ---------------------------------------------------------


def feed(r: TurretReader, image: np.ndarray, start: float, count: int, step: float = 0.3):
    states = None
    for i in range(count):
        states = r.read(image, start + i * step)
    return states


def standing_of(states, turret: Turret) -> bool | None:
    return next(s.standing for s in states if s.turret == turret)


def test_every_turret_is_unknown_until_called() -> None:
    r = reader()
    first = r.read(scene(), 0.0)
    assert len(first) == 22 and all(s.standing is None for s in first)
    assert all(s.standing is True for s in r.read(scene(), 0.3))


def test_nothing_read_yet_is_no_set() -> None:
    assert reader().states() is None


def test_a_turret_covered_all_along_stays_unknown_alone() -> None:
    """The cursor parked on one turret must not hold the other 21 back."""
    image = scene(set(TURRETS) - {BLUE_MID_INNER})
    cx, cy = centre(BLUE_MID_INNER)
    cv2.circle(image, (cx, cy), 14, (235, 235, 235), -1)
    states = feed(reader(), image, 0.0, 30)
    assert standing_of(states, BLUE_MID_INNER) is None
    assert sum(s.standing is True for s in states) == 21


def test_a_loss_is_adopted_only_after_the_hold() -> None:
    r = reader(destroy_hold=1.5, destroy_readings=3)
    feed(r, scene(), 0.0, 2)
    gone = scene(set(TURRETS) - {BLUE_MID_INNER})
    assert standing_of(feed(r, gone, 1.0, 5), BLUE_MID_INNER) is True  # 1.2s
    assert standing_of(feed(r, gone, 2.5, 1), BLUE_MID_INNER) is False


def test_a_standing_reading_restarts_the_hold() -> None:
    r = reader(destroy_hold=1.5)
    feed(r, scene(), 0.0, 2)
    gone = scene(set(TURRETS) - {BLUE_MID_INNER})
    feed(r, gone, 1.0, 4)
    feed(r, scene(), 2.2, 1)
    assert standing_of(feed(r, gone, 2.5, 5), BLUE_MID_INNER) is True
    assert standing_of(feed(r, gone, 4.0, 1), BLUE_MID_INNER) is False


def test_cover_neither_breaks_nor_advances_a_loss() -> None:
    """A champion walking over a fallen turret's spot says nothing about the
    turret; the readings around it still count."""
    r = reader(destroy_hold=1.5, destroy_readings=3)
    feed(r, scene(), 0.0, 2)
    gone = scene(set(TURRETS) - {BLUE_MID_INNER})
    x, y = ANCHORS[BLUE_MID_INNER]
    feed(r, gone, 1.0, 2)
    for i in range(5):
        r.read(gone, 1.6 + 0.3 * i, markers=[(x, y, 20.0)])
    assert standing_of(r.read(gone, 3.1), BLUE_MID_INNER) is False


def test_a_lane_turret_never_comes_back() -> None:
    r = reader(destroy_hold=1.5)
    feed(r, scene(), 0.0, 2)
    feed(r, scene(set(TURRETS) - {BLUE_MID_INNER}), 1.0, 8)
    assert standing_of(feed(r, scene(), 5.0, 10), BLUE_MID_INNER) is False


def test_a_nexus_turret_rebuilds() -> None:
    r = reader(destroy_hold=1.5, rebuild_hold=1.5)
    feed(r, scene(), 0.0, 2)
    down = scene(set(TURRETS) - {BLUE_NEXUS_TOP}, grey={BLUE_NEXUS_TOP})
    assert standing_of(feed(r, down, 1.0, 8), BLUE_NEXUS_TOP) is False
    assert standing_of(feed(r, scene(), 5.0, 3), BLUE_NEXUS_TOP) is False
    assert standing_of(feed(r, scene(), 5.9, 3), BLUE_NEXUS_TOP) is True


def test_a_turret_already_down_is_learned_as_down() -> None:
    r = reader()
    states = feed(r, scene(set(TURRETS) - {RED_BOT_OUTER}), 0.0, 20)
    assert standing_of(states, RED_BOT_OUTER) is False


def test_a_panel_showing_no_turret_at_all_is_not_evidence() -> None:
    """Whatever filled the panel, it was not the map losing all 22."""
    r = reader()
    feed(r, scene(), 0.0, 2)
    blank = np.full((*SIZE, 3), GROUND_BGR, np.uint8)
    states = feed(r, blank, 1.0, 20)
    assert all(s.standing for s in states)


def test_reset_forgets_the_game() -> None:
    r = reader()
    feed(r, scene(set(TURRETS) - {RED_BOT_OUTER}), 0.0, 20)
    r.reset()
    assert r.states() is None
    assert standing_of(feed(r, scene(), 10.0, 2), RED_BOT_OUTER) is True


@pytest.mark.parametrize("width", [400, 600])
def test_other_panel_sizes_scale_the_icons(width: int) -> None:
    factor = width / 486
    scaled = {
        key: cv2.resize(t, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        for key, t in TEMPLATES.items()
    }
    canvas = np.full((round(SIZE[0] * factor), width, 3), GROUND_BGR, np.uint8)
    scaled_anchors = {}
    for turret in TURRETS:
        x, y = ANCHORS[turret]
        scaled_anchors[turret] = (x * factor, y * factor)
        icon = scaled[(turret.team, turret.kind)]
        dx, dy = ICON_OFFSET["nexus" if turret.tier is Tier.NEXUS else "shield"]
        cx, cy = round((x + dx) * factor), round((y + dy) * factor)
        h, w = icon.shape[:2]
        canvas[cy - h // 2 : cy - h // 2 + h, cx - w // 2 : cx - w // 2 + w] = icon
    r = TurretReader(scaled_anchors, minimap_width=width)
    assert set(r.read_once(canvas).values()) == {Reading.STANDING}
