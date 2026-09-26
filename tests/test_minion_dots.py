"""Minion dots on the minimap.

Synthetic, like the marker tests: the colours are exact by construction, so a
failure is in the disc, surround or brightness logic rather than a band a few
degrees out. The dots are drawn the way the 486px minimap draws them -- a lit
core about six pixels across inside a dark outline.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.minimap.minions import (
    MinionDot,
    MinionDotDetector,
    scaled_dot_config,
)
from spectral_sight.types import Team

TERRAIN = (95, 125, 120)
BLUE = (235, 165, 40)
RED = (60, 50, 235)
OUTLINE = (30, 30, 35)


def terrain(size: int = 240) -> np.ndarray:
    rng = np.random.default_rng(3)
    canvas = np.tile(np.array(TERRAIN, np.int16), (size, size, 1))
    canvas += rng.integers(-8, 8, canvas.shape, dtype=np.int16)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def dot(canvas: np.ndarray, x: int, y: int, colour=BLUE) -> None:
    cv2.circle(canvas, (x, y), 4, OUTLINE, -1)
    cv2.circle(canvas, (x, y), 3, colour, -1)


def found(dots: list[MinionDot]) -> set[tuple[int, int, Team]]:
    return {(d.x, d.y, d.team) for d in dots}


def near(dots: list[MinionDot], x: int, y: int, team: Team) -> bool:
    return any(d.team is team and abs(d.x - x) <= 1 and abs(d.y - y) <= 1
               for d in dots)


def test_finds_dots_of_both_teams() -> None:
    canvas = terrain()
    dot(canvas, 40, 50)
    dot(canvas, 120, 90, RED)
    dot(canvas, 200, 200, RED)
    dots = MinionDotDetector().detect(canvas)
    assert len(dots) == 3
    assert near(dots, 40, 50, Team.BLUE)
    assert near(dots, 120, 90, Team.RED)
    assert near(dots, 200, 200, Team.RED)


def test_a_chain_of_touching_dots_is_each_dot() -> None:
    """A wave walking a lane is drawn as dots nose to tail."""
    canvas = terrain()
    for step in range(4):
        dot(canvas, 60, 60 + 11 * step)
    dots = MinionDotDetector().detect(canvas)
    assert len(dots) == 4
    assert all(near(dots, 60, 60 + 11 * s, Team.BLUE) for s in range(4))


def test_ignores_a_dim_icon_in_team_colours() -> None:
    """Turret shields and objective icons are the team hue, drawn darker."""
    canvas = terrain()
    dim = (150, 105, 25)  # BLUE at V ~150
    dot(canvas, 100, 100, dim)
    assert MinionDotDetector().detect(canvas) == []


def test_ignores_team_colour_over_an_area() -> None:
    """Base shading and the river are coloured all the way round."""
    canvas = terrain()
    cv2.rectangle(canvas, (50, 50), (150, 150), BLUE, -1)
    assert MinionDotDetector().detect(canvas) == []


def test_ignores_faint_lane_dots() -> None:
    canvas = terrain()
    cv2.circle(canvas, (100, 100), 4, (140, 150, 110), -1)
    assert MinionDotDetector().detect(canvas) == []


def test_skips_dots_inside_a_champion_marker() -> None:
    canvas = terrain()
    dot(canvas, 100, 100)
    dot(canvas, 180, 180)
    dots = MinionDotDetector().detect(canvas, markers=[(104, 98, 20.0)])
    assert found(dots) and all(abs(d.x - 180) <= 1 for d in dots)


def test_radii_scale_with_the_panel() -> None:
    assert scaled_dot_config(486).core_radius == pytest.approx(3.0)
    assert scaled_dot_config(972).core_radius == pytest.approx(6.0)


def test_rejects_non_bgr_input() -> None:
    with pytest.raises(ValueError):
        MinionDotDetector().detect(np.zeros((64, 64), np.uint8))
