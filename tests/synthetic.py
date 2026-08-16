"""Synthetic minimaps with known ground truth.

Real footage is what the HSV bands ultimately have to be fitted against, but it
cannot tell us whether the *geometry* logic is right, because a miss there looks
identical to a colour threshold being slightly off. Synthetic frames separate
those two failure modes: the colours here are exact by construction, so any test
failure is a bug in the ring/core reasoning rather than a tuning problem.

The distractors mirror the things the real minimap actually draws in team
colours -- solid turret glyphs, base shading, minion dots -- since those are
what the detector has to reject.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spectral_sight.types import Team

# Chosen to land inside the detector's default HSV bands.
TEAM_BGR: dict[Team, tuple[int, int, int]] = {
    Team.BLUE: (235, 165, 40),
    Team.RED: (40, 50, 240),
}
PORTRAIT_BGR = (90, 100, 110)
"""Desaturated grey-brown, standing in for champion portrait art."""


@dataclass(frozen=True, slots=True)
class Marker:
    x: int
    y: int
    team: Team


DEFAULT_MARKERS: tuple[Marker, ...] = (
    Marker(45, 40, Team.BLUE),
    Marker(110, 65, Team.BLUE),
    Marker(70, 150, Team.BLUE),
    Marker(160, 200, Team.BLUE),
    Marker(45, 230, Team.BLUE),
    Marker(215, 55, Team.RED),
    Marker(240, 130, Team.RED),
    Marker(180, 105, Team.RED),
    Marker(235, 235, Team.RED),
    Marker(130, 245, Team.RED),
)


def _background(size: int, seed: int) -> np.ndarray:
    """Dark, low-saturation terrain noise -- nowhere near the team hues."""
    rng = np.random.default_rng(seed)
    base = np.array([45, 55, 40], np.uint8)
    canvas = np.tile(base, (size, size, 1)).astype(np.int16)
    canvas += rng.integers(-18, 18, (size, size, 3), dtype=np.int16)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_champion(
    canvas: np.ndarray, x: int, y: int, team: Team, radius: int = 10
) -> None:
    """A portrait core inside a team-coloured ring -- the thing we detect."""
    cv2.circle(canvas, (x, y), radius, PORTRAIT_BGR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), radius, TEAM_BGR[team], 3, cv2.LINE_AA)


def draw_turret(canvas: np.ndarray, x: int, y: int, team: Team, radius: int = 8) -> None:
    """A solid team-coloured glyph: right colour, right size, no hole."""
    cv2.circle(canvas, (x, y), radius, TEAM_BGR[team], -1, cv2.LINE_AA)


def draw_minion(canvas: np.ndarray, x: int, y: int, team: Team) -> None:
    """A team-coloured dot below the radius floor."""
    cv2.circle(canvas, (x, y), 2, TEAM_BGR[team], -1)


def draw_base_shading(
    canvas: np.ndarray, x: int, y: int, team: Team, radius: int = 45
) -> None:
    """A large solid team-coloured area, as drawn around a base or turret range."""
    overlay = canvas.copy()
    cv2.circle(overlay, (x, y), radius, TEAM_BGR[team], -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)


def synthetic_minimap(
    size: int = 280,
    markers: tuple[Marker, ...] = DEFAULT_MARKERS,
    *,
    seed: int = 7,
    with_distractors: bool = True,
) -> tuple[np.ndarray, tuple[Marker, ...]]:
    """Build a minimap image and return it alongside its ground-truth markers."""
    canvas = _background(size, seed)

    if with_distractors:
        draw_base_shading(canvas, 20, size - 20, Team.BLUE)
        draw_base_shading(canvas, size - 20, 20, Team.RED)
        for i in range(6):
            draw_minion(canvas, 90 + i * 7, 90 + i * 7, Team.BLUE)
            draw_minion(canvas, 190 - i * 7, 190 - i * 7, Team.RED)
        draw_turret(canvas, 95, 195, Team.BLUE)
        draw_turret(canvas, 195, 95, Team.RED)

    for marker in markers:
        draw_champion(canvas, marker.x, marker.y, marker.team)

    return canvas, markers
