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

# Sampled from real footage rather than invented: the enemy ring reads as
# magenta (H~167), not red, and the ally ring sits around H~100.
TEAM_BGR: dict[Team, tuple[int, int, int]] = {
    Team.BLUE: (235, 165, 40),
    Team.RED: (120, 40, 230),
}
PORTRAIT_BGR = (90, 100, 110)
"""Desaturated grey-brown, standing in for champion portrait art."""

CHAMPION_RADIUS = 13
"""Marker radius on real footage ranged 11.5-15.0 at a 325px minimap."""


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


# -- nameplates -----------------------------------------------------------
#
# Sampled from real footage the same way the marker colours were: the hostile
# health bar is the same magenta-red as the enemy ring, the resource bar sits
# near H~100, and the level box is dark enough to read as unlit.
PLATE_HOSTILE_BGR = (40, 35, 190)
PLATE_ALLY_BGR = (40, 190, 60)
PLATE_RESOURCE_BGR = (200, 150, 70)
PLATE_BOX_BGR = (30, 25, 35)
PLATE_INK = (238, 238, 238)

PLATE_TICK_EVERY = 15
PLATE_TICK_WIDTH = 2
"""Health bars are divided by tick marks, which is why fill measurement has to
hop short gaps rather than take a connected component."""


def draw_nameplate(
    canvas: np.ndarray,
    x: int,
    y: int,
    *,
    health: float,
    resource: float,
    hostile: bool = True,
    level: int | None = None,
    bar_width: int = 100,
    bar_height: int = 10,
    resource_dy: int = 13,
    resource_height: int = 4,
    box_width: int = 22,
    ticks: bool = True,
) -> None:
    """A champion nameplate: level box, ticked health bar, resource bar.

    `x`, `y` are the health bar's top-left, matching what the reader reports.
    """
    box_right = x - 4
    cv2.rectangle(canvas, (box_right - box_width, y - 3),
                  (box_right, y + resource_dy + resource_height + 2),
                  PLATE_BOX_BGR, -1)
    if level is not None:
        text = str(level)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            canvas, text,
            (box_right - box_width // 2 - tw // 2, y + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLATE_INK, 1, cv2.LINE_AA,
        )

    colour = PLATE_HOSTILE_BGR if hostile else PLATE_ALLY_BGR
    filled = int(round(bar_width * health))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled - 1, y + bar_height - 1),
                      colour, -1)
    if ticks:
        for offset in range(PLATE_TICK_EVERY, filled, PLATE_TICK_EVERY):
            cv2.rectangle(
                canvas, (x + offset, y),
                (x + offset + PLATE_TICK_WIDTH - 1, y + bar_height - 1),
                PLATE_BOX_BGR, -1,
            )

    top = y + resource_dy
    filled = int(round(bar_width * resource))
    if filled > 0:
        cv2.rectangle(canvas, (x, top), (x + filled - 1, top + resource_height - 1),
                      PLATE_RESOURCE_BGR, -1)


def draw_minion_bar(
    canvas: np.ndarray, x: int, y: int, *, hostile: bool = True, width: int = 40
) -> None:
    """A health bar with no resource bar and no level box.

    The thing the reader has to reject: right colour, bar-shaped, and there are
    far more of them on screen than there are champions.
    """
    colour = PLATE_HOSTILE_BGR if hostile else PLATE_ALLY_BGR
    cv2.rectangle(canvas, (x, y), (x + width, y + 3), colour, -1)


def plate_scene(
    width: int = 640, height: int = 400, *, seed: int = 11
) -> np.ndarray:
    """A world-view background to draw nameplates onto."""
    rng = np.random.default_rng(seed)
    base = np.array([60, 68, 58], np.uint8)
    canvas = np.tile(base, (height, width, 1)).astype(np.int16)
    canvas += rng.integers(-14, 14, (height, width, 3), dtype=np.int16)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _background(size: int, seed: int) -> np.ndarray:
    """Dark, low-saturation terrain noise -- nowhere near the team hues."""
    rng = np.random.default_rng(seed)
    base = np.array([45, 55, 40], np.uint8)
    canvas = np.tile(base, (size, size, 1)).astype(np.int16)
    canvas += rng.integers(-18, 18, (size, size, 3), dtype=np.int16)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_champion(
    canvas: np.ndarray,
    x: int,
    y: int,
    team: Team,
    radius: int = CHAMPION_RADIUS,
    *,
    core_bgr: tuple[int, int, int] = PORTRAIT_BGR,
) -> None:
    """A portrait core inside a team-coloured ring -- the thing we detect.

    `core_bgr` exists so tests can reproduce the case that killed hole-based
    detection: portrait art that happens to match the team's own hue.
    """
    cv2.circle(canvas, (x, y), radius, core_bgr, -1, cv2.LINE_AA)
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
