"""Drawing detections onto frames so a human can check them."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from spectral_sight.types import Blip, Team

TEAM_COLORS: dict[Team, tuple[int, int, int]] = {
    Team.BLUE: (255, 200, 0),
    Team.RED: (0, 80, 255),
    Team.UNKNOWN: (160, 160, 160),
}


def draw_blips(
    image: np.ndarray,
    blips: Iterable[Blip],
    *,
    scale: float = 1.0,
    show_score: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with each blip circled and labelled."""
    canvas = image.copy()
    if scale != 1.0:
        canvas = cv2.resize(
            canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

    for blip in blips:
        color = TEAM_COLORS[blip.team]
        center = (int(round(blip.x * scale)), int(round(blip.y * scale)))
        radius = int(round(blip.radius * scale))
        cv2.circle(canvas, center, radius, color, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 4, 1)
        if show_score:
            cv2.putText(
                canvas,
                f"{blip.score:.2f}",
                (center[0] - radius, center[1] - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32 * scale,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_tracks(
    image: np.ndarray,
    tracks: Iterable,
    timestamp: float,
    *,
    scale: float = 1.0,
    self_track=None,
    lost_after: float = 0.5,
) -> np.ndarray:
    """Draw tracked champions with their names.

    Champions currently visible are drawn solid; those in fog are drawn hollow
    at their last known position, which is the state a player actually cares
    about -- "Yorick was bottom river four seconds ago" is the useful readout.
    """
    canvas = image.copy()
    if scale != 1.0:
        canvas = cv2.resize(
            canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

    for track in sorted(tracks, key=lambda t: t.age(timestamp), reverse=True):
        age = track.age(timestamp)
        visible = age < lost_after
        color = TEAM_COLORS[track.team]
        center = (int(round(track.x * scale)), int(round(track.y * scale)))
        radius = int(round(14 * scale))

        if not visible:
            color = tuple(int(c * 0.55) for c in color)
        cv2.circle(canvas, center, radius, color, 2 if visible else 1, cv2.LINE_AA)

        if track is self_track:
            cv2.circle(canvas, center, radius + 4, (255, 255, 255), 1, cv2.LINE_AA)

        label = track.identity or "?"
        if track is self_track:
            label = f"{label} (you)"
        if not visible:
            label = f"{label} {age:.0f}s"

        origin = (center[0] - radius, center[1] - radius - 4)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.40 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.40 * scale, color, 1, cv2.LINE_AA)
    return canvas


def stack_masks(masks: dict[Team, np.ndarray]) -> np.ndarray:
    """Compose per-team binary masks into one BGR image for eyeballing."""
    if not masks:
        raise ValueError("no masks to stack")
    height, width = next(iter(masks.values())).shape[:2]
    canvas = np.zeros((height, width, 3), np.uint8)
    for team, mask in masks.items():
        color = np.array(TEAM_COLORS[team], np.uint8)
        canvas[mask > 0] = color
    return canvas
