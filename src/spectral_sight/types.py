"""Core value types shared across the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Team(Enum):
    """Which side a minimap marker belongs to.

    League renders Order as blue and Chaos as red on the minimap. In spectator
    and replay mode both teams are always drawn (no fog of war), which is the
    property the tracker leans on downstream.
    """

    BLUE = "blue"
    RED = "red"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Frame:
    """A single captured frame plus its position in the source stream."""

    image: np.ndarray
    """BGR uint8 array of shape (H, W, 3)."""

    index: int
    """Zero-based frame counter within the source."""

    timestamp: float
    """Seconds since the start of the source."""

    captured_at: float | None = None
    """Wall-clock time the frame arrived from a live source, as epoch seconds.

    None for a recorded clip, which has no wall time -- its frames happened
    whenever the recording did. For a live window this is stamped on *arrival*,
    before the frame waits its turn in the mailbox, because the gap between
    arrival and processing is precisely the latency a downstream consumer needs
    to know about and the one a stamp taken any later would hide."""

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        height, width = self.image.shape[:2]
        return width, height


@dataclass(frozen=True, slots=True)
class Blip:
    """A class-agnostic champion marker found on the minimap.

    Coordinates are in minimap-crop pixel space. Converting to frame space or
    game-world space is the caller's job -- see `MinimapRegion.to_frame` and the
    world calibration that lands with stage 2.
    """

    x: float
    y: float
    radius: float
    team: Team
    score: float
    """Detection confidence in [0, 1]. Not a probability, just a ranking key."""

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y

    def distance_to(self, other: Blip) -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))
