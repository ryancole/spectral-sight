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
class GameArea:
    """Where the game picture sits inside a captured frame, in frame pixels.

    A window capture is the whole window, title bar and border included --
    Graphics Capture hands over the window's visible bounds, not its client
    area -- so on the kilrogg receiver at 96 DPI the game starts 31px down and
    1px in. Nothing in the pixels says so, and the chrome does not scale with
    the window: a layout measured as fractions of the *frame* is right for one
    title bar height and drifts with any other. Fractional layouts are
    measured against this rectangle instead, and it is published so a
    consumer placing anything on the game can find it.

    The whole frame when there is no window to ask.
    """

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def whole(cls, width: int, height: int) -> GameArea:
        return cls(0, 0, width, height)

    @classmethod
    def parse(cls, text: str) -> GameArea:
        """From `x,y,w,h`, the form `--game-area` takes."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError(f"expected x,y,w,h, got {text!r}")
        x, y, width, height = (int(p) for p in parts)
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError(f"not a rectangle in the frame: {text!r}")
        return cls(x, y, width, height)

    def clipped(self, width: int, height: int) -> GameArea:
        """This rectangle cut to a frame of `width` x `height`."""
        x0, y0 = min(max(self.x, 0), width), min(max(self.y, 0), height)
        x1 = min(max(self.x + self.width, x0), width)
        y1 = min(max(self.y + self.height, y0), height)
        return GameArea(x0, y0, x1 - x0, y1 - y0)

    def box(
        self, left: float, top: float, right: float, bottom: float
    ) -> tuple[int, int, int, int]:
        """(x, y, w, h) in frame pixels of a rectangle given as fractions of
        this area. Edges truncate, as a fraction of the frame always has."""
        x0 = self.x + int(left * self.width)
        y0 = self.y + int(top * self.height)
        x1 = self.x + int(right * self.width)
        y1 = self.y + int(bottom * self.height)
        return x0, y0, x1 - x0, y1 - y0

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width,
                "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> GameArea:
        return cls(int(data["x"]), int(data["y"]), int(data["width"]),
                   int(data["height"]))


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
