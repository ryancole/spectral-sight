"""Champion portraits drawn in the HUD.

Two places show them, and between them they cover your whole team:

- a row of four circular teammate portraits along the right edge, each with a
  level badge and health and mana bars
- the local player's own portrait in the centre HUD, which the row omits

Positions are fixed by resolution, so reading them is a crop, not a search.

The HUD is a *labelled reference gallery present in every frame*, which is why
identification can get started with no external assets at all. But it is the
weaker of the two available galleries, and the reason is worth stating plainly
so nobody rebuilds on it by mistake:

**Minimap icons are always stock champion art. HUD portraits are skin-specific.**

So the HUD only agrees with the minimap for a champion on their base skin. It
agreed for all four teammates in the sample footage and disagreed completely for
the local player, who was using a skin -- their two images share almost nothing.
That is a property of the skin, not of the matcher.

A static stock icon set is therefore the correct gallery for identification, and
covers enemies too, which the HUD never can because nothing here names them.
This panel remains useful for what it uniquely provides: a *labelled* roster of
who is on your team, plus per-ally health and level.

See `spectral_sight.perception.identity`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAYOUT_DIR = Path(__file__).resolve().parents[4] / "etc" / "hud"


def _square_crop(frame: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(width, int(round(cx)) + radius)
    y1 = min(height, int(round(cy)) + radius)
    return frame[y0:y1, x0:x1]


@dataclass(frozen=True, slots=True)
class PortraitLayout:
    """Where every friendly champion portrait sits, in frame pixel coordinates."""

    ally_first_center_x: float
    ally_center_y: float
    ally_spacing: float
    ally_radius: int
    self_center_x: float
    self_center_y: float
    self_radius: int
    ally_count: int = 4

    # -- allies -----------------------------------------------------------

    def ally_center(self, index: int) -> tuple[float, float]:
        if not 0 <= index < self.ally_count:
            raise IndexError(
                f"ally index {index} out of range 0..{self.ally_count - 1}"
            )
        return (
            self.ally_first_center_x + index * self.ally_spacing,
            self.ally_center_y,
        )

    def ally_crop(self, frame: np.ndarray, index: int) -> np.ndarray:
        cx, cy = self.ally_center(index)
        return _square_crop(frame, cx, cy, self.ally_radius)

    def ally_crops(self, frame: np.ndarray) -> list[np.ndarray]:
        return [self.ally_crop(frame, i) for i in range(self.ally_count)]

    # -- local player -----------------------------------------------------

    def self_crop(self, frame: np.ndarray) -> np.ndarray:
        return _square_crop(frame, self.self_center_x, self.self_center_y,
                            self.self_radius)

    def all_crops(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Every friendly portrait, keyed by a stable slot name."""
        crops = {f"ally{i + 1}": c for i, c in enumerate(self.ally_crops(frame))}
        crops["self"] = self.self_crop(frame)
        return crops

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ally_first_center_x": self.ally_first_center_x,
            "ally_center_y": self.ally_center_y,
            "ally_spacing": self.ally_spacing,
            "ally_radius": self.ally_radius,
            "ally_count": self.ally_count,
            "self_center_x": self.self_center_x,
            "self_center_y": self.self_center_y,
            "self_radius": self.self_radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> PortraitLayout:
        return cls(
            ally_first_center_x=float(data["ally_first_center_x"]),
            ally_center_y=float(data["ally_center_y"]),
            ally_spacing=float(data["ally_spacing"]),
            ally_radius=int(data["ally_radius"]),
            ally_count=int(data.get("ally_count", 4)),
            self_center_x=float(data["self_center_x"]),
            self_center_y=float(data["self_center_y"]),
            self_radius=int(data["self_radius"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PortraitLayout:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(cls, width: int, height: int) -> PortraitLayout:
        path = LAYOUT_DIR / f"{width}x{height}.json"
        if not path.exists():
            raise FileNotFoundError(f"no calibrated portrait layout at {path}")
        return cls.load(path)
