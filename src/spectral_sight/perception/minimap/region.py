"""Where the minimap lives inside a captured frame.

The minimap is anchored to the bottom-right of the HUD, but its size is driven
by an in-game scale slider, so it cannot be derived from resolution alone. We
resolve it once per (resolution, scale) via `tools/calibrate_minimap.py` and
persist the result under `etc/regions/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGION_DIR = Path(__file__).resolve().parents[4] / "etc" / "regions"


@dataclass(frozen=True, slots=True)
class MinimapRegion:
    """An axis-aligned crop rectangle in frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"minimap region must have positive extent, got {self}")

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the minimap sub-image. This is a view, not a copy."""
        frame_h, frame_w = image.shape[:2]
        if self.x + self.width > frame_w or self.y + self.height > frame_h:
            raise ValueError(
                f"region {self} does not fit inside a {frame_w}x{frame_h} frame"
            )
        return image[self.y : self.y + self.height, self.x : self.x + self.width]

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Map a minimap-crop coordinate back into frame coordinates."""
        return x + self.x, y + self.y

    def to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """Map a minimap-crop coordinate into [0, 1] across the panel.

        This is the form stage 2 hands to the world-space transform, since it is
        independent of the user's minimap scale setting.
        """
        return x / self.width, y / self.height

    @property
    def looks_square(self) -> bool:
        """Whether this is the shape a minimap panel actually is.

        The panel is square in the game. It is not quite square here, because
        the receiver fills its window by stretching without preserving aspect --
        the calibrated 2118x1354 panel is 325x322 -- so this is a loose check
        for a misdrag, not a claim about the aspect.
        """
        return abs(self.width - self.height) <= max(self.width, self.height) * 0.08

    @classmethod
    def select(
        cls, image: np.ndarray, *, window: str = "calibrate minimap"
    ) -> MinimapRegion | None:
        """Drag a box around the panel. None if the drag was cancelled.

        The fallback, not the route. `locate.locate_panel` finds the panel by
        recognising the map drawn on it, to within a pixel, and this is what
        happens when it declines -- on a frame with no panel in it, or a window
        shaped so oddly the search cannot express the panel's aspect.

        The reasoning that once made this the only option still holds and is
        worth keeping: a region that is merely close is not a slightly worse
        read, it is a confident read of the wrong pixels, and nothing downstream
        is in a position to notice. What retired it was not the locator being
        accurate but the locator being able to say when it is not.
        """
        import cv2

        box = cv2.selectROI(window, image, showCrosshair=False)
        cv2.destroyWindow(window)
        x, y, width, height = (int(v) for v in box)
        if width == 0 or height == 0:
            return None
        return cls(x=x, y=y, width=width, height=height)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> MinimapRegion:
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )

    @classmethod
    def parse(cls, spec: str) -> MinimapRegion:
        """Parse an ``x,y,width,height`` string, as accepted on the CLI."""
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"expected 'x,y,width,height', got {spec!r}")
        return cls(*(int(p) for p in parts))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MinimapRegion:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(
        cls, width: int, height: int, *, profile: str = "default"
    ) -> MinimapRegion:
        """Load the calibrated region for a resolution from ``etc/regions/``."""
        name = f"{width}x{height}" + ("" if profile == "default" else f".{profile}")
        path = REGION_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no calibrated minimap region at {path}. Run: python "
                f"tools/calibrate_minimap.py --image <clip, still or window:name>"
            )
        return cls.load(path)
