"""Minimap pixels to Summoner's Rift world units.

Stage 1 reports positions in minimap-crop pixels, which is the right coordinate
system for a detector and the wrong one for everything downstream. Crop pixels
depend on the resolution and on the user's minimap-scale slider, so two clips
are not comparable, and nothing in them is *addressable*: no predicate about
lanes, the river, a jungle quadrant, or the distance to a pit can be written
against a number whose meaning changes with a settings menu.

World units fix both. They are stable across captures, they make distances mean
something (a champion's body is a couple of hundred units, a lane is a few
thousand), and they let a speed be checked against the game's own movement
speeds -- which is the only external handle we have on whether the scale is
right at all.

**The map area is not the minimap crop.** The panel has an ornate frame and a
black gutter inside it, so the rendered map sits inside the region calibrated by
`tools/calibrate_minimap.py`. Assuming otherwise puts a fixed percentage error
into every position. Measured on the sample capture, marker centres span 9.5 to
298.5 px across a 325 px crop, and the slack is not symmetric.

**The rectangle is stored in frame coordinates, not crop coordinates.** Storing
it relative to the minimap crop would be more convenient and quietly wrong: the
crop is itself hand-calibrated, so re-dragging it would invalidate the world
calibration with no error and no warning. Where the map sits on the screen is a
fact about the screen, like the minimap region and the clock region, and it is
stored the same way.

**Y is flipped.** World +y runs toward the top of the minimap -- blue base sits
at the low corner of both axes and is drawn bottom-left -- while pixel y runs
downward.

On the bounds themselves: `SUMMONERS_RIFT` is the extent in wide community use
for this conversion, not a figure Riot publishes. Nothing structural depends on
it being exact, since it is a single scalar applied uniformly, but it is the one
number here that is taken on faith rather than measured. `tools/calibrate_world.py
--validate` is the check: it converts tracked champion motion into units per
second and compares it against the movement speeds the game actually has. A
scale that is wrong by a third shows up immediately as champions that sprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from spectral_sight.perception.minimap.region import MinimapRegion

WORLD_DIR = Path(__file__).resolve().parents[4] / "etc" / "world"


@dataclass(frozen=True, slots=True)
class WorldBounds:
    """The extent of a map in game units."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError(f"bounds must have positive extent, got {self}")

    @property
    def span_x(self) -> float:
        return self.max_x - self.min_x

    @property
    def span_y(self) -> float:
        return self.max_y - self.min_y

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> WorldBounds:
        return cls(
            min_x=float(data["min_x"]),
            min_y=float(data["min_y"]),
            max_x=float(data["max_x"]),
            max_y=float(data["max_y"]),
        )


SUMMONERS_RIFT = WorldBounds(min_x=0.0, min_y=0.0, max_x=14870.0, max_y=14980.0)
"""Playable extent of Summoner's Rift. See the module docstring: this is the
conventional figure, and the validation pass is what actually tests it."""


@dataclass(frozen=True, slots=True)
class WorldTransform:
    """Affine map between frame pixels and world units, with the y axis flipped.

    `x`, `y`, `width` and `height` describe the rendered map area in *frame*
    pixel coordinates -- the terrain square inside the minimap panel, not the
    panel and not the calibrated crop.
    """

    x: float
    y: float
    width: float
    height: float
    bounds: WorldBounds = SUMMONERS_RIFT

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"map area must have positive extent, got {self}")

    # -- conversion -------------------------------------------------------

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        """Frame pixel coordinate to world units."""
        u = (x - self.x) / self.width
        v = (y - self.y) / self.height
        return (
            self.bounds.min_x + u * self.bounds.span_x,
            self.bounds.min_y + (1.0 - v) * self.bounds.span_y,
        )

    def to_frame(self, world_x: float, world_y: float) -> tuple[float, float]:
        """World units back to frame pixel coordinates."""
        u = (world_x - self.bounds.min_x) / self.bounds.span_x
        v = 1.0 - (world_y - self.bounds.min_y) / self.bounds.span_y
        return self.x + u * self.width, self.y + v * self.height

    def from_minimap(
        self, region: MinimapRegion, x: float, y: float
    ) -> tuple[float, float]:
        """Minimap-crop coordinate to world units, via the crop's frame offset.

        This is the entry point stage 1 output goes through: `Blip` positions are
        in crop space, and the transform is anchored to the frame.
        """
        return self.to_world(*region.to_frame(x, y))

    # -- scale ------------------------------------------------------------

    @property
    def units_per_pixel(self) -> tuple[float, float]:
        return self.bounds.span_x / self.width, self.bounds.span_y / self.height

    @property
    def squareness(self) -> float:
        """Ratio of the x and y scales, as a check on the calibrated rectangle.

        The map renders square, so this reduces to the aspect of the bounds
        themselves: Summoner's Rift is 14870 by 14980 units, giving 0.993 for a
        correctly measured area. A value further from that than a percent or so
        means the rectangle is stretched, not that the map is.
        """
        ux, uy = self.units_per_pixel
        return ux / uy

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "bounds": self.bounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorldTransform:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            bounds=WorldBounds.from_dict(data["bounds"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WorldTransform:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(cls, width: int, height: int) -> WorldTransform:
        """Load the calibrated transform for a resolution from ``etc/world/``."""
        path = WORLD_DIR / f"{width}x{height}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no world calibration at {path}. "
                f"Run: python tools/calibrate_world.py --input <clip>"
            )
        return cls.load(path)

    @classmethod
    def assuming_crop(
        cls, region: MinimapRegion, bounds: WorldBounds = SUMMONERS_RIFT
    ) -> WorldTransform:
        """Treat the whole minimap crop as the map area.

        Wrong by however much frame and gutter the crop includes -- several
        percent on the sample capture -- but it needs no extra calibration step,
        and it is a useful starting point and a useful thing to measure against.
        """
        return cls(
            x=float(region.x),
            y=float(region.y),
            width=float(region.width),
            height=float(region.height),
            bounds=bounds,
        )
