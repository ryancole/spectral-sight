"""Derive a whole calibration set for a frame size nobody has calibrated.

Six calibrations sit under `etc/`, each a set of rectangles in frame pixels
filed under one exact frame size, and each was originally a person dragging a
box. That is five drags too many for a tool whose point is that you start it and
watch, and it is also unnecessary: the six are not independent.

They are all the same HUD, at one scale. The receiver stretches a fixed game
layout to fill its window, so a frame of any size is that one layout under a
scale and a shift -- and if the transform can be recovered, every rectangle
follows from the reference set without anyone pointing at anything.

Recovering it needs one thing found in the frame, and `minimap.locate` already
finds the panel to within a pixel. That gives four numbers, but they are not
equally good:

- **The horizontal scale does not come from the panel at all.** It is the frame
  width over the reference width, exactly, because the game fills the window's
  full width. Deriving it from the panel's width instead costs a pixel of
  measurement error on a 325px panel, which is 0.3% -- and 0.3% over the
  thousand pixels between the panel and the left of the HUD is a 15px miss.
  Measured, that is exactly what happened: elements near the panel landed within
  a pixel and the player's own portrait was out by 15.
- **The vertical scale does come from the panel**, because the frame's height is
  not all game: a window has a title bar, and it does not scale with the
  content. The panel's height is the only measurement of the content's height
  available.
- **The vertical offset absorbs the title bar** without ever measuring it. It
  falls out of where the panel sits, so a window with 31 pixels of chrome, 45,
  or none needs no special case. Measured across all three: within 1.4px.

Worst error over every HUD element at seven window sizes and three chrome
heights: 3.4 pixels, and that at the extreme top-left corner, furthest from the
panel in both axes. Every element that is actually read came in under 1.5.

**What this assumes is that the game's own layout has not changed** -- same
resolution and same UI scale on the machine being streamed. Window size, window
shape and title bar are all free, but a different in-game HUD scale would move
the pieces relative to each other and no amount of fitting here would say so.
The guard against that is the clock: it is a known string in a known font, so
the derived reader is tried on the frame it was derived from and kept only if it
reads. A layout that has moved fails that test, and the run then says game time
is unavailable instead of producing a plausible timeline of nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.hud.clock import (
    ClockReader,
    ClockRegion,
    GlyphSet,
    load_clock_reader,
    save_calibration,
    tighten,
)
from spectral_sight.perception.hud.clock import CLOCK_DIR as _CLOCK_DIR
from spectral_sight.perception.hud.portraits import LAYOUT_DIR as PORTRAIT_DIR
from spectral_sight.perception.hud.portraits import PortraitLayout
from spectral_sight.perception.hud.resources import LAYOUT_DIR as RESOURCE_DIR
from spectral_sight.perception.hud.resources import ResourceLayout
from spectral_sight.perception.minimap.locate import PanelMatch, locate_panel
from spectral_sight.perception.minimap.region import REGION_DIR, MinimapRegion
from spectral_sight.perception.minimap.world import WORLD_DIR, WorldTransform
from spectral_sight.perception.nameplates.plates import LAYOUT_DIR as NAMEPLATE_DIR
from spectral_sight.perception.nameplates.plates import NameplateLayout

PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "etc" / "map" / "profile.json"
)


@dataclass(frozen=True, slots=True)
class Reference:
    """The frame size whose calibrations everything else is derived from."""

    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @classmethod
    def load(cls, path: str | Path | None = None) -> Reference:
        path = Path(path) if path is not None else PROFILE_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"no reference profile at {path}. "
                f"Run: python tools/build_reference.py --input <calibrated clip>"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(width=int(data["width"]), height=int(data["height"]))

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else PROFILE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"width": self.width, "height": self.height}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return path

    def panel(self) -> MinimapRegion:
        return MinimapRegion.for_resolution(self.width, self.height)


@dataclass(frozen=True, slots=True)
class LayoutFit:
    """The scale and shift taking the reference layout onto this frame."""

    scale_x: float
    scale_y: float
    offset_y: float
    panel: MinimapRegion
    """The panel as found, used rather than derived -- it is a measurement, and
    a rounded copy of it would be strictly worse than the thing measured."""

    score: float

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x, y * self.scale_y + self.offset_y

    def box(
        self, x: float, y: float, width: float, height: float
    ) -> tuple[float, float, float, float]:
        px, py = self.point(x, y)
        return px, py, width * self.scale_x, height * self.scale_y

    def across(self, value: float) -> float:
        """A horizontal length, which has no offset."""
        return value * self.scale_x

    def down(self, value: float) -> float:
        """A vertical length, which has no offset."""
        return value * self.scale_y

    def mean(self, value: float) -> float:
        """A length with no axis -- a radius. Distorted by an uneven stretch
        either way, so the average is the least wrong single number."""
        return value * (self.scale_x + self.scale_y) / 2


def fit_layout(
    frame: np.ndarray, reference: Reference, match: PanelMatch | None = None
) -> LayoutFit | None:
    """Recover the transform from the reference layout onto this frame.

    None when the panel could not be found confidently, since every number here
    is derived from it and a guessed panel would silently misplace the whole HUD
    rather than misplace the minimap alone.
    """
    if match is None:
        match = locate_panel(frame)
    if match is None or not match.confident:
        return None

    height, width = frame.shape[:2]
    reference_panel = reference.panel()
    scale_x = width / reference.width
    scale_y = match.region.height / reference_panel.height
    return LayoutFit(
        scale_x=scale_x,
        scale_y=scale_y,
        offset_y=match.region.y - reference_panel.y * scale_y,
        panel=match.region,
        score=match.score,
    )


def _derive_world(fit: LayoutFit, source: WorldTransform) -> WorldTransform:
    x, y, width, height = fit.box(source.x, source.y, source.width, source.height)
    return replace(source, x=x, y=y, width=width, height=height)


def _derive_portraits(fit: LayoutFit, source: PortraitLayout) -> PortraitLayout:
    first_x, center_y = fit.point(
        source.ally_first_center_x, source.ally_center_y
    )
    self_x, self_y = fit.point(source.self_center_x, source.self_center_y)
    return replace(
        source,
        ally_first_center_x=first_x,
        ally_center_y=center_y,
        ally_spacing=fit.across(source.ally_spacing),
        ally_radius=max(1, round(fit.mean(source.ally_radius))),
        self_center_x=self_x,
        self_center_y=self_y,
        self_radius=max(1, round(fit.mean(source.self_radius))),
    )


def _derive_resources(fit: LayoutFit, source: ResourceLayout) -> ResourceLayout:
    def move(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(round(v) for v in fit.box(*box))  # type: ignore[return-value]

    return replace(source, health=move(source.health), mana=move(source.mana))


def _derive_nameplates(fit: LayoutFit, source: NameplateLayout) -> NameplateLayout:
    # `exclude` is fractions of the frame and the projection coefficients act on
    # normalised positions, so both are already scale-free and carry over as
    # they are. Only the pixel measurements move.
    def down(pair: tuple[int, int]) -> tuple[int, int]:
        return round(fit.down(pair[0])), round(fit.down(pair[1]))

    return replace(
        source,
        bar_width=max(1, round(fit.across(source.bar_width))),
        bar_height=max(1, round(fit.down(source.bar_height))),
        resource_dy=down(source.resource_dy),
        level_dx=(round(fit.across(source.level_dx[0])),
                  round(fit.across(source.level_dx[1]))),
        level_dy=down(source.level_dy),
    )


CLOCK_AGREEMENT = 0.6
"""Share of sample frames the derived clock must read to be kept.

Not a majority for its own sake: a clock reader that lands on some frames and
not others is what a slightly-wrong box looks like, while a correct one reads
nearly everything. One frame cannot tell those apart -- measured, a derivation
checked against the single frame it came from was kept at a window size where it
went on to read two frames in five.
"""


def _derive_clock(
    fit: LayoutFit, frames: Sequence[np.ndarray], reference: Reference
) -> tuple[ClockRegion, GlyphSet] | None:
    """The clock box and its glyphs, rescaled and then snapped to the digits.

    The snap is what makes this safe. Everything else here lands within a pixel
    or two, which is nothing against a portrait 52 pixels across but is most of
    a 13-pixel-tall row of digits. `tighten` already exists to shrink a
    generous hand-drawn box onto the glyphs actually inside it, and a derived
    box with padding is exactly a generous hand-drawn box -- so the prediction
    only has to be close enough to contain them, not to be right.
    """
    frame = frames[0]
    source = load_clock_reader(*reference.size)
    x, y, width, height = fit.box(
        source.region.x, source.region.y, source.region.width, source.region.height
    )
    pad_x, pad_y = max(4.0, width * 0.25), max(4.0, height * 0.6)
    # Clipped to the frame rather than trusted. A fit that is wrong enough to
    # put the timer off the edge should cost the clock and nothing else, and
    # `crop` raises on a box that does not fit -- which would take the whole run
    # down over the one calibration the run can manage without.
    frame_height, frame_width = frame.shape[:2]
    left = min(max(0, round(x - pad_x)), frame_width)
    top = min(max(0, round(y - pad_y)), frame_height)
    right = min(frame_width, round(x + width + pad_x))
    bottom = min(frame_height, round(y + height + pad_y))
    if right - left < 8 or bottom - top < 4:
        return None

    snapped = tighten(
        frame, ClockRegion(x=left, y=top, width=right - left, height=bottom - top)
    )
    if snapped is None:
        return None

    glyph_width = max(4, round(source.glyphs.size[0] * fit.scale_x))
    glyph_height = max(4, round(source.glyphs.size[1] * fit.scale_y))
    glyphs = GlyphSet(
        glyphs={
            label: cv2.resize(image, (glyph_width, glyph_height),
                              interpolation=cv2.INTER_AREA)
            for label, image in source.glyphs.glyphs.items()
        },
        size=(glyph_width, glyph_height),
    )

    # Try it before keeping it. Unlike every other piece here the clock has a
    # legibility floor as well as a position: shrink the window far enough and
    # the timer is a smudge that no calibration can read, whatever it is
    # pointed at. A calibration that cannot read is worse than none, because
    # the pipeline treats a present one as usable.
    reader = ClockReader(snapped, glyphs)
    read = sum(reader.read(f) is not None for f in frames)
    if read < max(1, round(len(frames) * CLOCK_AGREEMENT)):
        return None
    return snapped, glyphs


@dataclass(frozen=True, slots=True)
class Piece:
    """One calibration: what it buys, where it lives, and how it is derived.

    One table rather than two lists. The first version kept the set of
    calibrations and the set of derivations separately and they promptly
    disagreed about a name, which is the whole failure mode of parallel lists.

    `load` and `convert` are absent for the two that do not derive by
    transforming a rectangle: the minimap is measured rather than derived, and
    the clock carries learned glyph images as well as a box.
    """

    name: str
    directory: Path
    load: object = None
    convert: object = None

    def path(self, width: int, height: int) -> Path:
        return self.directory / f"{width}x{height}.json"


PIECES: tuple[Piece, ...] = (
    Piece("minimap", REGION_DIR),
    Piece("game time", _CLOCK_DIR),
    Piece("world units", WORLD_DIR, WorldTransform.for_resolution, _derive_world),
    Piece("deaths", PORTRAIT_DIR, PortraitLayout.for_resolution, _derive_portraits),
    Piece("nameplates", NAMEPLATE_DIR, NameplateLayout.for_resolution,
          _derive_nameplates),
    Piece("player bars", RESOURCE_DIR, ResourceLayout.for_resolution,
          _derive_resources),
)
"""Every calibration, by the name a person would use for what it buys them."""


def missing(width: int, height: int) -> list[str]:
    """Which calibrations this frame size does not have yet."""
    return [p.name for p in PIECES if not p.path(width, height).exists()]


def derive(
    frames: Sequence[np.ndarray], fit: LayoutFit, reference: Reference
) -> dict[str, Path]:
    """Write a full calibration set for this frame size. Returns what it wrote.

    Geometry comes from the first frame; the rest are only used to check that
    the derived clock reads more than once by luck. More frames make that check
    stronger and change nothing else, and one frame is allowed -- a still is a
    legitimate source.

    Nothing already on disk is overwritten. A calibration that exists was either
    made by hand or derived earlier and then possibly corrected by hand, and in
    both cases it is better evidence than a fresh derivation -- which is also
    what stops this from ever eating the reference set it derives from.

    Each piece is skipped rather than failed on when the reference does not have
    it, so a partially calibrated reference derives a partial set -- the same
    deal the pipeline already offers, where the optional stages are absent
    rather than fatal.
    """
    if not frames:
        raise ValueError("need at least one frame to derive from")
    height, width = frames[0].shape[:2]
    if (width, height) == reference.size:
        raise ValueError(
            f"{width}x{height} is the reference layout itself; deriving onto it "
            f"would replace the calibrations everything else is derived from"
        )
    written: dict[str, Path] = {}

    panel_piece = next(p for p in PIECES if p.name == "minimap")
    path = panel_piece.path(width, height)
    if not path.exists():
        fit.panel.save(path)
        written[panel_piece.name] = path

    for piece in PIECES:
        if piece.load is None or piece.path(width, height).exists():
            continue
        try:
            source = piece.load(*reference.size)
        except FileNotFoundError:
            continue
        if source is None:
            continue
        path = piece.path(width, height)
        path.parent.mkdir(parents=True, exist_ok=True)
        piece.convert(fit, source).save(path)
        written[piece.name] = path

    clock_piece = next(p for p in PIECES if p.name == "game time")
    try:
        clock = (None if clock_piece.path(width, height).exists()
                 else _derive_clock(fit, frames, reference))
    except FileNotFoundError:
        clock = None
    if clock is not None:
        region, glyphs = clock
        save_calibration(region, glyphs, width, height)
        written[clock_piece.name] = clock_piece.path(width, height)

    return written


MISSING_CLOCK = (
    "no game time: the derived timer did not read reliably. Either the window "
    "is too small for the digits to survive being scaled down -- a larger one "
    "would recover it -- or the streamed game's HUD is not the layout the "
    "reference was calibrated against, in which case the rest of the derived "
    "geometry is suspect too."
)
"""Why the derived clock was dropped. Both causes are named because nothing
here can tell them apart: the vertical scale looked like it might, but measured
across a range of window sizes the clock read at 0.70, failed at 0.68, read
again at 0.66 and 0.63, and failed below that. There is no floor to quote, so
quoting one would be inventing a diagnosis."""
