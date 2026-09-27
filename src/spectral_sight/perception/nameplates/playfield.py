"""The part of the frame a world-view bar can be read from.

Both bar readers build colour masks over the whole frame and then discard every
bar whose start lies in one of the layout's `exclude` regions -- the HUD strips
and the minimap, roughly two fifths of the frame. The masks over those regions
are computed only to be thrown away, and at full resolution the mask passes are
most of what the two readers cost.

So the per-pixel work is done only where it can matter: wherever a bar may
*start*, which is everywhere outside the excluded regions, grown by how far a
reader looks from a bar's start -- up to the health bar and left to the level
box for a plate, right along the fill for both. Outside that the masks are left
zero. That is not a guess about where champions are; it is the readers' own
rule applied before the work instead of after it.

Colour thresholds look at one pixel at a time, so splitting the region into
rectangles leaves no seams in a mask. Connected components does look at
neighbours, which is why the readers run it over `rows` -- one band spanning
every rectangle -- rather than per rectangle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Playfield:
    width: int
    height: int
    rects: tuple[tuple[int, int, int, int], ...]
    """Disjoint (x0, y0, x1, y1), half-open, covering every pixel a reader
    may look at in a mask."""

    rows: tuple[int, int]
    """The band of rows spanning every rectangle, half-open."""

    @classmethod
    def of(
        cls,
        exclude: Sequence[tuple[float, float, float, float]],
        width: int,
        height: int,
        *,
        left: int,
        right: int,
        above: int,
        below: int,
    ) -> Playfield:
        """Everything outside `exclude`, grown by how far a reader looks.

        `exclude` is in fractions of the frame, as the layout stores it, and a
        region is matched inclusively at both ends, as the readers' own test
        does -- so a pixel counts as excluded only when that test would
        reject a bar starting on it.
        """
        boxes = [
            (math.ceil(x0 * width), math.ceil(y0 * height),
             math.floor(x1 * width) + 1, math.floor(y1 * height) + 1)
            for x0, y0, x1, y1 in exclude
        ]
        xs = sorted({0, width, *(min(max(v, 0), width)
                                 for b in boxes for v in (b[0], b[2]))})
        ys = sorted({0, height, *(min(max(v, 0), height)
                                  for b in boxes for v in (b[1], b[3]))})

        # Where a bar may start: the grid cells no excluded region covers.
        # Each is grown by the reach and marked, and the marks are then read
        # back as rectangles -- done once per frame size, so a full-size
        # boolean is no cost worth avoiding.
        needed = np.zeros((height, width), dtype=bool)
        for y0, y1 in zip(ys, ys[1:]):
            for x0, x1 in zip(xs, xs[1:]):
                if any(bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1
                       for bx0, by0, bx1, by1 in boxes):
                    continue
                needed[max(y0 - above, 0):min(y1 + below, height),
                       max(x0 - left, 0):min(x1 + right, width)] = True

        rects: list[tuple[int, int, int, int]] = []
        runs: tuple[tuple[int, int], ...] | None = None
        start = 0
        for y in range(height + 1):
            here = _runs(needed[y]) if y < height else None
            if here != runs:
                if runs:
                    rects.extend((a, start, b, y) for a, b in runs)
                runs, start = here, y
        rows = (
            (min(r[1] for r in rects), max(r[3] for r in rects))
            if rects else (0, 0)
        )
        return cls(width, height, tuple(rects), rows)

    @property
    def coverage(self) -> float:
        """Share of the frame the masks are computed over."""
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in self.rects)
        return area / max(self.width * self.height, 1)

    def in_range(
        self,
        hsv: np.ndarray,
        *ranges: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> np.ndarray:
        """`cv2.inRange` over the playfield, OR-ed across `ranges`.

        uint8 0/255 like `cv2.inRange` itself, and zero off the playfield.
        """
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        for x0, y0, x1, y1 in self.rects:
            patch = hsv[y0:y1, x0:x1]
            mask = cv2.inRange(patch, *ranges[0])
            for lower, upper in ranges[1:]:
                mask |= cv2.inRange(patch, lower, upper)
            out[y0:y1, x0:x1] = mask
        return out


def _runs(row: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Half-open intervals of True in a boolean row."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
    return tuple(zip(edges[::2].tolist(), edges[1::2].tolist()))
