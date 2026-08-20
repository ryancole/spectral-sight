"""Find the minimap panel in a frame, without asking anyone where it is.

The panel's size is set by an in-game scale slider and its position by the
window's shape, so neither is derivable from resolution. But its *contents* are:
Summoner's Rift is the same picture in every game, at whatever size and stretch
the panel happens to be drawn at. So the panel can be found the way any known
picture is found -- correlate a reference against the frame across a range of
sizes and aspects, and take the peak.

Measured over 80 frames spanning four clips, three of which the reference was
not built from: every corner within one pixel of a hand-drawn calibration, with
correlation from 0.825 to 0.912.

That number is why this exists rather than a person dragging a box, and the
reversal is worth being precise about. The case against detecting the panel was
never that it could not be done -- it was that a region which is merely *close*
is not a worse read but a confident read of the wrong pixels, with nothing
downstream in a position to notice. That objection stands. What defeats it is
not the accuracy but the separation: on frames where the panel is absent, or
where the window is shaped so oddly that the search range cannot express it,
correlation falls to 0.00-0.74, against 0.825 at worst for a true find. There is
a wide empty band between "right to a pixel" and "wrong", with nothing observed
inside it. So this can decline, and `MIN_SCORE` sits in that band.

What lowers the score is the reference no longer describing the map: a new
season's art, a different map, a heavy colour filter. Each of those degrades
into declining rather than into a plausible wrong answer, and
`tools/build_reference.py` rebuilds the reference when the art moves on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.minimap.region import MinimapRegion

REFERENCE_PATH = (
    Path(__file__).resolve().parents[4] / "etc" / "map" / "reference.png"
)

MIN_SCORE = 0.78
"""Correlation below which the panel is reported as not found.

Sits in the observed gap: true finds bottom out at 0.825, and the best score on
a frame with no panel in it -- or with one the search range cannot express --
was 0.74. Being wrong here costs every downstream stage at once, so the
threshold leans towards declining.
"""

ASPECTS = (0.72, 0.80, 0.88, 0.94, 1.0, 1.06, 1.14, 1.24, 1.36)
"""Width-to-height ratios searched for the panel.

The panel is square in the game and arrives stretched, because the receiver
fills its window without preserving aspect. The ratio is therefore the window's
aspect divided by the streamed desktop's, and neither is known here -- so it is
searched. This range covers every window shape short of the absurd; past it the
score falls away and the answer becomes "ask a human" rather than a wrong
rectangle.
"""

SEARCH_QUADRANT = 0.55
"""Fraction of the frame, from the bottom-right corner, that is searched.

The panel is anchored to that corner in every League layout. Searching the whole
frame would cost several times as much to find the same thing, and would give
the correlator that much more unrelated art to accidentally like.
"""

MIN_SIDE, MAX_SIDE = 0.10, 0.45
"""Panel size searched, as a fraction of the frame's shorter side."""

COARSE = 6
"""Downscale factor for the first pass. The correlation peak is broad, so it
survives this, and the fine pass then has only one neighbourhood to search."""


@dataclass(frozen=True, slots=True)
class PanelMatch:
    """Where the panel is, and how strongly the evidence says so."""

    region: MinimapRegion
    score: float

    @property
    def confident(self) -> bool:
        return self.score >= MIN_SCORE


def load_reference(path: str | Path | None = None) -> np.ndarray:
    """The averaged map art that the panel is recognised by."""
    path = Path(path) if path is not None else REFERENCE_PATH
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"no minimap reference at {path}. "
            f"Run: python tools/build_reference.py --input <calibrated clip>"
        )
    return image


def _prepare(image: np.ndarray) -> np.ndarray:
    """Greyscale, slightly blurred.

    Colour is dropped because what is being matched is the map's *layout* --
    lanes, jungle walls, structures -- while the colour laid over it is the part
    that moves: fog, team tints, ability art. The blur costs nothing at these
    sizes and keeps the correlation peak broad enough to survive the coarse
    pass's downscale.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (3, 3), 0)


def _peak(
    view: np.ndarray, reference: np.ndarray, sizes: list[tuple[int, int]]
) -> tuple[float, int, int, int, int] | None:
    """Best (score, x, y, width, height) over a list of candidate sizes."""
    best = None
    for width, height in sizes:
        if width < 8 or height < 8:
            continue
        if height >= view.shape[0] or width >= view.shape[1]:
            continue
        template = cv2.resize(
            reference, (width, height), interpolation=cv2.INTER_AREA
        )
        response = cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if best is None or score > best[0]:
            best = (score, location[0], location[1], width, height)
    return best


def locate_panel(
    frame: np.ndarray, reference: np.ndarray | None = None
) -> PanelMatch | None:
    """Find the minimap panel. None if the search could not run at all.

    The caller decides what to do with a match that is not `confident`. This
    reports what it found and how strongly, rather than swallowing a weak
    answer, because "it is here" and "something here is a bit like it" want
    different handling and only the caller knows which.
    """
    if reference is None:
        reference = load_reference()

    frame_height, frame_width = frame.shape[:2]
    left = int(frame_width * (1 - SEARCH_QUADRANT))
    top = int(frame_height * (1 - SEARCH_QUADRANT))
    view = _prepare(frame[top:frame_height, left:frame_width])
    template = _prepare(reference)
    side = min(frame_height, frame_width)

    small = cv2.resize(
        view, None, fx=1 / COARSE, fy=1 / COARSE, interpolation=cv2.INTER_AREA
    )
    grid = [
        (max(8, int(size * aspect) // COARSE), max(8, size // COARSE))
        for size in range(int(side * MIN_SIDE), int(side * MAX_SIDE), 5)
        for aspect in ASPECTS
    ]
    rough = _peak(small, template, grid)
    if rough is None:
        return None

    x, y, width, height = (value * COARSE for value in rough[1:])
    score = 0.0

    # Width and height are refined one at a time rather than over a 2D grid. A
    # stretch scales the two independently, so alternating converges on the same
    # answer for a fraction of the work -- and two rounds are enough, because the
    # coarse pass already lands within a few pixels.
    for _ in range(2):
        for sizes in (
            [(width + step, height) for step in range(-8, 9, 2)],
            [(width, height + step) for step in range(-8, 9, 2)],
        ):
            pad = 3 * COARSE
            x0, y0 = max(0, x - pad), max(0, y - pad)
            patch = view[
                y0 : min(view.shape[0], y + height + pad),
                x0 : min(view.shape[1], x + width + pad),
            ]
            found = _peak(patch, template, sizes)
            if found is None:
                continue
            score, dx, dy, width, height = found
            x, y = x0 + dx, y0 + dy

    if width <= 0 or height <= 0:
        return None
    return PanelMatch(
        region=MinimapRegion(x=x + left, y=y + top, width=width, height=height),
        score=float(score),
    )
