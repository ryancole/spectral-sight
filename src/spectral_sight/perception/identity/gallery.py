"""Matching a minimap marker against a gallery of champion portraits.

The minimap always draws *stock* champion art, so the complete icon set is a
closed, known reference for every champion in the game -- enemies included, from
the first frame they are visible. That is the intended gallery source; see
`tools/fetch_icons.py`.

The HUD ally panel can also seed a gallery, and needs no download, but it is the
weaker option: HUD portraits are skin-specific, so they only agree with the
minimap for a champion on their base skin.

Both sides of the comparison show the same art, so this does not need a learned
embedding -- it needs the two crops put into a common frame and compared. Three
things have to be normalised away first, and each is a correctness issue rather
than a refinement:

**Scale.** A minimap marker is roughly 26px across; a stock icon is 48px and a
panel portrait about 52px. All are resampled to a fixed size and blurred equally
afterwards, so a sharper reference cannot carry detail the minimap could never
produce.

**Framing.** Stage 1 frames the *ring*, not the portrait, and the minimap's
circular crop of a square icon is not pixel-identical to the icon itself.
`describe_variants` searches scale and offset rather than trusting one crop.

**Exposure.** Minimap art is drawn dimmer than the source icons. Descriptors are
z-normalised per channel, so comparison depends on structure and relative colour
rather than absolute brightness.

Similarity is cosine distance over the masked, normalised pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PATCH_SIZE = 32
"""Common resample size. Large enough for facial structure, small enough that a
26px minimap marker is not being asked to invent detail."""

INNER_RADIUS = 0.78
"""Fraction of the patch half-width kept. Excludes the team-coloured ring."""

BADGE_CENTER = (0.0, -0.78)
BADGE_RADIUS = 0.42
"""Level badge occluding the top of a HUD panel portrait, in patch-local units
where 1.0 is the half-width."""

MATCH_BLUR = 0.8

ALIGN_SCALES: tuple[float, ...] = (0.80, 0.90, 1.0, 1.10, 1.20)
ALIGN_OFFSETS: tuple[float, ...] = (-1.5, 0.0, 1.5)
"""Crop variants tried when matching a marker whose framing is uncertain.

Stage 1 locates a marker to a pixel or two and refits its radius against the
ring, which is not the same as framing the *portrait*. Comparing a single crop
penalises correct identities for being slightly mis-framed. Measured on real
markers, searching these lifted true matches from 0.60 to 0.79 and 0.73 to 0.84
while leaving non-champions near 0.4 -- it widens separation rather than raising
every score.
"""


def _circle(size: int, radius: float) -> np.ndarray:
    axis = (np.arange(size) - (size - 1) / 2.0) / ((size - 1) / 2.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.hypot(xx, yy) <= radius


def build_mask(size: int = PATCH_SIZE, *, exclude_badge: bool = False) -> np.ndarray:
    """Which pixels a descriptor may use.

    `exclude_badge` drops the region a HUD level badge covers. Only enable it
    when the *gallery* is HUD-sourced: every descriptor in a comparison must use
    the same mask, and masking the badge out of stock icons that never had one
    just discards signal.
    """
    keep = _circle(size, INNER_RADIUS)
    if not exclude_badge:
        return keep
    axis = (np.arange(size) - (size - 1) / 2.0) / ((size - 1) / 2.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    badge = np.hypot(xx - BADGE_CENTER[0], yy - BADGE_CENTER[1]) <= BADGE_RADIUS
    return keep & ~badge


CIRCLE_MASK = build_mask()
HUD_MASK = build_mask(exclude_badge=True)


@dataclass(frozen=True, slots=True)
class PatchDescriptor:
    """A normalised, masked appearance vector for one circular champion icon."""

    vector: np.ndarray

    def similarity(self, other: PatchDescriptor) -> float:
        """Cosine similarity in [-1, 1]. Identical art scores near 1."""
        return float(np.dot(self.vector, other.vector))


def describe(patch: np.ndarray, mask: np.ndarray | None = None) -> PatchDescriptor:
    """Build a descriptor from a square BGR crop centred on a champion icon."""
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"expected a BGR patch, got shape {patch.shape}")
    if patch.size == 0:
        raise ValueError("empty patch")
    if mask is None:
        mask = CIRCLE_MASK

    resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    if MATCH_BLUR > 0:
        resized = cv2.GaussianBlur(resized, (0, 0), MATCH_BLUR)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    selected = lab[mask]

    # Per-channel z-normalisation: structure and relative colour, not exposure.
    centered = selected - selected.mean(axis=0)
    spread = centered.std(axis=0)
    spread[spread < 1e-6] = 1.0
    vector = (centered / spread).ravel()

    norm = np.linalg.norm(vector)
    if norm > 1e-6:
        vector = vector / norm
    return PatchDescriptor(vector=vector.astype(np.float32))


def describe_variants(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    mask: np.ndarray | None = None,
    scales: tuple[float, ...] = ALIGN_SCALES,
    offsets: tuple[float, ...] = ALIGN_OFFSETS,
) -> list[PatchDescriptor]:
    """Descriptors for several plausible framings of one marker."""
    height, width = image.shape[:2]
    variants: list[PatchDescriptor] = []
    for scale in scales:
        scaled = radius * scale
        for dx in offsets:
            for dy in offsets:
                x0 = int(round(cx + dx - scaled))
                y0 = int(round(cy + dy - scaled))
                x1 = int(round(cx + dx + scaled))
                y1 = int(round(cy + dy + scaled))
                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    continue
                patch = image[y0:y1, x0:x1]
                if patch.size == 0 or min(patch.shape[:2]) < 6:
                    continue
                variants.append(describe(patch, mask))
    return variants


@dataclass(frozen=True, slots=True)
class Match:
    """The gallery's answer for one query marker."""

    name: str
    similarity: float
    margin: float
    """Gap to the runner-up. Low margin means the gallery cannot separate them,
    which is a different failure from low similarity and worth acting on
    differently -- ambiguity should defer to motion, not be forced."""

    @property
    def confident(self) -> bool:
        return self.similarity >= 0.55 and self.margin >= 0.08


@dataclass
class Gallery:
    """Named reference descriptors, queried by nearest neighbour.

    Holds the references as one stacked matrix so a query is a single matmul.
    That matters at full size: the stock set is 173 champions, and every marker
    is compared as 15 framing variants.
    """

    entries: dict[str, PatchDescriptor] = field(default_factory=dict)
    mask: np.ndarray | None = None
    """Mask used for every descriptor this gallery builds or compares. Set to
    `HUD_MASK` when seeding from the HUD panel."""

    _names: list[str] = field(default_factory=list, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return list(self.entries)

    def add(self, name: str, patch: np.ndarray) -> None:
        self.add_descriptor(name, describe(patch, self.mask))

    def add_descriptor(self, name: str, descriptor: PatchDescriptor) -> None:
        self.entries[name] = descriptor
        self._matrix = None

    def _stack(self) -> tuple[list[str], np.ndarray]:
        if self._matrix is None:
            self._names = list(self.entries)
            self._matrix = np.stack(
                [self.entries[n].vector for n in self._names]
            ) if self._names else np.zeros((0, 1), np.float32)
        return self._names, self._matrix

    def match(self, patch: np.ndarray) -> Match | None:
        """Best gallery entry for `patch`, or None if the gallery is empty."""
        return self.match_descriptor(describe(patch, self.mask))

    def match_descriptor(self, query: PatchDescriptor) -> Match | None:
        if not self.entries:
            return None
        names, matrix = self._stack()
        scores = matrix @ query.vector
        return self._to_match(names, scores)

    @staticmethod
    def _to_match(names: list[str], scores: np.ndarray) -> Match:
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        runner_up = float(scores[order[1]]) if len(order) > 1 else -1.0
        return Match(
            name=names[best],
            similarity=float(scores[best]),
            margin=float(scores[best]) - runner_up,
        )

    def assign(
        self, patches: list[np.ndarray], *, min_similarity: float = 0.35
    ) -> list[Match | None]:
        """Match patches to gallery entries one-to-one, best pairs first."""
        if not self.entries or not patches:
            return [None] * len(patches)
        names, matrix = self._stack()
        scores = np.stack(
            [matrix @ describe(p, self.mask).vector for p in patches]
        )
        return self._resolve(names, scores, min_similarity)

    def assign_regions(
        self,
        image: np.ndarray,
        regions: list[tuple[float, float, float]],
        *,
        min_similarity: float = 0.35,
    ) -> list[Match | None]:
        """One-to-one assignment for markers given as (x, y, radius) in `image`.

        Prefer this over `assign` for stage 1 output: it searches crop framings
        per marker instead of trusting one crop, which is worth a large accuracy
        gain because stage 1 frames the ring, not the portrait.
        """
        if not self.entries or not regions:
            return [None] * len(regions)
        names, matrix = self._stack()

        rows = []
        for cx, cy, radius in regions:
            variants = describe_variants(image, cx, cy, radius, mask=self.mask)
            if not variants:
                rows.append(np.full(len(names), -1.0, np.float32))
                continue
            stacked = np.stack([v.vector for v in variants])
            rows.append((stacked @ matrix.T).max(axis=0))
        return self._resolve(names, np.stack(rows), min_similarity)

    def _resolve(
        self, names: list[str], scores: np.ndarray, min_similarity: float
    ) -> list[Match | None]:
        """Greedy one-to-one resolution over a (marker, entry) score matrix.

        Independent nearest-neighbour lets two markers claim the same champion,
        which is impossible -- a champion is in exactly one place. Resolving
        jointly means a marker that only weakly prefers some identity still
        lands on the right one once stronger claims are settled.

        Greedy rather than optimal: with at most ten identities in play the
        difference is immaterial, and greedy keeps the result explainable.
        """
        count = scores.shape[0]
        results: list[Match | None] = [None] * count
        claimed: set[int] = set()

        order = np.dstack(np.unravel_index(np.argsort(scores, axis=None)[::-1],
                                           scores.shape))[0]
        for marker, entry in order:
            score = float(scores[marker, entry])
            if score < min_similarity:
                break
            if results[marker] is not None or int(entry) in claimed:
                continue
            alternatives = np.delete(scores[marker], entry)
            runner_up = float(alternatives.max()) if alternatives.size else -1.0
            results[int(marker)] = Match(
                name=names[int(entry)],
                similarity=score,
                margin=score - runner_up,
            )
            claimed.add(int(entry))
        return results


def load_icon_gallery(directory: str | Path) -> Gallery:
    """Build a gallery from a directory of stock champion icons.

    Icons are square; the descriptor's circular mask does the cropping, matching
    how the minimap presents them.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"no icon set at {directory}. Run: python tools/fetch_icons.py"
        )

    gallery = Gallery()
    for path in sorted(directory.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gallery.add(path.stem, image)
    if not gallery:
        raise RuntimeError(f"no readable icons in {directory}")
    return gallery
