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


_BLUR_KSIZE = 7
"""The kernel OpenCV picks itself for `MATCH_BLUR` on 8-bit images, spelled
out so `describe_batch` knows how much border each patch needs."""


def describe(patch: np.ndarray, mask: np.ndarray | None = None) -> PatchDescriptor:
    """Build a descriptor from a square BGR crop centred on a champion icon."""
    return PatchDescriptor(vector=describe_batch([patch], mask)[0])


def describe_batch(
    patches: list[np.ndarray], mask: np.ndarray | None = None
) -> np.ndarray:
    """Descriptor vectors for many crops at once, one row per patch.

    Identical to calling `describe` on each, but the blur, colour conversion
    and normalisation run once over the whole batch. On 32px patches those
    calls are nearly all fixed overhead: a marker is described in 45 framings
    and a frame has about ten markers, and doing that one patch at a time
    cost about 55 ms of every frame.

    The patches are blurred as one tall image, each padded with the same
    reflected border OpenCV would give it alone, so no pixel's kernel reaches
    into a neighbouring patch.
    """
    if mask is None:
        mask = CIRCLE_MASK
    count = len(patches)
    if count == 0:
        return np.zeros((0, 3 * int(mask.sum())), np.float32)

    resized = np.empty((count, PATCH_SIZE, PATCH_SIZE, 3), np.uint8)
    for i, patch in enumerate(patches):
        if patch.ndim != 3 or patch.shape[2] != 3:
            raise ValueError(f"expected a BGR patch, got shape {patch.shape}")
        if patch.size == 0:
            raise ValueError("empty patch")
        resized[i] = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE),
                                interpolation=cv2.INTER_AREA)

    if MATCH_BLUR > 0:
        pad = _BLUR_KSIZE // 2
        side = PATCH_SIZE + 2 * pad
        # numpy's "reflect" is OpenCV's default BORDER_REFLECT_101.
        padded = np.pad(resized, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                        mode="reflect")
        blurred = cv2.GaussianBlur(padded.reshape(count * side, side, 3),
                                   (_BLUR_KSIZE, _BLUR_KSIZE), MATCH_BLUR)
        resized = np.ascontiguousarray(
            blurred.reshape(count, side, side, 3)[:, pad:-pad, pad:-pad]
        )

    lab = cv2.cvtColor(resized.reshape(count * PATCH_SIZE, PATCH_SIZE, 3),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    selected = lab.reshape(count, PATCH_SIZE * PATCH_SIZE, 3)[:, mask.ravel()]

    # Per-channel z-normalisation: structure and relative colour, not exposure.
    centered = selected - selected.mean(axis=1, keepdims=True)
    spread = centered.std(axis=1, keepdims=True)
    spread[spread < 1e-6] = 1.0
    vectors = (centered / spread).reshape(count, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms <= 1e-6] = 1.0
    return (vectors / norms).astype(np.float32)


def _variant_patches(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    scales: tuple[float, ...] = ALIGN_SCALES,
    offsets: tuple[float, ...] = ALIGN_OFFSETS,
) -> list[np.ndarray]:
    """Crops for several plausible framings of one marker."""
    height, width = image.shape[:2]
    patches: list[np.ndarray] = []
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
                patches.append(patch)
    return patches


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
    patches = _variant_patches(image, cx, cy, radius, scales, offsets)
    return [PatchDescriptor(vector=v) for v in describe_batch(patches, mask)]


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

        # Every framing of every marker is described in one batch, then each
        # marker takes its best score per entry over its own framings.
        patches: list[np.ndarray] = []
        owners: list[int] = []
        for index, (cx, cy, radius) in enumerate(regions):
            framings = _variant_patches(image, cx, cy, radius)
            patches.extend(framings)
            owners.extend([index] * len(framings))

        rows = np.full((len(regions), len(names)), -1.0, np.float32)
        if patches:
            scores = describe_batch(patches, self.mask) @ matrix.T
            np.maximum.at(rows, np.asarray(owners), scores)
        return self._resolve(names, rows, min_similarity)

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
