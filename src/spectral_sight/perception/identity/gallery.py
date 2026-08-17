"""Matching a minimap marker against a gallery of champion portraits.

Both sides of the comparison show the same circular champion art, so this does
not need a learned embedding -- it needs the two crops put into a common frame
and compared. Three things have to be normalised away first, and each one is a
correctness issue rather than a refinement:

**Scale.** A minimap marker is roughly 26px across; an ally panel portrait is
about 52px. Both are resampled to a fixed size, and both are blurred to the same
degree afterwards, so the downsampled portrait does not carry sharper detail than
anything the minimap could ever produce.

**Occlusion.** Panel portraits have a badge covering the top of the circle, and
minimap markers have a team-coloured ring around the outside. Neither is present
in the other, so a single shared mask excludes both regions from every
descriptor regardless of where it came from. Comparing like with like matters
more here than retaining every pixel.

**Exposure.** Minimap art is drawn dimmer than the panel. Descriptors are
z-normalised per channel, which makes the comparison depend on structure and
relative colour rather than absolute brightness.

Similarity is cosine distance over the masked, normalised pixels. A learned
embedding would buy robustness to art the gallery has never seen, which is not
the situation here: for allies the gallery *is* built from the current game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PATCH_SIZE = 32
"""Common resample size. Large enough for facial structure, small enough that a
26px minimap marker is not being asked to invent detail."""

INNER_RADIUS = 0.78
"""Fraction of the patch half-width kept. Excludes the team-coloured ring."""

BADGE_CENTER = (0.0, -0.78)
BADGE_RADIUS = 0.42
"""Level badge occluding the top of a panel portrait, in patch-local units where
1.0 is the half-width. Masked out of *both* sides so they stay comparable."""

MATCH_BLUR = 0.8

ALIGN_SCALES: tuple[float, ...] = (0.85, 1.0, 1.15)
ALIGN_OFFSETS: tuple[float, ...] = (-1.5, 0.0, 1.5)
"""Crop variants tried when matching a marker whose framing is uncertain.

Stage 1 locates a marker to a pixel or two and refits its radius against the
ring, which is not the same as framing the *portrait* the way the HUD does.
Comparing a single crop therefore penalises correct identities for being
slightly mis-framed. Measured on real markers, searching these variants lifted
true matches from 0.60 to 0.79 and 0.73 to 0.84 while leaving non-champions
around 0.4 -- so it widens the separation rather than just raising every score.
"""


def _shared_mask(size: int = PATCH_SIZE) -> np.ndarray:
    """The pixels every descriptor is allowed to use."""
    axis = (np.arange(size) - (size - 1) / 2.0) / ((size - 1) / 2.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")

    keep = np.hypot(xx, yy) <= INNER_RADIUS
    badge = np.hypot(xx - BADGE_CENTER[0], yy - BADGE_CENTER[1]) <= BADGE_RADIUS
    return keep & ~badge


_MASK = _shared_mask()


@dataclass(frozen=True, slots=True)
class PatchDescriptor:
    """A normalised, masked appearance vector for one circular champion icon."""

    vector: np.ndarray

    def similarity(self, other: PatchDescriptor) -> float:
        """Cosine similarity in [-1, 1]. Identical art scores near 1."""
        return float(np.dot(self.vector, other.vector))


def describe(patch: np.ndarray) -> PatchDescriptor:
    """Build a descriptor from a square BGR crop centred on a champion icon."""
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"expected a BGR patch, got shape {patch.shape}")
    if patch.size == 0:
        raise ValueError("empty patch")

    resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    if MATCH_BLUR > 0:
        resized = cv2.GaussianBlur(resized, (0, 0), MATCH_BLUR)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    selected = lab[_MASK]

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
    scales: tuple[float, ...] = ALIGN_SCALES,
    offsets: tuple[float, ...] = ALIGN_OFFSETS,
) -> list[PatchDescriptor]:
    """Descriptors for several plausible framings of one marker.

    Callers compare a reference against *all* of these and keep the best, which
    is what makes matching robust to stage 1's framing being a pixel or two off.
    """
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
                variants.append(describe(patch))
    return variants


@dataclass(frozen=True, slots=True)
class Match:
    """The gallery's answer for one query patch."""

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
    """Named reference descriptors, queried by nearest neighbour."""

    entries: dict[str, PatchDescriptor] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return list(self.entries)

    def add(self, name: str, patch: np.ndarray) -> None:
        self.entries[name] = describe(patch)

    def add_descriptor(self, name: str, descriptor: PatchDescriptor) -> None:
        self.entries[name] = descriptor

    def match(self, patch: np.ndarray) -> Match | None:
        """Best gallery entry for `patch`, or None if the gallery is empty."""
        return self.match_descriptor(describe(patch))

    def match_descriptor(self, query: PatchDescriptor) -> Match | None:
        if not self.entries:
            return None
        scored = sorted(
            ((query.similarity(ref), name) for name, ref in self.entries.items()),
            reverse=True,
        )
        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        return Match(
            name=best_name,
            similarity=best_score,
            margin=best_score - runner_up,
        )

    def assign(
        self, patches: list[np.ndarray], *, min_similarity: float = 0.35
    ) -> list[Match | None]:
        """Match patches to gallery entries one-to-one, best pairs first.

        Independent nearest-neighbour lets two markers claim the same champion,
        which is impossible -- a champion is in exactly one place. Resolving the
        assignment jointly means a marker that only weakly prefers some ally
        still lands on the right one once the stronger claims are settled.

        Greedy rather than optimal: with at most five identities the difference
        is immaterial, and greedy keeps the result explainable.

        Returns a list aligned with `patches`; None means no identity survived,
        which is the expected answer for an enemy marker or a stage 1 false
        positive.
        """
        if not self.entries or not patches:
            return [None] * len(patches)

        descriptors = [describe(p) for p in patches]
        similarity = {
            (i, name): descriptor.similarity(reference)
            for i, descriptor in enumerate(descriptors)
            for name, reference in self.entries.items()
        }

        return self._resolve(similarity, len(patches), min_similarity)

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
        gain because stage 1 frames the *ring*, not the portrait. See
        `ALIGN_SCALES`.
        """
        if not self.entries or not regions:
            return [None] * len(regions)

        similarity: dict[tuple[int, str], float] = {}
        for index, (cx, cy, radius) in enumerate(regions):
            variants = describe_variants(image, cx, cy, radius)
            for name, reference in self.entries.items():
                similarity[(index, name)] = max(
                    (variant.similarity(reference) for variant in variants),
                    default=-1.0,
                )

        return self._resolve(similarity, len(regions), min_similarity)

    def _resolve(
        self,
        similarity: dict[tuple[int, str], float],
        count: int,
        min_similarity: float,
    ) -> list[Match | None]:
        """Greedy one-to-one resolution over a precomputed similarity table."""
        results: list[Match | None] = [None] * count
        claimed: set[str] = set()
        for (index, name), score in sorted(
            similarity.items(), key=lambda kv: kv[1], reverse=True
        ):
            if score < min_similarity:
                break
            if results[index] is not None or name in claimed:
                continue
            # Margin against this marker's best *alternative* identity, which is
            # what says whether the gallery could really tell them apart.
            alternatives = [
                s for (i, other), s in similarity.items()
                if i == index and other != name
            ]
            runner_up = max(alternatives) if alternatives else -1.0
            results[index] = Match(
                name=name, similarity=score, margin=score - runner_up
            )
            claimed.add(name)
        return results
