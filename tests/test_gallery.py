"""Descriptor and gallery behaviour.

These use synthetic art rather than real crops, so they pin the invariances the
matcher claims -- scale, exposure, occlusion masking, one-to-one assignment --
rather than the accuracy of matching, which only real footage can measure.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.identity import (
    HUD_MASK,
    Gallery,
    describe,
    load_icon_gallery,
)
from spectral_sight.perception.identity.gallery import build_mask


def _art(seed: int, size: int = 64) -> np.ndarray:
    """A distinctive circular icon, standing in for champion portrait art."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size, 3), np.uint8)
    for _ in range(9):
        center = tuple(rng.integers(size // 4, 3 * size // 4, 2).tolist())
        color = rng.integers(40, 255, 3).tolist()
        cv2.circle(canvas, center, int(rng.integers(4, 13)), color, -1)
    return cv2.GaussianBlur(canvas, (0, 0), 1.0)


def test_identical_art_scores_near_one() -> None:
    patch = _art(1)
    assert describe(patch).similarity(describe(patch)) == pytest.approx(1.0, abs=1e-5)


def test_different_art_scores_lower_than_identical() -> None:
    a, b = describe(_art(1)), describe(_art(2))
    assert a.similarity(b) < a.similarity(describe(_art(1)))


def test_descriptor_is_scale_invariant() -> None:
    """A 26px minimap marker and a 52px panel portrait must be comparable."""
    art = _art(3)
    small = cv2.resize(art, (26, 26), interpolation=cv2.INTER_AREA)
    large = cv2.resize(art, (52, 52), interpolation=cv2.INTER_AREA)
    assert describe(small).similarity(describe(large)) > 0.8


def test_descriptor_is_exposure_invariant() -> None:
    """Minimap art is drawn dimmer than the HUD; that must not matter."""
    art = _art(4)
    dimmed = (art.astype(np.float32) * 0.6).astype(np.uint8)
    assert describe(art).similarity(describe(dimmed)) > 0.8


def test_gallery_finds_the_right_entry() -> None:
    gallery = Gallery()
    for i in range(5):
        gallery.add(f"champ{i}", _art(i))

    match = gallery.match(_art(3))
    assert match is not None
    assert match.name == "champ3"
    assert match.confident


def test_empty_gallery_returns_none() -> None:
    assert Gallery().match(_art(1)) is None


def test_assignment_is_one_to_one() -> None:
    """Two markers must never claim the same champion -- it is in one place."""
    gallery = Gallery()
    for i in range(4):
        gallery.add(f"champ{i}", _art(i))

    # Two near-copies of the same art compete; only one may win that identity.
    queries = [_art(0), cv2.GaussianBlur(_art(0), (0, 0), 2.0), _art(2)]
    results = gallery.assign(queries)

    claimed = [m.name for m in results if m is not None]
    assert len(claimed) == len(set(claimed))


def test_assignment_returns_none_for_unknown_art() -> None:
    gallery = Gallery()
    for i in range(3):
        gallery.add(f"champ{i}", _art(i))

    results = gallery.assign([_art(99)], min_similarity=0.9)
    assert results == [None]


def test_assignment_aligns_with_input_order() -> None:
    gallery = Gallery()
    for i in range(3):
        gallery.add(f"champ{i}", _art(i))

    results = gallery.assign([_art(2), _art(0)])
    assert [m.name if m else None for m in results] == ["champ2", "champ0"]


def test_rejects_non_bgr_patch() -> None:
    with pytest.raises(ValueError):
        describe(np.zeros((16, 16), np.uint8))


# -- masks ------------------------------------------------------------------


def test_badge_mask_is_a_subset_of_the_circle() -> None:
    """The HUD variant only ever removes pixels; it never adds any."""
    circle, hud = build_mask(), build_mask(exclude_badge=True)
    assert hud.sum() < circle.sum()
    assert not (hud & ~circle).any()


def test_gallery_applies_its_mask_to_queries_too() -> None:
    """Every descriptor in a comparison must use the same mask."""
    gallery = Gallery(mask=HUD_MASK)
    gallery.add("champ", _art(5))

    match = gallery.match(_art(5))
    assert match is not None
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


# -- icon set ---------------------------------------------------------------


def test_load_icon_gallery_reads_a_directory(tmp_path) -> None:
    for name in ("Aatrox", "Ahri", "Akali"):
        cv2.imwrite(str(tmp_path / f"{name}.png"), _art(hash(name) % 100))

    gallery = load_icon_gallery(tmp_path)
    assert sorted(gallery.names) == ["Aatrox", "Ahri", "Akali"]


def test_load_icon_gallery_reports_a_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_icons"):
        load_icon_gallery(tmp_path / "nope")


def test_assign_regions_finds_markers_in_place() -> None:
    """The stage 1 entry point: markers given as (x, y, radius) in an image."""
    gallery = Gallery()
    for i in range(4):
        gallery.add(f"champ{i}", _art(i))

    canvas = np.zeros((200, 200, 3), np.uint8)
    placements = [(50, 60, 2), (140, 130, 0)]
    for cx, cy, seed in placements:
        canvas[cy - 20 : cy + 20, cx - 20 : cx + 20] = cv2.resize(
            _art(seed), (40, 40), interpolation=cv2.INTER_AREA
        )

    results = gallery.assign_regions(
        canvas, [(float(cx), float(cy), 20.0) for cx, cy, _ in placements]
    )
    assert [m.name if m else None for m in results] == ["champ2", "champ0"]
