"""Deriving a calibration set for a frame size nobody calibrated.

The arithmetic is pinned here; the accuracy is not, because accuracy is a claim
about real footage and belongs to the measurement recorded in the module
docstring -- deaths agreeing on every sampled frame at every window size tried,
the world transform within 0.3%, the clock reading or being dropped.

What these do cover is the part that would be silent if it broke: that a
derivation never lands on top of a calibration someone already has, and never on
the reference set it derives *from*. That failure destroys work rather than
producing a bad answer, and it happened once during development.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_sight import calibration
from spectral_sight.calibration import (
    LayoutFit,
    Reference,
    derive,
    fit_layout,
    missing,
)
from spectral_sight.perception.minimap.locate import PanelMatch
from spectral_sight.perception.minimap.region import MinimapRegion

REFERENCE = Reference(width=2118, height=1354)
REFERENCE_PANEL = MinimapRegion(1787, 1020, 325, 322)


def fit(scale_x: float = 1.0, scale_y: float = 1.0,
        offset_y: float = 0.0) -> LayoutFit:
    return LayoutFit(scale_x=scale_x, scale_y=scale_y, offset_y=offset_y,
                     panel=REFERENCE_PANEL, score=0.9)


class TestLayoutFit:
    def test_identity_moves_nothing(self) -> None:
        assert fit().point(100, 200) == (100, 200)

    def test_scales_and_shifts(self) -> None:
        moved = fit(scale_x=0.5, scale_y=0.75, offset_y=10)
        assert moved.point(100, 200) == (50, 160)

    def test_lengths_take_no_offset(self) -> None:
        """The offset is the title bar; a bar's height does not contain one."""
        moved = fit(scale_y=0.5, offset_y=31)
        assert moved.down(20) == 10
        assert moved.across(20) == 20

    def test_a_radius_averages_the_two_scales(self) -> None:
        """A circle under an uneven stretch is an ellipse; this is the least
        wrong single number for something stored as one radius."""
        assert fit(scale_x=0.8, scale_y=1.2).mean(10) == pytest.approx(10.0)

    def test_a_box_scales_its_extent_without_the_offset(self) -> None:
        x, y, width, height = fit(scale_x=2, scale_y=2, offset_y=5).box(1, 1, 10, 10)
        assert (x, y, width, height) == (2, 7, 20, 20)


class TestFitLayout:
    def test_declines_when_the_panel_is_not_confident(self, monkeypatch) -> None:
        """Every number in the fit rides on the panel, so an unsure panel is not
        a slightly worse HUD placement -- it is a wrong one, everywhere."""
        weak = PanelMatch(region=REFERENCE_PANEL, score=0.1)
        frame = np.zeros((1354, 2118, 3), np.uint8)
        assert fit_layout(frame, REFERENCE, weak) is None

    def test_recovers_a_pure_scale(self, monkeypatch) -> None:
        monkeypatch.setattr(Reference, "panel", lambda self: REFERENCE_PANEL)
        frame = np.zeros((677, 1059, 3), np.uint8)          # exactly half
        found = PanelMatch(region=MinimapRegion(893, 510, 162, 161), score=0.9)
        got = fit_layout(frame, REFERENCE, found)
        assert got.scale_x == pytest.approx(0.5, abs=0.01)
        assert got.scale_y == pytest.approx(0.5, abs=0.01)
        assert got.offset_y == pytest.approx(0.0, abs=1.0)

    def test_the_offset_absorbs_a_title_bar(self, monkeypatch) -> None:
        """A window's chrome does not scale with its content, and is never
        measured -- it falls out of where the panel sits."""
        monkeypatch.setattr(Reference, "panel", lambda self: REFERENCE_PANEL)
        frame = np.zeros((1354, 2118, 3), np.uint8)
        shifted = PanelMatch(
            region=MinimapRegion(1787, 1020 + 40, 325, 322), score=0.9
        )
        got = fit_layout(frame, REFERENCE, shifted)
        assert got.offset_y == pytest.approx(40.0, abs=0.5)


class TestDerive:
    def test_refuses_to_derive_onto_the_reference_itself(self) -> None:
        """It would overwrite the calibrations everything else is derived from."""
        frame = np.zeros((1354, 2118, 3), np.uint8)
        with pytest.raises(ValueError, match="reference layout"):
            derive([frame], fit(), REFERENCE)

    def test_needs_a_frame(self) -> None:
        with pytest.raises(ValueError, match="at least one frame"):
            derive([], fit(), REFERENCE)

    def test_never_overwrites_an_existing_calibration(
        self, tmp_path, monkeypatch
    ) -> None:
        """An existing file is better evidence than a fresh derivation: it was
        either drawn by hand or derived and then corrected."""
        pieces = tuple(
            calibration.Piece(p.name, tmp_path / p.name, p.load, p.convert)
            for p in calibration.PIECES
        )
        monkeypatch.setattr(calibration, "PIECES", pieces)

        existing = next(p for p in pieces if p.name == "minimap")
        existing.directory.mkdir(parents=True)
        mine = MinimapRegion(1, 2, 3, 4)
        mine.save(existing.path(800, 600))

        frame = np.zeros((600, 800, 3), np.uint8)
        written = derive([frame], fit(), REFERENCE)

        assert "minimap" not in written
        assert MinimapRegion.load(existing.path(800, 600)) == mine


class TestMissing:
    def test_reports_everything_for_an_unknown_size(self, tmp_path, monkeypatch) -> None:
        pieces = tuple(
            calibration.Piece(p.name, tmp_path / p.name, p.load, p.convert)
            for p in calibration.PIECES
        )
        monkeypatch.setattr(calibration, "PIECES", pieces)
        assert missing(999, 999) == [p.name for p in calibration.PIECES]


class TestReference:
    def test_round_trips(self, tmp_path) -> None:
        path = tmp_path / "profile.json"
        Reference(1920, 1080).save(path)
        assert Reference.load(path) == Reference(1920, 1080)

    def test_says_how_to_make_one_when_absent(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="build_reference"):
            Reference.load(tmp_path / "nope.json")
