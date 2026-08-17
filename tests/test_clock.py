"""Reading the match timer.

The tests render their own digits rather than shipping crops of the game's
font. What is being pinned here is the machinery -- segmentation, matching,
parsing, and the filter that holds the clock through an unreadable frame -- and
that machinery has to work on any font it is calibrated against. Whether the
thresholds suit League's own font is a question about real footage, which
`tools/calibrate_clock.py --validate-only` answers over thousands of frames.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.clock import (
    COLON,
    ClockFilter,
    ClockReader,
    ClockRegion,
    GameClock,
    GlyphSet,
    glyph_boxes,
    segment_glyphs,
    tighten,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.9
THICKNESS = 2
PANEL = (24, 28, 26)
"""Dark, faintly coloured background, like the HUD strip the timer sits on."""

INK = (238, 238, 238)
LABELS = [*(str(d) for d in range(10)), COLON]


def _glyph(char: str) -> np.ndarray:
    (width, height), baseline = cv2.getTextSize(char, FONT, SCALE, THICKNESS)
    image = np.zeros((height + baseline + 4, width + 4, 3), np.uint8)
    cv2.putText(image, char, (2, height + 1), FONT, SCALE, INK, THICKNESS, cv2.LINE_AA)
    return image


def render(text: str, *, gap: int = 4, pad: int = 3) -> np.ndarray:
    """A timer strip reading `text`, with the glyphs cleanly separated."""
    glyphs = [_glyph(c) for c in text]
    height = max(g.shape[0] for g in glyphs) + 2 * pad
    width = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1) + 2 * pad
    strip = np.full((height, width, 3), PANEL, np.uint8)
    x = pad
    for glyph in glyphs:
        window = strip[pad : pad + glyph.shape[0], x : x + glyph.shape[1]]
        np.maximum(window, glyph, out=window)
        x += glyph.shape[1] + gap
    return strip


@pytest.fixture(scope="module")
def glyphs() -> GlyphSet:
    """A glyph set built the way calibration builds one: from rendered samples."""
    strips = {label: render(label) for label in LABELS}
    width = height = 0
    for strip in strips.values():
        for _, _, w, h in glyph_boxes(strip):
            width, height = max(width, w), max(height, h)
    size = (width + 2, height + 2)
    return GlyphSet(
        glyphs={l: segment_glyphs(s, size)[0] for l, s in strips.items()},
        size=size,
    )


def _reader(strip: np.ndarray, glyphs: GlyphSet) -> ClockReader:
    region = ClockRegion(x=0, y=0, width=strip.shape[1], height=strip.shape[0])
    return ClockReader(region, glyphs)


# -- segmentation ---------------------------------------------------------


def test_finds_one_run_per_character() -> None:
    assert len(glyph_boxes(render("12:34"))) == 5


def test_ignores_the_dark_panel_between_glyphs() -> None:
    """A value floor is doing this, not a fixed pitch: the gaps are not equal."""
    boxes = glyph_boxes(render("11:11"))
    assert len(boxes) == 5
    assert all(w > 0 and h > 0 for _, _, w, h in boxes)


def test_rejects_a_saturated_neighbour() -> None:
    """The gold clock icon sits beside the digits and must not become a glyph."""
    strip = render("12:34")
    cv2.circle(strip, (6, strip.shape[0] // 2), 4, (0, 190, 235), -1)
    assert len(glyph_boxes(strip)) == 5


# -- reading --------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("00:00", 0), ("12:34", 754), ("01:05", 65), ("59:59", 3599), ("07:00", 420)],
)
def test_reads_a_rendered_time(glyphs: GlyphSet, text: str, expected: int) -> None:
    reading = _reader(render(text), glyphs).read(render(text))
    assert reading is not None
    assert reading.total_seconds == expected


def test_reads_a_three_digit_minute(glyphs: GlyphSet) -> None:
    """Minutes are however many digits there are; only seconds are fixed at two."""
    strip = render("100:00")
    reading = _reader(strip, glyphs).read(strip)
    assert reading is not None
    assert reading.total_seconds == 6000


def test_rejects_an_impossible_seconds_value(glyphs: GlyphSet) -> None:
    strip = render("12:74")
    assert _reader(strip, glyphs).read(strip) is None


def test_rejects_a_strip_with_no_separator(glyphs: GlyphSet) -> None:
    strip = render("1234")
    assert _reader(strip, glyphs).read(strip) is None


def test_rejects_a_strip_with_one_seconds_digit(glyphs: GlyphSet) -> None:
    strip = render("12:3")
    assert _reader(strip, glyphs).read(strip) is None


def test_rejects_an_empty_strip(glyphs: GlyphSet) -> None:
    strip = np.full((20, 60, 3), PANEL, np.uint8)
    assert _reader(strip, glyphs).read(strip) is None


def test_survives_a_save_and_load(glyphs: GlyphSet, tmp_path) -> None:
    path = tmp_path / "glyphs.npz"
    glyphs.save(path)
    loaded = GlyphSet.load(path)
    assert sorted(loaded.glyphs) == sorted(glyphs.glyphs)
    assert loaded.size == glyphs.size
    strip = render("12:34")
    assert ClockReader(
        ClockRegion(0, 0, strip.shape[1], strip.shape[0]), loaded
    ).read(strip).total_seconds == 754


# -- the calibration box --------------------------------------------------


def test_tighten_shrinks_a_generous_box() -> None:
    strip = render("12:34")
    padded = cv2.copyMakeBorder(strip, 12, 12, 20, 20, cv2.BORDER_CONSTANT,
                                value=PANEL)
    box = ClockRegion(0, 0, padded.shape[1], padded.shape[0])
    tightened = tighten(padded, box)
    assert tightened is not None
    assert tightened.width < box.width
    assert tightened.height < box.height
    assert tightened.x >= 20 - 1


def test_tighten_returns_none_on_a_blank_box() -> None:
    blank = np.full((30, 80, 3), PANEL, np.uint8)
    assert tighten(blank, ClockRegion(0, 0, 80, 30)) is None


# -- formatting -----------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, text", [(0, "0:00"), (65, "1:05"), (754, "12:34"), (3599, "59:59")]
)
def test_formats_as_the_game_does(seconds: int, text: str) -> None:
    assert str(GameClock(seconds, confidence=1.0)) == text


# -- the filter -----------------------------------------------------------


def test_first_reading_is_taken_on_trust() -> None:
    clock = ClockFilter()
    assert clock.update(GameClock(100, 1.0), 0.0).total_seconds == 100


def test_carries_the_clock_through_an_unreadable_frame() -> None:
    clock = ClockFilter()
    clock.update(GameClock(100, 1.0), 0.0)
    held = clock.update(None, 4.0)
    assert held.total_seconds == 104
    assert not held.observed


def test_rejects_a_misread_that_jumps() -> None:
    """One wrong glyph moves the clock by tens of seconds; elapsed time cannot."""
    clock = ClockFilter()
    clock.update(GameClock(100, 1.0), 0.0)
    kept = clock.update(GameClock(700, 1.0), 1.0)
    assert kept.total_seconds == 101
    assert not kept.observed


def test_accepts_a_reading_that_agrees() -> None:
    clock = ClockFilter()
    clock.update(GameClock(100, 1.0), 0.0)
    fresh = clock.update(GameClock(110, 1.0), 10.0)
    assert fresh.total_seconds == 110
    assert fresh.observed


def test_resyncs_after_sustained_disagreement() -> None:
    """A pause or a seek is the world changing, not noise to be filtered out."""
    clock = ClockFilter(resync_after=3.0)
    clock.update(GameClock(100, 1.0), 0.0)
    assert not clock.update(GameClock(500, 1.0), 1.0).observed
    assert not clock.update(GameClock(502, 1.0), 3.0).observed
    adopted = clock.update(GameClock(504, 1.0), 5.0)
    assert adopted.total_seconds == 504
    assert adopted.observed


def test_a_single_disagreement_does_not_resync() -> None:
    clock = ClockFilter(resync_after=3.0)
    clock.update(GameClock(100, 1.0), 0.0)
    assert not clock.update(GameClock(500, 1.0), 1.0).observed
    back = clock.update(GameClock(102, 1.0), 2.0)
    assert back.total_seconds == 102
    assert back.observed
