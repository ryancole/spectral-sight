"""The player's gold, and the filter that decides which figure to publish.

Like the creep score tests, the reader is tested on digits rendered here
rather than crops of the game's font: pale yellow on a dark teal box, drawn
larger than the templates, since those are the two things the reader does
that the clock does not -- lift the saturation ceiling and rescale each glyph.
Whether the thresholds suit League's own font is a question about footage,
answered in the module docstring. The filter is pure logic over readings, so
it is fed readings directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.abilities import AbilityLayout
from spectral_sight.perception.hud.clock import (
    GlyphSet,
    glyph_boxes,
    segment_glyphs,
)
from dataclasses import replace

from spectral_sight.perception.hud.gold import GoldConfig, GoldFilter, GoldReader

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_FILL = (70, 55, 20)
"""Dark teal, BGR, like the gold box."""
INK = (170, 230, 235)
"""Pale yellow, BGR: saturation about 70, which the clock's ceiling of 45
would throw away."""
COIN = (40, 190, 230)
"""Saturated gold, for the coin left of the digits."""


@pytest.fixture(scope="module")
def glyphs() -> GlyphSet:
    """Templates in white at a smaller size than the gold is drawn -- the
    clock's relation to the gold box."""
    strips = {}
    for digit in "0123456789":
        strip = np.full((24, 18, 3), (24, 28, 26), np.uint8)
        cv2.putText(strip, digit, (2, 18), FONT, 0.55, (238, 238, 238), 2,
                    cv2.LINE_AA)
        strips[digit] = strip
    width = height = 0
    for strip in strips.values():
        for _, _, w, h in glyph_boxes(strip):
            width, height = max(width, w), max(height, h)
    size = (width + 2, height + 2)
    return GlyphSet(
        glyphs={d: segment_glyphs(s, size)[0] for d, s in strips.items()},
        size=size,
    )


def layout(slot: float = 40.0) -> AbilityLayout:
    """A panel whose first summoner slot is at (20, 20), `slot` px square.
    Only the summoner fields place the gold box."""
    return AbilityLayout(
        ability_first_x=0.0, ability_y=20.0, ability_width=slot,
        ability_height=slot, ability_spacing=slot,
        summoner_first_x=20.0, summoner_y=20.0, summoner_width=slot,
        summoner_height=slot, summoner_spacing=slot,
    )


def gold_frame(
    text: str,
    reader: GoldReader,
    *,
    scale: float = 1.0,
    gap: int = 3,
    bridge: bool = False,
    stray: bool = False,
) -> np.ndarray:
    """A frame with `text` in the reader's box, left-aligned after a coin.

    Each digit is drawn on its own and placed `gap` px after the last, so
    the spacing is controlled. `bridge` joins every pair of neighbours with
    a one-row stroke, as the lit box does to "44"; `stray` leaves a few lit
    pixels two rows under the digits.
    """
    frame = np.full((220, 320, 3), (15, 15, 15), np.uint8)
    x, y, w, h = reader.box
    frame[y - 3 : y + h + 3, x - 25 : x + w] = BOX_FILL
    cv2.circle(frame, (x - 12, y + h // 2), round(5 * scale), COIN, -1)

    size, thick = 0.7 * scale, max(1, round(2 * scale))
    left = x + round(7 * scale)
    top = y + round(2 * scale)
    # Every digit in this font stands as tall as the next, so tops aligned
    # is baselines aligned.
    spans = []
    for char in text:
        (cw, ch), base = cv2.getTextSize(char, FONT, size, thick)
        tile = np.zeros((ch + base + 4, cw + 4, 3), np.uint8)
        cv2.putText(tile, char, (2, ch + 1), FONT, size, INK, thick, cv2.LINE_AA)
        lit = tile.max(axis=2) > 0
        rows, cols = np.nonzero(lit.any(axis=1))[0], np.nonzero(lit.any(axis=0))[0]
        tile = tile[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        window = frame[top : top + tile.shape[0], left : left + tile.shape[1]]
        np.maximum(window, tile, out=window)
        spans.append((left, left + tile.shape[1]))
        left += tile.shape[1] + gap
    if bridge:
        row = top + round(9 * scale)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            frame[row, end - 2 : start + 2] = INK
    if stray and spans:
        rows = frame[y : y + h, spans[0][0] : spans[-1][1]].max(axis=(1, 2))
        bottom = y + int(np.nonzero(rows > 100)[0][-1])
        frame[bottom + 2, spans[1][0] if len(spans) > 1 else spans[0][0]] = INK
        frame[bottom + 2, spans[-1][0] + 1 : spans[-1][0] + 3] = INK
    return frame


def reader_for(glyphs: GlyphSet, slot: float = 40.0) -> GoldReader:
    return GoldReader.beside(layout(slot), glyphs)


# -- the reader -----------------------------------------------------------


@pytest.mark.parametrize("gold", ["0", "75", "500", "1325", "4444"])
def test_reads_the_yellow_figure(glyphs: GlyphSet, gold: str) -> None:
    reader = reader_for(glyphs)
    assert reader.read(gold_frame(gold, reader)) == int(gold)


def test_five_digits_fit_the_box(glyphs: GlyphSet) -> None:
    """Set tight: this font's digits are wider than the game's."""
    reader = reader_for(glyphs)
    assert reader.read(gold_frame("14250", reader, gap=1)) == 14250


def test_the_box_sits_at_the_measured_offset(glyphs: GlyphSet) -> None:
    """(138, 79) from the first summoner slot, 76 x 19, at a 40px slot."""
    assert reader_for(glyphs).box == (158, 99, 76, 19)


def test_touching_digits_are_split(glyphs: GlyphSet) -> None:
    """The lit box joins neighbours' crossbars; one wide glyph read as a
    single digit would sink the whole figure. The cut is placed by the
    digit pitch, a property of the face: this font advances its own height
    per digit where the game's advances 0.82 of it."""
    reader = reader_for(glyphs)
    reader = replace(reader, config=replace(GoldConfig(), pitch=1.0))
    assert reader.read(gold_frame("4444", reader, gap=1, bridge=True)) == 4444


def test_stray_pixels_under_the_digits_are_ignored(glyphs: GlyphSet) -> None:
    reader = reader_for(glyphs)
    assert reader.read(gold_frame("2760", reader, stray=True)) == 2760


def test_a_scaled_panel_is_read_through_the_scaled_layout(
    glyphs: GlyphSet,
) -> None:
    """The HUD scale grows the slot, and the box and the digits with it."""
    reader = reader_for(glyphs, slot=60.0)
    assert reader.box == (227, 138, 114, 28)
    assert reader.read(gold_frame("3076", reader, scale=1.5, gap=4)) == 3076


def test_an_empty_box_reads_nothing(glyphs: GlyphSet) -> None:
    reader = reader_for(glyphs)
    assert reader.read(gold_frame("", reader)) is None


def test_a_frame_without_the_panel_reads_nothing(glyphs: GlyphSet) -> None:
    reader = reader_for(glyphs)
    assert reader.read(np.zeros((220, 320, 3), np.uint8)) is None


def test_a_purchase_shows_as_a_drop(glyphs: GlyphSet) -> None:
    """Frames through reader and filter: gold climbing, a 850-gold buy,
    climbing again. The drop is published on the frame it is drawn."""
    reader = reader_for(glyphs)
    gold = GoldFilter()
    shown = [1325, 1325, 1326, 1326, 476, 476, 477, 477]
    published = [gold.update(reader.read(gold_frame(str(g), reader)))
                 for g in shown]
    assert published == [None, 1325, 1325, 1326, 476, 476, 476, 477]


# -- the filter -----------------------------------------------------------


def run(readings) -> list[int | None]:
    f = GoldFilter()
    return [f.update(r) for r in readings]


def test_the_first_figure_needs_two_readings() -> None:
    assert run([500, 500]) == [None, 500]


def test_a_rise_needs_two_readings() -> None:
    assert run([500, 500, 502, 502]) == [None, 500, 500, 502]


def test_a_fall_is_taken_on_one_reading() -> None:
    """A purchase must not leave the old figure up: reading high says the
    player can afford what they cannot."""
    assert run([1325, 1325, 476]) == [None, 1325, 476]


def test_a_single_high_misread_is_never_published() -> None:
    assert run([661, 661, 961, 661, 662, 662])[2:] == [661, 661, 661, 662]


def test_a_single_low_misread_is_published_and_recovers() -> None:
    """The filter leans low on purpose. Measured misreads came back low --
    2760 as 260 -- and cost a reading of "can't afford" at worst."""
    assert run([2760, 2760, 260, 2760, 2760]) == [None, 2760, 260, 260, 2760]


def test_no_rise_or_fall_limit_applies() -> None:
    """Gold has no direction rule: a kill, a sale or a big buy all jump."""
    assert run([300, 300, 3300, 3300, 0, 0])[-3:] == [3300, 0, 0]


def test_unreadable_frames_hold_the_figure_and_the_count() -> None:
    assert run([500, 500, None, 510, None, 510]) == [
        None, 500, 500, 500, 500, 510,
    ]


def test_reset_forgets_everything() -> None:
    f = GoldFilter()
    f.update(500)
    f.update(500)
    f.reset()
    assert f.value is None
    assert f.update(500) is None
