"""Where the game is inside a captured window frame.

The numbers are the kilrogg receiver's, measured at 96 DPI: Graphics Capture
hands over the window's visible bounds, 2117x1354, and the game is drawn in the
client area, 2115x1322 at (1, 31) inside them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectral_sight.capture import window as window_module
from spectral_sight.capture.client import client_area
from spectral_sight.capture.window import Mailbox, WindowSource
from spectral_sight.perception.nameplates.playfield import Playfield, excluded_regions
from spectral_sight.types import GameArea

RECEIVER = GameArea(1, 31, 2115, 1322)
FRAME = (2117, 1354)


# -- the client rectangle against the captured bounds ----------------------


def test_the_client_origin_less_the_frame_origin_is_the_offset() -> None:
    # The window at (640, 200) on screen; its client area starts 1px in, under
    # a 30px title bar and the 1px border above it.
    area = client_area((640, 200), (641, 231), (2115, 1322), FRAME)
    assert area == RECEIVER


def test_the_offset_does_not_depend_on_where_the_window_is() -> None:
    here = client_area((0, 0), (1, 31), (2115, 1322), FRAME)
    there = client_area((-1800, 950), (-1799, 981), (2115, 1322), FRAME)
    assert here == there == RECEIVER


def test_a_scaled_title_bar_is_measured_not_assumed() -> None:
    """At 150% the chrome is 46px, and only the measurement knows it."""
    area = client_area((100, 100), (102, 146), (3172, 1984), (3176, 2032))
    assert (area.x, area.y, area.width, area.height) == (2, 46, 3172, 1984)


def test_a_client_area_past_the_captured_bounds_is_cut_to_the_frame() -> None:
    area = client_area((0, 0), (1, 31), (2200, 1400), FRAME)
    assert area == GameArea(1, 31, 2116, 1323)


def test_a_borderless_window_is_the_whole_frame() -> None:
    area = client_area((0, 0), (0, 0), FRAME, FRAME)
    assert area == GameArea.whole(*FRAME)


# -- the rectangle itself --------------------------------------------------


def test_the_override_parses_x_y_width_height() -> None:
    assert GameArea.parse("1,31,2115,1322") == RECEIVER
    assert GameArea.parse(" 1, 31, 2115 ,1322 ") == RECEIVER


@pytest.mark.parametrize("text", ["1,31,2115", "1,31,0,1322", "-1,31,2115,1322",
                                  "a,b,c,d"])
def test_the_override_rejects_what_is_not_a_rectangle(text: str) -> None:
    with pytest.raises(ValueError):
        GameArea.parse(text)


def test_a_box_of_fractions_lands_inside_the_game() -> None:
    assert RECEIVER.box(0.0, 0.0, 1.0, 1.0) == (1, 31, 2115, 1322)
    assert RECEIVER.box(0.5, 0.5, 1.0, 1.0) == (1 + 1057, 31 + 661, 1058, 661)


def test_the_area_round_trips_as_the_meta_writes_it() -> None:
    assert GameArea.from_dict(RECEIVER.to_dict()) == RECEIVER


# -- fractional layouts measured against the game --------------------------

OLD_EXCLUDE = ((0.0, 0.0, 1.0, 0.035), (0.0, 0.0, 0.28, 0.36),
               (0.0, 0.78, 1.0, 1.0), (0.76, 0.6, 1.0, 1.0))
"""The nameplate exclusions as fractions of the frame, before they moved."""

NEW_EXCLUDE = ((0.0, 0.0, 1.0, 0.0124), (0.0, 0.0, 0.27979, 0.34526),
               (0.0, 0.77543, 1.0, 1.0), (0.76025, 0.59107, 1.0, 1.0))
"""The same regions as fractions of the receiver's game area, as `etc/`
now stores them."""


def test_the_converted_exclusions_test_the_same_interior_pixels() -> None:
    width, height = FRAME
    for old, new in zip(OLD_EXCLUDE, excluded_regions(NEW_EXCLUDE, *FRAME, RECEIVER)):
        before = (math.ceil(old[0] * width), math.ceil(old[1] * height),
                  math.floor(old[2] * width), math.floor(old[3] * height))
        after = (math.ceil(new[0]), math.ceil(new[1]),
                 math.floor(new[2]), math.floor(new[3]))
        # Edges on the frame border now stop at the game's; the border and
        # title bar are excluded by the strips below instead.
        for b, a, limit in zip(before, after, (0, 0, width, height)):
            if b not in (0, limit):
                assert a == b


def test_the_window_chrome_is_never_read_as_game() -> None:
    regions = excluded_regions((), *FRAME, RECEIVER)

    def excluded(x: int, y: int) -> bool:
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in regions)

    assert excluded(500, 0) and excluded(500, 30)  # the title bar
    assert excluded(0, 600)  # the left border
    assert excluded(2116, 600)  # the right border
    assert excluded(500, 1353)  # the bottom border
    assert not excluded(500, 31) and not excluded(1, 600)


def test_the_playfield_leaves_out_the_title_bar() -> None:
    field = Playfield.of((), *FRAME, left=0, right=0, above=0, below=0,
                         area=RECEIVER)
    assert field.rows == (31, 31 + 1322)


def test_with_no_area_the_exclusions_are_fractions_of_the_frame() -> None:
    assert excluded_regions(((0.0, 0.0, 0.5, 0.5),), 100, 80) == [
        (0.0, 0.0, 50.0, 40.0)
    ]


# -- the window source measures once, with the size -------------------------


def build_source(hwnd: int | None) -> WindowSource:
    source = WindowSource.__new__(WindowSource)
    source.title = "test"
    source.hwnd = hwnd
    source.startup_timeout = 1.0
    source._mailbox = Mailbox()
    source._size = None
    source._game_area = None
    source._control = None
    source._first = None
    return source


def test_the_window_source_measures_the_client_area_with_the_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = []

    def measure(hwnd: int, size: tuple[int, int]) -> GameArea:
        asked.append((hwnd, size))
        return RECEIVER

    monkeypatch.setattr(window_module, "window_game_area", measure)
    source = build_source(hwnd=42)
    source._mailbox.put(np.zeros((1354, 2117, 4), dtype=np.uint8))
    assert source.game_area == RECEIVER
    assert source.game_area == RECEIVER
    assert asked == [(42, FRAME)]


def test_a_window_windows_cannot_measure_is_all_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_module, "window_game_area", lambda *_: None)
    source = build_source(hwnd=42)
    source._mailbox.put(np.zeros((1354, 2117, 4), dtype=np.uint8))
    assert source.game_area == GameArea.whole(*FRAME)


def test_with_no_window_handle_nothing_is_measured() -> None:
    source = build_source(hwnd=None)
    source._mailbox.put(np.zeros((6, 4, 4), dtype=np.uint8))
    assert source.game_area == GameArea.whole(4, 6)
