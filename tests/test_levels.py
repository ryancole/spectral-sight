"""Holding a champion's level steady across misreads.

The filter is pure logic over integers, so these tests are the whole story --
there is no image here to be right or wrong about. What real footage decides is
the *rate* of misreads the filter has to absorb, which
`tools/calibrate_nameplates.py --validate` reports.
"""

from __future__ import annotations

from spectral_sight.perception.nameplates import LevelBook, LevelFilter


def feed(filter_: LevelFilter, readings: list[int | None]) -> list[int | None]:
    return [filter_.update(reading) for reading in readings]


# -- starting up ----------------------------------------------------------


def test_starts_with_no_level() -> None:
    assert LevelFilter().level is None


def test_a_single_reading_is_not_enough() -> None:
    """A series that adopts its first reading adopts a misread just as readily."""
    assert LevelFilter().update(3) is None


def test_adopts_a_level_once_confirmed() -> None:
    state = LevelFilter()
    state.update(3)
    assert state.update(3) == 3


def test_disagreeing_first_readings_restart_the_count() -> None:
    state = LevelFilter()
    assert feed(state, [3, 5, 3]) == [None, None, None]
    assert state.update(3) == 3


# -- holding through misreads ---------------------------------------------


def test_ignores_a_reading_that_decreases() -> None:
    """The failure that actually happens: a '3' read as a '1'."""
    state = LevelFilter()
    feed(state, [3, 3])
    assert state.update(1) == 3
    assert state.update(3) == 3


def test_ignores_a_single_spurious_low_reading_in_a_run() -> None:
    state = LevelFilter()
    feed(state, [4, 4])
    assert feed(state, [4, 1, 4, 1, 4]) == [4, 4, 4, 4, 4]


def test_ignores_a_jump_of_more_than_one() -> None:
    state = LevelFilter()
    feed(state, [3, 3])
    assert state.update(9) == 3


def test_a_missing_reading_changes_nothing() -> None:
    state = LevelFilter()
    feed(state, [3, 3])
    assert state.update(None) == 3


def test_none_before_anything_is_known_stays_none() -> None:
    assert LevelFilter().update(None) is None


# -- levelling up ---------------------------------------------------------


def test_accepts_a_confirmed_level_up() -> None:
    state = LevelFilter()
    feed(state, [4, 4])
    assert state.update(5) == 4
    assert state.update(5) == 5


def test_a_lone_reading_one_above_does_not_level_up() -> None:
    """A misread landing exactly one above would otherwise stick for good."""
    state = LevelFilter()
    feed(state, [4, 4])
    assert state.update(5) == 4
    assert state.update(4) == 4
    assert state.update(4) == 4


def test_climbs_one_level_at_a_time() -> None:
    state = LevelFilter()
    feed(state, [1, 1])
    for level in range(2, 7):
        assert state.update(level) == level - 1
        assert state.update(level) == level


# -- resynchronising ------------------------------------------------------


def test_adopts_a_sustained_jump_upward() -> None:
    """A champion seen again after time off screen has levelled more than once,
    and refusing that forever would pin the level at whatever it last saw."""
    state = LevelFilter()
    feed(state, [3, 3])
    assert state.update(9) == 3
    readings = [9] * 6
    assert feed(state, readings)[-1] == 9


def test_does_not_adopt_a_sustained_decrease() -> None:
    """Levels do not fall, so agreement is not evidence -- it is a stuck glyph."""
    state = LevelFilter()
    feed(state, [7, 7])
    assert feed(state, [2] * 10)[-1] == 7


# -- the book -------------------------------------------------------------


def test_book_keeps_a_level_per_track() -> None:
    book = LevelBook()
    for _ in range(2):
        book.update(1, 3)
        book.update(2, 8)
    assert book.level(1) == 3
    assert book.level(2) == 8


def test_book_returns_none_for_an_unseen_track() -> None:
    assert LevelBook().level(99) is None


def test_book_forgets_a_dropped_track() -> None:
    book = LevelBook()
    book.update(1, 3)
    book.update(1, 3)
    book.forget(1)
    assert book.level(1) is None


def test_book_does_not_leak_between_tracks() -> None:
    """Two champions on different levels must not confirm each other."""
    book = LevelBook()
    assert book.update(1, 4) is None
    assert book.update(2, 4) is None
    assert book.update(1, 4) == 4
    assert book.level(2) is None
