"""Reading the player's own health and mana off the HUD.

The pixel path is decided by real footage and reported by
`tools/validate_casts.py` -- measured on a 5.3-minute clip the mana line reads
on 100% of frames with its maximum never once stepping backwards across 3,187
readings. What these tests pin is the logic that turns a row of matched glyphs
into two numbers, which is where the reader can be wrong in a way no amount of
tuning would show.
"""

from __future__ import annotations

from spectral_sight.perception.hud.resources import (
    Reading,
    ResourceLayout,
    ResourceReader,
)


def boxes(*spans: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """(x, width) pairs as the (x, y, w, h) boxes the reader works in."""
    return [(x, 0, width, 10) for x, width in spans]


# -- finding the separator ------------------------------------------------
#
# `/` correlates with a digit template about as well as a digit does, so it
# cannot be found by matching. The gaps flanking it are 5-8px against 1-4px
# between digits, which is what the reader keys on instead.


def test_finds_the_separator_by_its_gaps() -> None:
    """`488 / 488`: three digits, a wide gap either side of the slash."""
    layout = boxes((0, 6), (8, 7), (16, 7), (28, 5), (39, 6), (47, 7), (55, 7))
    assert ResourceReader._separator(layout) == 3


def test_finds_the_separator_with_uneven_digit_counts() -> None:
    """`88 / 452` -- splitting at a fixed position would get this wrong."""
    layout = boxes((0, 6), (8, 7), (20, 5), (31, 6), (39, 7), (47, 7))
    assert ResourceReader._separator(layout) == 2


def test_the_separator_is_never_an_end_glyph() -> None:
    """A number cannot start or finish with a slash, so the ends are not
    candidates however wide the gap beside them."""
    layout = boxes((0, 6), (30, 5), (38, 6))
    assert ResourceReader._separator(layout) == 1


def test_too_few_glyphs_has_no_separator() -> None:
    assert ResourceReader._separator(boxes((0, 6), (8, 7))) is None
    assert ResourceReader._separator([]) is None


# -- what counts as a reading ---------------------------------------------


def test_current_above_maximum_is_a_misread() -> None:
    """The cheapest check there is, and it catches a transposed digit."""
    assert not Reading(current=4881, maximum=488).plausible
    assert Reading(current=488, maximum=488).plausible


def test_an_empty_pool_is_a_misread_not_a_state() -> None:
    """A champion can be at zero mana; the bar cannot have zero maximum."""
    assert not Reading(current=0, maximum=0).plausible
    assert Reading(current=0, maximum=488).plausible


def test_an_absurd_maximum_is_rejected() -> None:
    assert not Reading(current=1, maximum=999_999).plausible


def test_a_reading_converts_to_the_fraction_a_nameplate_would_give() -> None:
    """Which is the whole point of having it: the two are comparable."""
    assert Reading(current=244, maximum=488).fraction == 0.5


def test_a_zero_maximum_has_no_fraction() -> None:
    assert Reading(current=0, maximum=0).fraction is None


def test_a_reading_prints_the_way_the_hud_does() -> None:
    assert str(Reading(current=391, maximum=664)) == "391/664"


# -- calibration ----------------------------------------------------------


def test_a_layout_survives_a_round_trip(tmp_path) -> None:
    layout = ResourceLayout(health=(940, 1306, 130, 22), mana=(940, 1326, 130, 22))
    path = tmp_path / "2118x1354.json"
    layout.save(path)
    assert ResourceLayout.load(path) == layout


def test_an_uncalibrated_resolution_is_absent_rather_than_an_error() -> None:
    """Every other stage carries on without its calibration, and so does this
    one -- a run with no player numbers is worse than one with them, not
    broken."""
    assert ResourceLayout.for_resolution(1, 1) is None
