"""Reading casts out of a champion's resource series.

The detector is pure logic over floats, so these tests are the whole story --
there is no image here to be right or wrong about. What real footage decides is
the *rate* of drops the detector has to judge and how many of them arrive across
a gap, which `tools/detect_casts.py` reports.

Series are written as (video_time, resource) pairs at 0.1s spacing, which is the
10 Hz the pipeline samples at. None stands for a frame where the plate was not
readable.
"""

from __future__ import annotations

from spectral_sight.perception.nameplates import Cast, CastBook, CastConfig
from spectral_sight.perception.nameplates.casts import CastDetector


def feed(
    detector: CastDetector,
    series: list[tuple[float, float | None]],
    level: int | None = 3,
) -> list[Cast]:
    """Fold a whole series in and collect whatever settled."""
    casts = [
        detector.update(time, resource, None, level) for time, resource in series
    ]
    return [cast for cast in casts if cast is not None]


def steady(start: float, value: float, count: int) -> list[tuple[float, float]]:
    """`count` readings of an unchanging bar at 10 Hz."""
    return [(round(start + 0.1 * i, 2), value) for i in range(count)]


# -- a step that holds ----------------------------------------------------


def test_a_clean_step_is_a_cast() -> None:
    detector = CastDetector(track_id=1)
    casts = feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.72, 3))
    assert len(casts) == 1
    assert round(casts[0].drop, 2) == 0.18
    assert casts[0].resource_before == 0.90
    assert casts[0].resource_after == 0.72


def test_the_cast_is_stamped_when_the_drop_happened() -> None:
    """Not when it settled -- emission waits a reading to see the drop hold."""
    detector = CastDetector(track_id=1)
    casts = feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.72, 3))
    assert casts[0].at == 0.2


def test_a_step_within_a_continuous_series_is_confirmed() -> None:
    detector = CastDetector(track_id=1)
    casts = feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.72, 3))
    assert casts[0].confirmed
    assert casts[0].continuous


def test_two_casts_in_succession_are_both_reported() -> None:
    """A further drop is itself a hold: it does not rebound."""
    detector = CastDetector(track_id=1)
    casts = feed(
        detector,
        steady(0.0, 0.90, 2) + steady(0.2, 0.72, 2) + steady(0.4, 0.55, 2),
    )
    assert [round(cast.drop, 2) for cast in casts] == [0.18, 0.17]


# -- noise ----------------------------------------------------------------


def test_a_fall_below_the_threshold_is_not_a_cast() -> None:
    """One pixel of a 117px bar is 0.85%, and the bar jitters by about that."""
    detector = CastDetector(track_id=1)
    assert feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.885, 3)) == []


def test_a_fall_that_rebounds_is_rejected() -> None:
    """The whole point of holding a candidate: a misread reverts, a cast does not."""
    detector = CastDetector(track_id=1)
    series = steady(0.0, 0.90, 2) + [(0.2, 0.72)] + steady(0.3, 0.90, 3)
    assert feed(detector, series) == []


def test_a_rebound_within_tolerance_still_counts_as_holding() -> None:
    """A pixel of drift back up is not the bar refilling."""
    detector = CastDetector(track_id=1)
    series = steady(0.0, 0.90, 2) + [(0.2, 0.72), (0.3, 0.728)]
    assert len(feed(detector, series)) == 1


def test_regeneration_does_not_accumulate_into_a_cast() -> None:
    """Regen only ever pushes the series up, so it cannot make a step down."""
    detector = CastDetector(track_id=1)
    series = [(round(0.1 * i, 2), 0.50 + 0.0005 * i) for i in range(60)]
    assert feed(detector, series) == []


# -- gaps -----------------------------------------------------------------


def test_a_drop_across_a_gap_is_reported_but_not_continuous() -> None:
    """Most of the evidence arrives this way, so discarding it is not an option."""
    detector = CastDetector(track_id=1)
    casts = feed(detector, [(0.0, 0.90), (8.0, 0.60), (8.1, 0.60)])
    assert len(casts) == 1
    assert not casts[0].continuous
    assert casts[0].span == 8.0
    assert round(casts[0].drop, 2) == 0.30


def test_unreadable_frames_do_not_break_the_series() -> None:
    """None is a frame that was not looked at, not a reading of zero."""
    detector = CastDetector(track_id=1)
    series: list[tuple[float, float | None]] = [
        (0.0, 0.90), (0.1, None), (0.2, None), (0.3, 0.72), (0.4, 0.72),
    ]
    casts = feed(detector, series)
    assert len(casts) == 1
    assert casts[0].continuous, "0.3s of holes is still one continuous stretch"


def test_a_drop_with_no_follow_up_is_emitted_unconfirmed() -> None:
    """The champion walked off screen. That is not evidence against the cast."""
    detector = CastDetector(track_id=1)
    feed(detector, steady(0.0, 0.90, 2) + [(0.2, 0.72)])
    cast = detector.flush()
    assert cast is not None
    assert not cast.confirmed


def test_a_rebound_after_a_gap_does_not_reject_the_cast() -> None:
    """After seconds away a higher reading is as likely regen as a bad read."""
    detector = CastDetector(track_id=1)
    casts = feed(detector, [(0.0, 0.90), (0.1, 0.72), (9.0, 0.85)])
    assert len(casts) == 1
    assert not casts[0].confirmed


# -- levels ---------------------------------------------------------------


def test_a_level_up_alongside_a_drop_does_not_suppress_it() -> None:
    """Levelling grants current resource along with maximum, so the fraction
    holds or rises -- it cannot fall, and so cannot fake a cast. Measured on the
    sample clip the change at a level-up sat inside the noise floor."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.90, None, 3),
        detector.update(0.1, 0.72, None, 4),
        detector.update(0.2, 0.72, None, 4),
    ]
    settled = [cast for cast in casts if cast is not None]
    assert len(settled) == 1
    assert settled[0].level == 4


# -- the two ways a fake step gets made -----------------------------------
#
# Neither is visible in the resource series on its own, which is why `health`
# is passed in at all. Both were found by running the detector over a real clip
# and asking what the five worst casts had in common -- see
# `tools/detect_casts.py`.


def test_a_truncated_plate_is_not_read_as_a_cast() -> None:
    """A bar cut partway truncates *both* fills at the same column, so they
    come back a pixel apart. On the sample clip that accounted for three of the
    five false casts."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.98, 0.78),
        detector.update(0.1, 0.34, 0.33),
        detector.update(0.2, 0.34, 0.33),
    ]
    assert [cast for cast in casts if cast is not None] == []


def test_a_truncated_reading_does_not_break_the_series_either_side() -> None:
    """It is skipped, not merely rejected: measured *from*, it would invent a
    step back up that hides the real cast after it."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.98, 0.78),
        detector.update(0.1, 0.34, 0.33),   # truncated, skipped
        detector.update(0.2, 0.80, 0.78),   # the real cast, measured from 0.98
        detector.update(0.3, 0.80, 0.78),
    ]
    settled = [cast for cast in casts if cast is not None]
    assert len(settled) == 1
    assert round(settled[0].drop, 2) == 0.18


def test_a_full_bar_reading_equal_on_both_is_believed() -> None:
    """Full health and full mana is the commonest state in the game, and it
    reads equal on both bars without being truncated."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.99, 0.99),
        detector.update(0.1, 0.80, 0.99),
        detector.update(0.2, 0.80, 0.99),
    ]
    assert len([cast for cast in casts if cast is not None]) == 1


def test_both_bars_stepping_together_is_the_plate_changing_champion() -> None:
    """Association is geometric and occasionally lands on the wrong track. The
    next reading is then a different champion's bars, and both fills move by
    nearly the same amount at once."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.99, 0.82),
        detector.update(0.1, 0.36, 0.18),
        detector.update(0.2, 0.36, 0.18),
    ]
    assert [cast for cast in casts if cast is not None] == []


def test_casting_while_taking_damage_is_still_a_cast() -> None:
    """The guard is on the two falls *agreeing*, not on damage happening.
    Champions cast while being hit constantly, and rejecting those would throw
    away the fights, which is when abilities actually get used."""
    detector = CastDetector(track_id=1)
    casts = [
        detector.update(0.0, 0.90, 0.80),
        detector.update(0.1, 0.72, 0.55),   # -18% resource, -25% health
        detector.update(0.2, 0.72, 0.55),
    ]
    assert len([cast for cast in casts if cast is not None]) == 1


def test_health_is_optional() -> None:
    """A caller with no health reading gets the plain series behaviour rather
    than an error, since both guards are refinements on top of it."""
    detector = CastDetector(track_id=1)
    casts = feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.72, 3))
    assert len(casts) == 1


# -- the book -------------------------------------------------------------


def test_tracks_are_kept_apart() -> None:
    book = CastBook()
    book.update(1, 0.0, 0.90)
    book.update(2, 0.0, 0.20)
    assert book.update(1, 0.1, 0.72) is None
    assert book.update(2, 0.1, 0.20) is None
    assert book.update(1, 0.2, 0.72) is not None


def test_forgetting_a_track_releases_its_candidate() -> None:
    book = CastBook()
    book.update(1, 0.0, 0.90)
    book.update(1, 0.1, 0.72)
    cast = book.forget(1)
    assert cast is not None and not cast.confirmed


def test_a_reused_track_id_does_not_inherit_a_series() -> None:
    """Otherwise the new champion's first reading reads as one enormous step."""
    book = CastBook()
    book.update(1, 0.0, 0.90)
    book.forget(1)
    assert book.update(1, 5.0, 0.20) is None
    assert book.update(1, 5.1, 0.20) is None


def test_forgetting_an_unknown_track_is_harmless() -> None:
    assert CastBook().forget(99) is None


# -- configuration --------------------------------------------------------


def test_the_threshold_is_configurable() -> None:
    detector = CastDetector(track_id=1, config=CastConfig(min_drop=0.10))
    assert feed(detector, steady(0.0, 0.90, 2) + steady(0.2, 0.85, 3)) == []


def test_the_continuity_window_is_configurable() -> None:
    detector = CastDetector(track_id=1, config=CastConfig(continuity=2.0))
    casts = feed(detector, [(0.0, 0.90), (1.5, 0.60), (1.6, 0.60)])
    assert casts[0].continuous
