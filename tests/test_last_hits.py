"""The creep score, and minion deaths judged against it.

The reader is tested on digits rendered the way the nameplate tests render
them -- in yellow, since that is what separates the count from the clock. The
filter and the detector are pure logic over readings, so they are fed
readings directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.clock import (
    ClockReader,
    ClockRegion,
    GlyphSet,
    glyph_boxes,
    segment_glyphs,
)
from spectral_sight.perception.hud.creep_score import (
    CreepScoreFilter,
    CreepScoreReader,
)
from spectral_sight.perception.screen.last_hits import (
    LastHitConfig,
    LastHitDetector,
)

# -- the reader -----------------------------------------------------------

YELLOW = (60, 230, 240)


@pytest.fixture(scope="module")
def glyphs() -> GlyphSet:
    strips = {}
    for digit in "0123456789":
        strip = np.full((26, 20, 3), (24, 28, 26), np.uint8)
        cv2.putText(strip, digit, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (238, 238, 238), 2, cv2.LINE_AA)
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


def score_bar(count: str) -> np.ndarray:
    frame = np.full((80, 400, 3), (24, 28, 26), np.uint8)
    cv2.putText(frame, count, (205, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                YELLOW, 2, cv2.LINE_AA)
    return frame


def cs_reader(glyphs: GlyphSet) -> CreepScoreReader:
    clock = ClockReader(ClockRegion(x=306, y=45, width=47, height=20), glyphs)
    reader = CreepScoreReader.beside(clock)
    return CreepScoreReader(
        x=reader.x, y=reader.y, width=reader.width, height=26, glyphs=glyphs,
    )


@pytest.mark.parametrize("count", ["0", "7", "42", "108"])
def test_reads_the_yellow_count(glyphs: GlyphSet, count: str) -> None:
    assert cs_reader(glyphs).read(score_bar(count)) == int(count)


def test_an_empty_bar_reads_nothing(glyphs: GlyphSet) -> None:
    assert cs_reader(glyphs).read(score_bar("")) is None


# -- the filter -----------------------------------------------------------


def run(readings) -> list[int | None]:
    f = CreepScoreFilter()
    return [f.update(r) for r in readings]


def test_a_rise_of_one_needs_two_readings() -> None:
    assert run([5, 5, 6, 6, 6]) == [None, 5, 5, 6, 6]


def test_the_count_never_falls() -> None:
    assert run([5, 5, 4, 4, 4])[-1] == 5


def test_a_repeated_misread_does_not_jump_the_count() -> None:
    """Measured: "11" read as "14" twice running, between an 11 and a 12."""
    assert run([11, 11, 14, 14, 12, 12]) == [None, 11, 11, 11, 11, 12]


def test_a_wave_cleared_at_once_is_believed_once_it_holds() -> None:
    assert run([10, 10, 15, 15, 15, 15])[-1] == 15


def test_a_count_that_really_changed_resyncs() -> None:
    assert run([30, 30] + [3] * 6)[-1] == 3


def test_unreadable_frames_hold_the_count() -> None:
    assert run([5, 5, None, None])[-1] == 5


# -- the detector ---------------------------------------------------------

VIEW = (0, 0, 1000, 800)


def feed(detector, script):
    """`script` is a list of (time, minions, score); returns every result."""
    out = []
    for t, minions, score in script:
        detector.observe(t, minions, score)
        out += detector.resolve(t)
    return out + detector.resolve(1e9)


def dying(x, y, healths, start=0.0, step=0.1):
    """A minion at (x, y) whose health reads `healths` then disappears."""
    return [(start + i * step, [(x, y, h)]) for i, h in enumerate(healths)]


def with_score(frames, scores):
    return [(t, m, s) for (t, m), s in zip(frames, scores)]


def quiet(start, n, score, step=0.1):
    return [(start + i * step, [], score) for i in range(n)]


def test_a_low_minion_dying_with_the_score_rising_is_a_last_hit() -> None:
    frames = dying(500, 400, [0.6, 0.3, 0.1])
    script = with_score(frames, [3, 3, 3]) + quiet(0.3, 5, 3) + quiet(0.8, 20, 4)
    hits = feed(LastHitDetector(VIEW), script)
    assert [h.outcome for h in hits] == ["last_hit"]
    assert hits[0].at == pytest.approx(0.2)
    assert hits[0].health == pytest.approx(0.1)


def test_a_low_minion_dying_with_no_rise_is_missed() -> None:
    frames = dying(500, 400, [0.6, 0.3, 0.1])
    script = with_score(frames, [3, 3, 3]) + quiet(0.3, 30, 3)
    assert [h.outcome for h in feed(LastHitDetector(VIEW), script)] == ["missed"]


def test_no_score_means_unknown_not_missed() -> None:
    frames = dying(500, 400, [0.6, 0.3, 0.1])
    script = with_score(frames, [None] * 3) + quiet(0.3, 30, None)
    assert [h.outcome for h in feed(LastHitDetector(VIEW), script)] == ["unknown"]


def test_a_healthy_bar_vanishing_is_not_a_miss() -> None:
    """Most healthy bars that vanish were covered by another bar."""
    frames = dying(500, 400, [1.0, 0.9, 0.9])
    script = with_score(frames, [3, 3, 3]) + quiet(0.3, 30, 3)
    assert feed(LastHitDetector(VIEW), script) == []


def test_but_a_healthy_bar_vanishing_as_the_score_rises_is_an_ability_kill() -> None:
    frames = dying(500, 400, [1.0, 0.8, 0.6])
    script = with_score(frames, [3, 3, 3]) + quiet(0.3, 3, 3) + quiet(0.6, 20, 4)
    assert [h.outcome for h in feed(LastHitDetector(VIEW), script)] == ["last_hit"]


def test_a_bar_leaving_by_the_edge_is_not_a_death() -> None:
    frames = dying(20, 400, [0.3, 0.2, 0.1])
    script = with_score(frames, [3, 3, 3]) + quiet(0.3, 30, 3)
    assert feed(LastHitDetector(VIEW), script) == []


def test_a_bar_hidden_for_a_second_is_not_a_death() -> None:
    """Measured: a bar under a champion's translucent cape blinked out for
    1.1s and came back."""
    script = with_score(dying(500, 400, [0.3, 0.2, 0.2]), [3, 3, 3])
    script += quiet(0.3, 11, 3)
    script += [(1.4 + i * 0.1, [(505, 400, 0.2)], 3) for i in range(10)]
    assert feed(LastHitDetector(VIEW), script) == []


def test_one_point_of_score_credits_one_death() -> None:
    """Two minions dying together and one point gained: one hit, one miss."""
    script = [
        (0.0, [(300, 400, 0.2), (700, 400, 0.2)], 3),
        (0.1, [(300, 400, 0.1), (700, 400, 0.1)], 3),
    ] + quiet(0.2, 4, 3) + quiet(0.6, 30, 4)
    outcomes = sorted(h.outcome for h in feed(LastHitDetector(VIEW), script))
    assert outcomes == ["last_hit", "missed"]


def test_an_earlier_kill_off_screen_is_not_credited_to_a_later_miss() -> None:
    script = quiet(0.0, 5, 3) + quiet(0.5, 30, 4)
    script += with_score(dying(500, 400, [0.3, 0.2, 0.1], start=5.0), [4, 4, 4])
    script += quiet(5.3, 30, 4)
    assert [h.outcome for h in feed(LastHitDetector(VIEW), script)] == ["missed"]


def test_reset_forgets_the_bars() -> None:
    detector = LastHitDetector(VIEW)
    detector.observe(0.0, [(500, 400, 0.1)], 3)
    detector.observe(0.1, [(500, 400, 0.1)], 3)
    detector.reset()
    for i in range(30):
        detector.observe(0.2 + i * 0.1, [], 3)
    assert detector.resolve(1e9) == []


def test_config_defaults_are_the_measured_ones() -> None:
    cfg = LastHitConfig()
    assert cfg.low == 0.35 and cfg.settle == 1.5 and cfg.after == 1.0
