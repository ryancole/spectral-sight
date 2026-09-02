"""Reading the local player's ability casts off the HUD cooldown veil.

The pixel path -- what the veil looks like, where the band sits -- is decided by
real footage and reported by `tools/detect_abilities.py`: measured on the
2026-08-30 clip the reader catches 90% of the mana-fall casts with a
false-positive rate near one percent. What these tests pin is the state machine
that turns a stream of veil fractions into casts, which is where the reader goes
wrong in ways no threshold would show -- a cast counted twice, a mid-cooldown
flash read as a fresh cast, a candidate confirmed before the veil held.

Frames are painted rather than captured: a slot filled with the veil's blue
reads fully veiled, one filled with an out-of-band colour reads clear. That is
enough to exercise every transition without a single real pixel, the same way
the synthetic minimaps pin the tracker's geometry.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.abilities import (
    AbilityConfig,
    AbilityLayout,
    AbilityReader,
)
from spectral_sight.perception.hud.clock import (
    GlyphSet,
    glyph_boxes,
    segment_glyphs,
)

# A compact layout so a test frame is small: two ability slots at 10 and 70,
# the machinery is per-slot and indifferent to how many there are.
LAYOUT = AbilityLayout(
    ability_first_x=10, ability_y=10, ability_width=40, ability_height=40,
    ability_spacing=60,
    summoner_first_x=250, summoner_y=10, summoner_width=30, summoner_height=30,
    summoner_spacing=40,
)
FRAME_W, FRAME_H = 320, 60

VEIL = (105, 220, 150)   # HSV inside the band: hue 98-112, sat >= 150
READY = (24, 230, 150)   # gold, hue well outside the band -> reads clear
PARTIAL = "partial"      # a slot half in the band: a fraction in the deadband


def _bgr(hsv: tuple[int, int, int]) -> np.ndarray:
    patch = np.array([[list(hsv)]], dtype=np.uint8)
    return cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]


def frame_with(**slots: tuple[int, int, int] | str) -> np.ndarray:
    """A frame whose named slots hold the given colour; others read clear.

    A slot value of PARTIAL paints half the box veil-blue and half gold, which
    lands the veil fraction near 0.5 -- in the deadband between clear and
    veiled, the shape a shrinking cooldown wedge passes through.
    """
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    boxes = LAYOUT.boxes()
    for name, box in boxes.items():
        x, y, w, h = box
        value = slots.get(name, READY)
        if value == PARTIAL:
            frame[y : y + h, x : x + w] = _bgr(READY)
            frame[y : y + h, x : x + w // 3] = _bgr(VEIL)
        else:
            frame[y : y + h, x : x + w] = _bgr(value)
    return frame


def drive(reader: AbilityReader, sequence: list[tuple[float, dict]]) -> list:
    """Feed (timestamp, {slot: hsv}) frames, collect every settled cast."""
    casts = []
    for timestamp, slots in sequence:
        casts.extend(reader.read(frame_with(**slots), timestamp))
    return casts


def ready_then_veil(veil_from: float, step: float = 0.1) -> list[tuple[float, dict]]:
    """Clear long enough to arm, then a held veil -- the shape of one cast."""
    seq = [(round(t * step, 3), {}) for t in range(4)]  # 0.0..0.3 clear -> armed
    seq += [(round((4 + t) * step, 3), {"Q": VEIL}) for t in range(4)]
    return seq


def test_a_held_veil_after_ready_is_one_cast() -> None:
    casts = drive(AbilityReader(LAYOUT, glyphs=None), ready_then_veil(0.4))
    assert [c.slot for c in casts] == ["Q"]
    assert casts[0].confirmed
    # `at` is the first veiled reading, not the confirming one after it.
    assert casts[0].at == 0.4


def test_the_veil_persisting_does_not_re_fire() -> None:
    """A cooldown veil lasts seconds; the cast is the edge, counted once."""
    seq = ready_then_veil(0.4)
    seq += [(round((8 + t) * 0.1, 3), {"Q": VEIL}) for t in range(30)]
    casts = drive(AbilityReader(LAYOUT, glyphs=None), seq)
    assert [c.slot for c in casts] == ["Q"]


def test_a_one_frame_veil_flash_is_not_a_cast() -> None:
    """A projectile crossing the slot veils it for a single reading; without a
    holding follow-up the candidate is dropped, not emitted."""
    seq = [(round(t * 0.1, 3), {}) for t in range(4)]
    seq += [(0.4, {"Q": VEIL}), (0.5, {})]  # veil one frame, then clear
    casts = drive(AbilityReader(LAYOUT, glyphs=None), seq)
    assert casts == []


def test_a_mid_cooldown_flash_does_not_re_arm() -> None:
    """The failure the ready-hold exists for: an ability already veiled flashes
    clear for a frame or two when re-pressed, then the cooldown veil returns.
    That brief clear must not arm a second cast."""
    seq = ready_then_veil(0.4)  # one real cast, veiled from 0.4
    # two clear frames (a 0.1s flash) then veil resumes -- no genuine ready
    seq += [(0.8, {"Q": VEIL}), (0.9, {}), (1.0, {}),
            (1.1, {"Q": VEIL}), (1.2, {"Q": VEIL})]
    casts = drive(AbilityReader(LAYOUT, glyphs=None), seq)
    assert [c.slot for c in casts] == ["Q"]


def test_a_dead_band_reading_breaks_the_ready_run() -> None:
    """A reading that is neither clear nor veiled -- the veil washing out to low
    saturation -- must not count toward the ready hold."""
    reader = AbilityReader(LAYOUT, glyphs=None)
    # clear, clear, dead-band, clear, then veil: the run is broken at the
    # dead-band frame, so by the veil only 0.1s of clear has accrued -- under
    # the 0.25s hold -- and nothing arms.
    seq = [(0.0, {}), (0.1, {}), (0.2, {"Q": PARTIAL}), (0.3, {}),
           (0.4, {"Q": VEIL}), (0.5, {"Q": VEIL})]
    casts = drive(reader, seq)
    assert casts == []


def test_a_genuine_return_to_ready_allows_a_second_cast() -> None:
    """Cooldown ends, the slot is clear for real, the ability is cast again."""
    seq = ready_then_veil(0.4)                      # first cast
    seq += [(round((8 + t) * 0.1, 3), {}) for t in range(4)]   # clear -> re-arm
    seq += [(round((12 + t) * 0.1, 3), {"Q": VEIL}) for t in range(3)]  # second
    casts = drive(AbilityReader(LAYOUT, glyphs=None), seq)
    assert [c.slot for c in casts] == ["Q", "Q"]


def test_slots_are_independent() -> None:
    """W going on cooldown says nothing about Q."""
    seq = [(round(t * 0.1, 3), {}) for t in range(4)]
    seq += [(round((4 + t) * 0.1, 3), {"W": VEIL}) for t in range(3)]
    casts = drive(AbilityReader(LAYOUT, glyphs=None), seq)
    assert [c.slot for c in casts] == ["W"]


def test_reset_forgets_a_pending_candidate() -> None:
    """After a tear the held candidate must not confirm against new footage."""
    reader = AbilityReader(LAYOUT, glyphs=None)
    drive(reader, ready_then_veil(0.4)[:5])  # armed, one veiled frame: pending
    reader.reset()
    # a veiled frame that would have confirmed the pending candidate
    casts = reader.read(frame_with(Q=VEIL), 0.6)
    assert casts == []


def test_flush_releases_a_pending_candidate_unconfirmed() -> None:
    reader = AbilityReader(LAYOUT, glyphs=None)
    seq = [(round(t * 0.1, 3), {}) for t in range(4)] + [(0.4, {"Q": VEIL})]
    settled = drive(reader, seq)
    assert settled == []          # not yet confirmed
    released = reader.flush()
    assert [c.slot for c in released] == ["Q"]
    assert released[0].confirmed is False


def test_no_glyphs_means_no_countdown_but_still_a_cast() -> None:
    casts = drive(AbilityReader(LAYOUT, glyphs=None), ready_then_veil(0.4))
    assert casts[0].countdown is None


# -- the countdown, read with the clock's glyphs ---------------------------
#
# The digits on the veil are the timer's face rescaled, matched the way the
# resource text is. The pixel accuracy is a footage question; what is pinned
# here is that a digit centred on the veil is read, and one outside the band
# (the corner cost label) is not mistaken for the countdown.

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _digit(char: str, scale: float = 0.9, thickness: int = 2) -> np.ndarray:
    (w, h), baseline = cv2.getTextSize(char, FONT, scale, thickness)
    image = np.zeros((h + baseline + 4, w + 4, 3), np.uint8)
    cv2.putText(image, char, (2, h + 1), FONT, scale, (240, 240, 240),
                thickness, cv2.LINE_AA)
    return image


@pytest.fixture(scope="module")
def glyphs() -> GlyphSet:
    strips = {str(d): _digit(str(d)) for d in range(10)}
    width = height = 0
    for strip in strips.values():
        for _, _, w, h in glyph_boxes(strip):
            width, height = max(width, w), max(height, h)
    size = (width + 2, height + 2)
    return GlyphSet(
        glyphs={d: segment_glyphs(s, size)[0] for d, s in strips.items()},
        size=size,
    )


def _veil_with_digit(char: str, in_band: bool = True) -> np.ndarray:
    """A Q slot fully veiled with a white digit painted on it.

    In-band: a full-size digit centred, the countdown. Out-of-band: a small
    digit tucked into the top corner, the mana-cost label -- above the vertical
    band the countdown is read from.
    """
    frame = frame_with(Q=VEIL)
    x, y, w, h = LAYOUT.boxes()["Q"]
    glyph = _digit(char, scale=0.9 if in_band else 0.45)
    if in_band:
        gy = y + (h - glyph.shape[0]) // 2
        gx = x + (w - glyph.shape[1]) // 2
    else:
        gy, gx = y + 2, x + w - glyph.shape[1] - 2
    window = frame[gy : gy + glyph.shape[0], gx : gx + glyph.shape[1]]
    np.maximum(window, glyph, out=window)
    return frame


def test_reads_the_countdown_off_the_veil(glyphs: GlyphSet) -> None:
    reader = AbilityReader(LAYOUT, glyphs=glyphs)
    for t in range(4):
        reader.read(frame_with(), round(t * 0.1, 3))     # arm
    reader.read(_veil_with_digit("5"), 0.4)              # onset, pending
    settled = reader.read(_veil_with_digit("5"), 0.5)    # confirm
    assert [c.countdown for c in settled] == [5]


def test_a_digit_outside_the_band_is_not_the_countdown(glyphs: GlyphSet) -> None:
    """The mana-cost label sits in the top corner; it must not be read as
    seconds. A veil with only a corner digit yields a cast with no countdown."""
    reader = AbilityReader(LAYOUT, glyphs=glyphs)
    for t in range(4):
        reader.read(frame_with(), round(t * 0.1, 3))
    reader.read(_veil_with_digit("9", in_band=False), 0.4)
    settled = reader.read(_veil_with_digit("9", in_band=False), 0.5)
    assert [c.slot for c in settled] == ["Q"]
    assert settled[0].countdown is None


def test_layout_boxes_are_evenly_spaced() -> None:
    boxes = LAYOUT.boxes()
    assert set(boxes) == {"Q", "W", "E", "R", "D", "F"}
    assert boxes["Q"][0] == 10 and boxes["W"][0] == 70
    assert boxes["D"][0] == 250 and boxes["F"][0] == 290


def test_layout_round_trips_through_dict() -> None:
    assert AbilityLayout.from_dict(LAYOUT.to_dict()) == LAYOUT
