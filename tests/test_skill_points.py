"""Reading whether a skill point is waiting off the level-up chevrons.

The pixel path -- what a lit chevron correlates at, how much gold it holds --
is decided by real footage and written up in the module docstring. What these
tests pin is the state machine around it: a slot changes state only once two
readings agree, an abstaining reading neither advances nor breaks a change,
and the reader says nothing at all until every slot has settled.

Frames are painted: the template is a chevron drawn onto a dark box, a lit
slot is that same drawing in gold, an unlit slot is a flat dark box, and the
"unavailable" chevron is the drawing in grey -- the shape without the colour,
which is the one look-alike the colour gate exists for.
"""

from __future__ import annotations

import cv2
import numpy as np

from spectral_sight.perception.hud.abilities import AbilityLayout
from spectral_sight.perception.hud.skill_points import (
    SkillPointConfig,
    SkillPointReader,
    load_skill_point_reader,
)

LAYOUT = AbilityLayout(
    ability_first_x=10, ability_y=50, ability_width=40, ability_height=40,
    ability_spacing=50,
    summoner_first_x=250, summoner_y=50, summoner_width=30, summoner_height=30,
    summoner_spacing=40,
    point_y=10, point_height=32,
)
FRAME_W, FRAME_H = 320, 100

GOLD = (24, 220, 230)   # HSV: inside the gold band
GREY = (0, 0, 120)      # no saturation: the unavailable chevron's colour


def _bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    patch = np.array([[list(hsv)]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0])


def chevron(width: int, height: int, colour: tuple[int, int, int]) -> np.ndarray:
    """A box with a chevron drawn on it in `colour`, on a dark panel."""
    box = np.full((height, width, 3), 30, dtype=np.uint8)
    pts = np.array([
        [width * 0.2, height * 0.7], [width * 0.5, height * 0.25],
        [width * 0.8, height * 0.7],
    ], dtype=np.int32)
    cv2.polylines(box, [pts], False, _bgr(colour), thickness=max(2, height // 6))
    return box


TEMPLATE = cv2.cvtColor(chevron(40, 32, GOLD), cv2.COLOR_BGR2GRAY)


def frame_with(**slots: str) -> np.ndarray:
    """A frame whose named chevron boxes are 'lit', 'grey' or 'noise';
    unnamed boxes are flat dark (unlit)."""
    frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
    boxes = LAYOUT.point_boxes()
    assert boxes is not None
    rng = np.random.default_rng(0)
    for name, (x, y, w, h) in boxes.items():
        look = slots.get(name, "unlit")
        if look == "lit":
            frame[y : y + h, x : x + w] = chevron(w, h, GOLD)
        elif look == "grey":
            frame[y : y + h, x : x + w] = chevron(w, h, GREY)
        elif look == "noise":
            frame[y : y + h, x : x + w] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return frame


def reader(**config) -> SkillPointReader:
    """Frames arrive 0.1s apart, so a 0.15s settle reports a set on the
    second reading after it first appears -- short enough to drive by hand,
    long enough to test, and clear of 0.2 - 0.1 - 0.1 landing under 0.2 in
    floating point."""
    config.setdefault("settle", 0.15)
    return SkillPointReader(LAYOUT, TEMPLATE, SkillPointConfig(**config))


def drive(r: SkillPointReader, looks: list[dict]) -> list:
    return [r.read(frame_with(**look), i * 0.1) for i, look in enumerate(looks)]


def test_nothing_is_said_until_the_row_has_settled() -> None:
    """Two readings to settle each slot, then the set must hold for
    `settle` -- here two more readings."""
    out = drive(reader(), [{}, {}, {}, {}])
    assert out == [None, None, None, ()]


def test_a_lit_chevron_is_read_once_the_slots_and_the_set_agree() -> None:
    lit = {"Q": "lit", "W": "lit"}
    out = drive(reader(), [{}] * 4 + [lit] * 5)
    assert out[3:] == [(), (), (), (), ("Q", "W"), ("Q", "W")]


def test_a_one_frame_flash_changes_nothing() -> None:
    """The spend flash shows up for a single reading."""
    out = drive(reader(), [{}] * 4 + [{"Q": "lit"}, {}, {}])
    assert out[3:] == [(), (), (), ()]


def test_a_staggered_slide_in_is_one_change_not_three() -> None:
    """The row slides in left to right over successive readings; the sets
    on the way -- Q, then QW -- never last long enough to be reported."""
    q, qw, qwe = {"Q": "lit"}, {"Q": "lit", "W": "lit"}, {"Q": "lit", "W": "lit", "E": "lit"}
    out = drive(reader(), [{}] * 4 + [q, q, qw, qwe, qwe, qwe, qwe])
    reported = [o for o in out if o]
    assert reported and all(o == ("Q", "W", "E") for o in reported)


def test_a_staggered_slide_out_is_one_change_not_two() -> None:
    qwe, qe = {"Q": "lit", "W": "lit", "E": "lit"}, {"Q": "lit", "E": "lit"}
    out = drive(reader(), [qwe] * 4 + [qe, {}, {}, {}, {}, {}])
    assert ("Q", "E") not in out
    assert out[-1] == ()


def test_the_grey_unavailable_chevron_is_not_lit() -> None:
    """Same shape, no gold: R before level 6."""
    out = drive(reader(), [{"W": "lit", "R": "grey"}] * 4)
    assert out[-1] == ("W",)


def test_the_world_showing_through_is_not_lit() -> None:
    out = drive(reader(), [{"E": "noise"}] * 4)
    assert out[-1] == ()


def test_an_abstention_does_not_break_a_change_in_progress() -> None:
    """A reading between the thresholds is neither vote; the change already
    under way still lands on the next agreeing frame."""
    r = reader(lit_above=0.9, unlit_below=0.1)
    drive(r, [{}] * 4)
    assert r.learnable == ()
    r.read(frame_with(Q="lit"), 0.4)
    # A chevron half-buried in noise: correlates, but not to 0.9.
    muddy = frame_with()
    x, y, w, h = LAYOUT.point_boxes()["Q"]
    noise = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    muddy[y : y + h, x : x + w] = cv2.addWeighted(chevron(w, h, GOLD), 0.5, noise, 0.5, 0)
    assert 0.1 < r._score(muddy[y : y + h, x : x + w]) < 0.9
    r.read(muddy, 0.5)
    assert r._adopted() == ()
    r.read(frame_with(Q="lit"), 0.6)
    assert r._adopted() == ("Q",)


def test_spending_the_point_clears_the_set() -> None:
    out = drive(reader(), [{"Q": "lit"}] * 4 + [{}] * 4)
    assert out[3] == ("Q",) and out[-1] == ()


def test_reset_forgets_everything() -> None:
    r = reader()
    drive(r, [{"Q": "lit"}] * 4)
    assert r.learnable == ("Q",)
    r.reset()
    assert r.learnable is None
    assert r.read(frame_with(), 1.0) is None


def test_a_template_of_another_size_is_rescaled_to_the_box() -> None:
    big = cv2.resize(TEMPLATE, (80, 64), interpolation=cv2.INTER_CUBIC)
    r = SkillPointReader(LAYOUT, big, SkillPointConfig(settle=0.15))
    assert drive(r, [{"E": "lit"}] * 4)[-1] == ("E",)


def test_a_layout_without_the_row_gets_no_reader() -> None:
    bare = AbilityLayout(
        ability_first_x=10, ability_y=50, ability_width=40, ability_height=40,
        ability_spacing=50, summoner_first_x=250, summoner_y=50,
        summoner_width=30, summoner_height=30, summoner_spacing=40,
    )
    assert bare.point_boxes() is None
    assert load_skill_point_reader(bare) is None


def test_the_row_round_trips_through_the_layout_file() -> None:
    assert AbilityLayout.from_dict(LAYOUT.to_dict()) == LAYOUT
    bare = LAYOUT.to_dict()
    del bare["point_y"], bare["point_height"]
    assert AbilityLayout.from_dict(bare).point_boxes() is None
