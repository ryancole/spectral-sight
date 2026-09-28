"""Measuring the HUD scale, and moving the panel's readers to it.

What the border score peaks at on real footage -- 1.00 on the calibrated
recordings, 1.50 on a VOD drawn larger -- is written up in the module
docstring. These tests pin the rest: the transform is about the bottom
centre, a panel painted at a known scale is measured at it, a frame with no
panel is not believed, and the watch adopts a scale only from a full window
of agreeing readings.

Frames are painted: the six slots as bright framed squares on a dark ground,
beside a long horizontal bar -- the health bar's shape, which a scorer that
only wanted horizontal edges would lock onto.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spectral_sight.perception.hud.abilities import AbilityLayout
from spectral_sight.perception.hud.alive import AliveReader
from spectral_sight.perception.hud.portraits import PortraitLayout
from spectral_sight.perception.hud.resources import ResourceLayout
from spectral_sight.perception.hud.scale import (
    HudScale,
    HudScaleConfig,
    HudScaleWatch,
    HudScaleWatchConfig,
    measure_hud_scale,
    scale_abilities,
    scale_portraits,
    scale_resources,
)

WIDTH, HEIGHT = 800, 400
LAYOUT = AbilityLayout(
    ability_first_x=300, ability_y=320, ability_width=30, ability_height=30,
    ability_spacing=36,
    summoner_first_x=450, summoner_y=322, summoner_width=26, summoner_height=26,
    summoner_spacing=30,
    point_y=290, point_height=26,
)


def panel(scale: float) -> np.ndarray:
    """The slots drawn where `LAYOUT` puts them at `scale`, over noise."""
    rng = np.random.default_rng(0)
    frame = rng.integers(20, 60, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    hud = HudScale.for_frame(scale, WIDTH, HEIGHT)
    cv2.rectangle(frame, (250, 375), (560, 385), (60, 200, 60), -1)
    scaled = scale_abilities(LAYOUT, hud)
    for (x, y, w, h) in scaled.boxes().values():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    return frame


def test_the_panel_scales_about_the_bottom_centre() -> None:
    hud = HudScale.for_frame(1.5, WIDTH, HEIGHT)
    assert hud.x(WIDTH / 2) == WIDTH / 2
    assert hud.y(HEIGHT) == HEIGHT
    assert hud.x(300) == pytest.approx(400 - 100 * 1.5)
    assert hud.y(320) == pytest.approx(400 - 80 * 1.5)


def test_identity_leaves_every_layout_where_it_was() -> None:
    hud = HudScale.identity(WIDTH, HEIGHT)
    assert scale_abilities(LAYOUT, hud) == LAYOUT
    bars = ResourceLayout(health=(320, 360, 100, 12), mana=(320, 372, 100, 12))
    assert scale_resources(bars, hud) == bars


def test_only_the_self_portrait_follows_the_panel() -> None:
    portraits = PortraitLayout(
        ally_first_center_x=700, ally_center_y=100, ally_spacing=40,
        ally_radius=15, self_center_x=250, self_center_y=340, self_radius=20,
    )
    moved = scale_portraits(portraits, HudScale.for_frame(1.5, WIDTH, HEIGHT))
    assert moved.ally_center(0) == portraits.ally_center(0)
    assert moved.self_radius == 30
    assert moved.self_center_y == pytest.approx(400 - 60 * 1.5)


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.25, 1.5])
def test_a_painted_panel_is_measured_at_its_scale(scale: float) -> None:
    reading = measure_hud_scale(panel(scale), LAYOUT)
    assert reading is not None
    assert reading.confident(HudScaleConfig())
    assert reading.scale == pytest.approx(scale, abs=0.01)


def test_a_frame_with_no_panel_is_not_believed() -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    reading = measure_hud_scale(noise, LAYOUT)
    assert reading is None or not reading.confident(HudScaleConfig())


def test_the_watch_adopts_only_after_a_full_window() -> None:
    watch = HudScaleWatch(LAYOUT, HudScaleWatchConfig(window=3))
    frame = panel(1.5)
    assert watch.update(frame, 0.0) is None
    assert watch.update(frame, 1.0) is None
    adopted = watch.update(frame, 2.0)
    assert adopted == pytest.approx(1.5, abs=0.01)
    assert watch.settled


def test_the_watch_waits_out_its_interval() -> None:
    watch = HudScaleWatch(LAYOUT, HudScaleWatchConfig(window=1))
    assert watch.update(panel(1.0), 0.0) is None   # within tolerance of 1.0
    # Settled now, so a different panel half a second later is not looked at.
    assert watch.update(panel(1.5), 0.5) is None
    assert watch.scale == 1.0
    watch.unsettle()
    assert watch.update(panel(1.5), 0.6) == pytest.approx(1.5, abs=0.01)


def test_a_correct_scale_is_not_nudged_by_scatter() -> None:
    watch = HudScaleWatch(LAYOUT, HudScaleWatchConfig(window=1))
    assert watch.update(panel(1.01), 0.0) is None
    assert watch.scale == 1.0


def test_relayout_forgets_only_the_slots_that_moved() -> None:
    portraits = PortraitLayout(
        ally_first_center_x=700, ally_center_y=100, ally_spacing=40,
        ally_radius=15, self_center_x=250, self_center_y=340, self_radius=20,
    )
    reader = AliveReader(portraits)
    reader._baselines.update({"self": 0.5, "ally1": 0.4})
    reader.relayout(scale_portraits(portraits,
                                    HudScale.for_frame(1.5, WIDTH, HEIGHT)))
    assert reader.baselines == {"ally1": 0.4}
