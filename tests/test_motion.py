"""Camera motion from the terrain.

Which estimator to use was decided on footage (see the module: sparse-flow
median beat phase correlation on 98% of moving frame pairs). What is pinned
here is the contract -- a known shift comes back as itself, a repeat is
flagged rather than measured, time is accounted between distinct frames, and a
bare view declines -- on synthetic terrain where the answer is known exactly.
"""

from __future__ import annotations

import cv2
import numpy as np

from spectral_sight.perception.screen import CameraTracker, MotionConfig, WorldView

VIEW = WorldView(left=0.0, top=0.0, right=1.0, bottom=1.0)
SMALL = MotionConfig(repeat_pixels=400)
"""The repeat floor is sized to a 1608x1009 view; these frames are a third of
that and a lone change on them would be a repeat by the footage number."""


def terrain(seed: int = 1, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A textured ground: smoothed noise, so corners exist everywhere."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size, dtype=np.uint8)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)
    smooth = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


def test_recovers_a_known_shift() -> None:
    tracker = CameraTracker(VIEW, SMALL)
    ground = terrain()
    assert tracker.update(ground, 0.0) is None
    motion = tracker.update(shifted(ground, 7.0, -4.0), 1 / 30)
    assert motion is not None and motion.estimated and not motion.repeat
    assert abs(motion.dx - 7.0) < 0.5 and abs(motion.dy + 4.0) < 0.5
    assert motion.inliers > 0.9


def test_a_large_shift_is_within_reach() -> None:
    """The pyramid must reach the fastest camera motion seen: ~40 px/frame."""
    tracker = CameraTracker(VIEW, SMALL)
    ground = terrain(seed=2)
    tracker.update(ground, 0.0)
    motion = tracker.update(shifted(ground, -38.0, 12.0), 1 / 30)
    assert abs(motion.dx + 38.0) < 1.0 and abs(motion.dy - 12.0) < 1.0


def test_a_foreground_mover_does_not_sway_the_median() -> None:
    """A big object moving differently from the ground is an outlier."""
    tracker = CameraTracker(VIEW, SMALL)
    ground = terrain(seed=3)
    tracker.update(ground, 0.0)
    nxt = shifted(ground, 5.0, 0.0)
    # paint a large textured block that moved the other way
    block = terrain(seed=4)[:120, :160]
    nxt[200:320, 100:260] = block
    motion = tracker.update(nxt, 1 / 30)
    assert abs(motion.dx - 5.0) < 0.7 and abs(motion.dy) < 0.7


def test_a_repeat_frame_is_flagged_and_time_carries() -> None:
    """A repeat is not remembered: the next distinct frame is measured against
    the last distinct one, and the pair's dt spans the repeat."""
    tracker = CameraTracker(VIEW, SMALL)
    ground = terrain(seed=5)
    tracker.update(ground, 0.0)
    noisy = ground.copy()
    noisy[::7, ::7] += 1  # compression-noise scale
    repeat = tracker.update(noisy, 1 / 30)
    assert repeat.repeat and repeat.dx == 0.0 and not repeat.estimated
    motion = tracker.update(shifted(ground, 3.0, 2.0), 2 / 30)
    assert not motion.repeat
    assert abs(motion.dt - 2 / 30) < 1e-9
    assert abs(motion.dx - 3.0) < 0.5 and abs(motion.dy - 2.0) < 0.5


def test_a_bare_view_declines() -> None:
    tracker = CameraTracker(VIEW, MotionConfig(min_corners=20, repeat_pixels=400))
    black = np.zeros((480, 640, 3), np.uint8)
    tracker.update(black, 0.0)
    black2 = black.copy()
    black2[10:50, 10:50] = 200  # 1,600 px changed: not a repeat, still no texture
    motion = tracker.update(black2, 1 / 30)
    assert motion is not None and not motion.repeat
    assert not motion.estimated and motion.corners == 0


def test_stabilise_aligns_the_previous_view() -> None:
    tracker = CameraTracker(VIEW, SMALL)
    ground = terrain(seed=6)
    tracker.update(ground, 0.0)
    nxt = shifted(ground, 6.0, -3.0)
    motion = tracker.update(nxt, 1 / 30)
    aligned = tracker.stabilise(motion)
    raw = cv2.absdiff(tracker.current, tracker.before).mean()
    stabilised = cv2.absdiff(tracker.current, aligned).mean()
    assert stabilised < raw * 0.25


def test_world_view_box_is_the_interior_rectangle() -> None:
    x, y, w, h = WorldView().box(2116, 1354)
    assert (x, y) == (0, 47)
    assert x + w == int(0.76 * 2116) and y + h == int(0.78 * 1354)
