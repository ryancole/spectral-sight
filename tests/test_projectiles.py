"""Projectile candidates from the world view.

Which movers are bolts is a footage question, reported by
`tools/detect_projectiles.py`. What is pinned here is the tracking logic
that no threshold would expose: a mover yields one track rather than a
train of ghosts, a chance pairing dies at birth, links are mutual, and the
shape measures say what they claim on a path whose answer is known.
"""

from __future__ import annotations

import cv2
import numpy as np

from spectral_sight.perception.screen import (
    Blob,
    CameraTracker,
    MotionConfig,
    ProjectileConfig,
    ProjectileTrack,
    ProjectileTracker,
    WorldView,
)

VIEW = WorldView(left=0.0, top=0.0, right=1.0, bottom=1.0)
FPS = 30.0
MOTION = MotionConfig(repeat_pixels=400)
"""The repeat floor is a footage measurement (a stale refresh of a 1608x1009
view); a synthetic frame with one mover on it would be a repeat by that
number. The tests are about the tracking, so they lower it."""


def terrain(seed: int = 1, size: tuple[int, int] = (480, 800)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size, dtype=np.uint8)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)
    smooth = cv2.normalize(smooth, None, 40, 160, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)


def with_dot(ground: np.ndarray, x: float, y: float, size: int = 24) -> np.ndarray:
    """A bright square mover. 24px, because a frame whose only change is a
    tiny dot is, by the repeat rule, a repeat -- and a real bolt with its
    trail changes a thousand pixels, not a hundred."""
    frame = ground.copy()
    x0, y0 = int(round(x - size / 2)), int(round(y - size / 2))
    frame[y0 : y0 + size, x0 : x0 + size] = 255
    return frame


def tracker(**overrides) -> ProjectileTracker:
    return ProjectileTracker(
        config=ProjectileConfig(**overrides),
        camera=CameraTracker(VIEW, MOTION),
    )


def run(frames: list[np.ndarray]) -> list[ProjectileTrack]:
    tr = tracker()
    out = []
    for i, frame in enumerate(frames):
        out.extend(tr.update(frame, i / FPS))
    out.extend(tr.flush())
    return out


def test_a_bolt_is_one_projectile_track_not_a_train_of_ghosts() -> None:
    """A mover differences twice per frame, where it is and where it was.
    Without ghost suppression the second copy is a second track."""
    ground = terrain()
    frames = [with_dot(ground, 100 + 90 * i, 240) for i in range(7)]
    tracks = run(frames)
    bolts = [t for t in tracks if t.is_projectile(ProjectileConfig())]
    assert len(bolts) == 1
    bolt = bolts[0]
    assert len(bolt.points) >= 5
    assert abs(bolt.speed - 90 * FPS) < 0.15 * 90 * FPS
    assert bolt.straightness > 0.98 and bolt.rms < 3


def test_a_slow_mover_is_tracked_but_is_not_a_projectile() -> None:
    """A champion-sized thing walking. Large, because a small slow mover on a
    dead-still ground changes fewer pixels than the repeat floor and the
    frame is, by the rule, a repeat -- which is the documented trade."""
    ground = terrain(seed=2)
    frames = [with_dot(ground, 100 + 8 * i, 240, size=60) for i in range(8)]
    tracks = run(frames)
    assert tracks, "a slow mover still yields a track"
    assert not any(t.is_projectile(ProjectileConfig()) for t in tracks)


def test_a_jittering_blob_is_not_a_projectile() -> None:
    ground = terrain(seed=3)
    rng = np.random.default_rng(0)
    frames = [with_dot(ground, 300 + rng.uniform(-40, 40), 240 + rng.uniform(-40, 40))
              for _ in range(10)]
    tracks = run(frames)
    assert not any(t.is_projectile(ProjectileConfig()) for t in tracks)


def test_a_bolt_survives_camera_motion() -> None:
    """The ground scrolls under the bolt; stabilisation must remove the
    scroll or the whole terrain becomes movers."""
    ground = terrain(seed=4, size=(480, 1000))
    frames = []
    for i in range(7):
        matrix = np.float32([[1, 0, -6.0 * i], [0, 1, 3.0 * i]])
        scrolled = cv2.warpAffine(ground, matrix, (1000, 480), borderMode=cv2.BORDER_REFLECT)
        frames.append(with_dot(scrolled, 150 + 90 * i, 240))
    tracks = run(frames)
    bolts = [t for t in tracks if t.is_projectile(ProjectileConfig())]
    assert len(bolts) == 1


# -- association rules, driven with blob lists directly ----------------------


def feed(tr: ProjectileTracker, frames: list[list[Blob]]) -> list[ProjectileTrack]:
    out = []
    for i, blobs in enumerate(frames):
        out.extend(tr._associate(blobs, i / FPS))
    out.extend(tr.flush())
    return out


def test_a_chance_pair_dies_when_the_third_point_disagrees() -> None:
    """Two strangers 100px apart form a pair; a third blob off the line
    they predict must not extend it into a track."""
    tr = tracker()
    frames = [
        [Blob(0, 0, 50)],
        [Blob(100, 0, 50)],
        [Blob(200, 70, 50)],   # predicted (200, 0): 70px off, > max(20, 35)
        [Blob(300, 70, 50)],
        [Blob(400, 70, 50)],
    ]
    tracks = feed(tr, frames)
    assert all(len(t.points) < 4 or t.points[0][1] != 0 for t in tracks)


def test_a_third_point_within_the_relative_gate_confirms() -> None:
    """A bolt's centroid wobbles with its trail: 30px off a 100px step is
    within the 35% tolerance and must be accepted."""
    tr = tracker()
    frames = [
        [Blob(0, 0, 50)],
        [Blob(100, 0, 50)],
        [Blob(200, 30, 50)],
        [Blob(300, 30, 50)],
        [Blob(400, 30, 50)],
    ]
    tracks = feed(tr, frames)
    assert any(len(t.points) == 5 for t in tracks)


def test_links_are_mutual() -> None:
    """One blob wanted by two tracks goes to the nearer; the other misses."""
    tr = tracker()
    frames = [
        [Blob(0, 0, 50), Blob(0, 300, 50)],
        [Blob(100, 0, 50), Blob(100, 300, 50)],
        [Blob(200, 0, 50), Blob(200, 300, 50)],
        [Blob(300, 10, 50)],                    # only one blob for two tracks
        [Blob(400, 10, 50)],
        [Blob(500, 10, 50)],
    ]
    tracks = feed(tr, frames)
    long = [t for t in tracks if len(t.points) >= 5]
    assert len(long) == 1 and long[0].points[0][2] == 0


def test_area_continuity_blocks_a_link() -> None:
    tr = tracker()
    frames = [
        [Blob(0, 0, 50)],
        [Blob(100, 0, 50)],
        [Blob(200, 0, 50)],
        [Blob(300, 0, 900)],   # eighteen times the area: not the same thing
        [Blob(400, 0, 900)],
    ]
    tracks = feed(tr, frames)
    assert not any(len(t.points) >= 4 for t in tracks)


# -- shape measures --------------------------------------------------------


def line(n: int, step: float, start: tuple[float, float] = (0.0, 0.0)) -> ProjectileTrack:
    pts = tuple((i / FPS, start[0] + step * i, start[1]) for i in range(n))
    return ProjectileTrack(id=0, points=pts, area=50)


def test_speed_and_straightness_of_a_line() -> None:
    t = line(5, 100.0)
    assert abs(t.speed - 100.0 * FPS) < 1e-6
    assert t.straightness == 1.0 and t.rms == 0.0
    assert t.heading == (1.0, 0.0)


def test_approaches_reports_closest_approach_ahead_only() -> None:
    t = line(5, 100.0)
    assert t.approaches((1000.0, 30.0), within=100.0) == 30.0
    assert t.approaches((1000.0, 300.0), within=100.0) is None   # too wide
    assert t.approaches((-500.0, 0.0), within=100.0) is None     # behind
