"""Bolts at the player, resolved against health and motion.

Whether a candidate is a real threat is a footage question. What is pinned
here is the accounting: a bolt launched by the player is not a threat to them,
a bolt flying past is not one, arrival is where the geometry says, a health
fall in the window is a hit and a window with no reading is unknown, and the
response is the motion across the line and nothing else.
"""

from __future__ import annotations

from spectral_sight.perception.screen import CameraMotion, ProjectileTrack
from spectral_sight.export import Threat
from spectral_sight.perception.screen.threats import ThreatConfig, ThreatDetector

FPS = 30.0
ANCHOR = (500.0, 300.0)


def bolt(x0: float, y0: float, ux: float, uy: float, speed: float = 1500.0,
         n: int = 5, t0: float = 10.0) -> ProjectileTrack:
    step = speed / FPS
    pts = tuple((t0 + i / FPS, x0 + ux * step * i, y0 + uy * step * i) for i in range(n))
    return ProjectileTrack(id=0, points=pts, area=50)


def detector(**overrides) -> ThreatDetector:
    # gates that need enemy plates are off unless a test turns them on
    cfg = dict(max_origin=None, max_end=None)
    cfg.update(overrides)
    return ThreatDetector(ThreatConfig(**cfg))


def test_a_bolt_heading_for_the_player_is_a_threat_with_an_arrival() -> None:
    d = detector()
    # from 300px left of the anchor, moving right at 1500 px/s
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)
    threats = d.flush()
    assert len(threats) == 1
    t = threats[0]
    assert abs(t.arrival - (10.0 + 300 / 1500)) < 1e-6
    assert t.closest == 0.0


def test_a_bolt_launched_by_the_player_is_not_a_threat() -> None:
    d = detector()
    d.consider([bolt(520, 300, 1, 0)], ANCHOR)   # starts 20px from the anchor
    assert d.flush() == []


def test_a_bolt_moving_away_is_not_a_threat() -> None:
    d = detector()
    d.consider([bolt(200, 300, -1, 0)], ANCHOR)
    assert d.flush() == []


def test_a_bolt_passing_wide_is_not_a_threat() -> None:
    d = detector(radius=120.0)
    d.consider([bolt(200, 500, 1, 0)], ANCHOR)   # 200px below the line
    assert d.flush() == []


def test_a_health_fall_in_the_window_is_a_hit() -> None:
    d = detector()
    d.observe_health(9.9, 800)
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)   # arrives at 10.2
    d.observe_health(10.1, 800)
    d.observe_health(10.4, 650)
    threats = d.resolve(now=11.0)
    assert threats[0].outcome == "hit" and threats[0].damage == 150


def test_no_fall_in_the_window_is_a_dodge() -> None:
    d = detector()
    d.observe_health(9.9, 800)
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)
    d.observe_health(10.3, 800)
    assert d.resolve(now=11.0)[0].outcome == "dodged"


def test_no_reading_in_the_window_is_unknown_not_a_dodge() -> None:
    d = detector()
    d.observe_health(9.0, 800)
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)
    d.observe_health(11.5, 500)   # far after the window
    assert d.resolve(now=12.0)[0].outcome == "unknown"


def test_resolve_waits_for_the_window_to_close() -> None:
    d = detector()
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)   # arrival 10.2, window to 10.7
    assert d.resolve(now=10.5) == []
    assert len(d.resolve(now=10.8)) == 1


def test_response_is_the_motion_across_the_line_only() -> None:
    d = detector()
    d.consider([bolt(200, 300, 1, 0)], ANCHOR)   # heading +x, arrival 10.2
    # camera moved 30px in y (across) and 100px in x (along) during the flight
    for k in range(6):
        d.observe_motion(10.0 + k / FPS, CameraMotion(dx=100 / 6, dy=30 / 6, dt=1 / FPS,
                                                       inliers=0.9, corners=200, repeat=False))
    t = d.resolve(now=11.0)[0]
    assert abs(t.moved_across - 30.0) < 1e-6


def test_origin_gate_needs_an_enemy_nearby() -> None:
    d = detector(max_origin=160.0)
    d.consider([bolt(200, 300, 1, 0)], ANCHOR, enemies=[(190.0, 310.0)])
    assert len(d.flush()) == 1
    d.consider([bolt(200, 300, 1, 0)], ANCHOR, enemies=[(190.0, 900.0)])
    assert d.flush() == []
    d.consider([bolt(200, 300, 1, 0)], ANCHOR, enemies=[])
    assert d.flush() == []


def test_end_gate_needs_the_track_to_reach_the_player() -> None:
    d = detector(max_end=90.0)
    d.consider([bolt(200, 300, 1, 0, n=5)], ANCHOR)    # ends at x=400: 100px short
    assert d.flush() == []
    d.consider([bolt(200, 300, 1, 0, n=7)], ANCHOR)    # ends at x=500: on the player
    assert len(d.flush()) == 1


def test_threat_round_trips_through_dict() -> None:
    t = Threat(at=10.0, arrival=10.2, closest=12.5, speed=1500.0, heading=(1.0, 0.0),
               outcome="hit", damage=150, moved_across=30.0, origin=140.0)
    assert Threat.from_dict(t.to_dict()) == t
