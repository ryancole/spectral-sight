"""The player's own skillshots, joined from a cast, a bolt and a target.

Whether a shot really hit is a footage question, and the module says at
length why this project's footage cannot answer it. What is pinned here is
the accounting: a cast with no bolt beside the player is not a skillshot, a
bolt belongs to one cast and not two, the target is the enemy the line went
nearest in front of it, the verdict is the geometry rather than the health
bar, and a bar that moved is carried either way.
"""

from __future__ import annotations

from spectral_sight.export import Skillshot
from spectral_sight.perception.screen import CameraMotion, ProjectileTrack
from spectral_sight.perception.screen.aim import AimConfig, AimDetector, EnemyPlate

FPS = 30.0
ANCHOR = (500.0, 300.0)


def bolt(x0: float, y0: float, ux: float, uy: float, speed: float = 1000.0,
         n: int = 5, t0: float = 10.0, id: int = 0) -> ProjectileTrack:
    step = speed / FPS
    pts = tuple((t0 + i / FPS, x0 + ux * step * i, y0 + uy * step * i)
                for i in range(n))
    return ProjectileTrack(id=id, points=pts, area=50)


def detector(**overrides) -> AimDetector:
    return AimDetector(AimConfig(**overrides))


def enemy(d: AimDetector, x: float, y: float, health: float | None,
          t: float = 9.9) -> None:
    d.observe_enemies(t, [EnemyPlate(x, y, health)])


def test_a_cast_that_launched_nothing_is_not_a_skillshot() -> None:
    d = detector()
    d.observe_cast("Q", 10.0)
    shots = d.flush()
    assert len(shots) == 1
    assert shots[0].launched is None
    assert shots[0].outcome == "unknown"


def test_a_bolt_born_beside_the_player_after_the_cast_is_the_shot() -> None:
    d = detector()
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)   # 100px from the anchor
    shot = d.flush()[0]
    assert shot.slot == "Q"
    assert shot.launched == 10.0
    assert abs(shot.speed - 1000.0) < 1.0


def test_a_bolt_born_far_from_the_player_is_not_their_shot() -> None:
    d = detector(max_launch=250.0)
    d.observe_cast("Q", 9.95)
    d.consider([bolt(900, 300, 1, 0)], ANCHOR)   # 400px away
    assert d.flush()[0].launched is None


def test_a_bolt_outside_the_launch_window_is_not_their_shot() -> None:
    d = detector(launch_after=0.5)
    d.observe_cast("Q", 9.0)                      # bolt is a second late
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.flush()[0].launched is None


def test_a_summoner_spell_is_not_considered() -> None:
    d = detector()
    d.observe_cast("D", 10.0)
    assert d.flush() == []


def test_one_bolt_belongs_to_one_cast() -> None:
    """Two casts within the launch window of a single bolt: the first takes
    it, the second reports none, rather than both claiming the same shot."""
    d = detector()
    d.observe_cast("Q", 9.95)
    d.observe_cast("W", 10.05)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shots = sorted(d.flush(), key=lambda s: s.at)
    assert [s.slot for s in shots] == ["Q", "W"]
    assert shots[0].launched == 10.0
    assert shots[1].launched is None


def test_the_target_is_the_enemy_the_line_passes_nearest_in_front() -> None:
    d = detector()
    enemy(d, 900.0, 320.0, 0.8)                   # 20px off a line going right
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shot = d.flush()[0]
    assert abs(shot.miss - 20.0) < 1e-6
    assert abs(shot.flight - 0.3) < 1e-6          # 300px at 1000px/s


def test_an_enemy_behind_the_bolt_is_not_its_target() -> None:
    d = detector()
    enemy(d, 300.0, 300.0, 0.8)                   # behind a bolt going right
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shot = d.flush()[0]
    assert shot.miss is None
    assert shot.outcome == "unknown"


def test_an_enemy_beyond_the_bolt_s_flight_is_not_its_target() -> None:
    d = detector(max_flight=0.2)
    enemy(d, 1200.0, 300.0, 0.8)                  # 600px out, 0.6s away
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.flush()[0].miss is None


def test_a_line_through_the_target_is_a_hit() -> None:
    d = detector(hit_radius=130.0)
    enemy(d, 900.0, 310.0, 0.8)
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.flush()[0].outcome == "hit"


def test_a_line_wide_of_the_target_is_a_miss() -> None:
    d = detector(hit_radius=130.0)
    enemy(d, 900.0, 600.0, 0.8)                   # 300px off the line
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shot = d.flush()[0]
    assert shot.outcome == "missed"
    assert abs(shot.miss - 300.0) < 1e-6


def test_the_health_bar_does_not_decide_the_verdict() -> None:
    """A wide shot stays a miss even when the target's bar falls, and a shot
    through them stays a hit when it does not -- the module's whole point."""
    wide = detector(hit_radius=130.0)
    enemy(wide, 900.0, 600.0, 0.9)
    wide.observe_cast("Q", 9.95)
    wide.consider([bolt(600, 300, 1, 0)], ANCHOR)
    wide.observe_enemies(10.3, [EnemyPlate(900.0, 600.0, 0.5)])
    shot = wide.flush()[0]
    assert shot.outcome == "missed"
    assert shot.fall is not None and abs(shot.fall - 0.4) < 1e-6

    through = detector(hit_radius=130.0)
    enemy(through, 900.0, 310.0, 0.9)
    through.observe_cast("Q", 9.95)
    through.consider([bolt(600, 300, 1, 0)], ANCHOR)
    through.observe_enemies(10.3, [EnemyPlate(900.0, 310.0, 0.9)])
    shot = through.flush()[0]
    assert shot.outcome == "hit"
    assert shot.fall is None


def test_an_unreadable_bar_leaves_the_fall_unrecorded_not_the_verdict() -> None:
    d = detector(hit_radius=130.0)
    enemy(d, 900.0, 310.0, None)
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shot = d.flush()[0]
    assert shot.outcome == "hit"
    assert shot.fall is None


def test_resolve_waits_for_the_launch_window_then_the_arrival() -> None:
    d = detector()
    enemy(d, 900.0, 310.0, 0.8)
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.resolve(10.4) == []                  # launch window still open
    assert d.resolve(10.6) == []                  # bolt still in flight
    assert len(d.resolve(11.5)) == 1


def test_an_enemy_who_left_the_view_is_still_the_target() -> None:
    """The plate track is kept after the champion walks off screen: the cast
    settles a second later, and they were there when it was aimed."""
    d = detector()
    enemy(d, 900.0, 310.0, 0.8, t=9.9)
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    for t in (10.5, 11.0, 11.5):
        d.observe_enemies(t, [])                  # nobody on screen any more
    shot = d.resolve(11.5)[0]
    assert shot.miss is not None


def test_lead_is_signed_against_the_target_s_own_motion() -> None:
    """A target walking down (+y) and a bolt going right that passes above
    them: the shot went by on the side behind their movement."""
    d = detector(hit_radius=130.0, min_target_speed=10.0)
    for i, t in enumerate((9.6, 9.7, 9.8, 9.9)):
        d.observe_enemies(t, [EnemyPlate(900.0, 500.0 + 20 * i, 0.8)])
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    shot = d.flush()[0]
    assert shot.lead is not None and shot.lead < 0

    up = detector(hit_radius=130.0, min_target_speed=10.0)
    for i, t in enumerate((9.6, 9.7, 9.8, 9.9)):
        up.observe_enemies(t, [EnemyPlate(900.0, 500.0 - 20 * i, 0.8)])
    up.observe_cast("Q", 9.95)
    up.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert up.flush()[0].lead > 0


def test_a_still_target_gets_no_lead() -> None:
    d = detector(hit_radius=130.0)
    for t in (9.6, 9.7, 9.8, 9.9):
        d.observe_enemies(t, [EnemyPlate(900.0, 500.0, 0.8)])
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.flush()[0].lead is None


def test_the_camera_is_taken_out_of_the_target_s_motion() -> None:
    """A plate holding still on screen while the view scrolls is a champion
    running with it, not a champion standing."""
    d = detector(hit_radius=130.0, min_target_speed=10.0)
    for t in (9.6, 9.7, 9.8, 9.9):
        d.observe_enemies(t, [EnemyPlate(900.0, 500.0, 0.8)])
        d.observe_motion(t, CameraMotion(dx=0.0, dy=6.0, dt=1 / FPS,
                                         inliers=1.0, corners=200, repeat=False))
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], ANCHOR)
    assert d.flush()[0].lead is not None


def test_no_anchor_means_no_shot_can_be_attributed() -> None:
    d = detector()
    d.observe_cast("Q", 9.95)
    d.consider([bolt(600, 300, 1, 0)], None)
    assert d.flush()[0].launched is None


def test_skillshot_round_trips_through_dict() -> None:
    shot = Skillshot(
        slot="Q", at=10.0, launched=10.1, speed=1200.0, heading=(0.6, -0.8),
        miss=42.5, flight=0.3, outcome="hit", fall=0.12, lead=-42.5,
    )
    assert Skillshot.from_dict(shot.to_dict()) == shot
    bare = Skillshot(
        slot="R", at=10.0, launched=None, speed=None, heading=None,
        miss=None, flight=None, outcome="unknown", fall=None, lead=None,
    )
    assert Skillshot.from_dict(bare.to_dict()) == bare
