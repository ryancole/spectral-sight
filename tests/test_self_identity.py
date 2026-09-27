"""The local player's name through a death: held where they fell, found again
at the fountain, and never handed a teammate's nameplate.

Measured on a live lane clip before this existed: while the player was dead,
stray markers walked their track about 1,400 units, and after the respawn the
self row named a teammate -- Kennen, then Malphite, then Yunara -- for the two
minutes the clip had left, while the player's name rode a phantom.
"""

from __future__ import annotations

import cv2
import numpy as np

from spectral_sight.perception.minimap.viewport import Viewport
from spectral_sight.perception.nameplates import Nameplate
from spectral_sight.perception.nameplates.plates import Side
from spectral_sight.pipeline import SELF_SLOT
from spectral_sight.tracking import Track, TrackState
from spectral_sight.types import Team
from tests.synthetic import Marker
from tests.test_alive import REGION, build_pipeline, frame, name_tracks, run

PLAYER = Marker(60, 60, Team.BLUE)
TEAMMATE = Marker(150, 170, Team.BLUE)
FOUNTAIN = Marker(40, 230, Team.BLUE)


def camera_on(x: float, y: float, *, dead: tuple[str, ...] = (),
              markers: tuple[Marker, ...]) -> np.ndarray:
    """A frame with the camera rectangle centred on (x, y) of the minimap."""
    image = frame(dead=dead, markers=markers)
    cx, cy = REGION.x + int(x), REGION.y + int(y)
    cv2.rectangle(image, (cx - 40, cy - 25), (cx + 40, cy + 25),
                  (235, 235, 235), 1)
    return image


def settle(pipeline) -> float:
    """Play the player and one teammate long enough to be tracked and named,
    with the camera on the player. Returns the next timestamp."""
    t = 0.0
    for _ in range(6):
        pipeline.process(camera_on(PLAYER.x, PLAYER.y,
                                   markers=(PLAYER, TEAMMATE)), t)
        t += 0.1
    name_tracks(pipeline, "Zilean", "Ryze")
    pipeline._self_evidence = {"Zilean": 50}
    return t


def by_name(result, name: str):
    return next(row for row in result.observations if row.champion == name)


def test_a_dead_players_track_is_not_fed_by_a_stray_marker() -> None:
    """The corpse's marker is gone but stage 1's surplus is not, and a marker
    a few pixels from where the player fell used to walk their track off."""
    pipeline = build_pipeline()
    t = settle(pipeline)
    stray = Marker(PLAYER.x + 8, PLAYER.y + 6, Team.BLUE)

    result = None
    for _ in range(15):
        result = pipeline.process(
            camera_on(PLAYER.x, PLAYER.y, dead=(SELF_SLOT,),
                      markers=(stray, TEAMMATE)), t)
        t += 0.1

    zilean = by_name(result, "Zilean")
    assert abs(zilean.x - PLAYER.x) < 3 and abs(zilean.y - PLAYER.y) < 3, (
        "the corpse stays where the player fell"
    )
    assert zilean.alive is False
    assert zilean.is_self, "dead, the self row is the corpse"
    assert zilean.visible is False


def test_the_respawn_brings_the_players_name_to_the_fountain() -> None:
    """The respawned marker is found by the camera, not the gallery, and the
    name has to come with it -- not be left on the corpse, and not be lent
    by whichever teammate's track sat nearest."""
    pipeline = build_pipeline()
    t = settle(pipeline)
    for _ in range(15):
        pipeline.process(camera_on(PLAYER.x, PLAYER.y, dead=(SELF_SLOT,),
                                   markers=(TEAMMATE,)), t)
        t += 0.1

    result = None
    for _ in range(15):
        result = pipeline.process(
            camera_on(FOUNTAIN.x, FOUNTAIN.y, markers=(FOUNTAIN, TEAMMATE)), t)
        t += 0.1

    selves = [row for row in result.observations if row.is_self]
    assert len(selves) == 1
    assert selves[0].champion == "Zilean"
    assert abs(selves[0].x - FOUNTAIN.x) < 3 and abs(selves[0].y - FOUNTAIN.y) < 3
    assert selves[0].alive is True
    assert by_name(result, "Ryze").is_self is False


def test_a_camera_clamped_off_the_player_still_finds_them() -> None:
    """At the fountain the camera sits against the map corner while the
    player walks out, so the centre is not the player: the marker placed
    there fed only phantoms. The player's named track, on screen, answers."""
    pipeline = build_pipeline()
    t = settle(pipeline)
    result = None
    for step in range(1, 16):
        walker = Marker(PLAYER.x + 2 * step, PLAYER.y, Team.BLUE)
        result = pipeline.process(
            camera_on(PLAYER.x, PLAYER.y, markers=(walker, TEAMMATE)), t)
        t += 0.1
    assert result.self_blip.score == 0.0, "the centre found nobody and placed one"
    selves = [row for row in result.observations if row.is_self]
    assert [row.champion for row in selves] == ["Zilean"]
    assert selves[0].x > PLAYER.x + 25, "the row follows the player, not the centre"


def test_a_teammate_at_the_death_cam_is_not_the_self_row() -> None:
    """The death-cam can sit on a teammate, whose marker is then at the
    camera centre exactly where the player's usually is."""
    pipeline = build_pipeline()
    t = settle(pipeline)
    result = None
    for _ in range(15):
        result = pipeline.process(
            camera_on(TEAMMATE.x, TEAMMATE.y, dead=(SELF_SLOT,),
                      markers=(TEAMMATE,)), t)
        t += 0.1
    assert by_name(result, "Ryze").is_self is False
    assert by_name(result, "Zilean").is_self is True


def test_a_held_corpse_outlives_the_fog_timeout() -> None:
    """A late-game respawn timer runs past `forget_after`; the player's track
    has to be there to come back to."""
    pipeline = build_pipeline()
    t = settle(pipeline)
    forget = pipeline.tracker.config.forget_after
    for step in (0.0, forget / 2, forget + 5.0):
        result = pipeline.process(
            camera_on(PLAYER.x, PLAYER.y, dead=(SELF_SLOT,),
                      markers=(TEAMMATE,)), t + step)
    assert by_name(result, "Zilean").alive is False


# -- nameplates -------------------------------------------------------------


class StubPlates:
    def __init__(self, plates: list[Nameplate]) -> None:
        self.plates = plates

    def read(self, frame, hsv=None) -> list[Nameplate]:
        return self.plates


class Identity:
    """Projection that reads a plate's x, y as minimap pixels."""

    def to_minimap(self, plate, viewport, frame_size):
        return float(plate.x), float(plate.y)


def plate(x: int, y: int, side: Side, health: float, level: int) -> Nameplate:
    return Nameplate(x=x, y=y, width=100, health=health, resource=1.0,
                     side=side, level=level)


def pair(plates: list[Nameplate], tracks: list[Track], self_track: Track | None):
    pipeline = build_pipeline()
    pipeline.plate_reader = StubPlates(plates)
    pipeline.projection = Identity()
    pipeline.tracker.tracks = tracks
    _, pairing, _ = pipeline._read_plates(
        np.zeros((10, 10, 3), np.uint8), tracks, Viewport(0, 0, 80, 50), 0.0,
        self_track=self_track,
    )
    return {track_id: plates[index] for index, track_id in pairing.items()}


def blue_track(track_id: int, x: float, y: float) -> Track:
    return Track(id=track_id, team=Team.BLUE, x=x, y=y, last_seen=0.0,
                 state=TrackState.CONFIRMED)


def test_a_teammates_plate_never_lands_on_the_player() -> None:
    """With the green bar gone -- the instant of a death -- the nearest blue
    plate used to be paired to the player's track by distance, and a player
    on 23% read as 90% and a level up."""
    me, mate = blue_track(1, 50, 50), blue_track(2, 300, 300)
    paired = pair([plate(52, 50, Side.ALLY, 0.9, 2)], [me, mate], me)
    assert 1 not in paired


def test_the_green_plate_goes_to_the_player_wherever_it_projects() -> None:
    """Both plates sit nearer the other's track: distance would swap them."""
    me, mate = blue_track(1, 50, 50), blue_track(2, 62, 50)
    green = plate(61, 50, Side.SELF, 0.23, 1)
    blue = plate(51, 50, Side.ALLY, 0.9, 2)
    paired = pair([green, blue], [me, mate], me)
    assert paired[1] is green, "the green bar names the player outright"
    assert paired[2] is blue


def test_without_a_self_track_the_green_plate_goes_nowhere() -> None:
    """Better no reading than the player's health on a teammate's row."""
    mate = blue_track(2, 50, 50)
    paired = pair([plate(50, 50, Side.SELF, 0.5, 3)], [mate], None)
    assert paired == {}
