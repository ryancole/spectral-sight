"""Marking a cast on the minimap overlay.

Drawing is checked by asking what changed rather than by comparing against a
reference image, which would have to be regenerated every time a radius moved.
What matters here is not the exact pixels but three things that were wrong once:
that the mark appears at all, that it expires, and that it is not the same ink
as the marker already on the champion underneath it.
"""

from __future__ import annotations

import numpy as np

from spectral_sight.debug.overlay import (
    CAST_COLOR,
    CAST_FLASH,
    CastMark,
    draw_tracks,
)
from spectral_sight.types import Team


class FakeTrack:
    """The handful of attributes `draw_tracks` reads off a track."""

    def __init__(self, track_id: int = 1, x: float = 80, y: float = 80) -> None:
        self.id = track_id
        self.x = x
        self.y = y
        self.team = Team.BLUE
        self.identity = "Test"

    def age(self, timestamp: float) -> float:
        return 0.0


def blank() -> np.ndarray:
    return np.zeros((160, 160, 3), np.uint8)


def changed(before: np.ndarray, after: np.ndarray) -> int:
    return int((before != after).any(axis=2).sum())


def test_a_cast_marks_the_champion() -> None:
    track = FakeTrack()
    plain = draw_tracks(blank(), [track], 0.0)
    marked = draw_tracks(blank(), [track], 0.0, casts={1: CastMark(0.0, True)})
    assert changed(plain, marked) > 0


def test_the_mark_expires() -> None:
    """Drawn permanently it would stop meaning anything."""
    track = FakeTrack()
    plain = draw_tracks(blank(), [track], 0.0)
    stale = draw_tracks(
        blank(), [track], 0.0, casts={1: CastMark(CAST_FLASH + 0.1, True)}
    )
    assert changed(plain, stale) == 0


def test_the_mark_fades() -> None:
    track = FakeTrack()
    fresh = draw_tracks(blank(), [track], 0.0, casts={1: CastMark(0.0, True)})
    old = draw_tracks(
        blank(), [track], 0.0, casts={1: CastMark(CAST_FLASH * 0.75, True)}
    )
    assert fresh[80 - 16, 80].sum() > old[80 - 16, 80].sum() > 0


def test_the_mark_is_not_white() -> None:
    """White is the local player's ring, two pixels further out. A cast in the
    same colour is invisible on the champion who casts most and is watched
    hardest, which is how this shipped the first time."""
    assert CAST_COLOR != (255, 255, 255)
    track = FakeTrack()
    marked = draw_tracks(blank(), [track], 0.0, casts={1: CastMark(0.0, True)})
    assert tuple(int(v) for v in marked[80 - 16, 80]) == CAST_COLOR


def test_the_mark_is_visible_on_the_local_player() -> None:
    """The case that matters: the player already wears a white ring."""
    track = FakeTrack()
    just_self = draw_tracks(blank(), [track], 0.0, self_track=track)
    self_casting = draw_tracks(
        blank(), [track], 0.0, self_track=track, casts={1: CastMark(0.0, True)}
    )
    assert changed(just_self, self_casting) > 100


def test_a_cast_across_a_gap_is_drawn_more_faintly() -> None:
    """Its timing is loose -- the champion was away, and the cast happened
    somewhere in a window that can be seconds wide. Drawing it identically to a
    cast pinned to one frame would put a vague claim on screen in the same ink
    as a precise one."""
    track = FakeTrack()
    plain = draw_tracks(blank(), [track], 0.0)
    solid = draw_tracks(blank(), [track], 0.0, casts={1: CastMark(0.0, True)})
    thin = draw_tracks(blank(), [track], 0.0, casts={1: CastMark(0.0, False)})
    assert changed(plain, thin) < changed(plain, solid)


def test_a_cast_on_one_track_does_not_mark_another() -> None:
    tracks = [FakeTrack(1, 50, 50), FakeTrack(2, 110, 110)]
    plain = draw_tracks(blank(), tracks, 0.0)
    marked = draw_tracks(blank(), tracks, 0.0, casts={1: CastMark(0.0, True)})
    difference = np.argwhere((plain != marked).any(axis=2))
    assert len(difference) > 0
    # Everything that moved is near the first track, not the second.
    assert np.hypot(*(difference - [50, 50]).T).max() < 30


def test_no_casts_draws_what_it_always_did() -> None:
    track = FakeTrack()
    assert changed(
        draw_tracks(blank(), [track], 0.0),
        draw_tracks(blank(), [track], 0.0, casts={}),
    ) == 0
