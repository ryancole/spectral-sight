"""Perceptual events, derived from the envelope stream and nothing else.

A frame envelope says what is true; an event says what *changed*, and a
consumer reacting in real time mostly wants the changes -- a death, a cast, a
champion slipping into fog -- without diffing ten rows against their
predecessors itself, badly, in every downstream tool at once.

**Everything here is a pure function of the rows.** The deriver never sees the
tracker, the filters or a pixel; its whole world is `FrameState`. That is a
harder constraint than it looks and it buys the property the live design rests
on: replaying a recorded timeline through this class produces the events the
live run produced, so the event logic is testable against the clips in `data/`
and a downstream tool can be developed against a replay it can trust. State
that leaks in from anywhere else is state a replay cannot reproduce.

**These are perceptual events, not interpretations.** A cast, a death, a
vanish are things the vision concluded; "gank incoming" and "objective
contested" are things they might *mean*, and meaning is the analysis tool's
job. This module publishes what was seen and how far the evidence goes, and
holding that line is what keeps the schema stable while downstream opinions
change.

The kinds, and what each is grounded in:

- `cast` -- a row carrying `cast_drop`. The pipeline already emits a cast on
  exactly one row, the one where the fall was confirmed to hold, so the row is
  the event and the deriver adds nothing but the wrapping.
- `death` / `respawn` -- `alive` turning definitely False / definitely True.
  Keyed by champion rather than track, because a corpse's track can be dropped
  and the respawn arrive on a fresh one. None never transitions anything: it
  means the HUD and the minimap disagreed, and a disagreement is not a death.
- `vanished` / `reappeared` -- `visible` transitions, already debounced
  upstream by the tracker's `lost_after`. Suppressed on the frame a death or
  respawn explains them, since "the corpse left the minimap" is not news.
- `level_up` -- the filtered `level` rising. First knowledge of a level is not
  an event: learning that someone *is* level 3 is state, reaching 4 is change.
- `identified` -- a track's champion becoming known, or changing. It can
  change before the roster locks, and re-announcing with `replaces` is honest
  where staying silent would leave consumers holding a name the pipeline no
  longer believes.
- `roster` -- a team showing five distinct named champions at once. Re-emitted
  if the set later changes, for the same reason `identified` is.

`seq` on an event is the envelope it was derived from, which makes it a
transport key, not a durable one: a live frame that produced no rows writes
nothing to the timeline, so a replay cannot count it. `video_time` and
`game_time` are the keys that survive the round trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectral_sight.export import Observation
from spectral_sight.feed import FrameState
from spectral_sight.types import Team

KINDS = (
    "cast",
    "death",
    "respawn",
    "vanished",
    "reappeared",
    "level_up",
    "identified",
    "roster",
)

_COMMON = frozenset(
    ("t", "kind", "seq", "video_time", "game_time", "team", "champion",
     "track_id")
)
"""Envelope keys a `detail` entry must not shadow -- see `Event.to_dict`."""


@dataclass(frozen=True, slots=True)
class Event:
    """One change, wrapped the way every message on the feed is wrapped.

    The common fields answer who and when for every kind; `detail` carries
    what only this kind knows, flattened onto the wire form so a consumer
    reads one level of keys rather than two.
    """

    kind: str
    seq: int
    video_time: float
    game_time: int | None
    team: Team | None
    champion: str | None
    track_id: int | None
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        assert not (_COMMON & self.detail.keys()), (
            f"detail shadows envelope keys: {_COMMON & self.detail.keys()}"
        )
        return {
            "t": "event",
            "kind": self.kind,
            "seq": self.seq,
            "video_time": round(float(self.video_time), 3),
            "game_time": self.game_time,
            "team": None if self.team is None else self.team.value,
            "champion": self.champion,
            "track_id": self.track_id,
            **self.detail,
        }


class EventDeriver:
    """Folds envelopes in, hands changes back. One per run, fed in order.

    The remembered state is deliberately small -- last definite verdicts, last
    announced names -- and keyed to survive the churn the rows actually have:
    liveness by champion because tracks do not outlive a death, everything
    else by track id because before a name exists there is nothing better.
    """

    def __init__(self) -> None:
        self._alive: dict[str, bool] = {}
        """Last *definite* verdict per champion. None rows leave no mark."""
        self._died_at: dict[str, float] = {}
        self._visible: dict[int, bool] = {}
        self._vanished_at: dict[int, float] = {}
        self._levels: dict[int, int] = {}
        self._named: dict[int, str] = {}
        self._rosters: dict[Team, frozenset[str]] = {}

    def update(self, state: FrameState) -> list[Event]:
        """Derive this frame's events. Order is deterministic: rows arrive
        sorted by track id, each row's events in a fixed kind order, roster
        checks last -- so a live run and a replay agree to the byte."""
        events: list[Event] = []
        for row in state.champions:
            events.extend(self._from_row(state, row))
        events.extend(self._rosters_from(state))
        return events

    def _from_row(self, state: FrameState, row: Observation) -> list[Event]:
        events: list[Event] = []
        # Everything remembered or arithmetic is done on values rounded the
        # way the timeline rounds them, because a live row carries raw floats
        # where a replayed one carries the file's. Deriving from the raw
        # values would let a `down_for` disagree in its last decimal between
        # a live run and its own replay, and byte-equal is the property the
        # determinism test holds.
        now = round(float(row.video_time), 3)

        def event(kind: str, **detail: object) -> Event:
            return Event(
                kind=kind,
                seq=state.seq,
                video_time=row.video_time,
                game_time=state.game_time,
                team=row.team,
                champion=row.champion,
                track_id=row.track_id,
                detail=detail,
            )

        # Identity first: if this row also carries a death or a cast, the
        # consumer should already have been told who it is talking about.
        if row.champion is not None and self._named.get(row.track_id) != row.champion:
            previous = self._named.get(row.track_id)
            detail: dict[str, object] = {"is_self": row.is_self}
            if previous is not None:
                detail["replaces"] = previous
            self._named[row.track_id] = row.champion
            events.append(event("identified", **detail))

        if row.level is not None:
            known = self._levels.get(row.track_id)
            self._levels[row.track_id] = row.level
            if known is not None and row.level > known:
                events.append(event("level_up", level=row.level))

        died, revived = self._liveness_of(row)
        if died:
            self._died_at[self._life_key(row)] = now
            events.append(event("death", allies_dead=state.allies_dead))
        if revived:
            detail = {}
            when = self._died_at.pop(self._life_key(row), None)
            if when is not None:
                detail["down_for"] = round(now - when, 2)
            events.append(event("respawn", **detail))

        was_visible = self._visible.get(row.track_id)
        self._visible[row.track_id] = row.visible
        if was_visible is not None and row.visible != was_visible:
            if not row.visible:
                # What is remembered is when they were last *seen*, not when
                # this event fired: the tracker waits `lost_after` before
                # calling a champion unaccounted for, and the row's own
                # `seconds_since_seen` carries that wait. A `gone_for`
                # measured event-to-event would under-report every stay in
                # fog by exactly that debounce.
                self._vanished_at[row.track_id] = round(
                    now - round(row.seconds_since_seen, 2), 3
                )
                # A death explains the disappearance; the fog did not take
                # them, they are on a gray screen. Same for the return.
                if not died:
                    events.append(event(
                        "vanished",
                        x=round(row.x, 2), y=round(row.y, 2),
                        **(
                            {"world_x": round(row.world_x, 1),
                             "world_y": round(row.world_y, 1)}
                            if row.world_x is not None and row.world_y is not None
                            else {}
                        ),
                    ))
            elif not revived:
                detail = {}
                last_seen = self._vanished_at.pop(row.track_id, None)
                if last_seen is not None:
                    detail["gone_for"] = round(now - last_seen, 2)
                events.append(event("reappeared", **detail))

        if row.cast_drop is not None:
            events.append(event(
                "cast",
                drop=round(float(row.cast_drop), 3),
                at=None if row.cast_at is None else round(float(row.cast_at), 3),
                span=(
                    None if row.cast_span is None
                    else round(float(row.cast_span), 3)
                ),
                continuous=row.cast_continuous,
                confirmed=row.cast_confirmed,
            ))

        return events

    def _liveness_of(self, row: Observation) -> tuple[bool, bool]:
        """(died, revived) this row. Only definite verdicts move the needle:
        a None between two Falses is one death, not two, and a None between
        True and False delays the event rather than inventing a second one."""
        if row.alive is None:
            return False, False
        key = self._life_key(row)
        before = self._alive.get(key)
        self._alive[key] = row.alive
        if row.alive:
            return False, before is False
        # A first-ever verdict of False is still a death: joining a stream
        # mid-corpse is late knowledge of a real event, not a non-event.
        return before is not False, False

    @staticmethod
    def _life_key(row: Observation) -> str:
        """Champion when named, track otherwise. A corpse's track is often
        dropped before the respawn, so the name is the identity that survives
        being dead; the track id is what exists before there is a name."""
        return row.champion if row.champion is not None else f"#{row.track_id}"

    def _rosters_from(self, state: FrameState) -> list[Event]:
        events: list[Event] = []
        for team in (Team.BLUE, Team.RED):
            named = frozenset(
                row.champion for row in state.champions
                if row.team is team and row.champion is not None
            )
            if len(named) == 5 and self._rosters.get(team) != named:
                self._rosters[team] = named
                events.append(Event(
                    kind="roster",
                    seq=state.seq,
                    video_time=state.video_time,
                    game_time=state.game_time,
                    team=team,
                    champion=None,
                    track_id=None,
                    detail={"champions": sorted(named)},
                ))
        return events
