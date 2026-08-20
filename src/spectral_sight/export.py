"""The pipeline's output: one row per champion per frame, written to disk.

Everything upstream of this module is perception, and perception that is only
ever drawn on screen cannot be built on. A clip runs at roughly 17 fps, so
asking a question about five minutes of footage costs minutes of vision every
time it is asked. Extracting once and querying a file afterwards turns that into
a read, which is the difference between an experiment that is worth running and
one that is not.

**An observation is flat and self-contained.** A `Track` is a live object with
velocity, per-name identity evidence and a lifecycle; none of that survives a
process boundary meaningfully, and freezing it into the file format would tie
every future reader to the tracker's internals. What a consumer needs is where a
named champion was at a known time, so that is what a row is.

**Positions are written in both spaces.** World units are the useful ones and
the whole reason the transform exists, but they are only available when the
world is calibrated, and a row with no position at all would be worse than one
in crop pixels. So crop pixels are always present and world units are present
when they can be.

**Absence is recorded, not skipped.** A row is written for every confirmed
track, including champions currently in fog, carrying `visible` and
`seconds_since_seen`. Dropping unseen champions would make the file silent about
the difference between "elsewhere" and "not looked at", which is most of what
player-perspective footage is actually about.

The format is JSONL: the first line is a `TimelineMeta` object describing the
capture, and every line after it is an `Observation`. One file rather than a
sidecar, because a timeline whose scale and calibration live in a separate file
is one careless copy away from being unreadable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from spectral_sight.types import Team

SCHEMA = 1
"""Bumped when a change would break a reader written against the old format.
Adding an optional field does not count; removing or repurposing one does."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One champion at one instant.

    Times come in two flavours and they are not interchangeable. `video_time` is
    position within this recording and is always present; `game_time` is match
    time and is the only one that means anything outside the clip, but it needs
    a calibrated clock.
    """

    video_time: float
    """Seconds since the start of the source."""

    track_id: int
    team: Team
    x: float
    y: float
    """Position in minimap-crop pixels. Always present."""

    visible: bool
    """Seen on this frame, rather than carried at a last known position.
    For an enemy, False means fog. For an ally, who is never fogged, it means
    death -- but this module does not make that call, it just records the fact
    the caller can make it from. See `alive`, which does make it."""

    seconds_since_seen: float
    """Zero while visible; grows while a champion is unaccounted for."""

    game_time: int | None = None
    """Match time in seconds, when the clock is calibrated and reading."""

    game_time_observed: bool = False
    """False when `game_time` was carried through a frame the reader could not
    resolve rather than read off the screen. Always False when there is no
    game time at all, so a consumer can filter on it without a None check."""

    champion: str | None = None
    """None while a track has not accumulated enough identity evidence."""

    world_x: float | None = None
    world_y: float | None = None
    """Position in Summoner's Rift units, when the world is calibrated."""

    is_self: bool = False
    """The local player, resolved from the camera viewport rather than by
    appearance."""

    alive: bool | None = None
    """Corroborated by the HUD, which draws a dead champion's portrait grey.

    None means unknown, and it is the honest answer far more often than not: no
    HUD panel names an enemy, so every enemy row carries None, and an ally row
    carries None whenever the portraits and the minimap disagree about how many
    teammates are missing.

    That disagreement is the reason this is a separate field rather than a
    rename of `visible`. An ally absent from the minimap is usually dead, but
    sometimes the tracker simply lost them -- and a consumer reading absence as
    death has no way to tell those apart. Here, one of them says None."""

    health: float | None = None
    resource: float | None = None
    """Fill fractions from the champion's floating nameplate, in [0, 1].

    Present only while the champion is on screen *and* their plate could be
    matched to this track, which is a far narrower window than the rest of a
    row: measured on a coop clip, an enemy is inside the camera view in 29% of
    frames against 88% having one somewhere on the minimap. So None here means
    "not looked at", not "unchanged" -- a consumer carrying a value forward must
    decide for itself how long that stays credible.

    `resource` is the only evidence this project has that an enemy used an
    ability, since nothing in the game displays enemy cooldowns: a step down
    that holds is a cast. It is absent for champions on energy, rage or no
    resource at all. The series is kept here raw as well as read, so a consumer
    can call casts differently without re-running the vision -- see the `cast_*`
    fields, and `perception.nameplates.casts` for how one is called."""

    level: int | None = None
    """Champion level, filtered across frames rather than read fresh.

    Raw per-frame readings are wrong about one time in ten -- see
    `LevelFilter` -- so this is the filtered value, and it lags a genuine
    level-up by a frame or two rather than flickering."""

    cast_drop: float | None = None
    """A fall in `resource` judged to be a cast, as a fraction of the pool.

    Present on the row where the cast *settled*, which is a frame or more after
    the fall itself: a step is only a cast once it has held, so the following
    reading is what decides it. `cast_at` says when the fall actually happened,
    and the other three fields say how far the evidence goes."""

    cast_at: float | None = None
    """`video_time` of the reading the fall was measured to. The cast lies
    somewhere in `[cast_at - cast_span, cast_at]`."""

    cast_span: float | None = None
    """Seconds between the two readings the fall was measured across."""

    cast_continuous: bool | None = None
    """Whether that interval was tight enough for this to be a single cast.

    False means the champion was away in between, so the row records the *net*
    spend over `cast_span` -- possibly several abilities, and reduced by
    whatever regenerated. At 0.40 readable enemy plates per frame most evidence
    arrives this way, so these are the bulk of the rows rather than the
    exception."""

    cast_confirmed: bool | None = None
    """Whether the post-drop level held on a continuous follow-up reading.

    False means no such reading arrived -- the champion left view -- not that
    the fall was contradicted. A fall that was contradicted never becomes a cast
    at all, so this never marks one the detector doubts, only one it could not
    finish checking."""

    allies_dead: int | None = None
    """How many of your five teammates were dead on this frame.

    Frame-level rather than per-champion, and repeated on every row of the
    frame the same way `game_time` is. It is a count the HUD gives directly and
    reliably, so it stays readable in the frames where `alive` cannot name
    anyone -- which is what stops "one teammate was down, we cannot say which"
    from being indistinguishable from "nobody was down"."""

    def to_dict(self) -> dict[str, object]:
        """Rounded to the precision the numbers actually carry.

        Sub-hundredth-of-a-pixel detection jitter is noise, and writing it out
        at full float width triples the file size to record it.
        """
        row: dict[str, object] = {
            "video_time": round(float(self.video_time), 3),
            "game_time": self.game_time,
            "game_time_observed": self.game_time_observed,
            "track_id": int(self.track_id),
            "team": self.team.value,
            "champion": self.champion,
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "visible": bool(self.visible),
            "seconds_since_seen": round(float(self.seconds_since_seen), 2),
            "is_self": bool(self.is_self),
            "alive": None if self.alive is None else bool(self.alive),
            "allies_dead": (
                None if self.allies_dead is None else int(self.allies_dead)
            ),
        }
        if self.world_x is not None and self.world_y is not None:
            row["world_x"] = round(float(self.world_x), 1)
            row["world_y"] = round(float(self.world_y), 1)
        # Omitted rather than written as null, because these are absent on most
        # rows -- an enemy is on screen far less often than on the minimap --
        # and a null per field per row would be most of the file.
        if self.health is not None:
            row["health"] = round(float(self.health), 3)
        if self.resource is not None:
            row["resource"] = round(float(self.resource), 3)
        if self.level is not None:
            row["level"] = int(self.level)
        # A cast lands on a handful of rows in a clip, so the whole group is
        # omitted rather than written as five nulls on every other one.
        if self.cast_drop is not None:
            row["cast_drop"] = round(float(self.cast_drop), 3)
            row["cast_at"] = round(float(self.cast_at or 0.0), 3)
            row["cast_span"] = round(float(self.cast_span or 0.0), 3)
            row["cast_continuous"] = bool(self.cast_continuous)
            row["cast_confirmed"] = bool(self.cast_confirmed)
        return row

    @classmethod
    def from_dict(cls, data: dict) -> Observation:
        return cls(
            video_time=float(data["video_time"]),
            track_id=int(data["track_id"]),
            team=Team(data["team"]),
            x=float(data["x"]),
            y=float(data["y"]),
            visible=bool(data["visible"]),
            seconds_since_seen=float(data["seconds_since_seen"]),
            game_time=None if data.get("game_time") is None else int(data["game_time"]),
            game_time_observed=bool(data.get("game_time_observed", False)),
            champion=data.get("champion"),
            world_x=None if data.get("world_x") is None else float(data["world_x"]),
            world_y=None if data.get("world_y") is None else float(data["world_y"]),
            is_self=bool(data.get("is_self", False)),
            health=None if data.get("health") is None else float(data["health"]),
            resource=None if data.get("resource") is None
            else float(data["resource"]),
            level=None if data.get("level") is None else int(data["level"]),
            alive=None if data.get("alive") is None else bool(data["alive"]),
            cast_drop=(
                None if data.get("cast_drop") is None else float(data["cast_drop"])
            ),
            cast_at=None if data.get("cast_at") is None else float(data["cast_at"]),
            cast_span=(
                None if data.get("cast_span") is None else float(data["cast_span"])
            ),
            cast_continuous=(
                None if data.get("cast_continuous") is None
                else bool(data["cast_continuous"])
            ),
            cast_confirmed=(
                None if data.get("cast_confirmed") is None
                else bool(data["cast_confirmed"])
            ),
            allies_dead=(
                None if data.get("allies_dead") is None
                else int(data["allies_dead"])
            ),
        )


@dataclass(frozen=True, slots=True)
class TimelineMeta:
    """What a reader needs to interpret the rows that follow.

    Deliberately about the *capture*, not about results. Anything derivable
    from the rows themselves -- champions seen, duration, frame count -- is left
    out, so the header cannot come to disagree with the body.
    """

    source: str
    """Basename of the clip, not its path. A timeline is a shareable artefact
    and there is no reason for it to carry someone's directory layout."""

    width: int
    height: int
    stride: int
    """Source frames per processed frame. 3 is 10 Hz on 30 fps footage."""

    created: str = ""
    """UTC ISO 8601. Stamped by the writer when left empty."""

    has_game_time: bool = False
    """Whether the clock was calibrated for this run. When False, every row's
    `game_time` is None and the timeline joins to nothing outside itself."""

    has_liveness: bool = False
    """Whether the HUD portraits were calibrated for this run. When False every
    row's `alive` is None because nothing was read, which a consumer must not
    confuse with the None meaning the HUD was read and disagreed."""

    has_nameplates: bool = False
    """Whether nameplates were read for this run, needing both a nameplate
    calibration and a fitted projection. When False every row is missing
    `health`, `resource` and `level` because nothing looked -- as against the
    usual case, where they are missing because the champion was not on screen.
    The distinction matters: a consumer measuring how often an enemy is
    observable would otherwise read a missing calibration as a quiet game."""

    world_bounds: dict[str, float] | None = None
    world_units_per_pixel: list[float] | None = None
    """The world calibration in force, or None if positions are crop pixels
    only. Recorded because a scale is not recoverable from the rows, and two
    timelines extracted under different calibrations are not comparable."""

    schema: int = SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "created": self.created,
            "has_game_time": self.has_game_time,
            "has_liveness": self.has_liveness,
            "has_nameplates": self.has_nameplates,
            "world_bounds": self.world_bounds,
            "world_units_per_pixel": self.world_units_per_pixel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimelineMeta:
        return cls(
            source=str(data["source"]),
            width=int(data["width"]),
            height=int(data["height"]),
            stride=int(data["stride"]),
            created=str(data.get("created", "")),
            has_game_time=bool(data.get("has_game_time", False)),
            has_liveness=bool(data.get("has_liveness", False)),
            has_nameplates=bool(data.get("has_nameplates", False)),
            world_bounds=data.get("world_bounds"),
            world_units_per_pixel=data.get("world_units_per_pixel"),
            schema=int(data.get("schema", SCHEMA)),
        )

    def stamped(self) -> TimelineMeta:
        """A copy with `created` filled in, if it was not already."""
        if self.created:
            return self
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return TimelineMeta(
            source=self.source,
            width=self.width,
            height=self.height,
            stride=self.stride,
            created=now,
            has_game_time=self.has_game_time,
            has_liveness=self.has_liveness,
            has_nameplates=self.has_nameplates,
            world_bounds=self.world_bounds,
            world_units_per_pixel=self.world_units_per_pixel,
            schema=self.schema,
        )


class TimelineWriter:
    """Streams observations to a JSONL file.

    Streaming rather than collecting: extraction runs for minutes, and a crash
    or a Ctrl-C at frame 3,000 should leave 3,000 usable frames behind rather
    than nothing. The header goes down first for the same reason -- a truncated
    file is still interpretable.
    """

    def __init__(self, path: str | Path, meta: TimelineMeta) -> None:
        self.path = Path(path)
        self.meta = meta.stamped()
        self.rows = 0
        self._file: IO[str] | None = None

    def __enter__(self) -> TimelineWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self._file.write(json.dumps(self.meta.to_dict()) + "\n")
        self._file.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, observations: Iterable[Observation]) -> None:
        """Append one frame's worth of rows, and flush.

        Flushing per frame rather than per buffer is what makes the paragraph
        above true rather than aspirational -- rows still sitting in a buffer
        are not on disk. At ten frames a second against a pipeline that spends
        60ms in vision per frame, the cost does not register.
        """
        if self._file is None:
            raise RuntimeError("TimelineWriter must be used as a context manager")
        for observation in observations:
            self._file.write(json.dumps(observation.to_dict()) + "\n")
            self.rows += 1
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def iter_timeline(path: str | Path) -> Iterator[Observation]:
    """Stream a timeline's observations, skipping the header.

    Use this over `read_timeline` for a long clip: a five-minute extraction is
    tens of thousands of rows, and most questions do not need them all resident.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header.strip():
            raise ValueError(f"{path} is empty")
        _check_schema(json.loads(header), path)
        for line in handle:
            if line.strip():
                yield Observation.from_dict(json.loads(line))


def read_timeline(path: str | Path) -> tuple[TimelineMeta, list[Observation]]:
    """Load a whole timeline: its header and every row."""
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header.strip():
            raise ValueError(f"{path} is empty")
        data = json.loads(header)
        _check_schema(data, path)
        meta = TimelineMeta.from_dict(data)
        rows = [
            Observation.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]
    return meta, rows


def _check_schema(header: dict, path: str | Path) -> None:
    """Fail loudly on a format this reader does not understand.

    A newer file read by older code would otherwise deserialise into something
    plausible and subtly wrong, which is the failure mode worth spending an
    exception on.
    """
    if "schema" not in header:
        raise ValueError(f"{path} has no timeline header; is it a timeline file?")
    schema = int(header["schema"])
    if schema > SCHEMA:
        raise ValueError(
            f"{path} is schema {schema}, but this build understands {SCHEMA}"
        )
