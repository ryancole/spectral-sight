"""Turrets on the minimap: which of the 22 are still standing.

The minimap draws every turret of both teams at a fixed place, all game, fog
or not. A lane turret is a shield in its team's colour (with a digit on it,
see below) and it is simply gone once destroyed. A nexus turret is a small
upright diamond beside its nexus, and it is *not* removed when destroyed: it
turns into a grey outline, and it comes back in colour when the turret
rebuilds -- nexus turrets regenerate, and on the 2026-09-25 lane clip the blue
one next to the top lane came back about a minute into the recording. Every
other turret, once destroyed, stays destroyed.

It needs the minimap turned up, like the minion dots: the defaults were
measured at the 486px panel, where a shield is about 23x30 pixels.

**A turret reads standing when its icon's shape and its team colour are both
there.** Each turret is looked for in a small window around a fixed anchor
(its world position projected through the world calibration, nudged to the
icon's centre) by correlating against an averaged icon
(`etc/turrets/<team>_<kind>.png`, averaged over every standing sighting in
the two 2026-09-25 clips), and then the pixels the icon fills in its team's
colour are checked in the frame. On those clips, sampled at 3 Hz:

- every standing icon scored at least 0.5 on shape when uncovered, and at
  least 0.49 on colour (the blue nexus turrets are the weakest; red shields
  sit at 0.8-1.0);
- no empty turret spot scored as much as 0.5 on shape -- a clump of minion
  dots on a fallen turret's spot got as far as 0.46;
- the grey outline of a destroyed nexus turret scores up to 0.97 on shape --
  it *is* the same shape -- but no more than 0.33 on colour.

These numbers are in-sample: the averaged icons were built from the same
footage. On one frame of a third game (the live receiver, same day) all 22
read right, 16 standing and 6 not.

**A missing icon is only evidence when the spot shows bare map.** Anything
drawn over an icon takes its shape and colour with it, and a lot gets drawn
over turrets: champion markers (two or three stacked in a fight, which the
marker detector does not find), map pings, and on the live receiver the
mouse cursor, parked on one turret for the whole of a five-minute
recording. So a reading is `GONE` only when the pixels the icon would fill
hold none of what cover looks like -- white brighter than any icon (the
cursor), saturated colour brighter than any icon (marker rings, pings,
minion dots) or vivid colour in neither team's hue (portraits) -- and
otherwise it is `UNKNOWN`. Team-coloured pixels as dark as an icon are left
alone, because an inhibitor sits against three of the turrets per side. On
the 2026-09-25 live recording, 882 readings of standing turrets that failed
the icon match were all covered, and 7 of them passed as bare map; none of
the 7 was more than 2s from the first. Fallen turrets passed on 50-100% of
readings at open spots, less where something sat on them.

Then a loss must hold: `destroy_readings` bare-map readings spanning
`destroy_hold` seconds, with no standing reading in between. On the three
2026-09-25 clips this was enough that no standing turret was ever taken as
lost; the cost is that a fallen turret with something parked on its spot is
not called until the spot clears -- the player standing on it, say. A lane
turret's loss is never reverted. A nexus turret's is, once it has read
standing for `rebuild_hold`.

The digit on a shield changed as the recorded turrets were hit -- the blue
mid-lane inhibitor turret went 5, 4, 2 and then vanished -- so it looks like
a coarse health reading. It is not read: nothing here has checked what the
digit counts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.types import Team

TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "etc" / "turrets"

REFERENCE_MINIMAP_WIDTH = 486
"""Panel width the templates and offsets were captured at."""


class Tier(Enum):
    OUTER = "outer"
    INNER = "inner"
    INHIBITOR = "inhibitor"
    NEXUS = "nexus"


@dataclass(frozen=True, slots=True)
class Turret:
    """One of the 22 turrets: who owns it and where it stands."""

    team: Team
    lane: str
    """"top", "mid", "bot", or "base" for the nexus turrets."""

    tier: Tier
    world: tuple[float, float]
    """Summoner's Rift units."""

    side: str | None = None
    """For the nexus turrets only, which have the same lane and tier:
    "top" for the one on the top-lane side of the nexus, "bot" for the
    other."""

    @property
    def rebuilds(self) -> bool:
        return self.tier is Tier.NEXUS

    @property
    def kind(self) -> str:
        """Which template draws it."""
        return "shield" if self.tier is not Tier.NEXUS else f"nexus_{self.side}"


def _lane_turrets(
    team: Team, positions: dict[tuple[str, Tier], tuple[float, float]]
) -> list[Turret]:
    return [Turret(team, lane, tier, world) for (lane, tier), world in positions.items()]


TURRETS: tuple[Turret, ...] = (
    *_lane_turrets(Team.BLUE, {
        ("top", Tier.OUTER): (981, 10441),
        ("top", Tier.INNER): (1512, 6699),
        ("top", Tier.INHIBITOR): (1169, 4287),
        ("mid", Tier.OUTER): (5846, 6396),
        ("mid", Tier.INNER): (5048, 4812),
        ("mid", Tier.INHIBITOR): (3651, 3696),
        ("bot", Tier.OUTER): (10504, 1029),
        ("bot", Tier.INNER): (6919, 1483),
        ("bot", Tier.INHIBITOR): (4281, 1253),
    }),
    Turret(Team.BLUE, "base", Tier.NEXUS, (1748, 2270), side="top"),
    Turret(Team.BLUE, "base", Tier.NEXUS, (2177, 1807), side="bot"),
    *_lane_turrets(Team.RED, {
        ("top", Tier.OUTER): (4318, 13875),
        ("top", Tier.INNER): (7943, 13411),
        ("top", Tier.INHIBITOR): (10481, 13650),
        ("mid", Tier.OUTER): (8955, 8510),
        ("mid", Tier.INNER): (9767, 10113),
        ("mid", Tier.INHIBITOR): (11134, 11207),
        ("bot", Tier.OUTER): (13866, 4505),
        ("bot", Tier.INNER): (13327, 8226),
        ("bot", Tier.INHIBITOR): (13624, 10572),
    }),
    Turret(Team.RED, "base", Tier.NEXUS, (12611, 13084), side="top"),
    Turret(Team.RED, "base", Tier.NEXUS, (13052, 12612), side="bot"),
)
"""The 22 turrets in the order the feed reports them. World positions are
the game's own; projected through the 2026-09-25 world calibration every one
lands within a few pixels of its icon."""

ICON_OFFSET = {"shield": (3.0, -6.0), "nexus": (3.0, -5.0)}
"""Icon centre minus projected world position, in minimap pixels at the
reference width. A shield is drawn above the point it marks."""



@dataclass(frozen=True, slots=True)
class TurretConfig:
    """Thresholds for `TurretReader`. Pixel sizes are at
    `REFERENCE_MINIMAP_WIDTH` and scaled with the panel."""

    search: float = 5.0
    """How far from its anchor an icon is looked for."""

    min_shape: float = 0.5
    """Correlation with the averaged icon for a standing reading."""

    min_colour: float = 0.4
    """Share of the icon's filled pixels in team colour for a standing
    reading. The grey outline of a destroyed nexus turret stops at 0.33."""

    max_grey: float = 0.2
    """Below this colour share a matched shape reads destroyed. Between it
    and `min_colour` the reading is inconclusive -- neither state."""

    blue_hue: tuple[int, int] = (90, 112)
    red_hue: tuple[int, int] = (168, 10)
    """Wraps past 179, like every red in this project."""

    min_saturation: int = 90
    min_value: int = 40
    """The icon's dark outline has full saturation and sits under 40."""

    cover_margin: float = 16.0
    """How close a champion marker's rim may come to an anchor before the
    turret counts as covered: about half an icon, since a marker clipping
    the icon's edge is enough to break the shape. Measured on the live
    receiver, a marker whose rim stopped 9px short of the anchor still
    turned a standing shield into a miss."""

    max_clutter: float = 0.02
    """Share of the icon's pixels that may look like cover in a `GONE`
    reading -- see `_clutter`."""

    cover_white: int = 200
    cover_bright: int = 190
    """Brightness above which unsaturated / saturated pixels are cover.
    Turret and inhibitor icons stay under 185."""

    destroy_hold: float = 5.0
    destroy_readings: int = 5
    """Bare-map readings needed, and the span they must cover, before a loss
    is adopted. The longest run of false bare-map readings measured was five
    over 2.0s."""

    stand_readings: int = 2
    """Consecutive standing readings for a first verdict of standing."""

    rebuild_hold: float = 1.5
    """How long a destroyed nexus turret must read standing again before it
    is taken to have rebuilt."""


class Reading(Enum):
    STANDING = "standing"
    GONE = "gone"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TurretState:
    """One turret's verdict, as the feed reports it."""

    turret: Turret
    standing: bool | None
    """None until the turret has been seen clearly enough to call -- the
    live receiver's mouse cursor sat on one for five minutes."""


def load_templates(
    directory: Path = TEMPLATE_DIR,
) -> dict[tuple[Team, str], np.ndarray]:
    """The averaged icons, keyed by (team, kind) -- `<team>_<kind>.png`."""
    templates = {}
    for key in {(t.team, t.kind) for t in TURRETS}:
        path = directory / f"{key[0].value}_{key[1]}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        templates[key] = image
    return templates


class _Verdict:
    """One turret's filter: readings in, a settled state out."""

    __slots__ = ("standing", "_stand_streak", "_stand_since", "_gone_since", "_gone_count")

    def __init__(self) -> None:
        self.standing: bool | None = None
        self._stand_streak = 0
        self._stand_since: float | None = None
        self._gone_since: float | None = None
        self._gone_count = 0

    def update(
        self, reading: Reading, timestamp: float, turret: Turret, cfg: TurretConfig
    ) -> None:
        if reading is Reading.STANDING:
            self._gone_since = None
            self._gone_count = 0
            self._stand_streak += 1
            if self._stand_since is None:
                self._stand_since = timestamp
            if self.standing is None and self._stand_streak >= cfg.stand_readings:
                self.standing = True
            elif (
                self.standing is False
                and turret.rebuilds
                and timestamp - self._stand_since >= cfg.rebuild_hold
                and self._stand_streak >= cfg.stand_readings
            ):
                self.standing = True
        elif reading is Reading.GONE:
            self._stand_streak = 0
            self._stand_since = None
            if self._gone_since is None:
                self._gone_since = timestamp
            self._gone_count += 1
            if (
                self.standing is not False
                and self._gone_count >= cfg.destroy_readings
                and timestamp - self._gone_since >= cfg.destroy_hold
            ):
                self.standing = False
        # UNKNOWN interrupts a standing streak but not a destroyed one: a
        # marker passing over a fallen turret's spot says nothing about it.
        else:
            self._stand_streak = 0
            self._stand_since = None


class TurretReader:
    """Minimap crops in, the state of all 22 turrets out."""

    def __init__(
        self,
        anchors: dict[Turret, tuple[float, float]],
        minimap_width: int = REFERENCE_MINIMAP_WIDTH,
        config: TurretConfig | None = None,
        templates: dict[tuple[Team, str], np.ndarray] | None = None,
    ) -> None:
        """`anchors` are the turrets' world positions projected to
        minimap-crop pixels -- see `project_anchors`."""
        self.config = config or TurretConfig()
        self.scale = minimap_width / REFERENCE_MINIMAP_WIDTH
        raw = templates or load_templates()
        self._templates: dict[tuple[Team, str], np.ndarray] = {}
        self._fills: dict[tuple[Team, str], np.ndarray] = {}
        for key, image in raw.items():
            if self.scale != 1.0:
                image = cv2.resize(
                    image, None, fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_AREA,
                )
            self._templates[key] = image.astype(np.float32)
            self._fills[key] = self._team_mask(image, key[0])
        self._anchors = {}
        for turret, (x, y) in anchors.items():
            dx, dy = ICON_OFFSET["nexus" if turret.tier is Tier.NEXUS else "shield"]
            self._anchors[turret] = (x + dx * self.scale, y + dy * self.scale)
        self._verdicts = {turret: _Verdict() for turret in TURRETS}
        self._read = False

    def reset(self) -> None:
        """Forget every verdict -- a different game is on screen."""
        self._verdicts = {turret: _Verdict() for turret in TURRETS}
        self._read = False

    def _team_mask(self, image: np.ndarray, team: Team) -> np.ndarray:
        cfg = self.config
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (s >= cfg.min_saturation) & (v >= cfg.min_value)
        lo, hi = cfg.blue_hue if team is Team.BLUE else cfg.red_hue
        hue = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return hue & lit

    def read_once(
        self,
        minimap: np.ndarray,
        markers: Iterable[tuple[float, float, float]] = (),
    ) -> dict[Turret, Reading]:
        """This crop's reading of every turret, unfiltered.

        `markers` are champion markers as (x, y, radius); a turret under one
        reads `UNKNOWN`.
        """
        if minimap.ndim != 3 or minimap.shape[2] != 3:
            raise ValueError(f"expected a BGR image, got shape {minimap.shape}")
        cfg = self.config
        markers = list(markers)
        image = minimap.astype(np.float32)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        masks = {team: self._team_mask(minimap, team) for team in (Team.BLUE, Team.RED)}
        height, width = minimap.shape[:2]
        search = max(1, round(cfg.search * self.scale))
        margin = cfg.cover_margin * self.scale

        readings = {}
        for turret in TURRETS:
            key = (turret.team, turret.kind)
            template = self._templates[key]
            th, tw = template.shape[:2]
            ax, ay = self._anchors[turret]
            x0 = round(ax - tw / 2) - search
            y0 = round(ay - th / 2) - search
            if x0 < 0 or y0 < 0 or x0 + tw + 2 * search > width or y0 + th + 2 * search > height:
                readings[turret] = Reading.UNKNOWN
                continue
            window = image[y0 : y0 + th + 2 * search, x0 : x0 + tw + 2 * search]
            scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, shape, _, (lx, ly) = cv2.minMaxLoc(scores)
            patch = masks[turret.team][y0 + ly : y0 + ly + th, x0 + lx : x0 + lx + tw]
            colour = float(patch[self._fills[key]].mean())

            if shape >= cfg.min_shape and colour >= cfg.min_colour:
                readings[turret] = Reading.STANDING
            elif any(
                np.hypot(ax - mx, ay - my) < mr + margin for mx, my, mr in markers
            ):
                readings[turret] = Reading.UNKNOWN
            elif shape >= cfg.min_shape and colour > cfg.max_grey:
                readings[turret] = Reading.UNKNOWN
            elif self._clutter(hsv, turret, key) > cfg.max_clutter:
                readings[turret] = Reading.UNKNOWN
            else:
                readings[turret] = Reading.GONE
        return readings

    def _clutter(self, hsv: np.ndarray, turret: Turret, key: tuple[Team, str]) -> float:
        """Share of the pixels this turret's icon would fill, at its anchor,
        that look like something drawn over the map rather than the map."""
        cfg = self.config
        fill = self._fills[key]
        th, tw = fill.shape
        ax, ay = self._anchors[turret]
        x0, y0 = round(ax - tw / 2), round(ay - th / 2)
        patch = hsv[y0 : y0 + th, x0 : x0 + tw]
        h = patch[..., 0].astype(np.int16)
        s, v = patch[..., 1], patch[..., 2]
        lo, hi = cfg.blue_hue
        team = (h >= lo) & (h <= hi)
        lo, hi = cfg.red_hue
        team |= (h >= lo) | (h <= hi)
        cover = (
            ((s < 60) & (v >= cfg.cover_white))
            | ((s >= 100) & (v >= cfg.cover_bright))
            | ((s >= 100) & (v >= 100) & ~team)
        )
        return float(cover[fill].mean())

    def read(
        self,
        minimap: np.ndarray,
        timestamp: float,
        markers: Iterable[tuple[float, float, float]] = (),
    ) -> tuple[TurretState, ...] | None:
        """Update from one crop; all 22 states, or None when no crop has
        been evidence yet.

        A crop where no turret at all reads standing is not taken as
        evidence: something other than the map is in the panel. The two
        nexus turrets alone keep this from triggering in a real game until
        its last seconds.
        """
        readings = self.read_once(minimap, markers)
        if Reading.STANDING in readings.values():
            self._read = True
            for turret, reading in readings.items():
                self._verdicts[turret].update(reading, timestamp, turret, self.config)
        return self.states()

    def states(self) -> tuple[TurretState, ...] | None:
        if not self._read:
            return None
        return tuple(TurretState(t, self._verdicts[t].standing) for t in TURRETS)


def project_anchors(
    to_crop: Callable[[float, float], tuple[float, float]],
) -> dict[Turret, tuple[float, float]]:
    """Every turret's world position in minimap-crop pixels, given the
    world-to-crop mapping in force."""
    return {turret: to_crop(*turret.world) for turret in TURRETS}
