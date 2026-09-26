"""The health bars drawn over minions in the world view.

The champion reader in `plates.py` goes out of its way to reject these; this
module is the other half, and reads them for themselves. A minion's bar is the
only thing on screen that says where a minion is *and* how close it is to
dying, which is what last-hitting and wave management are made of.

**A minion bar is a fixed-size box, not a fill.** The client draws a dark frame
the full width of the bar whatever the minion's health, and paints the health
into it from the left: measured on the 2026-08-30 session at 2116x1354, a
66px-by-4px fill inside a one-pixel frame, with no tick marks and no level box.
The frame is what makes a nearly dead minion findable at all -- a 5% minion has
three pixels of colour, which alone is indistinguishable from a spark, but
three pixels of colour followed by exactly sixty-three pixels of dark and then
*not* dark is a minion bar and nothing else.

**Blue is ours, red is theirs**, the same two hues the champion plates use: an
ally minion's bar is the resource-bar blue, an enemy's is the hostile red. That
blue is the reason width matters as much as it does. A champion's resource bar
is the same colour and the same four pixels tall, so the only thing separating
a champion at half mana from a minion at full health is that the resource bar's
empty tail runs on to 117px where a minion's stops at 66. A champion's *health*
bar is eleven pixels tall and never gets this far.

**Overlap is common and only half handled.** A wave stands in a clump, and the
bars of a clump overlap. League draws the nearer minion's bar on top, so the
bar behind loses its right-hand end, and the fill it reports is the fill up to
where the other bar starts -- a confident reading of the wrong thing, exactly
the champion reader's occlusion problem. When another bar's colour is found
inside the dark tail, the minion is kept (its position is still right) and its
health is withheld. What is not handled is two same-team bars at the same
height touching end to end: their fills merge into one run too wide to be a
minion, and both are lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spectral_sight.perception.nameplates.plates import NameplateLayout
from spectral_sight.types import Team


@dataclass(frozen=True, slots=True)
class MinionConfig:
    """Thresholds that do not depend on the resolution.

    Geometry -- the bar's width and height -- lives on `NameplateLayout`.
    """

    red_hue: tuple[int, int] = (168, 4)
    """Wraps past 179: the enemy bar is pinker than it looks, like the champion
    plate's and the minimap ring's."""

    blue_hue: tuple[int, int] = (92, 110)
    min_saturation: int = 70
    min_value: int = 90
    """Lower than the champion plate's floors. A minion bar's fill is lit on
    only its top three rows; the bottom row is shaded down to V 70-80, and
    compression bleeds the frame into the fill's ends."""

    height_slack: int = 2
    width_slack: int = 3
    """Pixels a measured fill may exceed the layout's width by, from
    antialiasing at the frame -- and the tolerance on where the tail ends."""

    min_fill: int = 2
    """Narrowest fill kept. Anything this short is only believed with the
    whole frame around it; see the module docstring."""

    frame_dark: int = 75
    """Brightest a frame pixel can be. The frame is one pixel wide and
    compression softens it to V 35-65 against terrain at 70-90, so this is a
    weak test on its own -- it rules out bright surroundings, not dark ones."""

    tail_dark: float = 50.0
    tail_flat: float = 15.0
    """Brightest mean, and largest spread, the empty part of the bar can have.
    Measured, a real tail is a flat V 30-43 with a spread of 2-11. Scenery
    that happens to be dark beside a speck of colour is rarely that uniform."""

    tail_contrast: float = 20.0
    """How much brighter the ground two pixels above and below the tail must
    be than the tail itself. This is what dark terrain cannot fake: real bars
    measured 35-50 levels darker than their surroundings, the false ones on
    the sample frames within 15."""

    end_edge: float = 20.0
    """The step up in brightness just past the bar's right end. Real bars
    measured +40 to +100; a longer bar -- a champion's resource bar, a
    turret's -- carries on dark, and scenery has no edge there at all."""

    covered: float = 0.8
    """Share of the tail, from the first coloured column rightwards, that must
    be coloured for it to be read as another bar drawn across this one. A bar
    in front covers the right-hand end solidly; chat text and sparks do not."""

    min_shown: int = 4
    """Pixels of empty tail that must be visible before the tail is judged --
    and before a covering bar is believed. Chat text, whose player names are
    in the team colours, puts one letter stroke two pixels after another."""


@dataclass(frozen=True, slots=True)
class Minion:
    """One minion's health bar, as read off a single frame."""

    x: int
    y: int
    """Top-left of the fill, in frame pixels."""

    width: int
    """The layout's bar width, carried so a minion is self-contained."""

    team: Team
    """Blue for the local player's team, red for the enemy."""

    health: float | None
    """Fill fraction in [0, 1], or None when the bar is covered or cut.

    None rather than the measured number for the reason `Nameplate.health` is:
    a bar truncated by the one in front of it reads as a plausible low health,
    and low health is precisely what a last-hit consumer is watching for."""

    occluded: bool = False
    """Another bar is drawn across this one's empty tail."""

    clipped: bool = False
    """The bar runs off the frame or under a HUD panel."""

    @property
    def center(self) -> tuple[float, float]:
        """Bar centre. The minion hangs below it."""
        return self.x + self.width / 2.0, float(self.y)


class MinionReader:
    """Frame in, minion health bars out."""

    def __init__(
        self, layout: NameplateLayout, config: MinionConfig | None = None
    ) -> None:
        if layout.minion_width is None or layout.minion_height is None:
            raise ValueError(
                "nameplate layout has no minion bar geometry "
                "(minion_width / minion_height)"
            )
        self.layout = layout
        self.config = config or MinionConfig()
        self.width = layout.minion_width
        self.height = layout.minion_height

    def _masks(
        self, frame: np.ndarray, hsv: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        if hsv is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat, val = cfg.min_saturation, cfg.min_value
        lo, hi = cfg.red_hue
        red = cv2.inRange(hsv, (lo, sat, val), (179, 255, 255)) | cv2.inRange(
            hsv, (0, sat, val), (hi, 255, 255)
        )
        lo, hi = cfg.blue_hue
        blue = cv2.inRange(hsv, (lo, sat, val), (hi, 255, 255))
        # uint8 0/255 throughout: converting a full frame's mask to bool and
        # back cost as much as building it.
        return red, blue, hsv[..., 2]

    def _excluded(self, x: int, y: int, width: int, height: int) -> bool:
        for x0, y0, x1, y1 in self.layout.exclude:
            if x0 * width <= x <= x1 * width and y0 * height <= y <= y1 * height:
                return True
        return False

    def read(
        self, frame: np.ndarray, hsv: np.ndarray | None = None
    ) -> list[Minion]:
        """Every minion health bar visible in the frame.

        `hsv` is the frame already converted, for a caller that has it --
        the pipeline shares one conversion with the champion plate reader."""
        height, width = frame.shape[:2]
        red, blue, value = self._masks(frame, hsv)
        either = cv2.bitwise_or(red, blue)
        minions: list[Minion] = []
        for team, mask in ((Team.RED, red), (Team.BLUE, blue)):
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            # A full frame of team colour is thousands of specks; the size
            # gates `_judge` opens with are applied here first, in bulk.
            rows = stats[1:]
            cfg = self.config
            h = rows[:, cv2.CC_STAT_HEIGHT]
            w = rows[:, cv2.CC_STAT_WIDTH]
            sized = (
                (np.abs(h - self.height) <= cfg.height_slack)
                & (w >= cfg.min_fill)
                & (w <= self.width + cfg.width_slack)
            )
            for row in rows[sized]:
                minion = self._judge(
                    team, row, value, either, width, height
                )
                if minion is not None:
                    minions.append(minion)
        return minions

    def _judge(
        self,
        team: Team,
        stats: np.ndarray,
        value: np.ndarray,
        either: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Minion | None:
        cfg, bar = self.config, self.width
        x, y = int(stats[cv2.CC_STAT_LEFT]), int(stats[cv2.CC_STAT_TOP])
        w, h = int(stats[cv2.CC_STAT_WIDTH]), int(stats[cv2.CC_STAT_HEIGHT])
        if abs(h - self.height) > cfg.height_slack:
            return None
        if w < cfg.min_fill or w > bar + cfg.width_slack:
            return None
        if int(stats[cv2.CC_STAT_AREA]) < 0.6 * w * h:
            return None
        if x < 4 or y < 3 or y + h + 3 > frame_height:
            return None
        if self._excluded(x, y, frame_width, frame_height):
            return None

        rows = slice(y, y + h)
        profile = value[rows].mean(axis=0)

        # The left frame: something dark within a few pixels of the fill's
        # start. Compression can leave a lighter pixel between the two.
        if profile[x - 4 : x].min() > cfg.frame_dark:
            return None

        right = x + bar
        clipped = right + 5 > frame_width or self._excluded(
            right, y, frame_width, frame_height
        )
        end = min(right, frame_width)

        # The frame's bottom edge, along the whole bar.
        below = value[y + h : y + h + 3, x:end].min(axis=0)
        if np.median(below) > cfg.frame_dark:
            return None

        occluded = False
        if not clipped:
            tail_end = right - 1
            if w < bar - cfg.width_slack:
                tail = either[rows, x + w + 1 : tail_end].any(axis=0)
                lit = np.flatnonzero(tail)
                dark_end = tail_end
                if lit.size:
                    first = int(lit[0])
                    if tail[first:].mean() < cfg.covered:
                        return None
                    occluded = True
                    dark_end = x + w + 1 + first
                # A covered bar still has to show some of its own empty tail
                # before the one in front starts: colour straight after colour
                # is text or a spark, and a bar cut exactly at its fill would
                # report no health anyway.
                shown = dark_end - (x + w + 1)
                if occluded and shown < cfg.min_shown:
                    return None
                if shown >= cfg.min_shown and not self._dark_tail(
                    value, x + w + 1, dark_end, y, h
                ):
                    return None
            if not occluded:
                # Past the frame the bar must stop. Still dark means a longer
                # bar: a champion's resource bar with its empty tail, a turret's.
                inside = value[rows, right - 3 : right].mean()
                beyond = value[rows, right + 2 : right + 5].mean()
                if beyond - inside < cfg.end_edge and w < bar - cfg.width_slack:
                    return None
                if beyond <= cfg.tail_dark:
                    return None

        unreadable = occluded or clipped
        return Minion(
            x=x,
            y=y,
            width=bar,
            team=team,
            health=None if unreadable else min(w / bar, 1.0),
            occluded=occluded,
            clipped=clipped,
        )

    def _dark_tail(
        self, value: np.ndarray, x0: int, x1: int, y: int, h: int
    ) -> bool:
        """Is `[x0, x1)` the empty part of a bar: flat, dark, and darker than
        the ground just above and below it?"""
        cfg = self.config
        tail = value[y : y + h, x0:x1].astype(np.float32)
        if tail.mean() > cfg.tail_dark or tail.std() > cfg.tail_flat:
            return False
        above = value[max(y - 4, 0) : y - 2, x0:x1].mean()
        below = value[y + h + 2 : y + h + 4, x0:x1].mean()
        return (above + below) / 2 - tail.mean() >= cfg.tail_contrast
