"""The floating bars drawn over champions in the world view.

Everything else this project reads is either on the minimap or in a fixed HUD
panel. A nameplate is neither: it follows a champion around the 3D view and only
exists while that champion is on screen. That is a much smaller window than the
minimap offers -- measured on a coop-vs-AI clip, at least one enemy sits inside
the camera viewport in 29% of frames against 88% having one somewhere on the
minimap -- but it is the only window in which an enemy's *resource* is legible,
and the resource bar is the only champion-agnostic evidence of a cast that the
client renders. Nothing in the game displays an enemy's cooldowns.

**Anchored on the resource bar, not the health bar.** Minions, wards, turrets
and monsters all draw a health bar; none draw a resource bar directly beneath
one. Pairing the two is what makes a bar a champion's bar. The resource run is
also the cleaner anchor geometrically, being unbroken, while a champion health
bar is divided by tick marks into a dozen disconnected pieces.

**A real bar start carries a level box.** This is what separates a resource run
that a champion model or a spell effect has split in two from the run belonging
to the next champion along. Both are short, collinear and nearby, so no
threshold on gap or width tells them apart. Measured, the box is 81-89% dark
pixels against 24% at a point partway along a lit bar. Without this test the
right-hand fragment anchors a plate of its own, measured from the wrong left
edge; linked across frames the pair reads as a large sudden drop, which is
indistinguishable from a cast and far more common than one. On the sample clip
it accounted for essentially every candidate cast above 4%.

**The bar's total width is fixed**, whatever the champion's maximum health or
mana -- which is what makes a fill *fraction* recoverable at all, since a filled
run of 92px says nothing without a denominator. It is a calibration constant,
not a derivable one, so it lives in ``etc/nameplates/`` beside the minimap and
clock calibrations.

**Fills are measured by walking from the left edge with a gap tolerance**, not
by taking a connected component. The tick marks are the reason: they are gaps of
two or three pixels *inside* the filled region, so a measurement that stops at
the first gap reports the health at the first tick instead of the health.

Two things are deliberately not read. A shield renders as a violet segment
appended to the health bar, so `health` excludes it and a champion gaining one
shows no change here. And champions on energy, rage, or no resource at all draw
no blue bar, so they yield no `resource` and no cast evidence by this route.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.hud.clock import (
    ClockConfig,
    GlyphSet,
    centred,
    glyph_boxes,
    lit_mask,
)

LAYOUT_DIR = Path(__file__).resolve().parents[4] / "etc" / "nameplates"


@dataclass(frozen=True, slots=True)
class NameplateConfig:
    """Thresholds that do not depend on the resolution.

    Colours and tolerances, as against `NameplateLayout`, which is the geometry
    and does depend on it.
    """

    health_hue: tuple[int, int] = (168, 180)
    """Hostile bars are magenta-red, wrapping past 180 into the low hues -- the
    same band the minimap's enemy ring sits in, and for the same reason: a naive
    0-10 red window mostly misses it."""

    ally_hue: tuple[int, int] = (35, 85)
    resource_hue: tuple[int, int] = (94, 106)

    min_saturation: int = 150
    ally_min_saturation: int = 120
    resource_min_saturation: int = 100
    min_value: int = 80
    resource_min_value: int = 60

    tick_gap: int = 5
    """Longest run of empty pixels treated as a tick mark rather than the end of
    the fill."""

    min_bar_width: int = 12
    """Shortest resource run that can anchor a plate.

    This is a floor on the champion's *resource*, and therefore a blind spot:
    below roughly a tenth of their bar there is not enough blue left to anchor
    on, and at zero there is none at all, so the plate is not found. That is
    exactly the champion who has just spent everything, which is unfortunate --
    but the alternative is anchoring on something that is not a champion. The
    level box test catches most of what a lower floor would let through, which
    is why this sits at 12px rather than the 20 it needed before."""

    max_bar_height: int = 8
    min_health_pixels: int = 12

    level_dark_fraction: float = 0.6
    """How much of the level box must be dark for a bar start to be real."""

    fragment_dy: int = 4
    min_plate_separation: int = 30
    """Closest two distinct plates may sit before being treated as one. Nearer
    than this and the plate behind is almost entirely covered anyway, so nothing
    readable is lost by collapsing the pair."""

    obstruction_slack: int = 2
    """How close the two fills may sit, in pixels, before the bar is read as cut
    rather than believed. Two independent quantities do not agree to a pixel by
    accident; a bar cut at one column reports that column twice."""

    obstructed_below: float = 0.95
    """Fraction of the bar below which the obstruction test applies.

    A champion at full health and full mana genuinely reads equal on both bars,
    and that is the commonest state in the game. A cut this near the right end
    is also nearly harmless, since what it reports is close to the truth."""

    level_stroke: int = 13
    level_min_score: float = 0.45
    level_min_margin: float = 0.02
    """The level glyphs are the clock's face at a smaller size -- 7x10 against
    9x13 -- so they are rescaled and matched against the glyph set that already
    exists rather than needing one of their own. Rescaling costs accuracy, hence
    thresholds looser than the clock's 0.55/0.04. Roughly one reading in ten
    still comes back '1' when the champion is on 3, 4 or 5, and those are full
    height rather than clipped fragments, so no size check removes them --
    `LevelFilter` is what does."""


@dataclass(frozen=True, slots=True)
class NameplateLayout:
    """Where a nameplate's parts sit, for one resolution.

    Hand-calibrated like the minimap panel and the clock strip, because the
    numbers are fixed for a resolution but not derivable from it.
    """

    bar_width: int
    """Total width of the bar, full or empty. The fill denominator."""

    bar_height: int
    """Height of the health bar, sampled as a band and OR-reduced so a single
    row cannot land on a tick mark or an antialiased edge."""

    resource_dy: tuple[int, int]
    """Range of offsets from the health bar's top to the resource bar's top."""

    level_dx: tuple[int, int]
    level_dy: tuple[int, int]
    """The level box, relative to the bar's left edge and the *resource* bar's
    top -- the resource run is what the reader anchors on, so offsets are
    expressed from it rather than from the health bar."""

    exclude: tuple[tuple[float, float, float, float], ...] = ()
    """Screen regions that draw bar-like art of their own: the HUD strips, the
    minimap, the death recap. Fractions of the frame, since they track the HUD
    layout rather than the pixel grid."""

    projection_x: tuple[float, float, float] | None = None
    projection_y: tuple[float, float, float] | None = None
    """Coefficients on (u, v, 1) mapping screen position to a position within
    the minimap viewport rectangle. See `projection.py`. Optional because the
    geometry above is usable on its own, and fitting these needs a second pass
    over footage."""

    def to_dict(self) -> dict[str, object]:
        return {
            "bar_width": self.bar_width,
            "bar_height": self.bar_height,
            "resource_dy": list(self.resource_dy),
            "level_dx": list(self.level_dx),
            "level_dy": list(self.level_dy),
            "exclude": [list(box) for box in self.exclude],
            "projection_x": None if self.projection_x is None
            else list(self.projection_x),
            "projection_y": None if self.projection_y is None
            else list(self.projection_y),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NameplateLayout:
        return cls(
            bar_width=int(data["bar_width"]),
            bar_height=int(data["bar_height"]),
            resource_dy=tuple(int(v) for v in data["resource_dy"]),
            level_dx=tuple(int(v) for v in data["level_dx"]),
            level_dy=tuple(int(v) for v in data["level_dy"]),
            exclude=tuple(
                tuple(float(v) for v in box) for box in data.get("exclude", [])
            ),
            projection_x=None if data.get("projection_x") is None
            else tuple(float(v) for v in data["projection_x"]),
            projection_y=None if data.get("projection_y") is None
            else tuple(float(v) for v in data["projection_y"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NameplateLayout:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(cls, width: int, height: int) -> NameplateLayout:
        """Load the calibrated layout for a resolution from ``etc/nameplates/``."""
        path = LAYOUT_DIR / f"{width}x{height}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no nameplate calibration at {path}. "
                f"Run: python tools/calibrate_nameplates.py --input <clip>"
            )
        return cls.load(path)


@dataclass(frozen=True, slots=True)
class Nameplate:
    """One champion's floating bar, as read off a single frame."""

    x: int
    y: int
    """Top-left of the health bar, in frame pixels."""

    width: int
    """The layout's bar width, carried so a plate is self-contained."""

    health: float | None
    resource: float | None
    """Fill fractions in [0, 1], or None when another plate overlaps this one.

    None rather than the measured number, because the measured number is not
    merely noisy -- it is a confident reading of the wrong thing. Two champions
    standing together draw overlapping plates, and the one behind is truncated
    at whatever pixel the plate in front begins, which reads as a sharp drop to
    a plausible fill. A consumer watching for sharp drops cannot tell that from
    real damage, or from the resource step this module exists to find."""

    hostile: bool
    """Red bar rather than green. An enemy champion only ever renders hostile."""

    occluded: bool = False
    """Another plate is drawn across this one."""

    clipped: bool = False
    """The bar runs off the frame, or into a region the HUD owns.

    Kept apart from `occluded` even though both blank the fills, because the
    two have different signatures and only one of them is obvious. A clipped
    bar is cut at a fixed column, so *both* fills truncate at the same place
    and come back equal -- 0.222 and 0.222 for a champion at the right-hand
    screen edge, which reads as a wounded champion low on mana rather than as
    an artefact. It was worth a field to be able to say which happened."""

    obstructed: bool = False
    """The bar is cut partway along by something drawn in the world.

    A champion model, a spell effect, a jungle camp -- anything that is neither
    another plate nor the frame edge, and so is caught by neither `occluded` nor
    `clipped`. It gives itself away because whatever covers the bar covers both
    of them at the same column, so the two fills come back equal: on a sample
    clip the readings that produced false casts had fills 0.008-0.009 apart, one
    pixel of a 117px bar, against 0.07-0.26 for readings that held up.

    A champion genuinely at equal health and mana is blanked too, and the clip
    says how often that costs anything: 90 of 2,840 readings are caught, 3.2%,
    and they fall into 18 runs of which the longest are 23, 21 and 11
    consecutive frames. 87% sit inside a run of three or more. A coincidence
    breaks the moment either bar moves, so a 2.3-second run of exact one-pixel
    agreement is an obstruction; the eight isolated frames are the real cost,
    0.3% of readings, against a third of the false casts this removes.

    Kept as its own flag for the same reason `clipped` is: all three blank the
    fills, and which one happened is the difference between a champion who is
    somewhere unreadable and one who is behind a wall of minions."""

    level: int | None = None
    """Champion level, when the box left of the bar resolves.

    Worth having for itself, and for reading an ability's cost against the pool
    it came out of. It is *not* needed to keep a level-up from being read as a
    cast: levelling grants current resource along with maximum, so the fraction
    holds or rises rather than falling -- measured across seven level-ups on the
    sample clip, from -0.9% to +5.2%. Raw readings are unreliable enough that a
    consumer should put them through `LevelFilter` rather than trusting one
    frame."""

    @property
    def center(self) -> tuple[float, float]:
        """Bar centre. The champion's model hangs below this point rather than
        sitting at it -- the plate floats overhead by a champion-dependent
        amount, which is part of why `projection` is fitted and not derived."""
        return self.x + self.width / 2.0, float(self.y)


class NameplateReader:
    """Frame in, champion nameplates out."""

    def __init__(
        self,
        layout: NameplateLayout,
        glyphs: GlyphSet | None = None,
        config: NameplateConfig | None = None,
    ) -> None:
        self.layout = layout
        self.glyphs = glyphs
        """The clock's glyph set. Levels are read only when it is supplied."""
        self.config = config or NameplateConfig()
        self._clock_config = ClockConfig()

    # -- masks ------------------------------------------------------------

    def _masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = cfg.health_hue
        red = (((h >= lo) & (h <= hi)) | (h <= 3)) & (s > cfg.min_saturation) & (
            v > cfg.min_value
        )
        lo, hi = cfg.ally_hue
        green = (h >= lo) & (h <= hi) & (s > cfg.ally_min_saturation) & (
            v > cfg.min_value
        )
        lo, hi = cfg.resource_hue
        blue = (h >= lo) & (h <= hi) & (s > cfg.resource_min_saturation) & (
            v > cfg.resource_min_value
        )
        return red, green, blue

    def _excluded(self, x: int, y: int, width: int, height: int) -> bool:
        for x0, y0, x1, y1 in self.layout.exclude:
            if x0 * width <= x <= x1 * width and y0 * height <= y <= y1 * height:
                return True
        return False

    def _clipped(self, x: int, y: int, width: int, height: int) -> bool:
        """Does the bar run off the frame, or into a region the HUD owns?

        `_excluded` drops a bar whose *start* is inside a HUD region, since
        that is HUD art rather than a champion. This is the other half: a bar
        that starts in the world and ends underneath a panel is a real
        champion whose fills cannot be read.
        """
        right = x + self.layout.bar_width
        if x < 0 or right > width:
            return True
        return self._excluded(right, y, width, height)

    def _obstructed(self, health: int, resource: int) -> bool:
        """Do both fills stop at the same column, short of the end of the bar?

        Then something in the world is drawn across the plate and cut them both
        there. `_clipped` catches the frame edge and the HUD panels because it
        can reason about where those are; nothing can be assumed about where a
        champion model is, so this reads the symptom instead.
        """
        limit = self.layout.bar_width * self.config.obstructed_below
        return (
            abs(health - resource) <= self.config.obstruction_slack
            and max(health, resource) < limit
        )

    def _fill(self, band: np.ndarray) -> int:
        """Filled width in pixels, walking from the left and hopping ticks."""
        on = band.any(axis=0)
        filled = gap = 0
        for index, lit in enumerate(on):
            if lit:
                filled, gap = index + 1, 0
            else:
                gap += 1
                if gap > self.config.tick_gap:
                    break
        return filled

    # -- level ------------------------------------------------------------

    def _anchored(self, frame: np.ndarray, x: int, resource_top: int) -> bool:
        """Is there a level box to the left, marking a real bar start?"""
        box = self._level_box(frame, x, resource_top)
        if box is None:
            return False
        value = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)[..., 2]
        return bool((value < 70).mean() >= self.config.level_dark_fraction)

    def _level_box(
        self, frame: np.ndarray, x: int, resource_top: int
    ) -> np.ndarray | None:
        y0 = resource_top + self.layout.level_dy[0]
        y1 = resource_top + self.layout.level_dy[1]
        x0 = x + self.layout.level_dx[0]
        x1 = x + self.layout.level_dx[1]
        if y0 < 0 or x0 < 0 or y1 > frame.shape[0] or x1 > frame.shape[1]:
            return None
        if y1 <= y0 or x1 <= x0:
            return None
        return frame[y0:y1, x0:x1]

    def read_level(
        self, frame: np.ndarray, x: int, resource_top: int
    ) -> int | None:
        """Champion level from the box left of the bar, or None if unreadable."""
        if self.glyphs is None:
            return None
        box = self._level_box(frame, x, resource_top)
        if box is None or box.size == 0:
            return None

        cfg = self.config
        mask = lit_mask(box, self._clock_config)
        grey = np.where(mask, cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), 0).astype(
            np.uint8
        )

        digits: list[str] = []
        for gx, gy, gw, gh in glyph_boxes(box, self._clock_config):
            if gh < 4:
                continue
            patch = grey[gy : gy + gh, gx : gx + gw]
            scale = cfg.level_stroke / gh
            grown = cv2.resize(
                patch,
                (max(1, round(gw * scale)), cfg.level_stroke),
                interpolation=cv2.INTER_CUBIC,
            )
            label, score, margin = self.glyphs.match(
                centred(grown, self.glyphs.size)
            )
            if not label.isdigit():
                return None
            if score < cfg.level_min_score or margin < cfg.level_min_margin:
                return None
            digits.append(label)

        if not digits or len(digits) > 2:
            return None
        value = int("".join(digits))
        return value if 1 <= value <= 18 else None

    # -- reading ----------------------------------------------------------

    def read(self, frame: np.ndarray) -> list[Nameplate]:
        """Every champion nameplate visible in the frame."""
        cfg, layout = self.config, self.layout
        height, width = frame.shape[:2]
        red, green, blue = self._masks(frame)

        binary = (blue.astype(np.uint8)) * 255
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
        )
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

        bars = []
        for row in stats[1:]:
            bx, by = int(row[cv2.CC_STAT_LEFT]), int(row[cv2.CC_STAT_TOP])
            bw, bh = int(row[cv2.CC_STAT_WIDTH]), int(row[cv2.CC_STAT_HEIGHT])
            if bw < cfg.min_bar_width or bh > cfg.max_bar_height or bw < bh * 4:
                continue
            bars.append((bx, by, bw, bh))

        plates: list[Nameplate] = []
        for bx, by, bw, bh in self._merge_fragments(frame, bars):
            if self._excluded(bx, by, width, height):
                continue
            top = self._health_top(red | green, bx, by)
            if top is None:
                continue
            band = (
                slice(top, top + layout.bar_height),
                slice(bx, bx + layout.bar_width),
            )
            red_fill = self._fill(red[band])
            green_fill = self._fill(green[band])
            if max(red_fill, green_fill) < cfg.min_health_pixels:
                continue
            resource = self._fill(
                blue[slice(by, by + bh), slice(bx, bx + layout.bar_width)]
            )
            clipped = self._clipped(bx, top, width, height)
            health_fill = max(red_fill, green_fill)
            obstructed = self._obstructed(health_fill, resource)
            unreadable = clipped or obstructed
            plates.append(
                Nameplate(
                    x=bx,
                    y=top,
                    width=layout.bar_width,
                    health=None if unreadable
                    else min(health_fill / layout.bar_width, 1.0),
                    resource=None if unreadable
                    else min(resource / layout.bar_width, 1.0),
                    hostile=red_fill >= green_fill,
                    clipped=clipped,
                    obstructed=obstructed,
                    level=self.read_level(frame, bx, by),
                )
            )
        return self._mark_occlusion(plates)

    def _health_top(
        self, health: np.ndarray, x: int, resource_top: int
    ) -> int | None:
        """First row of the health bar sitting above a resource bar.

        Two things this must not do, both of which a simpler version does.

        It must not try each offset in `resource_dy` and take the first that
        sees any health: a band starting anywhere inside a ten-pixel bar still
        contains health, so the smallest offset always "works" and the plate's
        y comes out several pixels low -- which then shifts the level box
        window and every occlusion comparison with it.

        And it must not walk a contiguous run upward. Overlapping plates have
        touching health bars, so the run climbs out of this plate and into its
        neighbour's, and the offset check then rejects one of the two. So the
        *bottom* of the run is found and the top derived from the known bar
        height, sampling a narrow strip at the left edge -- the part of a bar
        least likely to have another plate drawn across it, since the plate in
        front is the one further down the screen and to the right.
        """
        layout = self.layout
        window_top = max(resource_top - layout.resource_dy[1] - layout.bar_height, 0)
        if window_top >= resource_top:
            return None
        strip = min(layout.bar_width, 24)
        rows = health[window_top:resource_top, x : x + strip].any(axis=1)
        lit = np.nonzero(rows)[0]
        if lit.size == 0:
            return None

        health_top = window_top + int(lit[-1]) - layout.bar_height + 1
        if health_top < 0:
            return None
        offset = resource_top - health_top
        if not layout.resource_dy[0] <= offset <= layout.resource_dy[1]:
            return None
        return health_top

    # -- internals --------------------------------------------------------

    def _merge_fragments(
        self, frame: np.ndarray, bars: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Fold split resource runs back into the bar they belong to."""
        cfg, layout = self.config, self.layout
        anchors, loose = [], []
        for bar in sorted(bars, key=lambda b: (b[1], b[0])):
            target = anchors if self._anchored(frame, bar[0], bar[1]) else loose
            target.append(bar)

        # The level box test has a few pixels of slack: a fragment starting just
        # right of a real bar start still sees most of that bar's box and
        # passes. Two anchors this close are one plate, and the leftmost is the
        # edge the fill has to be measured from.
        deduped: list[tuple[int, int, int, int]] = []
        for bx, by, bw, bh in anchors:
            for index, (mx, my, mw, mh) in enumerate(deduped):
                if (abs(my - by) <= cfg.fragment_dy
                        and abs(mx - bx) <= cfg.min_plate_separation):
                    left = min(mx, bx)
                    deduped[index] = (
                        left, my, max(mw, bx + bw - left), max(mh, bh)
                    )
                    break
            else:
                deduped.append((bx, by, bw, bh))

        merged = [list(bar) for bar in deduped]
        for bx, by, bw, bh in loose:
            for bar in merged:
                if (abs(bar[1] - by) <= cfg.fragment_dy
                        and 0 < bx - bar[0] < layout.bar_width):
                    bar[2] = max(bar[2], bx + bw - bar[0])
                    break
        return [tuple(bar) for bar in merged]

    def _mark_occlusion(self, plates: list[Nameplate]) -> list[Nameplate]:
        """Blank the fills of any plate another plate is drawn over.

        Which plate is in front is decided by draw order: League draws the
        nearer champion's plate on top, and nearer means further down the
        screen, so the larger y wins. Ties go to the left, which is arbitrary
        but stable. The level survives, since it is read from the box left of
        the bar -- a plate covering that covers the whole plate, and then there
        is no detection to blank.
        """
        out = []
        for plate in plates:
            hidden = False
            for other in plates:
                if other is plate:
                    continue
                if abs(other.y - plate.y) > self.layout.bar_height:
                    continue
                in_front = (other.y, -other.x) > (plate.y, -plate.x)
                if in_front and abs(other.x - plate.x) < self.layout.bar_width:
                    hidden = True
                    break
            out.append(
                plate if not hidden else Nameplate(
                    x=plate.x, y=plate.y, width=plate.width,
                    health=None, resource=None, hostile=plate.hostile,
                    occluded=True, clipped=plate.clipped,
                    obstructed=plate.obstructed, level=plate.level,
                )
            )
        return out
