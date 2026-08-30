"""The game clock in the top-right HUD strip.

Everything else this project extracts is stamped with a *video* timestamp, which
is only meaningful inside one recording. The game clock converts those into
*game* time, and game time is the key that joins minimap positions to everything
outside the footage: objective spawns, level and item timings, ward duration,
and any comparison between two clips. It also detects pauses, and makes a clip
that was trimmed or seeked into self-describing rather than silently offset.

**Reading it needs no OCR dependency and no font asset**, because the clock
carries enough structure to bootstrap its own templates. The seconds ones-digit
counts 0-9 in order, once per second, forever, and it rolls over to zero exactly
when the tens-digit changes. So walking a clip frame by frame and watching which
glyph cells change is enough to label all ten digits with no human input --
`tools/calibrate_clock.py` does this in about twenty seconds of footage. That is
the same trick as the automatic ground truth in the README: the game is already
telling us the answer, we just have to notice.

Two measured properties of the strip make segmentation easy:

- **The digits are near-white and the adjacent clock icon is gold.** Measured on
  real footage the digits sit at S~11 while the icon is at S~61, so a saturation
  ceiling separates them cleanly and a sloppy calibration box that includes the
  icon still works.
- **The panel behind them is dark and flat**, V~29 against V>120 for the glyph
  strokes, so a value floor finds the text with no adaptive thresholding.

Glyphs are segmented as runs of columns containing text rather than assumed to
sit on a fixed pitch. The HUD font is not quite tabular -- a '1' is a pixel or
two narrower than a '0' -- so fixed cells drift as the digits change, while
column runs do not care.

Because the font size is fixed once the resolution is, glyphs are compared at
their native size: each one is centred into a canvas big enough for the largest
of them, and matched by normalised cross-correlation. No rescaling, so nothing
is thrown away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

CLOCK_DIR = Path(__file__).resolve().parents[4] / "etc" / "clock"

COLON = ":"
"""Label used for the separator glyph, alongside the digit labels '0'-'9'."""


@dataclass(frozen=True, slots=True)
class GameClock:
    """One reading of the match timer."""

    total_seconds: int
    confidence: float
    """Smallest per-glyph match margin in the reading. Not a probability."""

    observed: bool = True
    """False when this value was carried forward by `ClockFilter` through a
    frame the reader could not resolve, rather than read off the screen."""

    @property
    def minutes(self) -> int:
        return self.total_seconds // 60

    @property
    def seconds(self) -> int:
        return self.total_seconds % 60

    def __str__(self) -> str:
        return f"{self.minutes:d}:{self.seconds:02d}"


@dataclass(frozen=True, slots=True)
class ClockRegion:
    """Where the timer digits sit, in frame pixel coordinates.

    Tight to the glyphs rather than to the HUD panel, because a tight box is
    what makes the run segmentation unambiguous.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"clock region must have positive extent, got {self}")

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the timer sub-image. This is a view, not a copy."""
        frame_h, frame_w = image.shape[:2]
        if self.x + self.width > frame_w or self.y + self.height > frame_h:
            raise ValueError(
                f"region {self} does not fit inside a {frame_w}x{frame_h} frame"
            )
        return image[self.y : self.y + self.height, self.x : self.x + self.width]

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> ClockRegion:
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(frozen=True, slots=True)
class ClockConfig:
    """Thresholds for pulling the glyphs out of the strip.

    The digits are light grey text on a dark, flat panel, so they are found by
    brightness. `max_saturation` is what keeps the gold clock icon beside them
    out of the result even when the calibrated box includes it.
    """

    min_value: int = 110
    max_saturation: int = 45

    min_glyph_width: int = 2
    """Below this a column run is antialiasing or a stray pip, not a glyph."""

    min_glyph_pixels: int = 6
    """Total lit pixels required before a run is considered a glyph at all."""

    min_score: float = 0.55
    """Floor on the best normalised correlation for a glyph to be accepted."""

    min_margin: float = 0.04
    """How far the best match must beat the runner-up.

    Margin rather than raw score, for the reason the gallery uses it: a blurred
    glyph correlates decently with several templates at once, and it is the gap
    that says whether the winner means anything.
    """


def lit_mask(strip: np.ndarray, config: ClockConfig) -> np.ndarray:
    """Boolean mask of glyph pixels: bright and unsaturated.

    Public because the champion level box holds the same face on the same dark
    ground, so the nameplate reader segments its digits with this rather than
    growing a second copy that could drift from it.
    """
    if strip.ndim != 3 or strip.shape[2] != 3:
        raise ValueError(f"expected a BGR image, got shape {strip.shape}")
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] >= config.min_value) & (hsv[..., 1] <= config.max_saturation)


def _column_runs(mask: np.ndarray, config: ClockConfig) -> list[tuple[int, int]]:
    """Half-open column spans containing text, left to right."""
    lit = mask.any(axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(lit):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(lit)))
    return [
        (a, b)
        for a, b in runs
        if b - a >= config.min_glyph_width
        and int(mask[:, a:b].sum()) >= config.min_glyph_pixels
    ]


def centred(patch: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Place a tight glyph in the centre of a fixed canvas, as float32 [0, 1].

    Glyphs are kept at native size because the HUD font does not change scale
    within a resolution. Anything that does not fit is shrunk to fit, which only
    happens if the calibration is wrong.

    Public for the same reason as `lit_mask`. Note that a caller matching glyphs
    of a *different* size against this glyph set -- the level box is the case
    that exists -- has to rescale before centring, since centring alone would
    compare a small digit against a large template.
    """
    width, height = size
    patch = patch.astype(np.float32) / 255.0
    ph, pw = patch.shape[:2]
    if ph > height or pw > width:
        scale = min(height / ph, width / pw)
        patch = cv2.resize(
            patch, (max(1, int(pw * scale)), max(1, int(ph * scale))),
            interpolation=cv2.INTER_AREA,
        )
        ph, pw = patch.shape[:2]
    canvas = np.zeros((height, width), np.float32)
    top = (height - ph) // 2
    left = (width - pw) // 2
    canvas[top : top + ph, left : left + pw] = patch
    return canvas


def glyph_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised correlation in [-1, 1]."""
    av = a.ravel() - a.mean()
    bv = b.ravel() - b.mean()
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denominator < 1e-9:
        return 0.0
    return float(np.dot(av, bv) / denominator)


@dataclass(slots=True)
class GlyphSet:
    """Labelled glyph images, at one fixed canvas size.

    Built by `tools/calibrate_clock.py` and persisted per resolution, because
    the HUD font scales with the display.
    """

    glyphs: dict[str, np.ndarray]
    size: tuple[int, int]
    """(width, height) of the canvas every glyph is centred into."""

    def __len__(self) -> int:
        return len(self.glyphs)

    def match(self, canvas: np.ndarray) -> tuple[str, float, float]:
        """Best label for a glyph canvas, with its score and its margin."""
        scored = sorted(
            ((glyph_similarity(canvas, g), label) for label, g in self.glyphs.items()),
            reverse=True,
        )
        best_score, best_label = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        return best_label, best_score, best_score - runner_up

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(list(self.glyphs), dtype=object),
            **{f"g{i}": g for i, g in enumerate(self.glyphs.values())},
        )

    @classmethod
    def load(cls, path: str | Path) -> GlyphSet:
        with np.load(path, allow_pickle=True) as data:
            labels = [str(v) for v in data["labels"]]
            glyphs = {label: data[f"g{i}"] for i, label in enumerate(labels)}
        first = next(iter(glyphs.values()))
        return cls(glyphs=glyphs, size=(first.shape[1], first.shape[0]))


class ClockReader:
    """Frame in, game clock out."""

    def __init__(
        self,
        region: ClockRegion,
        glyphs: GlyphSet,
        config: ClockConfig | None = None,
    ) -> None:
        self.region = region
        self.glyphs = glyphs
        self.config = config or ClockConfig()

    def glyph_canvases(self, frame: np.ndarray) -> list[np.ndarray]:
        """Segment the strip into normalised glyph canvases, left to right.

        Exposed because the calibration tool needs exactly this, one frame at a
        time, before any labels exist to match against.
        """
        strip = self.region.crop(frame)
        return segment_glyphs(strip, self.glyphs.size, self.config)

    def read(self, frame: np.ndarray) -> GameClock | None:
        """Read the timer, or None if the strip does not resolve to a time."""
        canvases = self.glyph_canvases(frame)
        if not canvases:
            return None

        labels: list[str] = []
        confidence = 1.0
        for canvas in canvases:
            label, score, margin = self.glyphs.match(canvas)
            if score < self.config.min_score or margin < self.config.min_margin:
                return None
            labels.append(label)
            confidence = min(confidence, margin)

        return _parse(labels, confidence)


def _boxes_from_mask(
    mask: np.ndarray, config: ClockConfig
) -> list[tuple[int, int, int, int]]:
    boxes = []
    for x0, x1 in _column_runs(mask, config):
        rows = np.nonzero(mask[:, x0:x1].any(axis=1))[0]
        boxes.append((x0, int(rows[0]), x1 - x0, int(rows[-1] - rows[0] + 1)))
    return boxes


def glyph_boxes(
    strip: np.ndarray, config: ClockConfig | None = None
) -> list[tuple[int, int, int, int]]:
    """Tight (x, y, width, height) per glyph in a timer strip, left to right.

    Separate from `segment_glyphs` because calibration has to measure how big
    the glyphs are before it can choose the canvas everything is centred into.
    """
    cfg = config or ClockConfig()
    return _boxes_from_mask(lit_mask(strip, cfg), cfg)


def segment_glyphs(
    strip: np.ndarray,
    size: tuple[int, int],
    config: ClockConfig | None = None,
) -> list[np.ndarray]:
    """Cut a timer strip into fixed-size glyph canvases, left to right.

    Each run is tightened vertically as well as horizontally before centring, so
    a glyph's position within the canvas does not depend on where the
    calibration box happens to sit.
    """
    cfg = config or ClockConfig()
    mask = lit_mask(strip, cfg)
    grey = np.where(mask, cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), 0).astype(np.uint8)
    return [
        centred(grey[y : y + h, x : x + w], size)
        for x, y, w, h in _boxes_from_mask(mask, cfg)
    ]


def _parse(labels: list[str], confidence: float) -> GameClock | None:
    """Turn MM:SS glyph labels into a clock, rejecting anything malformed."""
    if labels.count(COLON) != 1:
        return None
    split = labels.index(COLON)
    minute_digits = labels[:split]
    second_digits = labels[split + 1 :]
    if not minute_digits or len(second_digits) != 2:
        return None

    minutes = int("".join(minute_digits))
    seconds = int("".join(second_digits))
    if seconds >= 60:
        return None
    return GameClock(total_seconds=minutes * 60 + seconds, confidence=confidence)


@dataclass(slots=True)
class ClockFilter:
    """Holds the clock steady across frames the reader cannot resolve.

    The match timer is the one signal in this project whose future value is
    known exactly: it advances one second per second. So a frame where the
    digits are occluded by a ping or a damage flash does not have to produce a
    gap -- the previous reading plus elapsed video time *is* the answer, and it
    is marked `observed=False` rather than dressed up as a fresh read.

    The same predictability catches misreads. A reading that disagrees with the
    prediction is rejected, since a single-glyph error moves the clock by tens of
    seconds and elapsed time never does.

    Pauses and seeks break the prediction legitimately, so persistent
    disagreement is treated as the world having changed rather than as noise:
    once a reading has contradicted the prediction for `resync_after` seconds it
    is adopted.
    """

    max_drift: float = 2.0
    """Seconds of disagreement tolerated before a reading is called wrong."""

    resync_after: float = 3.0
    """Seconds of sustained disagreement before the prediction gives way."""

    resynced: bool = False
    """True just after an update adopted a reading that contradicted the
    prediction -- a pause ending somewhere else, a seek, or different footage
    spliced into the same capture. Cleared by the next update. Exposed because
    continuity is a promise other state is built on: the HUD portrait
    baselines are evidence about the footage before the tear, and this flag is
    the one moment that says the footage changed."""

    _seconds: float | None = None
    _at: float = 0.0
    _disagreeing_since: float | None = None

    def update(self, reading: GameClock | None, timestamp: float) -> GameClock | None:
        """Fold one frame's reading in and return the clock to trust."""
        self.resynced = False
        if self._seconds is None:
            if reading is not None:
                self._seconds, self._at = float(reading.total_seconds), timestamp
            return reading

        predicted = self._seconds + (timestamp - self._at)

        if reading is None:
            return GameClock(int(predicted), confidence=0.0, observed=False)

        if abs(reading.total_seconds - predicted) <= self.max_drift:
            self._seconds, self._at = float(reading.total_seconds), timestamp
            self._disagreeing_since = None
            return reading

        if self._disagreeing_since is None:
            self._disagreeing_since = timestamp
        elif timestamp - self._disagreeing_since >= self.resync_after:
            self._seconds, self._at = float(reading.total_seconds), timestamp
            self._disagreeing_since = None
            self.resynced = True
            return reading

        return GameClock(int(predicted), confidence=0.0, observed=False)

    def reset(self) -> None:
        self._seconds = None
        self._at = 0.0
        self._disagreeing_since = None
        self.resynced = False


# -- persistence ----------------------------------------------------------


def _paths(width: int, height: int) -> tuple[Path, Path]:
    return CLOCK_DIR / f"{width}x{height}.json", CLOCK_DIR / f"{width}x{height}.npz"


def save_calibration(
    region: ClockRegion, glyphs: GlyphSet, width: int, height: int
) -> tuple[Path, Path]:
    """Persist a calibration for one resolution under ``etc/clock/``."""
    region_path, glyph_path = _paths(width, height)
    region_path.parent.mkdir(parents=True, exist_ok=True)
    region_path.write_text(
        json.dumps(region.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    glyphs.save(glyph_path)
    return region_path, glyph_path


def load_clock_reader(
    width: int, height: int, config: ClockConfig | None = None
) -> ClockReader:
    """Load the calibrated reader for a resolution from ``etc/clock/``."""
    region_path, glyph_path = _paths(width, height)
    if not region_path.exists() or not glyph_path.exists():
        raise FileNotFoundError(
            f"no clock calibration for {width}x{height} in {CLOCK_DIR}. "
            f"Run: python tools/calibrate_clock.py --input <clip>"
        )
    region = ClockRegion.from_dict(
        json.loads(region_path.read_text(encoding="utf-8"))
    )
    return ClockReader(region, GlyphSet.load(glyph_path), config)


def tighten(
    frame: np.ndarray, box: ClockRegion, config: ClockConfig | None = None
) -> ClockRegion | None:
    """Shrink a hand-drawn box to the glyphs actually inside it.

    Calibration asks for a rough box around the timer, not a precise one. The
    saturation ceiling drops the gold icon on its left, so the tightened result
    is the digits alone even when the drag was generous.
    """
    cfg = config or ClockConfig()
    mask = lit_mask(box.crop(frame), cfg)
    runs = _column_runs(mask, cfg)
    if not runs:
        return None
    rows = np.nonzero(mask.any(axis=1))[0]
    left, right = runs[0][0], runs[-1][1]
    return replace(
        box,
        x=box.x + int(left),
        y=box.y + int(rows[0]),
        width=int(right - left),
        height=int(rows[-1] - rows[0] + 1),
    )
