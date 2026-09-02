"""How the camera moved between two frames, from the terrain.

Everything that reads the 3D view at speed needs the world held still first. A
projectile is a small thing moving fast against a background that is itself
scrolling, and the raw difference between two frames is mostly the scroll: on
the 2026-08-30 clip the camera moves on 64% of distinct frame pairs, at a
median 256 px/s and a p90 of 628. Subtract that motion and what remains is
what moved *relative to the ground* -- champions, minions, spell effects, and
the projectiles this exists to find.

**Sparse flow with a median, not global correlation.** The obvious tool is
phase correlation over the whole view, and it does not work here: judged by
the residual left after stabilising, it beat doing nothing on 48% of moving
frame pairs -- a coin toss -- because a fight fills the view with foreground
that moves differently from the terrain, and a global correlation has no way
to prefer one over the other. Tracking a few hundred corners with
Lucas-Kanade and taking the *median* vector does: terrain corners are the
majority (inlier share p50 0.65, p10 0.27), so the median is the terrain's
motion and the champions are outliers it never sees. Measured on the same
pairs it halved the residual (mean |diff| 6.1 to 3.1) and beat the raw
difference on every single moving pair, and phase correlation on 98% of them.

**A third of the frames are repeats.** The recording runs at 30 fps but the
picture changes on about two thirds of them; the rest differ from their
predecessor by compression noise only. A repeat is not a frame: differencing
against it finds nothing, and counting it as a time step halves every
velocity computed across it. So repeats are flagged and the caller is
expected to skip them, with time accounted between *distinct* frames -- 33
or 67 ms apart, not always 33.

A repeat is recognised by *how many* pixels changed, not by how much the view
changed on average, and there are two kinds. Measured over 2,700 pairs, 609
changed fewer than 100 pixels by more than 12 levels -- exact repeats, block
noise only -- a thin smear of some 400 more ran from there up to about 8,000,
and every pair on which the picture had actually moved changed at least
18,000. The smear is the second kind: a **stale refresh**, a frame on which
the world view was not redrawn but something small was -- the cursor, a HUD
digit. It gives itself away in the motion log: zero camera shift and a
handful of blobs, sandwiched between frames shifting ten or twenty pixels
each. Treated as a frame it plants a stationary point in every track that
crosses it, which is enough to break a four-point bolt. So the floor sits at
8,000 changed pixels: below it the world did not move, whatever else did. On
this footage no genuine update comes within a factor of two of it. A window
much smaller than the calibrated one would scale that figure down, and it
is the first thing to re-measure on new footage.

**Locked camera, so this is also the player's motion.** The receiver's
capture of a locked camera keeps the champion at a fixed screen position --
their own nameplate sits at x 966-976, y 479-504 across sixteen minutes of the
2026-08-30 clip -- so the terrain's motion is the player's motion with the
sign flipped, at sub-pixel precision and 30 Hz. That is a far finer
measurement of a dodge than the minimap's 48 units per pixel at 10 Hz, and it
is the reason this module exists beyond stabilisation.

Coordinates: a `CameraMotion` reports the shift that takes the *previous*
frame onto the *current* one, in full-resolution frame pixels. Warping the
previous frame by (dx, dy) aligns its terrain with the current frame's.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class WorldView:
    """The part of the frame that is the game world, as fractions of it.

    The HUD occupies the bottom strip, the minimap and portrait column on the
    right, and the top bar; the same exclusions the nameplate reader carries,
    reduced to the one rectangle that avoids them all. Fractions rather than
    pixels, like the nameplate exclusions, because the receiver stretches one
    layout to any window and a fraction survives that where a pixel does not.
    """

    left: float = 0.0
    top: float = 0.035
    right: float = 0.76
    bottom: float = 0.78

    def box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) in pixels for a frame of this size."""
        x0, y0 = int(self.left * width), int(self.top * height)
        x1, y1 = int(self.right * width), int(self.bottom * height)
        return x0, y0, x1 - x0, y1 - y0

    def crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x, y, w, h = self.box(width, height)
        return frame[y : y + h, x : x + w]


@dataclass(frozen=True, slots=True)
class MotionConfig:
    repeat_level: int = 12
    repeat_pixels: int = 8000
    """A frame is a repeat of its predecessor when fewer than `repeat_pixels`
    pixels of the view differ by more than `repeat_level`. Exact repeats
    clear 12 levels on a few dozen pixels; stale refreshes -- the world not
    redrawn, a cursor moved -- on up to a few thousand; a frame on which the
    picture moved on 18,000 at the least. See the module docstring."""

    downscale: int = 2
    """Corners are found and tracked at half resolution. Terrain texture
    survives it, the cost quarters, and the shift is scaled back up."""

    max_corners: int = 600
    quality: float = 0.01
    min_distance: int = 12
    """Shi-Tomasi corner detection: many corners spread over the view, so
    the terrain outnumbers any one moving object however large."""

    window: int = 21
    levels: int = 3
    """Lucas-Kanade pyramid. Three levels at half resolution reach a shift of
    roughly 80 full-resolution pixels per frame, past the fastest camera
    motion measured (p99 of 1,109 px/s is 37 px per 33 ms frame)."""

    min_corners: int = 20
    """Fewer tracked corners than this and no estimate is made. A view this
    bare is a loading screen, a black death-cam fade, or a shop overlay, and
    a median of a handful of vectors is a guess wearing a number."""

    inlier_radius: float = 2.0
    """How close a corner's vector must sit to the median to count as agreeing
    with it. The inlier share is the estimate's own confidence."""


@dataclass(frozen=True, slots=True)
class CameraMotion:
    """The camera's shift between the previous distinct frame and this one."""

    dx: float
    dy: float
    """Shift taking the previous frame onto this one, full-resolution pixels.
    Zero on a repeat."""

    dt: float
    """Seconds since the previous distinct frame. The repeat's own time is
    absorbed here rather than lost."""

    inliers: float
    """Share of tracked corners within `inlier_radius` of the median. The
    terrain's vote: high when the view is mostly ground, low in a fight."""

    corners: int
    """Corners tracked into this frame. Zero when no estimate was possible."""

    repeat: bool
    """This frame is a repeat of the previous one -- compression noise only.
    Skip it: there is nothing to difference against and no time has passed
    in the picture."""

    @property
    def estimated(self) -> bool:
        return self.corners > 0

    @property
    def speed(self) -> float:
        """Pixels per second, for the pair this was measured across."""
        return float(np.hypot(self.dx, self.dy) / self.dt) if self.dt > 0 else 0.0


class CameraTracker:
    """Frames in, camera motion out. Stateful; feed it frames in order."""

    def __init__(
        self, view: WorldView | None = None, config: MotionConfig | None = None
    ) -> None:
        self.view = view or WorldView()
        self.config = config or MotionConfig()
        self._prev: np.ndarray | None = None
        self._prev_small: np.ndarray | None = None
        self._prev_time: float | None = None
        self._before: np.ndarray | None = None
        """The distinct frame before `_prev` -- the one the last reported
        motion was measured *from*, and so the one `stabilise` warps."""
        self._lk = dict(
            winSize=(self.config.window, self.config.window),
            maxLevel=self.config.levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @property
    def current(self) -> np.ndarray | None:
        """The latest distinct frame's world view, greyscale."""
        return self._prev

    @property
    def before(self) -> np.ndarray | None:
        """The distinct frame before `current` -- the one the last motion was
        measured from, and the one `stabilise` aligns onto `current`."""
        return self._before

    def reset(self) -> None:
        self._prev = self._prev_small = self._before = None
        self._prev_time = None

    def update(self, frame: np.ndarray, timestamp: float) -> CameraMotion | None:
        """Fold one frame in. None for the first frame, which has no pair."""
        gray = cv2.cvtColor(self.view.crop(frame), cv2.COLOR_BGR2GRAY)
        if self._prev is None:
            self._remember(gray, timestamp)
            return None

        dt = timestamp - self._prev_time
        changed = cv2.countNonZero(
            cv2.compare(cv2.absdiff(gray, self._prev), self.config.repeat_level,
                        cv2.CMP_GT)
        )
        if changed < self.config.repeat_pixels:
            # Not remembered: the next distinct frame is measured against the
            # last distinct one, and the repeat's time rides on that pair.
            return CameraMotion(0.0, 0.0, dt, 0.0, 0, repeat=True)

        small = self._small(gray)
        dx, dy, inliers, corners = self._flow(self._prev_small, small)
        self._remember(gray, timestamp, small)
        return CameraMotion(dx, dy, dt, inliers, corners, repeat=False)

    def stabilise(self, motion: CameraMotion) -> np.ndarray | None:
        """`before` warped onto `current`'s ground, for differencing.

        `absdiff(current, result)` is then motion relative to the terrain.
        The border the shift uncovers is filled from `current` itself, so it
        differences to zero rather than to a black stripe. None before two
        distinct frames exist.
        """
        if self._before is None or self._prev is None:
            return None
        matrix = np.float32([[1, 0, motion.dx], [0, 1, motion.dy]])
        height, width = self._prev.shape[:2]
        aligned = self._prev.copy()
        cv2.warpAffine(
            self._before, matrix, (width, height), dst=aligned,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        return aligned

    def _remember(
        self, gray: np.ndarray, timestamp: float, small: np.ndarray | None = None
    ) -> None:
        self._before = self._prev
        self._prev = gray
        self._prev_small = small if small is not None else self._small(gray)
        self._prev_time = timestamp

    def _small(self, gray: np.ndarray) -> np.ndarray:
        d = self.config.downscale
        if d <= 1:
            return gray
        return cv2.resize(
            gray, (gray.shape[1] // d, gray.shape[0] // d),
            interpolation=cv2.INTER_AREA,
        )

    def _flow(
        self, prev: np.ndarray, cur: np.ndarray
    ) -> tuple[float, float, float, int]:
        cfg = self.config
        corners = cv2.goodFeaturesToTrack(
            prev, maxCorners=cfg.max_corners, qualityLevel=cfg.quality,
            minDistance=cfg.min_distance, blockSize=7,
        )
        if corners is None or len(corners) < cfg.min_corners:
            return 0.0, 0.0, 0.0, 0
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, corners, None, **self._lk)
        kept = status.reshape(-1) == 1
        vectors = (moved - corners).reshape(-1, 2)[kept] * cfg.downscale
        if len(vectors) < cfg.min_corners:
            return 0.0, 0.0, 0.0, 0
        median = np.median(vectors, axis=0)
        agree = np.hypot(*(vectors - median).T) < cfg.inlier_radius
        return float(median[0]), float(median[1]), float(agree.mean()), int(len(vectors))
