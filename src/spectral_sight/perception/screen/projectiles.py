"""Projectiles on the world view: small things moving fast across the ground.

Nothing in the client labels a projectile, so this is built the other way
round from every HUD reader: not "find where the answer is drawn" but "find
what moves like one". Three stages, each cheap and each measured on the
2026-08-30 clip against the one free ground truth the footage offers -- the
local player's own Q and W, which `perception.hud.abilities` timestamps to a
frame and which each launch a bolt from the player's nameplate.

1. **Stabilise.** The camera scrolls; see `motion`. Differencing the previous
   distinct frame warped onto this one leaves only motion relative to the
   terrain.
2. **Blobs.** Threshold the residual, open it, take connected components.
   A fight produces a hundred of them a frame (p50 107, p90 205): champions,
   minions, spell effects, floating damage numbers, swaying brush, the click
   marker. Almost none are projectiles, and this stage does not try to tell.
   One thing it does remove: the *ghost*. A difference paints every mover
   twice, where it is and where it was, and the second copy trails the
   first by a frame as a second, equally fast, equally straight track.
   Where a mover was is exactly the previous frame's foreground carried
   onto this one, so that mask -- warped by the camera motion and grown a
   little -- is cleared from the current one. A bolt never overlaps its own
   previous position, so it loses nothing; a slow mover loses its overlap,
   and slow movers are not what this is for.
3. **Tracks.** Link blobs frame to frame under constant velocity, and keep
   the tracks that move like a bolt. This is where the separation happens,
   and the measurements that shape it are worth stating:

   - **A bolt is fast.** Ezreal's Q crosses the view at ~2,300 px/s, his W
     at 1,000-1,700; the champions, minions and effects around him top out
     near 600. Speed alone is most of the gate.
   - **A bolt is brief.** Q's range is ~600 px of screen at 2,300 px/s: a
     quarter of a second, four to six distinct frames at the recording's
     effective ~20 fps. Tracks are short *by nature*, so nothing here asks
     for a long one -- and 10 Hz sampling, which would see two frames of
     it, was never an option.
   - **Chance chains are the enemy.** A hundred blobs a frame and a first
     link that must reach 140 px (a bolt's step) means unrelated blobs pair
     up constantly, and some of those pairs get continued by luck. Two rules
     kill most of them at birth: a link is accepted only when the track and
     the blob are each other's nearest candidate, and a two-point track is
     kept only if its third point lands where the first two predict, within
     a tolerance that scales with the step -- a bolt's centroid wobbles
     10-17 px along a 100 px step, so a fixed tolerance either loses bolts
     or admits chains.

   What survives is a *candidate*: a fast, straight, brief mover. On a
   3-minute stretch with 25 self casts, 24 launched a candidate beside the
   player's nameplate, and about 210 candidates a minute launched from
   elsewhere -- many of them genuine (minion and turret bolts are
   projectiles too), the rest brush and effects that happened to line up.
   Telling those apart is the classifier's job, and the classifier needs
   labelled footage this project does not have yet; see the plan. This
   module's contract is recall: a real bolt becomes a candidate.

Coordinates are world-view pixels -- the crop `WorldView` makes -- not frame
pixels. `WorldView.box` gives the offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from spectral_sight.perception.screen.motion import CameraMotion, CameraTracker


@dataclass(frozen=True, slots=True)
class ProjectileConfig:
    diff_threshold: int = 40
    """Residual intensity a pixel must exceed to be a mover. Stabilised
    ground differences by a few units; a bolt, a champion and a spell effect
    by far more. Raising it to 60 thins the blob field by half and loses a
    third of the bolts' tails with it."""

    min_area: int = 25
    """Smallest component kept, in pixels. Below this is noise and the
    fragments of anti-aliased edges."""

    gate_first: float = 140.0
    """How far a lone blob may be from a one-point track to start a pair --
    a bolt's step between distinct frames, up to 150 px across a 67 ms gap."""

    gate_predicted: float = 45.0
    """How far a blob may land from a track's constant-velocity prediction."""

    birth_gate: float = 20.0
    birth_fraction: float = 0.35
    """A two-point track's third point must fall within the larger of these
    -- an absolute floor, or this fraction of the step just taken -- of the
    prediction. Relative because a bolt's residual centroid wobbles with its
    trail: measured 10-17 px along 100 px steps, which a 20 px gate rejected
    on the very casts used to check it."""

    area_ratio: tuple[float, float] = (0.4, 2.5)
    """How much a blob's area may change between links. Appearance
    continuity, coarse: a bolt and its trail stay the same order of size, a
    chain through a minion and then a spark does not."""

    max_miss: int = 1
    """Frames a track survives without a link. One, because the effective
    frame rate already halves across a repeat and a bolt is gone in six."""

    min_points: int = 4
    """Shortest track reported. Three points is one prediction confirmed;
    four is a second, and the first at which speed and straightness mean
    anything."""

    min_speed: float = 800.0
    """Chord speed, px/s, below which a track is not a bolt. Champions and
    the effects on them measured up to ~600 px/s. The speed floor is the
    gate's one real lever: swept on the 2026-08-30 clip, 800 caught 24 of 25
    self casts at ~210 candidates a minute, 900 caught 20 at ~150, and 1,100
    caught 13 -- Ezreal's W sits at 800-950 px/s and a floor above it loses
    every W. Recall is this stage's contract, so the floor sits below W."""

    max_rms: float = 25.0
    """Perpendicular RMS residual from the chord, px, above which a track is
    not straight. Bolts measured 1-25 -- the higher values are W's orb and
    trail wobbling about their shared centroid. Swept, this barely moves
    either recall or the rate above 25."""

    min_straightness: float = 0.85
    """Chord over path length. A bolt is a line; a jitter is a scribble."""

    suppress_ghosts: bool = True
    """Clear the previous frame's foreground from this one's -- see the
    module docstring. A switch because it is a trade, and the measurement
    tool wants both sides of it."""


@dataclass(frozen=True, slots=True)
class Blob:
    x: float
    y: float
    area: int


@dataclass(slots=True)
class _Live:
    id: int
    points: list[tuple[float, float, float]]
    area: int
    miss: int = 0

    def predict(self, t: float) -> tuple[float, float, tuple[float, float] | None]:
        if len(self.points) < 2:
            _, x, y = self.points[-1]
            return x, y, None
        (t1, x1, y1), (t2, x2, y2) = self.points[-2], self.points[-1]
        dt = t2 - t1
        vx, vy = (x2 - x1) / dt, (y2 - y1) / dt
        return x2 + vx * (t - t2), y2 + vy * (t - t2), (vx, vy)

    @property
    def step(self) -> float:
        (_, x1, y1), (_, x2, y2) = self.points[-2], self.points[-1]
        return float(np.hypot(x2 - x1, y2 - y1))


@dataclass(frozen=True, slots=True)
class ProjectileTrack:
    """One mover followed across frames, finished."""

    id: int
    points: tuple[tuple[float, float, float], ...]
    """(video_time, x, y) in world-view pixels, one per distinct frame."""

    area: int
    """Blob area at the last link, pixels."""

    @property
    def start(self) -> tuple[float, float, float]:
        return self.points[0]

    @property
    def end(self) -> tuple[float, float, float]:
        return self.points[-1]

    @property
    def duration(self) -> float:
        return self.points[-1][0] - self.points[0][0]

    @property
    def chord(self) -> float:
        (_, x0, y0), (_, x1, y1) = self.points[0], self.points[-1]
        return float(np.hypot(x1 - x0, y1 - y0))

    @property
    def speed(self) -> float:
        """Chord over duration, px/s. The chord rather than the path, so a
        jittering blob does not read as fast."""
        d = self.duration
        return self.chord / d if d > 0 else 0.0

    @property
    def heading(self) -> tuple[float, float]:
        """Unit vector from start to end; (0, 0) for a stationary track."""
        c = self.chord
        if c < 1e-6:
            return 0.0, 0.0
        (_, x0, y0), (_, x1, y1) = self.points[0], self.points[-1]
        return (x1 - x0) / c, (y1 - y0) / c

    @property
    def straightness(self) -> float:
        p = np.array([(x, y) for _, x, y in self.points])
        path = float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
        return self.chord / path if path > 0 else 0.0

    @property
    def rms(self) -> float:
        """Perpendicular RMS distance of the points from the chord."""
        c = self.chord
        if c < 1:
            return 0.0
        ux, uy = self.heading
        (_, x0, y0) = self.points[0]
        perp = [abs((x - x0) * uy - (y - y0) * ux) for _, x, y in self.points]
        return float(np.sqrt(np.mean(np.square(perp))))

    def is_projectile(self, config: ProjectileConfig) -> bool:
        """Fast, straight and brief: a candidate. See the module docstring
        for what this does and does not claim."""
        return (
            len(self.points) >= config.min_points
            and self.speed >= config.min_speed
            and self.rms <= config.max_rms
            and self.straightness >= config.min_straightness
        )

    def approaches(
        self, target: tuple[float, float], within: float
    ) -> float | None:
        """Closest approach of the track's line to `target`, in px, if the
        track is heading toward it; None if it is moving away.

        The line, not the segment: a bolt's track ends when it leaves the
        view or its trail fades, and where it was *going* is the question
        a threat asks."""
        ux, uy = self.heading
        if ux == 0.0 and uy == 0.0:
            return None
        (_, x0, y0) = self.points[0]
        ax, ay = target[0] - x0, target[1] - y0
        along = ax * ux + ay * uy
        if along <= 0:
            return None
        perp = abs(ax * uy - ay * ux)
        return perp if perp <= within else None


@dataclass
class ProjectileTracker:
    """Frames in, finished tracks out.

    Owns a `CameraTracker`; every frame goes through it first, repeats are
    dropped there, and the stabilised residual is what gets segmented.
    """

    config: ProjectileConfig = field(default_factory=ProjectileConfig)
    camera: CameraTracker = field(default_factory=CameraTracker)
    _live: list[_Live] = field(default_factory=list)
    _next_id: int = 0
    _kernel: np.ndarray = field(
        default_factory=lambda: np.ones((3, 3), np.uint8)
    )
    _ghost_kernel: np.ndarray = field(
        default_factory=lambda: np.ones((7, 7), np.uint8)
    )
    _prev_mask: np.ndarray | None = None
    """Last distinct frame's foreground, in its own coordinates; warped onto
    the next frame it marks where every mover *was* -- the ghosts."""

    last_motion: CameraMotion | None = None
    last_blobs: int = 0
    """The most recent frame's camera motion and blob count, for callers
    reporting what the stage saw."""

    def update(
        self, frame: np.ndarray, timestamp: float
    ) -> list[ProjectileTrack]:
        """Fold one frame in. Returns the tracks that *finished* on it, of
        any shape -- filter with `ProjectileTrack.is_projectile`."""
        motion = self.camera.update(frame, timestamp)
        self.last_motion = motion
        if motion is None or motion.repeat or not motion.estimated:
            # No pair, a repeat, or a view too bare to stabilise: nothing to
            # difference. Live tracks are not aged either -- no picture, no
            # time, the same accounting the camera tracker keeps.
            return []
        aligned = self.camera.stabilise(motion)
        residual = cv2.absdiff(self.camera.current, aligned)
        blobs = self._blobs(residual, motion)
        self.last_blobs = len(blobs)
        return self._associate(blobs, timestamp)

    def flush(self) -> list[ProjectileTrack]:
        """Finish every live track, for the end of a run."""
        out = [self._finish(tr) for tr in self._live
               if len(tr.points) >= self.config.min_points]
        self._live.clear()
        return out

    def reset(self) -> None:
        self._live.clear()
        self._prev_mask = None
        self.camera.reset()

    def _blobs(self, residual: np.ndarray, motion: CameraMotion) -> list[Blob]:
        cfg = self.config
        _, mask = cv2.threshold(residual, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        foreground = mask
        if (cfg.suppress_ghosts and self._prev_mask is not None
                and self._prev_mask.shape == mask.shape):
            height, width = mask.shape
            matrix = np.float32([[1, 0, motion.dx], [0, 1, motion.dy]])
            ghosts = cv2.warpAffine(self._prev_mask, matrix, (width, height))
            ghosts = cv2.dilate(ghosts, self._ghost_kernel)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(ghosts))
        self._prev_mask = foreground
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        return [
            Blob(float(centroids[i][0]), float(centroids[i][1]),
                 int(stats[i, cv2.CC_STAT_AREA]))
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= cfg.min_area
        ]

    def _associate(
        self, blobs: list[Blob], t: float
    ) -> list[ProjectileTrack]:
        cfg = self.config
        lo, hi = cfg.area_ratio

        # Each track's nearest admissible blob.
        wanted: dict[int, tuple[int, float]] = {}
        for ti, tr in enumerate(self._live):
            px, py, v = tr.predict(t)
            gate = cfg.gate_predicted if v is not None else cfg.gate_first
            best, best_d = None, gate
            for bi, b in enumerate(blobs):
                ratio = b.area / tr.area if tr.area else 1.0
                if not lo <= ratio <= hi:
                    continue
                d = float(np.hypot(b.x - px, b.y - py))
                if d < best_d:
                    best, best_d = bi, d
            if best is not None:
                wanted[ti] = (best, best_d)

        # Mutual: a blob wanted by several tracks goes to the nearest only.
        claimed: dict[int, tuple[int, float]] = {}
        for ti, (bi, d) in wanted.items():
            if bi not in claimed or d < claimed[bi][1]:
                claimed[bi] = (ti, d)

        linked: set[int] = set()
        used: set[int] = set()
        for bi, (ti, d) in claimed.items():
            tr = self._live[ti]
            if len(tr.points) == 2:
                # The third point decides whether the pair was a real mover
                # or two strangers: it must land where they predict.
                if d > max(cfg.birth_gate, cfg.birth_fraction * tr.step):
                    continue
            b = blobs[bi]
            tr.points.append((t, b.x, b.y))
            tr.area = b.area
            tr.miss = 0
            linked.add(ti)
            used.add(bi)

        finished: list[ProjectileTrack] = []
        survivors: list[_Live] = []
        for ti, tr in enumerate(self._live):
            if ti not in linked:
                tr.miss += 1
            # A pair that failed its birth test, or any track past its
            # allowance, ends here.
            dead = tr.miss > cfg.max_miss or (len(tr.points) == 2 and tr.miss > 0)
            if dead:
                if len(tr.points) >= cfg.min_points:
                    finished.append(self._finish(tr))
            else:
                survivors.append(tr)
        self._live = survivors

        for bi, b in enumerate(blobs):
            if bi not in used:
                self._live.append(_Live(self._next_id, [(t, b.x, b.y)], b.area))
                self._next_id += 1
        return finished

    @staticmethod
    def _finish(tr: _Live) -> ProjectileTrack:
        return ProjectileTrack(id=tr.id, points=tuple(tr.points), area=tr.area)
