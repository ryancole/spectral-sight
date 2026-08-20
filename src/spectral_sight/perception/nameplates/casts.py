"""Turning a champion's resource series into the abilities they cast.

Nothing in the game displays an enemy's cooldowns. Every earlier stage worked by
finding where the client had already drawn the answer; there is no such place
for ability usage, so it has to be inferred, and the resource bar on the
nameplate is the only champion-agnostic evidence the client renders. A step down
that holds is a cast.

**A step is a fall that holds, not a fall.** The measurement floor is one pixel
of a 117px bar -- 85.8% of consecutive readings on an unchanging bar are
identical, with a p05-p95 spread of +-0.85% -- but a bare threshold above that
floor still fires on the occasional bad read. What separates a cast from a
misread is not size but persistence: noise reverts on the next look, a cast does
not. So a drop is held as a candidate and the following reading decides it, the
same move `LevelFilter` makes against misread digits.

**The series is mostly holes, and that shapes the output.** An enemy is inside
the camera view in 29% of frames against 88% having a marker somewhere on the
minimap, averaging 0.40 readable enemy plates per frame. A champion routinely
leaves view, casts three times, and returns eight seconds later with a third of
their mana gone. Suppressing that would discard most of the evidence this stage
exists to collect, so it is emitted as one `Cast` carrying `span` and
`continuous=False` -- interval evidence, honestly labelled, in the same spirit
as `alive` and `game_time_observed` elsewhere.

For the same reason a candidate that never gets a continuous follow-up is
emitted with `confirmed=False` rather than dropped. Requiring confirmation
outright would throw away the drops seen in the last frame before a champion
walks off screen, which is a large share of them.

**Two things make a convincing fake step, and neither is visible in the
resource series.** Run over the 5.3-minute sample clip before either was
handled, the detector found ten casts and five of them were artefacts -- and
both kinds gave themselves away in what the *health* bar did at the same
moment, which is why one is passed in.

- **A truncated plate.** A bar cut partway along by a champion model or a spell
  effect -- not by the frame edge or a HUD panel, which the reader's own
  clipping test already catches -- truncates both fills at the same column, so
  they come back equal. The three worst false casts landed on readings whose
  fills were 0.008-0.009 apart, one pixel of a 117px bar, against 0.07-0.26 for
  the five that survived scrutiny. Such a reading is skipped rather than
  rejected, because measured *from* it invents a step down and measured *to* it
  invents the step back up that would hide the real cast after it.
- **A plate that landed on the wrong track.** Association is geometric and only
  approximately so, so the series occasionally jumps to a different champion's
  bars -- and then both fills move by nearly the same amount at once. The
  remaining two false casts moved health and resource within 0.033 and 0.010 of
  each other while dropping 60%. The test is on the two falls *agreeing*, not
  on damage happening at all: champions cast while being hit constantly, and
  rejecting those would throw away the fights.

**Levelling cannot fake a cast.** It raises maximum resource and grants the
same amount to current, so the fraction holds or rises. Measured across seven
level-ups straddling consecutive readings on the sample clip the change ran
from -0.9% to +5.2%, mean +1.0% -- the one fall being a single pixel, three
times below the threshold here. A level-up *can* mask a cast by cancelling part
of it, which is the safe direction and is left alone.

**Regeneration only ever biases against detection.** Mana regen runs roughly
0.3-0.6% of a pool per second, well under the noise floor frame to frame, so it
cannot manufacture a step. Across a long `span` it partly refills what was
spent, so `drop` understates the true cost and increasingly so as `span` grows.
That is the safe direction to be wrong in, and it is left uncorrected rather
than modelled: the correction would need a per-champion, per-level regen figure
this project has no way to observe.

Champions on energy, rage or no resource at all draw no blue bar, so they
produce no readings here and no casts -- see `NameplateConfig.resource_hue`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class CastConfig:
    """Thresholds for calling a fall in the resource bar a cast."""

    min_drop: float = 0.03
    """Smallest fall counted, as a fraction of the pool.

    Sits between the +-0.85% measurement spread and the smallest real costs
    observed -- Ezreal's clustered at 4-7% on the sample clip -- so it is above
    the noise without cutting into the cheapest abilities. Typical costs run
    7-25%, so most casts clear it comfortably."""

    continuity: float = 0.5
    """Longest gap between two readings still treated as consecutive.

    At the 10 Hz the pipeline samples at this is five frames, which is generous
    for a plate that flickers out for a frame or two behind a spell effect, and
    far short of the seconds a champion spends off screen."""

    tolerance: float = 0.01
    """How far a value may rebound above the post-drop level and still count as
    holding. One pixel of bar is 0.85%, so this is a hair over one pixel."""

    truncation_gap: float = 0.02
    """How close `health` and `resource` may sit before the reading is thrown
    out as a truncated plate rather than believed.

    A bar cut partway along -- by a champion model, a spell effect, anything
    the reader's own clipping test does not see because it is in the world
    rather than at the frame edge -- truncates *both* fills at the same column,
    so they come back equal. Measured on the sample clip the three worst false
    casts all landed on readings whose two fills were 0.008-0.009 apart, one
    pixel of a 117px bar, against 0.07-0.26 for the casts that survived
    scrutiny. Two independent quantities do not agree to a pixel by accident."""

    full_bar: float = 0.95
    """Above this the truncation test is not applied.

    A champion at full health and full mana genuinely reads equal on both bars,
    and that is the single most common state in the game. Truncation to a value
    this high is also nearly harmless, since it is close to the truth."""

    comovement: float = 0.05
    """How closely a fall in `health` may track a fall in `resource` before the
    pair is read as the series changing plates rather than as a cast.

    Plate-to-track association is geometric and only approximately so, so a
    plate occasionally lands on the wrong track -- and then the next reading is
    a different champion's bars entirely, which shows up as both fills stepping
    by nearly the same amount at nearly the same instant. On the sample clip the
    two worst false casts moved health and resource within 0.033 and 0.010 of
    each other while dropping 60%.

    This deliberately does not reject *any* cast that coincides with damage.
    Champions cast while being hit constantly, and there the two falls have
    nothing to do with one another; it is their agreeing to within a few percent
    that is the artefact."""


@dataclass(frozen=True, slots=True)
class Cast:
    """One observed fall in a champion's resource bar.

    Read as evidence about an interval, not an instant. `continuous` says
    whether that interval was tight enough for the fall to be a single cast;
    when it is False the champion was away and this is the net spend across
    however long `span` covers, which may be several abilities and is reduced by
    whatever regenerated in between.
    """

    track_id: int

    drop: float
    """Fall as a fraction of the pool, always positive."""

    resource_before: float
    resource_after: float

    at: float
    """`video_time` of the reading the fall was measured *to*.

    The cast is therefore somewhere in `[at - span, at]`. This is not the time
    the `Cast` was returned: emission waits for the following reading to decide
    whether the fall held, so it comes back a frame or more later than this."""

    span: float
    """Seconds between the two readings the fall was measured across."""

    continuous: bool
    """`span` was within `CastConfig.continuity`, so this is one cast rather
    than a net change across a stretch of not looking."""

    confirmed: bool
    """The post-drop level held on a continuous follow-up reading.

    False means no such reading arrived -- the champion left view, or the plate
    stopped being readable -- not that the fall was contradicted. A fall that
    *was* contradicted is discarded and never becomes a `Cast` at all."""

    level: int | None
    """Filtered level at the time, for reading a cost against a pool size."""


@dataclass(slots=True)
class CastDetector:
    """One champion's resource series, folded in a reading at a time."""

    track_id: int
    config: CastConfig = field(default_factory=CastConfig)

    _last: float | None = None
    _last_time: float | None = None
    _last_health: float | None = None
    _pending: Cast | None = None

    def update(
        self,
        video_time: float,
        resource: float | None,
        health: float | None = None,
        level: int | None = None,
    ) -> Cast | None:
        """Fold one frame's reading in, returning a cast if one just settled.

        `resource` of None means the plate was not read this frame, which is the
        common case. It is not a value: the series simply skips, and the next
        real reading is compared against the last real one however long ago that
        was. Carrying a value forward instead would invent readings and turn one
        cast across a gap into a step at an arbitrary frame.

        `health` is not read for itself. It is here because the two failure modes
        that produce a convincing fake step -- a truncated plate and a plate that
        landed on the wrong track -- both show up in what the *health* bar did at
        the same moment, and neither is visible in the resource series alone.
        """
        if resource is None or self._truncated(resource, health):
            return None

        settled = self._resolve(video_time, resource)

        if self._last is not None and self._last_time is not None:
            fall = self._last - resource
            if fall >= self.config.min_drop and not self._swapped(fall, health):
                span = video_time - self._last_time
                self._pending = Cast(
                    track_id=self.track_id,
                    drop=fall,
                    resource_before=self._last,
                    resource_after=resource,
                    at=video_time,
                    span=span,
                    continuous=span <= self.config.continuity,
                    confirmed=False,
                    level=level,
                )

        self._last, self._last_time = resource, video_time
        self._last_health = health
        return settled

    def _truncated(self, resource: float, health: float | None) -> bool:
        """Is this reading a bar cut off partway, rather than a bar?

        Treated as a frame that was not looked at rather than as a rejected
        cast, because the value poisons the series in both directions: measured
        *from*, it invents a step down; measured *to*, it invents the step back
        up that then rejects the real cast before it.
        """
        if health is None:
            return False
        return (
            abs(resource - health) <= self.config.truncation_gap
            and resource < self.config.full_bar
        )

    def _swapped(self, fall: float, health: float | None) -> bool:
        """Did the health bar fall by the same amount at the same instant?

        Then this is almost certainly the plate association moving to a
        different champion, whose bars are simply at different levels, rather
        than a champion who spent mana.
        """
        if health is None or self._last_health is None:
            return False
        health_fall = self._last_health - health
        return abs(health_fall - fall) <= self.config.comovement

    def _resolve(self, video_time: float, resource: float) -> Cast | None:
        """Decide a candidate held over from an earlier reading.

        A continuous follow-up is the only thing that can *reject* a candidate,
        because only there does a rebound mean the fall was not real. After a
        gap a rebound is just as likely to be regeneration, so the candidate is
        released as it stands rather than thrown away on evidence that cannot
        tell those apart.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return None

        if video_time - pending.at > self.config.continuity:
            return pending

        if resource > pending.resource_after + self.config.tolerance:
            return None

        return replace(pending, confirmed=True)

    def flush(self) -> Cast | None:
        """Release a candidate that will never get a follow-up.

        For the end of a clip, and for a track the tracker is about to discard.
        Without it the last cast of every series is silently lost.
        """
        pending, self._pending = self._pending, None
        return pending


@dataclass(slots=True)
class CastBook:
    """A `CastDetector` per track, mirroring `LevelBook`.

    Keyed by track id for the same reason: a plate is a per-frame observation
    with no memory, and what a resource series belongs to is the champion the
    tracker is following.
    """

    config: CastConfig = field(default_factory=CastConfig)
    detectors: dict[int, CastDetector] = field(default_factory=dict)

    def update(
        self,
        track_id: int,
        video_time: float,
        resource: float | None,
        health: float | None = None,
        level: int | None = None,
    ) -> Cast | None:
        state = self.detectors.get(track_id)
        if state is None:
            state = self.detectors[track_id] = CastDetector(
                track_id=track_id, config=self.config
            )
        return state.update(video_time, resource, health, level)

    def forget(self, track_id: int) -> Cast | None:
        """Drop a track's series, releasing any candidate it was holding.

        Called when the tracker discards the track. A reused id must not inherit
        a stranger's resource level, which would read as one enormous step the
        moment the new champion's plate is first seen.
        """
        state = self.detectors.pop(track_id, None)
        return None if state is None else state.flush()
