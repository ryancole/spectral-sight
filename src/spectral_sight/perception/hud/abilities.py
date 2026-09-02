"""The local player's ability slots: which ability was cast, read off the HUD.

The nameplate route infers a cast from a step in a resource fraction, which
says *that* something was cast and never *what*. For the local player the
client already draws the answer: the moment an ability is used, its HUD slot
stops showing the ability's art and shows the cooldown instead. Reading that
transition names the slot -- Q, W, E, R, or a summoner spell -- and the slot
plus the champion names the ability, with no icon matching anywhere.

**The signal is the veil, not the darkness.** The obvious reading -- the slot
gets darker when it goes on cooldown -- is wrong for exactly the champion this
was built against. Measured on the 2026-08-30 session, Ezreal's W slot fell
from V=152 to V=106 on cast, but his Q slot went from V=103 to V=109: the icon
art is itself blue-teal, and the cooldown overlay is no darker than it is. What
actually changes is that the art is *replaced* -- by a flat, saturated blue
field (H 103-105, S 210-230) with a countdown printed on it, whatever the art
was. So the reading is the fraction of the slot inside that blue band:

- ready icons measured 0.00-0.05, whatever their colour -- even Ezreal's teal
  Q sits at hue 90, outside the band
- the veil at cast onset measured 0.78-0.88
- the insufficient-mana tint, the one state that also looks "dark and blue",
  measured 0.00-0.03 in the band: it desaturates (S 39-101) where the veil
  saturates, so the S >= 150 floor excludes it entirely
- the tail of a cooldown decays through everything in between as the wedge
  shrinks, which is why a cast is a *jump* -- clear one reading, veiled soon
  after -- and not a level. The wedge only ever shrinks, so nothing but a
  fresh cast moves the fraction upward through the gap.

**A cast is a jump that holds.** One veiled reading is a candidate; the next
reading decides it, the same move every filter here makes. A real veil lasts
the cooldown -- seconds, dozens of readings -- so the confirmation costs one
frame of latency and removes one-frame flashes (a projectile crossing the HUD,
a transition sparkle) entirely.

**The countdown is read with the clock's glyphs, and it is optional.** The
digits on the veil are the timer's face again, white on a saturated field,
about 12px against the clock's 13 -- rescaled and matched exactly the way the
resource reader matches the mana text, margin-gated for the same reason.
Measured at onset they read '7', '2', '5' correctly against known cooldowns.
A veil whose digits cannot be read is still a veil, so the cast is emitted
with `countdown=None` rather than suppressed; the digits corroborate and
enrich, they do not gate.

**What this does not need is a baseline.** Death does not touch the slots
(measured: mid-death, undimmed, a cooldown still ticking), and the states that
do dim them -- insufficient mana, an unleveled ability -- either stay out of
the blue band or never produce the clear-to-veiled jump. A slot whose ready
art genuinely lives in the band (an ice-blue icon on some other champion)
never reads clear, so it emits nothing rather than emitting wrongly -- the
same decline-over-guess every reader here prefers. That is the known limit:
this reader can be blind to a slot, it should not lie about one.

Only the local player has these. The client never draws an enemy's cooldowns,
which is why the enemy half of ability naming is a different project entirely.

Measured on the 2026-08-30 session -- a human Ezreal, whose Q/W/E/R are all
skillshots -- against the printed-mana ground truth: recall 90% of the 196
mana-fall casts, and of the confirmed casts essentially all are real, with a
false-positive rate near one percent.

### Known limits

- **Re-pressing an ability already on cooldown can flash a phantom cast.**
  Pressing a slot that is on cooldown flashes its icon, and a flash that
  outlasts `ready_hold` re-arms the slot mid-cooldown. `ready_hold` clears the
  common 0.1-0.2s flash; a longer one slips through, which is the bulk of the
  residual false positives. The sturdy fix is to compare a cast's countdown
  against the slot's observed maximum -- a fresh cast starts at the full
  cooldown, a mid-cooldown re-fire does not -- but that needs the countdown
  read reliably and a per-champion, per-level cooldown this project cannot yet
  observe, so it is left for when that footage exists.
- **A frame-perfect re-cast can be missed.** An ability re-cast within
  `ready_hold` of coming off cooldown never accumulates the ready hold, so the
  second cast is dropped. This is the price of the flash rejection above, and
  it costs a few percent of recall on a clip full of rapid combos.
- **The countdown is a bonus, not a guarantee.** It reads on most Q and W
  onsets and rarely on E and R, whose veils carry larger or differently placed
  digits; a cast with `countdown=None` is no less a cast.
- **A slot whose ready art lives in the veil's own blue band reads clear and
  emits nothing.** No champion in the footage so far does, but an ice-blue
  ability icon on some other champion would -- the reader would decline rather
  than fire wrongly, the same trade every reader here makes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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

LAYOUT_DIR = Path(__file__).resolve().parents[4] / "etc" / "abilities"

ABILITY_SLOTS = ("Q", "W", "E", "R")
SUMMONER_SLOTS = ("D", "F")
SLOTS = ABILITY_SLOTS + SUMMONER_SLOTS


@dataclass(frozen=True, slots=True)
class AbilityLayout:
    """Where the six castable slots sit, in frame pixels, for one resolution.

    Parametric like the portrait row: the four ability slots are evenly
    spaced, and so are the two summoner slots beside them, so each row is a
    first position, a size and a spacing rather than a chance to mistype four
    rectangles. Sizes carry both axes because the receiver stretches without
    preserving aspect -- a square slot in the reference layout is a rectangle
    in a derived one.
    """

    ability_first_x: float
    ability_y: float
    ability_width: float
    ability_height: float
    ability_spacing: float
    summoner_first_x: float
    summoner_y: float
    summoner_width: float
    summoner_height: float
    summoner_spacing: float

    def boxes(self) -> dict[str, tuple[int, int, int, int]]:
        """Every slot's (x, y, width, height), keyed Q W E R D F."""
        out: dict[str, tuple[int, int, int, int]] = {}
        for i, name in enumerate(ABILITY_SLOTS):
            out[name] = (
                round(self.ability_first_x + i * self.ability_spacing),
                round(self.ability_y),
                round(self.ability_width),
                round(self.ability_height),
            )
        for i, name in enumerate(SUMMONER_SLOTS):
            out[name] = (
                round(self.summoner_first_x + i * self.summoner_spacing),
                round(self.summoner_y),
                round(self.summoner_width),
                round(self.summoner_height),
            )
        return out

    def to_dict(self) -> dict[str, float]:
        return {
            "ability_first_x": self.ability_first_x,
            "ability_y": self.ability_y,
            "ability_width": self.ability_width,
            "ability_height": self.ability_height,
            "ability_spacing": self.ability_spacing,
            "summoner_first_x": self.summoner_first_x,
            "summoner_y": self.summoner_y,
            "summoner_width": self.summoner_width,
            "summoner_height": self.summoner_height,
            "summoner_spacing": self.summoner_spacing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AbilityLayout:
        return cls(**{key: float(data[key]) for key in (
            "ability_first_x", "ability_y", "ability_width", "ability_height",
            "ability_spacing", "summoner_first_x", "summoner_y",
            "summoner_width", "summoner_height", "summoner_spacing",
        )})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AbilityLayout:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_resolution(cls, width: int, height: int) -> AbilityLayout | None:
        path = LAYOUT_DIR / f"{width}x{height}.json"
        return cls.load(path) if path.exists() else None


@dataclass(frozen=True, slots=True)
class AbilityConfig:
    """Thresholds for reading the cooldown veil."""

    veil_hue: tuple[int, int] = (98, 112)
    veil_saturation: int = 150
    """The veil's colour band. Measured at onset: H 103-105, S 210-230, on
    both a gold icon and a teal one. The saturation floor is what excludes the
    insufficient-mana tint, which lands in the same hues at S 39-101."""

    veiled_above: float = 0.5
    """Band fraction at which a slot reads veiled. Onsets measured 0.78-0.88;
    everything that is not a fresh cooldown measured 0.26 or less."""

    clear_below: float = 0.12
    """Band fraction below which a slot reads clear. Ready icons measured
    0.00-0.05; the decaying wedge passes through here only on its way down."""

    ready_hold: float = 0.25
    """How long a slot must read clear *continuously* before it can arm a new
    cast. A ready ability sits clear for its whole cooldown -- seconds -- so
    this costs a real cast nothing. What it removes is the UI flash: an ability
    already on cooldown brightens and desaturates for a frame or two when it is
    re-pressed or a pip animates, which reads as clear (S drops below the veil
    floor) without the ability ever being ready. Measured, those flashes span
    0.1-0.2s bracketed by veil on both sides; a genuine return to ready does
    not. A between-readings value breaks the clear run rather than extending
    it, so a flash's two clear frames never accumulate the hold."""

    hold_above: float = 0.35
    """What the follow-up reading must still show for a candidate to confirm.
    Looser than the onset bar because the wedge starts shrinking immediately,
    and a real cooldown a tenth of a second in is still far above this."""

    continuity: float = 0.5
    """Longest stretch between the last clear reading and the veiled one still
    read as a transition. The same figure the cast detector uses, for the same
    reason: five frames at 10 Hz forgives a blink without letting two distant
    states pretend to be adjacent."""

    inset_fraction: float = 0.11
    """Share of the slot box trimmed from each side before measuring, past the
    gold border and the corner adornments. The mana-cost label printed in the
    icon's top-right corner survives the trim, which is fine: it is a static
    dozen pixels, identical in the clear and veiled states it would have to
    tell apart."""

    digit_min_value: int = 190
    digit_max_saturation: int = 90
    """The countdown's white strokes over the saturated veil. Looser than the
    clock's 110/45 for the same reason the resource reader's are: the ground
    is bright and saturated rather than a flat dark panel."""

    digit_min_height: int = 8
    """Shortest box taken as a countdown digit. The veil's digits stand about
    12px at the calibrated resolution; the sweep's edge highlight and the
    corner label's remnants come in under this."""

    digit_band: tuple[float, float] = (0.25, 0.80)
    """Vertical span of the slot, as fractions, a digit's centre must fall in.
    The countdown is printed in the middle of the veil; the cost label sits in
    the top corner and the keybind letter along the bottom edge, and neither
    has any business being read as seconds."""

    digit_min_score: float = 0.42
    digit_min_margin: float = 0.04
    """Match floors for the rescaled glyphs, the resource reader's argument at
    the resource reader's values: a rescaled glyph correlates worse than a
    native one, and it is the margin over the runner-up that separates the
    digits a score cannot."""

    max_countdown: int = 999
    """Longest cooldown worth believing, in seconds. Three digits covers any
    real cooldown; a fourth is a misread."""


@dataclass(frozen=True, slots=True)
class AbilityCast:
    """One slot observed going from ready to cooldown."""

    slot: str
    """Q, W, E, R for abilities; D, F for summoner spells."""

    at: float
    """`video_time` of the first veiled reading. Emission is one reading later
    -- the follow-up that confirmed the veil held -- so this is earlier than
    the frame the cast appears on, the same way `cast_at` is."""

    countdown: int | None
    """Seconds printed on the veil when the digits could be read, which at the
    onset is the ability's cooldown. None when they could not; the veil is the
    evidence and the digits are a bonus."""

    confirmed: bool
    """The veil was still there on the following reading. False only for a
    candidate flushed at the end of a run, never for one that was contradicted
    -- a contradicted candidate was a one-frame flash, not a cast."""


@dataclass(slots=True)
class _SlotState:
    clear_since: float | None = None
    """When the current unbroken run of clear readings began, or None if the
    slot is not currently reading clear. A cast can arm only once this run has
    lasted `ready_hold`."""

    armed_at: float | None = None
    """`video_time` of the last reading at which the slot was armed -- clear
    long enough to be genuinely ready. The veil onset must follow within
    `continuity` of this for a cast to fire."""

    pending: AbilityCast | None = None


class AbilityReader:
    """Frame in, the local player's confirmed casts out. Stateful per slot;
    feed it frames in order."""

    def __init__(
        self,
        layout: AbilityLayout,
        glyphs: GlyphSet | None = None,
        config: AbilityConfig | None = None,
    ) -> None:
        self.layout = layout
        self.glyphs = glyphs
        self.config = config or AbilityConfig()
        self._boxes = layout.boxes()
        self._slots: dict[str, _SlotState] = {name: _SlotState() for name in SLOTS}
        self._digit_config = ClockConfig(
            min_value=self.config.digit_min_value,
            max_saturation=self.config.digit_max_saturation,
        )

    def read(self, frame: np.ndarray, timestamp: float) -> list[AbilityCast]:
        """Fold one frame in, returning the casts that settled on it."""
        settled: list[AbilityCast] = []
        for name in SLOTS:
            crop = self._crop(frame, name)
            if crop is None:
                continue
            veil = self._veil_fraction(crop)
            state = self._slots[name]

            if state.pending is not None:
                pending, state.pending = state.pending, None
                if veil >= self.config.hold_above:
                    countdown = pending.countdown
                    if countdown is None:
                        countdown = self._countdown(crop)
                    settled.append(
                        replace(pending, countdown=countdown, confirmed=True)
                    )
                # A veil gone by the very next reading was never a cooldown;
                # the candidate is dropped without ceremony.

            elif (
                veil >= self.config.veiled_above
                and state.armed_at is not None
                and timestamp - state.armed_at <= self.config.continuity
            ):
                state.pending = AbilityCast(
                    slot=name,
                    at=timestamp,
                    countdown=self._countdown(crop),
                    confirmed=False,
                )
                # Disarm. The veil then persists for the whole cooldown --
                # seconds, dozens of readings -- and without this every one of
                # them would re-fire the same cast. The slot must become ready
                # again before another cast can arm, which is exactly what a
                # real re-cast requires: the cooldown ends, the art returns,
                # the button is pressed again.
                state.armed_at = None

            # Track the run of clear readings, and arm once it has held long
            # enough to be a genuine ready state rather than a UI flash. A
            # reading in the deadband (neither clear nor veiled) breaks the
            # run without arming.
            if veil < self.config.clear_below:
                if state.clear_since is None:
                    state.clear_since = timestamp
                if timestamp - state.clear_since >= self.config.ready_hold:
                    state.armed_at = timestamp
            else:
                # Anything above clear -- veil or the deadband between -- breaks
                # the run. A flash's two clear frames never accumulate the hold.
                state.clear_since = None

        return settled

    def flush(self) -> list[AbilityCast]:
        """Release candidates that will never get a follow-up reading.

        For the end of a clip. Without it a cast on the final sampled frame is
        silently lost, which is the cast detector's argument verbatim.
        """
        released = []
        for state in self._slots.values():
            if state.pending is not None:
                released.append(state.pending)
                state.pending = None
        return released

    def reset(self) -> None:
        """Forget every slot's history, for when the footage tears wholesale.

        After a seek or a splice the last clear reading describes footage that
        ended, and a candidate held across the tear would confirm against a
        different game's pixels."""
        for state in self._slots.values():
            state.clear_since = None
            state.armed_at = None
            state.pending = None

    def _crop(self, frame: np.ndarray, name: str) -> np.ndarray | None:
        x, y, width, height = self._boxes[name]
        inset_x = round(width * self.config.inset_fraction)
        inset_y = round(height * self.config.inset_fraction)
        crop = frame[y + inset_y : y + height - inset_y,
                     x + inset_x : x + width - inset_x]
        return crop if crop.size else None

    def _veil_fraction(self, crop: np.ndarray) -> float:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation = hsv[..., 0], hsv[..., 1]
        low, high = self.config.veil_hue
        return float(np.mean(
            (hue >= low) & (hue <= high)
            & (saturation >= self.config.veil_saturation)
        ))

    def _countdown(self, crop: np.ndarray) -> int | None:
        """The seconds printed on the veil, or None if they cannot be read.

        Matched against the clock's glyph set rescaled, exactly as the level
        box and the resource text are. Every accepted glyph must be a digit
        and clear the margin gate; a glyph that cannot sinks the whole number
        rather than being guessed at, because a countdown with one digit
        quietly wrong still looks like a countdown.
        """
        if self.glyphs is None:
            return None
        cfg = self.config
        height = crop.shape[0]
        low, high = cfg.digit_band

        mask = lit_mask(crop, self._digit_config)
        grey = np.where(mask, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0)
        grey = grey.astype(np.uint8)

        boxes = [
            box for box in glyph_boxes(crop, self._digit_config)
            if box[3] >= cfg.digit_min_height
            and low <= (box[1] + box[3] / 2) / height <= high
        ]
        if not 1 <= len(boxes) <= 3:
            return None
        boxes.sort(key=lambda box: box[0])

        glyph_width, glyph_height = self.glyphs.size
        digits = []
        for gx, gy, gw, gh in boxes:
            scale = glyph_height / gh
            resized = cv2.resize(
                grey[gy : gy + gh, gx : gx + gw],
                (max(1, round(gw * scale)), glyph_height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            label, score, margin = self.glyphs.match(
                centred(resized, (glyph_width, glyph_height))
            )
            if (not label.isdigit() or score < cfg.digit_min_score
                    or margin < cfg.digit_min_margin):
                return None
            digits.append(label)

        value = int("".join(digits))
        return value if 0 < value <= cfg.max_countdown else None


def load_ability_reader(
    width: int, height: int, glyphs: GlyphSet | None
) -> AbilityReader | None:
    """The reader for a resolution, or None if it is not calibrated.

    Works without the clock's glyphs -- casts still read, countdowns do not --
    unlike the resource reader, whose whole output is text.
    """
    layout = AbilityLayout.for_resolution(width, height)
    if layout is None:
        return None
    return AbilityReader(layout, glyphs)
