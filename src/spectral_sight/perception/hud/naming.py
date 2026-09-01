"""Which champion each ally portrait slot belongs to, learned from deaths.

The HUD's four teammate portraits are in a fixed order for the whole match, so
naming a slot once names it forever -- but nothing on screen states the order.
The portrait art itself cannot: it is skin-specific, which is the established
reason HUD portraits are rejected as an identity source (see `portraits`). And
matching per-frame counts -- one dead slot, one track missing -- attributes
deaths to whichever living ally the detector happened to blink on that instant,
which is the failure `Pipeline._attribute_deaths` records in detail.

What does work is the same move the rest of the project makes everywhere:
accumulate evidence until a handful of bad frames cannot outvote it. An ally
is drawn on the minimap exactly while alive, and measured over three of the
local player's own deaths -- 388 frames of a champion known to be off the map
-- the gallery never once confidently matched the absent champion's icon
elsewhere. So a champion confidently matched while a slot reads dead is alive,
and is not that slot's corpse. Each unnamed slot carries a table: how long it
has been confirmed dead, across every death it has (the answer cannot change
mid-match), and how many of those seconds each candidate champion -- blue
roster, minus the local player, minus champions already named to other slots
-- was matched for.

**A candidate is judged against their own match rate, not against a fixed
bar.** How often the gallery vouches for a champion varies enormously with the
champion: measured on the same session, one mid-laner was matched every few
frames all game while the support went unmatched for most of a minute at a
stretch, standing close enough to their carry that the matcher would not swear
to the blob. Absolute silence is therefore meaningless from the support and
damning from the mid-laner, and the icon confusion cuts the other way too: the
matcher sometimes read that support *as* another teammate, so the corpse of a
well-matched champion can still show a trickle of phantom sightings. The rule
that survives both: a candidate has proven themselves alive during a slot's
dead footage once they were matched for `SEEN_MIN` absolute seconds *and* for
`ALIVE_FRACTION` of what their own running rate says those seconds should
have produced. A slot locks when exactly one candidate has failed to prove
alive over `DEAD_SECONDS` or more of dead footage. Anything else -- two
quiet candidates, or none -- is ambiguity, and the slot waits.

Waiting works because names cascade. Every lock shrinks the candidate pool,
so each lock re-judges every remaining table: the support dying cleanly names
their own slot, which strips them from the pool and lets the table where they
and a phantom-ridden teammate were indistinguishable settle at last. And when
three slots are named the fourth needs no death at all -- one slot, one
candidate, locked by elimination.

Naming is sticky once locked, like the roster: portrait order cannot change
mid-match, so a name that could drift is a name that was never evidence.

**Slot liveness is debounced here, not taken raw.** The raw reading flaps in
exactly the places a counter can shrug off but a named event cannot: the HUD
warming up read four teammates dead for under a second, and the post-game
screen flaps every slot sub-second indefinitely. A real death holds for a full
respawn timer, so a state must hold for `CONFIRM_SECONDS` of trusted footage
before it is believed -- symmetrically, so a death and its respawn shift by
the same amount and `down_for` stays honest. Frames the caller does not trust
(the match timer did not resolve, so the HUD is not proven on screen) freeze
the debounce rather than feeding it, which is what keeps the post-game screen
-- where the timer is gone -- from confirming anything ever again.
"""

from __future__ import annotations

from spectral_sight.perception.hud.alive import Liveness

SELF_SLOT = "self"
"""The slot holding the local player, as named by `PortraitLayout.all_crops`.
Debounced like every other slot -- the post-game screen flaps it exactly as
hard -- but never a naming candidate: the viewport names the local player, and
their raw reading keeps gating the self-identity vote in the pipeline."""

CONFIRM_SECONDS = 1.0
"""How long a slot's reading must hold before the change is believed.

The warm-up flicker and the post-game flapping both jump multiple slots within
a second; the shortest real death timer holds for ten. One second sits far
from both, and costs a named death event exactly that much latency."""

DEAD_SECONDS = 5.0
"""Confirmed-dead footage a slot must accumulate before it may lock. A single
death offers ten seconds and up, so this is a fraction of the very first one
-- late enough that the rivals have had a fair chance to be seen, early
enough that the death that taught the mapping is still in progress and gets
emitted."""

SEEN_MIN = 1.0
"""Absolute sighted-seconds a candidate needs, across a slot's dead footage,
to prove themselves alive. One confident frame could be the matcher's
mistake; ten of them are a champion walking around."""

ALIVE_FRACTION = 0.25
"""Share of their own expected sightings a candidate must reach to prove
themselves alive. The corpse of a well-matched champion can show phantom
sightings -- a teammate's icon misread as theirs -- but measured, those ran
at about a seventh of the champion's living rate, while genuinely living
champions stayed near their norm. A quarter splits the two regimes."""

MAX_STEP = 1.0
"""Cap on the seconds a single frame may contribute. A sparse stream still
accrues time between frames, but a long gap is one frame of evidence, not
however many seconds the capture happened to skip."""


class SlotNaming:
    """Debounced ally-slot liveness, and the learned slot-to-champion map.

    Stateful across frames, like every learner in the pipeline: feed it each
    frame's readings in order via `update`, ask it `state` and `name`.
    """

    def __init__(
        self,
        *,
        confirm_seconds: float = CONFIRM_SECONDS,
        dead_seconds: float = DEAD_SECONDS,
        seen_min: float = SEEN_MIN,
        alive_fraction: float = ALIVE_FRACTION,
    ) -> None:
        self.confirm_seconds = confirm_seconds
        self.dead_seconds = dead_seconds
        self.seen_min = seen_min
        self.alive_fraction = alive_fraction
        self._confirmed: dict[str, bool] = {}
        """Debounced liveness per ally slot. Absent until first confirmed."""
        self._pending: dict[str, tuple[bool, float]] = {}
        """Per slot: the reading disagreeing with `_confirmed`, and how many
        trusted seconds it has held so far."""
        self._slots: set[str] = set()
        """Every ally slot the layout has shown us, for the elimination case."""
        self._elapsed: float = 0.0
        """Interpretable seconds so far -- roster locked, local player named --
        the denominator under every champion's running match rate."""
        self._champion_seen: dict[str, float] = {}
        """Of those seconds, how many each champion was confidently matched
        for. `_champion_seen[c] / _elapsed` is the rate their absence is
        judged against."""
        self._dead_time: dict[str, float] = {}
        """Per unnamed slot: seconds it has spent confirmed dead, across every
        death, while the evidence was interpretable."""
        self._sighted: dict[str, dict[str, float]] = {}
        """Per unnamed slot: of that dead time, how much of it each champion
        was confidently matched on the minimap for -- the exclusion table."""
        self._names: dict[str, str] = {}
        """Locked slot-to-champion pairs. Sticky."""
        self._last: float | None = None

    # -- what is known -----------------------------------------------------

    def state(self, slot: str) -> bool | None:
        """Debounced liveness for an ally slot, None until first confirmed."""
        return self._confirmed.get(slot)

    def name(self, slot: str) -> str | None:
        """The champion a slot has locked onto, if it has."""
        return self._names.get(slot)

    def names(self) -> dict[str, str]:
        """Every locked slot-to-champion pair. For inspection."""
        return dict(self._names)

    # -- learning ----------------------------------------------------------

    def update(
        self,
        liveness: Liveness,
        seen: set[str],
        roster: frozenset[str] | None,
        self_champion: str | None,
        timestamp: float,
        *,
        trusted: bool = True,
    ) -> None:
        """Fold in one frame.

        `seen` is the champions confidently gallery-matched on the minimap
        this frame, blue team only. `trusted` says the in-game HUD is proven
        on screen (the pipeline proves it by the match timer resolving) -- an
        untrusted frame is ignored outright, freezing both the debounce and
        the evidence rather than feeding either with a screen that may not be
        showing portraits at all.
        """
        if not trusted:
            return
        step = 0.0
        if self._last is not None:
            step = max(0.0, min(timestamp - self._last, MAX_STEP))
        self._last = timestamp

        raw: dict[str, bool | None] = {}
        for slot in liveness.slots:
            if slot.slot != SELF_SLOT:
                self._slots.add(slot.slot)
            raw[slot.slot] = slot.alive
            self._debounce(slot.slot, slot.alive, step)
        self._observe(raw, seen, roster, self_champion, step)

    def reset(self) -> None:
        """Forget everything, for when the footage tears wholesale. A splice
        puts different champions in the same boxes, and both the held states
        and the learned names describe footage that ended."""
        self._confirmed.clear()
        self._pending.clear()
        self._slots.clear()
        self._elapsed = 0.0
        self._champion_seen.clear()
        self._dead_time.clear()
        self._sighted.clear()
        self._names.clear()
        self._last = None

    # -- internals ---------------------------------------------------------

    def _debounce(self, slot: str, reading: bool | None, step: float) -> None:
        if reading is None:
            # An unreadable slot neither confirms nor refutes; the held state
            # stands and the pending one waits.
            return
        if reading == self._confirmed.get(slot):
            self._pending.pop(slot, None)
            return
        pending = self._pending.get(slot)
        if pending is None or pending[0] != reading:
            # The step just elapsed belongs to whatever the slot read before
            # this frame, so a fresh disagreement starts from nothing.
            self._pending[slot] = (reading, 0.0)
            return
        held = pending[1] + step
        if held >= self.confirm_seconds:
            self._confirmed[slot] = reading
            self._pending.pop(slot, None)
        else:
            self._pending[slot] = (reading, held)

    def _observe(
        self,
        raw: dict[str, bool | None],
        seen: set[str],
        roster: frozenset[str] | None,
        self_champion: str | None,
        step: float,
    ) -> None:
        if roster is None or self_champion is None or step <= 0.0:
            return
        self._elapsed += step
        for champion in seen & roster:
            self._champion_seen[champion] = (
                self._champion_seen.get(champion, 0.0) + step
            )
        # Every dead slot accrues in parallel: a champion matched right now
        # is alive, which excludes them from being *any* of the current
        # corpses. Both readings must agree the slot is dead: the debounced
        # state trails the raw one by the confirm hold, and in that lag after
        # a respawn the champion is already back on the map -- at the
        # fountain, isolated, matching beautifully -- which would hand the
        # corpse a second of sightings against itself every time it revived.
        for slot, held in self._confirmed.items():
            if held is not False or raw.get(slot) is not False:
                continue
            if slot in self._names or slot == SELF_SLOT:
                continue
            self._dead_time[slot] = self._dead_time.get(slot, 0.0) + step
            table = self._sighted.setdefault(slot, {})
            for champion in seen & roster:
                table[champion] = table.get(champion, 0.0) + step
        self._resolve(roster, self_champion)

    def _resolve(self, roster: frozenset[str], self_champion: str) -> None:
        """Lock every slot the evidence now settles, letting each lock
        re-judge the rest: a claimed champion leaves the candidate pool, which
        is what breaks the ties a single death cannot."""
        while True:
            candidates = (
                set(roster) - {self_champion} - set(self._names.values())
            )
            unnamed = self._slots - set(self._names)
            if not candidates or not unnamed:
                return
            if len(candidates) == 1 and len(unnamed) == 1:
                self._lock(next(iter(unnamed)), next(iter(candidates)))
                continue
            for slot in sorted(unnamed):
                if self._dead_time.get(slot, 0.0) < self.dead_seconds:
                    continue
                quiet = [
                    c for c in candidates if not self._proven_alive(slot, c)
                ]
                if len(quiet) == 1:
                    self._lock(slot, quiet[0])
                    break
            else:
                return

    def _proven_alive(self, slot: str, champion: str) -> bool:
        """Whether this champion was matched enough, while `slot` was dead,
        to rule them out as its corpse -- judged against their own rate."""
        sighted = self._sighted.get(slot, {}).get(champion, 0.0)
        if sighted < self.seen_min:
            return False
        expected = (
            self._champion_seen.get(champion, 0.0) / self._elapsed
            * self._dead_time.get(slot, 0.0)
        )
        return sighted >= self.alive_fraction * expected

    def _lock(self, slot: str, champion: str) -> None:
        self._names[slot] = champion
        self._dead_time.pop(slot, None)
        self._sighted.pop(slot, None)
