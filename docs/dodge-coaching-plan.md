# Dodge / aim coaching — plan

The feature: read what the player threw and what was thrown at them, then say
how well they aimed and how well they dodged. Two new kinds of perception carry
it — the player's own ability HUD, and projectiles on the main screen — plus a
coaching layer that turns those into events.

This document is the plan of record. Each phase ships something measurable on
its own; the order exists so that no phase waits on footage or infrastructure
it does not strictly need.

## The insight the design rests on

**The camera is locked to the player.** The pipeline already depends on this —
it is how the viewport names the local player — and it has two consequences
that shape everything below:

1. **The player does not move on screen; the world does.** A dodge is
   invisible as player motion but shows up as global camera translation.
   Estimating camera motion between consecutive frames (phase correlation on
   terrain, masked by the existing HUD exclusion rects) measures the player's
   velocity at sub-pixel precision — far finer than the minimap's 48 units/px,
   and dodge timing lives or dies on that measurement.
2. **Projectile screen motion is contaminated by camera motion.** Stabilizing
   frames against the estimated camera motion first makes projectiles pop out
   as residual movers, which is what makes classical candidate detection
   viable before any ML.

So camera-motion estimation is a foundation stage of projectile work, not an
optimization.

A corollary sets the coordinate frame: dodge analysis happens in **screen
space**. With the camera locked, "toward the player" is "toward a fixed screen
anchor", which sidesteps the known-weak screen-to-minimap projection entirely
for v1.

## Sampling rate

10 Hz is not enough for projectiles. A skillshot crosses the camera view in
well under a second — 7–10 samples at 10 Hz, with 100 ms timing granularity
against human reactions of ~200 ms. The recordings are 30 fps, so the VOD path
(`--input --stride 1`) already carries the needed rate; 33 ms granularity and
20–30 frames per projectile are adequate. The consequence is accepted rather
than fought: **this feature is offline-VOD first.** Identification already
does not run live, and the projectile stage will not either. The live-window
path can later run the projectile stage on a center crop, or the receiver can
play at reduced speed.

## Phases

### Phase 0 — purpose-built footage with automatic ground truth

All current footage has bot opponents, and none was recorded to exercise
projectiles. Record Practice Tool / custom clips through the receiver as
usual: one enemy champion repeatedly firing one known skillshot, the player
alternately eating it and dodging it.

The automatic ground truth is the same move the clock and the cast validator
made: the player's HP is readable **as printed text**, and in a controlled
clip only the skillshot deals damage — so "hit at time t" is labeled for
free, frame-accurately, with no hand labels. That yields training labels for
the Phase 2 classifier and precision/recall denominators for Phase 3.

Phase 0 is only on the critical path for Phase 3 scoring and ML training.
Phases 1 and 2 develop against existing footage.

### Phase 1 — self ability tracking (`perception/hud/abilities.py`)

The easy half and independently shippable. The ability slots (Q W E R and the
two summoner spells) are fixed HUD rects, derivable from the existing
single-fit calibration like every other panel. A cast is the slot's cooldown
overlay beginning — a sharp darkening that then ramps back — read against a
per-slot learned "ready" baseline, the same move `alive.py` makes for
portraits. Slot position gives the key; the roster gives the champion; the
two together name the ability without any icon matching.

Validation needs no labels: the existing mana-drop cast detector and the
printed-mana reader are two independent observers of the same casts. Score
agreement on `Recording 2026-08-30 200315` — the one clip with a human
player, on Ezreal, whose Q/W/R are all skillshots. Manaless costs (E at zero
mana, leveling edge cases) are where the HUD route sees casts the mana route
cannot, which is the point of building it.

Output: `cast` events gain the slot, and self casts stop being anonymous.

### Phase 2 — projectile detection: classical candidates, ML gate

Runs at full VOD rate on the main screen minus the existing exclusion rects.

1. **Stabilize** — estimate camera translation per frame pair (phase
   correlation on terrain). This also *is* the player-motion measurement
   Phase 3 consumes.
2. **Candidates** — frame-difference the stabilized pair; moving blobs.
3. **Tracks** — constant-velocity association, the tracker's existing
   playbook at smaller scale. Keep tracks that are fast and linearly
   coherent; champions, minions and camera jitter do not survive that filter.
4. **ML gate** — the survivors still over-produce (autoattacks, pets, wards
   flying, particles). A small CNN classifies track chips projectile / not,
   trained on Phase 0's auto-labeled clips. Train in PyTorch offline; ship
   inference on ONNX Runtime so the package grows one wheel, not a framework.

Escalate to a full learned detector only if candidate *recall* — measured on
clips where every cast is known — proves insufficient. Precision problems
stay in the gate.

### Phase 3 — dodge coaching

Screen space only. A projectile track whose extrapolated path passes within a
threat radius of the player anchor is a `threat`. Outcome from the printed HP
text — the same signal that was the training label, now the scorer: a fall in
the window around closest approach is a hit; passage without one is a dodge.
The player's response comes from the camera-motion track: reaction latency
from threat onset to velocity change, movement direction against the
projectile's perpendicular.

New event kinds (`threat`, resolved hit/dodged, carrying reaction ms) join
`docs/output-format.md` and flow through feed, replay and dashboard unchanged.

### Phase 4 — aim coaching

Phase 1 says which skillshot the player cast and when; Phase 2 catches the
projectile leaving the player anchor; the outcome is an enemy nameplate HP
drop coinciding with the track ending on that plate. Emits `skillshot` events
with hit/miss and target lead/lag. `Recording 2026-08-30 200315` is the
validation clip — a human Ezreal is as skillshot-dense as footage gets.

### Phase 5 — enemy ability naming (later)

Classify projectile chips per champion. The roster lock is the same gift it
was for identification: once locked, the candidate space is five enemy kits,
not all of League. This upgrades "you got hit" to "you got hit by Morgana Q".
It needs labeled footage per champion, so it grows with the clip library
rather than blocking anything.

## Risks, named now

- **Live mode stays behind.** Accepted above; VOD-offline first.
- **The ML dependency is new.** ONNX Runtime for inference keeps runtime
  dependencies to one wheel; training code stays out of the package.
- **Unlocked-camera footage breaks the screen-space shortcut.** The same
  assumption `viewport.py` already makes, now doing more work. State it; a
  camera-motion estimate that disagrees wildly with the minimap self-track is
  the detectable symptom.
- **Not every ability is a traveling missile.** Ground-targeted AoEs and
  instant lines need telegraph detection, not projectile tracking — out of
  scope for v1, in scope for the Phase 5 classifier.
- **Enemy behavior in current footage is bots.** Dodge-side conclusions do
  not generalize until humans-vs-humans footage exists; say so in any report.

## Status

- Phase 0: not started (needs recording time, not code)
- **Phase 1: done.** `perception/hud/abilities.py` reads the local player's
  casts off the cooldown veil, named to a slot, with the countdown when
  legible. Wired through the pipeline (`abilities` on the self row, gated on
  trusted frames and self-alive), the timeline schema (`abilities` field,
  `has_abilities` meta), the feed (unchanged — it wraps the row), and events (a
  new `ability` kind). Measured on `Recording 2026-08-30 200315` against the
  printed-mana ground truth: **90% recall of the 196 mana-fall casts, ~1%
  false-positive rate.** Inspect/score with `tools/detect_abilities.py`. Known
  residual: a long re-press flash on an already-cooling slot can emit a phantom
  (documented in the module).
- Phases 2–5: not started
