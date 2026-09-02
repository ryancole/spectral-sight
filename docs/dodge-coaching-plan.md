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
that shape everything below. It was checked rather than assumed, because a
human player might unlock it: on `Recording 2026-08-30 200315` the player's own
nameplate sits at x 966–976, y 479–504 across sixteen minutes, which is a
locked camera and nothing else. (The minimap viewport route resolved the player
on only 20% of that clip's frames — a weakness of that route, not of the
camera, and the green self nameplate is the better anchor; see the spawned
follow-ups.)

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

10 Hz is not enough for projectiles. Measured, a bolt is *briefer* than the
plan first guessed: Ezreal's Q crosses the view at ~2,300 px/s and is gone in
a quarter of a second. The recordings are 30 fps, but about 30% of frames are
repeats or stale refreshes (the world view not redrawn — see
`perception/screen/motion.py`), so the effective distinct rate is ~20 fps and a
bolt is **four to six distinct frames**. That is enough to track; 10 Hz would
have seen two. The consequence is accepted rather than fought: **this feature
is offline-VOD first** (`--input --stride 1`). Identification already does not
run live, and the projectile stage will not either. The live-window path can
later run the projectile stage on a center crop, or the receiver can play at
reduced speed.

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

Runs at full VOD rate on the world view (`perception/screen/`).

1. **Stabilize** — camera translation per distinct frame pair. Planned as
   phase correlation; built as sparse Lucas-Kanade flow with a median, because
   measured on footage phase correlation beat doing nothing on 48% of moving
   pairs (a fight's foreground outvotes the terrain) while the flow median
   halved the residual on every one. This also *is* the player-motion
   measurement Phase 3 consumes.
2. **Candidates** — difference the stabilized pair; moving blobs, with the
   previous frame's foreground cleared so every mover is not painted twice.
3. **Tracks** — constant-velocity association with mutual-nearest-neighbour
   links and a step-relative birth test, because with 40–100 blobs a frame
   the enemy is not champions but chance chains. Keep tracks that are fast
   (≥ 800 px/s; champions and effects top out ~600, Ezreal's W starts at
   ~800), straight and brief.
4. **ML gate** — the survivors still over-produce, and many of them are
   *real* (minion and turret bolts are projectiles). A small CNN classifies
   track chips, trained on Phase 0's auto-labeled clips. Train in PyTorch
   offline; ship inference on ONNX Runtime so the package grows one wheel,
   not a framework.

Escalate to a full learned detector only if candidate *recall* — measured on
clips where every cast is known — proves insufficient. Precision problems
stay in the gate.

### Phase 3 — dodge coaching

Screen space only. A projectile track whose extrapolated path passes within a
threat radius of the player anchor is a `threat`. Outcome from the printed HP
text — the same signal that was the training label, now the scorer: a fall in
the window around arrival is a hit; none is a dodge; no reading is unknown.
The player's response comes from the camera-motion track, as displacement
across the bolt's line between onset and arrival.

Two things the footage changed. **Reaction latency is not measurable as
planned**: a bolt gives ~0.26 s of warning from first sighting, the edge of
human reaction, so "onset to velocity change" is not a coaching number — the
dodge is a response to the enemy's *cast*, which is Phase 5's signal. What is
measurable is whether the player was already moving across the line. And
**the origin gate is what makes a threat mean anything**: a candidate that
launched within 160 px of an enemy champion's plate is followed by a health
fall three times as often as the baseline; where the track ended is useless.

The `threat` event (with `at`, `arrival`, `closest`, `speed`, `heading`,
`outcome`, `damage`, `moved_across`, `origin`) is in `docs/output-format.md`
and rides the self row through feed, replay and dashboard unchanged. The
pipeline gained a two-rate mode (`--coach`): every frame fed, minimap stages
sampled every `--stride`.

### Phase 4 — aim coaching

Phase 1 says which skillshot the player cast and when; Phase 2 catches the
projectile leaving the player anchor; the outcome was to be an enemy nameplate
HP drop coinciding with the track ending on that plate. Emits `skillshot`
events with hit/miss and target lead/lag. `Recording 2026-08-30 200315` is the
validation clip — a human Ezreal is as skillshot-dense as footage gets.

*Built, and the outcome half of that paragraph did not survive the footage —
see the status below. The track's end is useless and the health drop is not a
label while anything else on the map is dealing damage, so the verdict is the
bolt's geometry and the health fall rides along as corroboration.*

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
- **Phase 2: classical stage done, ML gate pending footage.**
  `perception/screen/motion.py` (camera motion by sparse-flow median, repeat
  and stale-frame skipping) and `perception/screen/projectiles.py`
  (stabilised residual → ghost-suppressed blobs → mutual-nearest-neighbour
  constant-velocity tracks → fast/straight/brief gate). Measured on
  150–330s of `Recording 2026-08-30 200315` against the player's own Q/W
  casts: **24 of 25 casts launched a candidate, ~210 candidates/min**
  overall, ~20/min heading for the player. Not yet on the wire — what a
  consumer needs (a `threat`, not a track) is Phase 3's decision. The ML
  gate waits on Phase 0 footage. `tools/detect_projectiles.py --sweep`.
- **Phase 3: built.** `perception/screen/threats.py` judges candidates
  against the player's model, resolves outcomes off the printed health and
  the response off the camera track; `threats` ride the self row, `threat`
  events flow through feed/replay; `--coach` is the two-rate pipeline mode.
  Measured on 150–330s of `Recording 2026-08-30 200315`: 4.7 threats/min
  (14: 5 hit, 4 dodged, 5 unknown), 0.40 s median warning. Reaction latency
  was dropped as unmeasurable at that warning; `moved_across` replaces it.
  The origin gate (bolt launched at an enemy plate) is on, the end gate off.
  What a threat lacks is a label — Phase 0 footage and the classifier.
- **Phase 4: built.** `perception/screen/aim.py` joins the cast, the bolt it
  launched and the enemy plate it went at; `skillshots` ride the self row, a
  `skillshot` event flows through feed and replay, and `has_skillshots` is on
  for a `--coach` run with the ability HUD and nameplates calibrated. Measured
  on 150-700 s of `Recording 2026-08-30 200315`: **111 casts, 72 launching a
  bolt, 24 with an enemy on screen in front of it** -- 17 called hits, 7
  misses, at a median miss of 77 px. `tools/detect_skillshots.py --sweep`.

  **The phase changed shape once measured, and the reason is worth carrying
  forward.** The plan above said the outcome was "an enemy nameplate HP drop
  coinciding with the track ending on that plate". Neither half survived
  contact with the footage. Where a track *ends* is useless, the same finding
  the threat stage records from the other direction: a bolt's track stops when
  its residual fades, at a median 0.62 of the way to its target, and the
  near/wide split on that is 0.75 against 0.52. And the health drop is **not a
  label on this footage at all**: an enemy's bar falls in **52% of every
  0.65-second window they are on screen** (1,327 windows, 62 plate tracks),
  because a bot-game lane trade is continuous damage from minions, the turret,
  autos and the player's other abilities. No window or threshold fixes it --
  tightening until the baseline is informative drops the near-miss rate to
  chance with it.

  So the verdict is geometric -- did the bolt's line pass within a hit radius
  of the target's model -- and the health fall is carried as `fall`,
  corroboration rather than answer. It leans the right way (82% of the 17
  called hits, 43% of the 7 called misses) but seven wide shots is not a
  result. The hit radius sits in a gap in the measured miss distances (nothing
  between 124 px and 183 px) and is the least settled number in the module.

  Two smaller things measured along the way. A bolt is claimed by one cast
  only: without that, three of twenty-four shots were two casts within half a
  second both credited with the same bolt. And the miss is measured against
  where the target stood at the *launch* rather than at arrival -- the two
  differ by a median 12 px and disagreed on 1 verdict in 24, and the launch
  position is available on every shot where the arrival position is available
  on five sixths of them.

  **This makes Phase 0 the critical path for Phase 4 as well as Phase 3.** One
  enemy, one skillshot, nothing else dealing damage: then the baseline is zero,
  the bar is the label it was supposed to be, and the hit radius can be swept
  against it instead of chosen in a gap.
- Phase 5: not started
