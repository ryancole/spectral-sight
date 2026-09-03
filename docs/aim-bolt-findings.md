# Aim stage: where the bolts go — measurements, 2026-09-02

A record of three investigations into `perception/screen/aim.py` and the
projectile stage it joins against, so none of them has to be repeated. All
numbers are from `Recording 2026-08-30 200315` (a human Ezreal, whose Q and W
always launch a bolt), on two ranges: **200–470 s**, the window mind-control's
`coach-sample.jsonl` fixture covers, and **150–700 s**, the range the plan
doc's Phase 4 status quotes. Nothing here is from labelled footage; the one
free ground truth is that every Q/W cast launched *something*.

## Method, so it can be re-run

1. **Dump the stage once, replay the join offline.** A per-frame dump of the
   ability reader's casts (with the frame they were read on), the sampled
   nameplates (self anchor and enemy plates), every frame's camera-motion
   flags, and every finished track of three or more points with speed
   ≥ 500 px/s. The tracker was run with `min_points=3` so tracks that died
   one short of the gate are in the dump too; `is_projectile` with the default
   config is applied offline. Everything below replays `AimDetector` over that
   dump in the pipeline's order (casts observed on their read frame,
   `consider` before the sampled frame's plates update the anchor), which
   reproduces the pipeline's own output exactly.
2. **Stray-claim rate.** Fake casts placed every 0.5 s at moments with no real
   cast within 1.5 s, run through the same join. The share credited a bolt is
   the probability a cast is credited an unrelated track. Ezreal's autos are a
   bolt from the model too, so this is an upper bound on strays that are not
   the player's, not a pure false-positive rate.
3. **Trace-back test.** A shot the player fired must have a line that passes
   back through the player's model, with the model *behind* its first tracked
   point: perpendicular distance from the anchor to the track's line ≤ a
   tolerance, and the anchor-to-start vector pointing along the heading. The
   anchor (self plate centre + 95 px) was checked: on frame crops the marker
   sits on the champion, and the origin fitted by least squares from the real
   bolt lines lands within 30 px of it.
4. **Frame crops.** 600 px crops around the anchor, six consecutive frames,
   with the credited track drawn, for any case that needed eyes.

## 1. Recall: why 15 of 51 Q/W casts on the fixture carried no bolt

The fixture's count reproduces (36 of 51 Q+W credited; 73 of 111 casts on
150–700 s against the doc's 72). The plan doc's Phase 2 "24 of 25" is a
different metric — 400 px radius, −0.15 to +0.6 s, casts with no self plate on
the cast frame dropped from the denominator — and on the same 25 casts the aim
window (250 px, −0.1 to +0.5 s) sees 20. Nothing regressed.

The "settled late" rows in the fixture (the Q at 381.6 on the row at 385.6,
the Q at 253.733 on 259.3) are the minimap self track blinking out: the
pipeline holds ability and skillshot events until a self row exists. The
reader itself emits every cast one frame after `at`.

Per cast:

| Cast | Nearest evidence | Why no bolt | Class |
|---|---|---|---|
| W 204.567 | 652 and 678 px/s tracks at +0.43 s, 75–94 px | below the 800 px/s floor | detection |
| W 213.833 | 3-point tracks at +0.00 (2721 px/s) and +0.33 (1733) | died at three points | detection |
| W 228.8 | candidate at +0.57 s, 916 px/s, 224 px | 0.07 s past `launch_after` | join: window |
| Q 250.033 | candidate at +0.07 s, 884 px/s, 120 px | anchor None on its frame (two SELF plates read) | join: anchor — **fixed** |
| W 266.0 | candidate at +0.30 s, 1196 px/s, 239 px | anchor None on its frame | join: anchor — **fixed** |
| Q 328.9 | candidate at +0.53 s, 1347 px/s, 159 px | one frame past `launch_after` | join: window |
| Q 339.367 | candidate at −0.100 s, 1044 px/s, 137 px | exactly three frames early, off the inclusive edge by float error | join: edge — **fixed** |
| Q 359.3 | candidate at −0.100 s, 818 px/s, 122 px | same | join: edge — **fixed** |
| W 367.3 | 3-point at +0.10 s, 961 px/s, 144 px | died at three points | detection |
| Q 377.167 | 3-point at −0.03 s, 890 px/s, 126 px | died at three points | detection |
| Q 381.6 | 3-point tracks at −0.13 and +0.27 s | died at three points | detection |
| Q 440.0 | nothing fast within 250 px | no track | detection |
| W 447.1 | 3-point tracks at +0.03 and +0.10 s, ~900 px/s | died at three points | detection |
| Q 460.4 | candidates at +0.20 s, 335–346 px | past `max_launch` | join: distance |
| R 467.567 | 3-point tracks at −0.07 and +0.07 s | died at three points | detection |
| Q 468.3 | nothing fast within 250 px; the R missile is on screen | no track | detection |

Split: 8 detection, 7 join. The two anchor cases and the two edge cases are
fixed in commit `5a1d741` (last good anchor stands in when a frame's plate
read resolves no player; the launch window's edges carry 0.005 s of slack).
That took 150–700 s from 72 to 83 credited bolts with no already-credited bolt
changed except the R at 385.533, which took a nearer early stray — and see §3
for why that R's bolt was never real anyway.

Measured and **not** taken, with the stray rate as the cost:

| Change | Fixture Q+W of 51 | 150–700 s of 111 | Stray rate |
|---|---|---|---|
| none | 36 | 73 | 22.3% |
| anchor fallback + edge slack (taken) | 40 | 83 | 22.3% per anchored cast |
| `min_points` 3 | 46 | 96 | 38.3%, and 15 credited bolts change |
| `launch_after` 0.7 | 38 | 78 | 23.3%, and the W at 272.267 steals the Q at 272.767's bolt |
| `max_launch` 350 | 39 | 82 | 27.2% |

## 2. Why bolts die at three points, and four remedies that do not pay

Instrumented the tracker over 195–475 s and looked at every fast track with
exactly three points born beside a cast (115 of them) on the frame it first
failed to link:

| First failed link | Count |
|---|---|
| A rival track took the blob under mutual-nearest — usually a shorter chance track a few px nearer | 33 |
| The ghost mask erased the bolt's own next blob: the dilated trail of a slow-enough bolt overlaps its next position | 30 |
| The residual genuinely faded below `diff_threshold` | 28 |
| Prediction landed outside the 45 px gate | 13 |
| Only blobs with an out-of-range area ratio | 11 |

So `projectiles.py`'s claim that a bolt never overlaps its previous position
is false for bolts with trails (W's orb, R), and mutual-nearest has no notion
of a confirmed track outranking a pair. Both mechanisms are real. The catch is
that the same split holds for the 1,556 fast three-point tracks *anywhere* —
whatever helps a bolt link also helps a chance chain.

Remedies, scored on the aim join over 200–470 s (Q+W bolts of 53 casts here,
because the dump includes two casts at 401.967 and 402.2 the pipeline dropped
as untrusted):

| Tracker variant | Q+W bolts | candidates/min | stray rate | bolts lost |
|---|---|---|---|---|
| base | 42 | 219 | 26.9% | — |
| clear the ghost mask in a 45 px disc around every fast track's prediction | 41 | 345 | 36.8% | 4 |
| confirmed fast track with nothing admissible gets a second look at the raw (pre-ghost) blobs, which never enter the pool | 38 | 215 | 26.5% | 5 |
| seniority in every conflict (more points wins, distance breaks ties) | 44 | 268 | 33.2% | 2 |
| seniority only when a confirmed (≥3) track meets an unconfirmed one | 45 | 254 | 35.6% | 1 |
| both narrow forms | 42 | 229 | 32.0% | 4 |

The ghost fixes lose bolts outright: extending a track into its own trail
blob bends or slows it and the gates reject the whole track. The only recall
gain is confirmed-first (+3 of the 11 silent Q/W casts), at +9 points of
stray claims; of its 17 changed bolts, 15 are different tracks and five head
the opposite way. Nothing adopted. The fourth-point conflict needs a
discriminator other than distance — appearance, or the ML gate, which needs
Phase 0 labels.

## 3. Most credited bolts are not the player's shot

This is the finding that reframes the rest. Trace-back test on every credited
bolt, 200–470 s:

| Credited bolts | trace back through the model (45 / 60 / 80 px) | total |
|---|---|---|
| real Q/W casts | 8 / 9 / 12 | 42 |
| real casts, all slots | 9 / 10 / 13 | 46 |
| stray claims on fake casts | 12 / 14 / 21 | 68 |
| real casts that received a verdict | 4 / 5 / 8 | 19 |

Three frame crops confirmed it:

- **Q at 214.1**, called a hit with a 6 px miss: the credited track ran
  horizontally 240 px above the player, at the enemy's position. The real Q,
  yellow, left the model to the right at +0.43 s.
- **Q at 203.4**: the credited track was an enemy's blue crescent gliding
  *into* the player.
- **Q at 398.467**: the credited track was the genuine bolt leaving the model
  toward the upper right.

The fitted origin of the real bolt lines sits at anchor + (−11, −30) px, so
the anchor is not the explanation; the credited tracks are enemy projectiles
arriving, allied bolts passing by, and effects near the model. The stray
rate's original suspect — Ezreal's auto-attacks — is at most the residual: the
strays that *do* trace back (about a fifth) end at the same distances as real
bolts and show no attack cadence in a sample of 21, so autos can be neither
confirmed nor excluded there.

An origin gate in `consider` (perpendicular ≤ tol, anchor behind the start),
measured:

| Gate | 150–700 s: credited of 111 | aimed | hit | missed | stray rate |
|---|---|---|---|---|---|
| none (current) | 83 | 28 | 20 | 8 | 25.3% |
| 45 px | 26 | 9 | 8 | 1 | 5.1% |
| 60 px | 30 | 11 | 10 | 1 | 5.6% |
| 80 px | 38 | 15 | 14 | 1 | 7.3% |

Seven of the eight `missed` verdicts vanish with the gate. **Nearly every wide
shot the stage has reported was a stray**, and so was the gap in miss
distances between 124 and 183 px that `AimConfig.hit_radius` cites as its
justification. The Phase 4 numbers in the plan doc and the aim module's
docstring (24 aimed, 17 hit, 7 missed, median miss 77 px, the 82%/43% fall
split) are built on that contaminated set and should not be quoted until
re-based. For mind-control: a `missed` verdict from the current stage is
mostly not the player's shot; the silence problem from §1 is the smaller
effect.

**Recommendation, not applied:** add the origin gate at 80 px (which admits
the W's large orb where 45 px does not), re-base the docstring and plan doc on
the gated run, and accept that credited bolts fall to about a third of casts
— the stage then sees the shot in a third of cases and says nothing otherwise.

## 4. The choice rule is moot once the gate is in

`_settle` picks the candidate nearest the model, earliest to break ties.
Alternatives measured: earliest-first, and smallest trace-back miss.

- **Without the gate** the rules reshuffle 9–11 of 46 credited bolts and none
  raises the share that traces back above 28%. Moving verdicts among strays.
- **With the gate at 80 px** all three rules credit the same 15 bolts and
  differ on one cast each, because the gate leaves most casts a single
  admissible candidate. The R at 385.533 gets no bolt under any rule.

Widening the window under the gate, 150–700 s:

| `launch_after` | credited of 111 | casts with >1 admissible offer | aimed | missed | stray rate |
|---|---|---|---|---|---|
| 0.5 s (current) | 38 | 13 | 15 | 1 | 7.3% |
| 0.7 s | 44 | 20 | 16 | 4 | 7.8% |
| 0.9 s | 48 | 19 | 17 | 4 | 8.2% |

Six to ten more bolts, contested casts up by half, two credited bolts change,
three new misses of unknown provenance. Three of the gains are E casts —
Arcane Shift's homing bolt from the landing point, which is real and which
the gate handles. Leave the window alone until the gate is in and Phase 0
footage can say what the new misses are.

## What is still true after all of this

- The anchor is right. The camera is locked; the model anchor moved under
  8 px across the clip.
- The silence of a cast with no bolt is honest and errs toward under-counting
  shots thrown.
- The verdicts are not, yet. The gate is the fix, and it is one config field
  and one condition in `consider`.
- Phase 0 footage remains the only way to turn any of these proxies into a
  precision or recall number.
