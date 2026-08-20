# spectral-sight

Vision-only, real-time game state extraction from League of Legends replays. No
Riot APIs, no process interaction — pixels in, structured state out.

## Status

**Stage 1 — class-agnostic minimap marker detection.** Working. Retuned against
5.3 minutes of footage using an automatic ground truth (see below): finds at
least all five allies in 87% of frames, averaging 5.8 blue markers against 5
expected, so it over-produces slightly by design.

**Stage 2 — champion identification.** Working for both teams. Markers are
matched against the stock champion icon set (173 icons, `tools/fetch_icons.py`);
the local player is resolved geometrically instead (see below).

Run blind against all 173 icons on the sample clip, the full ten-champion roster
falls out cleanly — four teammates and five enemies dominate, with a sharp cliff
to noise below them and no false enemy names at all. Restricting the gallery to
that discovered roster then lifts accuracy further: at least three of five
allies known in 63% of frames, four in 28%, and 1.6 enemies identified per frame
against a fog-limited maximum.

That makes enemy identification a solved problem rather than the open one it
looked like: no scoreboard frame is needed and no incremental discovery, because
the stock icon set already contains every champion.

**Minimap icons are always stock champion art, never skin-specific.** This is
the fact the design should rest on, and getting it backwards costs a lot:

- A complete stock icon set is a closed, known reference for *every* champion,
  so enemy identification needs no scoreboard frame and no incremental
  discovery. This is the intended gallery source.
- The HUD bootstrap is the weaker option, not the clever one. HUD portraits *are*
  skin-specific, so they only agree with the minimap for a champion on their base
  skin. It worked here because all four teammates were on base skins.
- It failed completely for the local player, who was using a skin — their HUD
  portrait and minimap marker share almost nothing.

So the local player is not identified by appearance at all. The minimap's camera
viewport rectangle is located instead, and since the camera is locked to the
player they sit at its centre. The nearest marker lands 5–7px from that centre
with the runner-up 38–88px away, so it is effectively exact — found in 98% of
frames and resolving the player in 85%. That route is immune to skins, gallery
coverage and fog alike.

**Tracker.** Working, and it is where most of the identification accuracy
actually comes from. Champions are followed across frames with a constant
velocity model, and identity is accumulated over a track's life rather than
decided fresh each frame — so a champion stays identified through the frames
where its 26px marker is unreadable or stage 1 blinks.

Measured over 1,200 frames at 10 Hz against the blind 173-icon gallery:

| | Per-frame | Tracked |
|---|---|---|
| Mean allies identified (of 5) | 2.52 | **4.55** |
| Mean enemies identified (of 5) | 1.61 | **3.19** |
| ≥3 allies known | 50% | **100%** |
| ≥4 allies known | 24% | **100%** |
| All 5 allies known | 4% | **56%** |

Fog is handled by one gate that grows with elapsed time,
`blink_distance + max_speed × seconds_since_seen`. Between frames it is a tight
leash; after seconds in fog it covers anywhere the champion could have walked
but still far less than the map. Frame-to-frame association and
re-identification after fog are therefore the same operation with the same
parameters, rather than two mechanisms that can disagree.

Identity evidence is weighted by each match's *margin* over its runner-up, not
by raw similarity. Similarity alone let popular icons act as a sink: many
markers weakly prefer the same champion, and three concurrent tracks each
accumulated enough to claim Galio.

**Roster locking.** The ten champions in the game are discovered from the
footage, then used as a constraint. Evidence accumulates per team; once a team's
top five are clearly ahead of the sixth, the gallery restricts to those five and
the tracker holds the team to five tracks. On the sample clip blue locks at 66s
and red at 209s, both correct.

Locking is deliberately conservative, because a wrong lock is far worse than a
late one — it makes the right answer permanently unreachable. Two constraints
carry that:

- The fifth slot must be both well supported *and* clearly ahead of the sixth.
- **A champion plays for exactly one team.** Omitting this is not academic: the
  enemy roster locked with an *ally* in it and permanently displaced the
  champion who was really there. Names one team has locked are struck from the
  other's candidate pool.

Track count is capped at five per team from the first frame, which needs no
roster — a team fields five champions whether or not their names are known yet.

Measured over the whole 5.3-minute clip at 10 Hz, against the blind 173-icon
gallery:

| | Value |
|---|---|
| Mean allies identified (of 5) | 4.79 |
| Mean enemies identified (of 5) | 4.03 |
| ≥4 allies known | 100% |
| All 5 allies known | 79% |
| ≥3 enemies known | 92% |
| False identities | **0** |
| Live tracks | mean 9.3, max **10** |

**Game clock.** Working, and exact everywhere it has been checked: **14,744
frames across all three recordings, read in 100% of them, with zero
backwards steps.** Two of those clips were never seen while the digit templates
were built.

The clock is what turns a video timestamp into a *game* timestamp, which is the
key that joins anything here to anything outside the footage — objective spawns,
level and item timings, ward duration, or a second clip of the same match. The
three sample recordings start at +22.40s, +24.00s and +160.20s of game time, and
none of that is recoverable from the video alone.

No OCR dependency and no font asset, because **the clock can bootstrap its own
templates**. The seconds ones-digit counts 0–9 in order once per second, and it
returns to zero exactly on the frame the tens-digit changes — so watching which
glyph cells change from frame to frame labels all ten digits with no human
input. Twenty seconds of footage is enough. It is the same move as the automatic
ground truth below: the game is already telling us the answer.

Checking it is free for the same reason. The clock advances one second per
second of video, so `clock − video_time` is constant for a correct reader, and
its spread across thousands of frames is a real accuracy number nobody had to
label. Two things are measured and they are not the same:

- **Running backwards is an error.** A single misclassified glyph moves the
  clock by seconds or minutes, so almost any misread trips it. Zero occurred.
- **Drifting is not.** A step in the offset means the game stopped and the video
  did not. The long clip contains one such episode, 59 frames where the offset
  slides to −3.0s, and it is a real two-second client freeze — the frames are
  static to within compression noise and then jump. The reader was correctly
  reporting a frozen screen. An earlier version of this check also required that
  the clock never *gain* on video time, which charged the catch-up after that
  freeze as an error; stalls are now reported rather than counted.

**World coordinates.** Minimap positions are now convertible to Summoner's Rift
world units, which makes them comparable across captures and, more importantly,
*addressable* — no predicate about lanes, river, jungle quadrants or distance to
a pit can be written against a number that moves when the user drags a settings
slider.

**The map area is not the minimap crop.** The panel has an ornate frame and an
inner border band, so the rendered terrain sits inside the calibrated region.
Measured on the sample capture the map is **310×310 px at (1791, 1027)** while
the calibrated crop is 325×322 at (1787, 1020) — treating the crop as the map
puts a 4.8% scale error and a several-pixel offset into every position.

That rectangle is measured by hand, once per resolution, and deliberately so.
Three ways to find it automatically were tried and all three returned a
confident answer; they disagreed with each other by about 5%, which is the size
of the error the calibration exists to remove:

- *Bounding box of non-black pixels* catches the ornate frame and comes back
  20px too large.
- *Bounding box of pixels that change over time* finds where the game happened,
  not where the map is — 16px short at the top, where the enemy base sits under
  permanent fog and nothing ever moves.
- *Rows and columns uniform along their length* looked principled, since the
  border is uniform and terrain is textured. But the map's outer ring is
  unwalkable black void, which is also uniform, so it lands 8–9px inside the
  true edge on every side.

What worked is the brightness profile across the panel edge, averaged along it:
ornament, then a flat border band, then map. Reading the band's inner edge off
that gave 310px on each axis *independently*, and nothing forced those to agree.

Scale comes out at **48.0 × 48.3 units/px**. Validating it is a different
problem from everything else here, because a coordinate scale makes no visible
prediction — the pixels look identical whether the map is 15,000 units across or
30,000. What it does predict is *motion*: champions move at a few hundred units
per second, so tracked positions either convert into that band or they do not.

Measured over 1,200 frames, the p90 speed across one-second windows is **394
u/s**, with no position landing outside the map bounds. Two caveats are worth
more than the number:

- **Measure over a second, not between frames.** At 48 units/px and 10 Hz, one
  pixel of jitter reads as 480 u/s — more than a champion actually moves. The
  frame-to-frame distribution is mostly noise.
- **This rules out gross error, not fine error.** On the same footage the
  untuned whole-crop transform scores 377 u/s against the calibrated 394, and
  both are plausible, because they differ by under 5%. A pass means the bounds
  and scale are right to within roughly ten percent. It does not confirm the
  last few pixels, and should not be quoted as if it did.

**Champion nameplates.** Working, within the window they exist in. The bars
floating over a champion give health, resource and level for anyone currently on
screen — which is 29% of frames for an enemy, against 88% having one somewhere on
the minimap. The resource bar is the only evidence this project can gather that
an enemy used an ability, since the client never displays enemy cooldowns, and
its precision is about one pixel of a 117px bar. Levels reuse the glyph set the
clock reader already learned, filtered by the fact that a level only ever rises
by one, which takes bad readings from 20.9% to zero. Plates are matched to tracks
by a fitted screen-to-minimap projection accurate to ~12px p90 in a ~74px
viewport — enough to gate on, not enough to decide alone. See *Reading a
champion's bars* below.

**Timeline output.** The pipeline now writes what it sees: one row per tracked
champion per frame, as JSONL. Until this existed nothing downstream could be
built, because every question about five minutes of footage cost minutes of
vision to ask again. Extracting once turns that into a file read.

A row is flat and self-contained — game time, champion, team, position, whether
they were visible — rather than a serialised `Track`. Freezing the tracker's
velocity and per-name evidence into the format would tie every future reader to
internals that exist to serve the next frame, not the next question.

Three choices are load-bearing:

- **Absence is recorded, not skipped.** A champion in fog still gets a row, at
  their last known position, carrying `seconds_since_seen`. Writing only what is
  currently visible would make the file silent about the difference between
  *elsewhere* and *not looked at*, which is most of what player-perspective
  footage has to say.
- **Positions are written in both spaces.** World units are the useful ones and
  the reason the transform exists, but crop pixels are always present, so an
  uncalibrated capture still produces a usable timeline instead of rows with no
  position at all.
- **The header describes the calibration.** Scale and bounds are not recoverable
  from the rows, and two timelines extracted under different calibrations are not
  comparable. It is the first line of the same file rather than a sidecar,
  because a timeline whose meaning lives in a second file is one careless copy
  away from being unreadable.

Measured on 400 frames of the sample clip: 2,149 rows at 17.2 fps, game time on
every row and monotonic, world units on every row, a champion named on 97%, and
20% of rows recording someone in fog.

**Death.** An ally is never hidden by fog, so an ally missing from the minimap
is dead — that was the reasoning, and the timeline could not act on it, so a
dead ally read as a champion nobody had seen for twenty-four seconds. The HUD
portraits now settle it: a dead champion's portrait is drawn grey.

Measured over both long recordings — 4,804 frames, 24,000 ally rows — 97.5% of
ally rows now carry a verdict, with no false deaths. The portraits found five
deaths across the two clips, four of them lasting 11.9–12.0s and one cut short
by the end of its clip; the four the timeline can attribute to a champion match
those readings exactly.

The interesting number is not the deaths. It is that **20.9% of ally rows are
champions the tracker had lost who were provably alive.** One ally row in five
was previously indistinguishable from a death, and the reasoning above would
have called every one of them a kill.

**Death is read from the portrait, not the health bar.** The bar is the obvious
place to look and it is a trap. A dead teammate's slot chrome is removed
entirely — portrait ring, bar box, borders — and the game world shows through
where the box was. The bar does not go empty, it stops existing, and what gets
measured is whatever terrain is behind it. Measured: an ally eleven seconds dead
read as *full health* in the frames where grass sat behind the missing box.

The two slot kinds also fail differently, which rules out fixing that with a
better bar reader. A teammate's box vanishes; the local player's box stays put
and reads `0 / 746`. The portrait means the same thing in both places.

Two details are load-bearing:

- **The threshold is relative to the slot's own history.** Portrait saturation
  is a property of the champion's art — living slots measured anywhere from 52
  to 161 while dead ones read 1 to 29 — so an absolute floor clearing a dead
  portrait sits uncomfortably close to a legitimately drab champion. Each slot
  learns what it looks like alive, the same move as the clock bootstrapping its
  digits. Against that baseline the tightest living slot sits at 0.78 and the
  loosest dead one at 0.24, and the threshold is halfway between.
- **Median saturation, not mean.** A dead portrait is not blank: the respawn
  countdown is drawn across it in saturated red. On the local player's larger
  portrait those digits pulled the mean to within **0.02** of the living
  threshold and the reading flickered as the number changed width. The digits
  are a small minority of the disc, so the median does not see them.

### Counting the dead does not name them

The HUD knows how many teammates are down but not which, since portrait art is
skin-specific. The obvious join is to the minimap: one dead portrait should mean
one ally track missing. **Measured, that names the wrong champion**, and it is
worth recording why.

The marker really does disappear — blue detections fall from a mean of 5.84 per
frame to 5.24 when a teammate dies. But stage 1 over-produces by about one
marker per frame *by design*, so five candidates remain, and the tracker, capped
at five per team, keeps feeding all five tracks. No track ever goes quiet. What
the counts then match on is ordinary frame-to-frame blinking, which hands the
dead champion's name to whichever living ally the detector dropped that instant.
On the sample clip that turned one twelve-second death into a dozen fragments
spread across three champions who were never dead at all.

So a death is only attributed to a champion that can be named outright, and
there is exactly one such route: the local player, resolved from the camera
viewport rather than by appearance. Their portrait is a known slot, so when it
greys out the champion at the centre of the camera is the one who died.

That name has to be *accumulated*, not read fresh, for two reasons. The viewport
finds the player by the marker at the camera centre, and a dead player has no
marker — measured, it resolves in **none** of the frames where the self portrait
reads dead. And its answer drifts: over the long clip it named the right
champion 84.6% of the time but sat on a teammate for seconds at a stretch, and
taking the latest value attributed the player's second death to two champions
who were alive throughout. Accumulating instead is the same move the tracker
makes for a marker's identity, and Zilean beat the runner-up 1,926 to 133.

A frame therefore resolves when every dead portrait is one that can be named,
which covers two cases and most of the footage: nobody dead, which needs no
naming at all and is four frames in five; and only the local player dead, which
names the casualty and so clears everyone else. A teammate down alongside — or
instead — leaves a death nobody can attribute, and those rows report `alive:
null` while `allies_dead` still carries the count. That is 2.5% of ally rows.

### Reading a champion's bars, and what that says about abilities

**Nothing in the game displays an enemy's cooldowns.** There is no HUD panel to
calibrate against, which is what every earlier stage relied on. So ability usage
has to be inferred, and the only champion-agnostic evidence the client renders is
the *resource bar* on the nameplate floating over a champion: a step down that
holds is a cast.

That window is much narrower than the minimap's, and it is the pipeline that
reads it: `perception/nameplates/casts.py` folds each champion's resource series
and writes what it finds onto the timeline's `cast_*` fields, with
`tools/detect_casts.py` reporting over a clip.

That window is much narrower than the minimap's. Measured on the coop-vs-AI clip
at 10 Hz, an enemy is somewhere on the minimap in **87.9%** of frames but inside
the camera view in only **29.4%**, averaging 0.40 readable enemy plates per
frame. Nameplates of some kind appear in 92% of frames. So the honest output is a
*cast event log*, not a cooldown state — a consumer that wants "is their ult up"
has to carry its own uncertainty.

Within that window the measurement is close to exact. On frames where a
champion's health did not change, **85.8%** of consecutive resource readings are
identical to within one pixel, and the p05–p95 spread is ±0.85% — one pixel of a
117px bar. Typical ability costs run 7–25% of a pool, comfortably clear of that
floor.

**A step is a fall that holds, not a fall.** A drop past the threshold is held
as a candidate and the following reading decides it — noise reverts, a cast does
not — which is the same move the level and clock filters make. A candidate that
never gets a continuous follow-up is still emitted, flagged `confirmed=False`,
because the champion walking off screen is not evidence against the cast and
that is where a large share of them are seen.

**Two things make a convincing fake step, and neither shows up in the resource
series.** Run over the sample clip before either was handled, the detector found
ten casts and five were artefacts. Both gave themselves away in what the health
bar did at the same instant, which is why the detector reads both bars:

- **A truncated plate** — one cut partway along by a champion model or a spell
  effect, rather than by the frame edge or a HUD panel the reader's own clipping
  test already catches — truncates both fills at the same column, so they come
  back equal. Three of the five had fills 0.008–0.009 apart, one pixel of a 117px
  bar, against 0.07–0.26 for the five that survived. Such a reading is skipped
  rather than rejected: measured *from* it invents a step down, and measured *to*
  it invents the step back up that would hide the real cast after it.
- **A plate on the wrong track** — association is geometric and only
  approximately so, so the series occasionally jumps to a different champion's
  bars, and both fills move by nearly the same amount at once. The other two
  moved health and resource within 0.033 and 0.010 of each other while dropping
  60%. The test is on the two falls *agreeing*, not on damage happening:
  champions cast while being hit constantly.

**Levelling cannot fake a cast**, which is the opposite of what this section
used to claim. It raises maximum resource and grants the same amount to current,
so the fraction holds or rises. Across seven level-ups straddling consecutive
readings the change ran −0.9% to +5.2%, mean +1.0% — the one fall being a single
pixel, three times below the threshold. It can *mask* a cast by cancelling part
of it, which is the safe direction and is left alone.

What survives on the 5.3-minute clip is six casts, all continuous and five of
them confirmed. They cluster the way repeated use of one ability should:
Xerath's four came in at 9.4%, 10.3%, 22.2% and 28.2% — two abilities, not four
— with Zilean at 9.4% and Sivir at 16.2%.

### What the score actually is

Both halves are now measured, because the HUD prints the local player's mana as
text — `488 / 488` — so their resource is known exactly on every frame and a
cast is simply a fall in that number. See below for how that is read.

Across three clips the HUD saw eight real casts:

| | |
|---|---|
| Precision | **7/7 — 100%** |
| Recall | **7/8 — 88%** |

On the clip recorded specifically to exercise this — 1.6 minutes in which the
player actually used abilities — it is 6/6 and 6/6. The single miss across all
footage was a frame where the plate could not be read at all.

Where the cast is continuous the size is close: 60 mana of a 452 pool is 13.3%
and the detector reported 12.8%. Where it spans a gap the size decays exactly as
designed, because regeneration refills part of what was spent while nobody was
looking — a 48-mana cast seen across 4.1 seconds came back as 5.1% against a
true 9.6%. The event is found; the magnitude is a floor, not a measurement.

Two of the six were located only as intervals, one of them 9.1 seconds wide.
That is the honest output for a champion who cast and then walked off screen,
and it is why `span` and `continuous` are on the row rather than a single
timestamp.

**Eight casts is still a sample**, and it should not be quoted as a rate.

Two earlier versions of this measurement were wrong, both times because the
check was trusted before its own error rate was known.

- The first reported 14%, because the reader below misread a `9` as a `4` in the
  tens digit, inventing a fifty-unit fall and recovery on alternating frames.
  Five of its seven casts were phantoms. Chasing the resulting "misses" produced
  a confident and wrong conclusion about plate association, which is in fact
  fine on the player's track — it agrees with printed mana to a median of one
  pixel, against 36.5% agreement for every other track.
- The second scored a cast as *both* a miss and a false positive whenever it
  spanned a gap, by matching against a fixed window around `at` instead of
  against `[at - span, at]` — the interval this very document says a cast
  occupies. Two events, counted four times, in opposite directions.

And the ground truth itself dropped a real cast because the maximum beside it
read 502 on one frame and 501 on the next, which the level-up test read as the
pool changing. One digit of flicker, and the check quietly discarded an event it
existed to catch. That is what `MaximumFilter` is for.

### Reading the player's own numbers

The numbers on the player's HUD bars are the timer's face at a smaller size,
roughly 10px against 15, exactly as the champion level box is — so they are
grown to a fixed stroke height and matched against the glyph set the clock
already bootstrapped. The stroke is fitted rather than guessed: swept from 10 to
16 over 50 frames, mean match score peaks sharply at 13.

The `/` is found by the gaps around it rather than by matching, because a
diagonal stroke correlates with a digit template about as well as a digit does.
Measured across frames the flanking gaps run 5–8px against 1–4px between
digits, with no overlap, and taking the widest pair works whatever the digit
counts — where splitting at a fixed position would not.

**A glyph is accepted on its margin, not its score.** Rescaled to 13px a `9`
and a `4` correlate almost identically: on the frame that exposed this the true
`9` scored 0.56 and the `4` it was read as scored 0.54, so no floor on score
separates them. Their margins over the runner-up do — 0.045 against 0.008 — and
a glyph that cannot clear the margin sinks the whole line rather than being
guessed at. A number with one digit quietly wrong still looks like a number, and
nothing downstream can catch it.

Checking it needs no labels, the same way the clock's check does. Two free
signals: the maximum only moves when the champion levels or buys an item, and a
current value that leaves and returns within a frame is impossible — mana falls
in steps and recovers by slow regeneration, never both at once. Measured over
the whole clip:

| | Mana | Health |
|---|---|---|
| Frames read | 93.3% | 80.4% |
| Maximum stepping backwards | **0** | 0 |
| Value leaving and returning | **0** | 1 (0.04%) |

Before the margin gate the mana line read on 100% of frames and spiked four
times; the gate trades 6.7% of readings for none. That is the right trade here,
because the phantom casts those four spikes produced cost far more than a
missing frame does. The four mana maxima reported across the clip are 452, 488,
525 and 565 — Zilean's pool at successive levels, in order.

None of this generalises to an enemy. The client never draws their numbers, and
this is the local player only.

Three things about the bars themselves had to be got right, and each was wrong
first:

- **Ticks break the health bar** into a dozen components, so fills are measured
  by walking from the left edge and hopping short gaps.
- **A split resource run spawns a phantom plate** measured from the wrong left
  edge, which linked across frames reads as a large sudden drop. This accounted
  for *essentially every* false cast above 4%. The fix is structural, not a
  threshold: a real bar start has a level box beside it, 81–89% dark against 24%
  partway along a lit bar.
- **Truncated bars read plausibly.** A plate behind another is cut where the
  front one begins; a plate at the screen edge is cut by the frame. The latter
  truncates *both* fills at the same column, so they come back equal — 0.222 and
  0.222 reads as a wounded champion low on mana. Both cases now report null and
  say which happened.

Levels come from the same plate, matched against the glyph set the clock reader
already learned, so no new font asset is needed. Rescaling 7×10 digits against
9×13 templates costs accuracy: about one reading in ten comes back `1` for a
champion on 3, 4 or 5, and those are full height rather than clipped, so no size
check removes them. What removes them is the constraint — a level starts at 1,
never falls, and rises by one. Over 516 transitions that took readings which
decrease or skip from **20.9% to zero**.

Matching a plate to a track is geometric and only approximately so. Scaling
screen position into the viewport rectangle leaves a p90 error of 29 minimap
pixels in a viewport ~74 wide, with the residual correlating **−0.69** with
screen height — that is the camera tilt. Fitted cross terms cut p90 to 12px,
held out at a median of 7.2px. Not enough to assign by geometry alone when two
enemies stand close, but enough to gate, and the track lineage carries the
identity. Against a proximity linker this collapsed 20 fragmented series into 3
coherent ones.

### Known limits

- **Identification is not real-time.** Roughly 17 fps of processing at 10 Hz
  sampling, dominated by the gallery pass. Fine for offline VOD analysis; not
  yet fast enough to run live alongside a game.
- **The screen-to-minimap projection is fitted, not derived.** It absorbs the
  camera tilt, the plate's float above the model, and the viewport rectangle's
  drawing conventions all at once. The coefficients wobble between fits, and the
  fit only ever saw plates in the upper half of the screen, so the vertical
  extrapolation below that is unvalidated. A projection derived from the camera
  angle would be sturdier.
- **A champion below about a tenth of their resource is not seen at all.** The
  reader anchors on the resource bar, and at zero there is no bar to anchor on —
  which is exactly the champion who just spent everything.
- **Energy, rage and manaless champions yield no cast evidence** by this route,
  and shields are invisible to it: a shield renders as a violet segment appended
  to the health bar, so gaining one shows no change.
- **Cast attribution inherits the gallery's coverage.** On the sample clip only
  two of five enemies were ever named, so a third track's casts are recorded
  against a track id and no champion.
- **Plate-to-track association is still the dominant source of false casts**,
  and only its blatant form is handled: when the series jumps to a champion
  whose bars sit at a *similar* level the co-movement test has nothing to catch.
  Note this is about *other* tracks — checked against printed mana, the plate on
  the player's own track is right, agreeing to a median of one pixel.
- **Recall rests on eight events and is not a rate.** Both halves of the score
  can be measured, but only for the local player and only where they actually
  cast. More footage is the only thing that turns 7/8 into a number worth
  quoting, and it has to be footage of somebody using their abilities.
- **A cast that spans a gap is located, not timed.** One of the six on the new
  clip is pinned only to a 9.1-second window. That is honest rather than wrong,
  but a consumer that treats `at` as the moment of the cast will be seconds out
  — `span` is not decoration.
- **Ability naming is closer than it was.** Five of the six casts on the new
  clip cost 59-60 mana of the same pool, which is one ability used five times,
  and the sixth cost 48. The ratio between two such clusters is independent of
  the pool size, so matching ratios against published costs would name abilities
  without ever needing a denominator. What blocks it now is that a gap-spanning
  cast understates its own size, so the clusters smear — the clean readings are
  the continuous ones, and there are three of those.
- **A truncated plate is found by the detector, not by the reader.** The reader
  nulls fills for a bar clipped at the frame edge or under a HUD panel, but a bar
  cut by a champion model in the middle of the world is not flagged — the
  detector notices the two fills agreeing and skips the reading. That leaves
  `health` on such a row still carrying the truncated number for anyone else who
  reads it.
- **The player's HUD health line reads on 80% of frames against mana's 93%.**
  It changes constantly and shares its strip with the regeneration figure, so
  more of its glyphs fail the margin gate. Nothing filters what survives; the
  constraint that would — a maximum that only rises — is the one `LevelFilter`
  already uses.
- **The HUD numbers are the local player's only.** The client never draws an
  enemy's, so nothing about this route generalises, and every other champion's
  resource remains a fraction with no denominator and no ground truth.
- **All footage so far is against bots.** The coverage figures in particular
  should be expected to move on a real game.
- **Enemy coverage is bounded by vision, not by the tracker.** All five enemies
  are known in 36% of frames; the rest of the time some have simply never been
  seen recently enough to place.
- **The world bounds are taken on faith.** The Summoner's Rift extent is the
  figure in wide community use, not one Riot publishes. It is a single scalar
  applied uniformly, so nothing structural depends on it, but the speed check
  only pins it to about ten percent.
- **Both new calibrations are per resolution and hand-made.** The digit
  templates and the map rectangle each need one pass on new footage at a new
  resolution. A map skin that redraws the panel border would move the rectangle;
  a client UI update would move the clock.
- **Nothing handles the absence of a clock.** Loading screens and the
  pre-game lobby have no timer, and the reader simply returns nothing there
  rather than knowing why.
- **A teammate's death cannot be attributed to a champion.** The HUD says
  someone is down and the count is reliable; naming them needs a slot-to-champion
  mapping that does not exist yet. Matching each portrait against just the five
  locked roster icons is the obvious route — a five-way closed-set assignment
  rather than the 173-way one, with the local player already pinned by the
  viewport and skinned slots recoverable by elimination — and it would close the
  last 2.5% of ally rows. That is the next thing to fix.
- **Health is read as alive or dead, not as a number.** The bars carry an exact
  figure and the local player's even carries it as text, but nothing reads it,
  so "low and retreating" is not expressible.
- **Nothing reads the respawn countdown**, which is drawn on every dead
  portrait and would say when a champion is coming back rather than only that
  they are gone.

### Automatic ground truth

Hand-labelling was mostly avoidable. Living allies are always drawn on the
minimap, and the HUD health bars show that no ally dies anywhere in the sample
clip — so the expected blue marker count is exactly 5 for every frame outside
the two stretches where the local player is dead (visible as the self portrait
desaturating from V≈76 to V≈30). That gives a per-frame recall target across
thousands of frames for free, which is what stage 1 was retuned against.

It measures recall, not precision — a false positive and a miss cancel out — so
it is a tuning signal, not a substitute for labelled positions.

## Input assumptions

The target input is a **VOD from a player's perspective** — not a Riot `.rofl`
replay, and not a spectator feed. This is a fixed property of the problem, not a
limitation of the sample footage, and it drives the design:

- **Fog of war is permanent.** Enemy champions are only on the minimap while your
  team has vision of them. Measured average is 6.4 markers per frame against a
  cap of 10.
- **Allies are not subject to it.** Your own team is always drawn on the minimap
  regardless of vision, so the five ally identities are continuously observable.
  An ally leaving the minimap means death, not fog.
- **Nothing may assume ten visible identities.** Tracking needs track birth and
  death plus re-identification on reappearance, not a fixed assignment over a
  closed set.

### The input is a window, not a file

This is a real-time review tool. It captures the VOD viewer's window while the
VOD plays, rather than reading a recording of it — `tools/watch.py --window
kilrogg`. The clips in `data/` are the development path and nothing more: a
detector is only tunable when its input is byte-identical across runs, which is
worth keeping for that alone.

They are also, literally, recordings of the same window. Every clip is
2118x1354 with a "kilrogg" title bar in the top-left corner, so every
calibration in `etc/` was derived from receiver pixels well before anything read
them live. The move to window capture changes where the pixels come from and
nothing about what they contain.

Two consequences follow, and both are load-bearing:

- **The window's size is part of the calibration.** Every rectangle in `etc/` —
  minimap region, clock box, portrait centres, plate geometry — is filed under
  one exact frame size. The receiver fills its window by stretching, without
  preserving aspect (`DXGI_SCALING_STRETCH`, in the receiver's presenter), which
  is why the calibrated minimap panel is 325x322 rather than square. A resize
  therefore does not merely move those rectangles, it reshapes the picture
  underneath them — and nothing downstream could notice, since the pipeline would go on
  emitting confident observations of whatever now sits where the minimap was.
  So a mid-session resize is a hard stop, not a warning.
- **Frames arrive whether or not anyone is ready for them.** A file waits for
  its reader; a window does not. Frames the pipeline cannot keep up with are
  dropped on arrival rather than queued, because a queued frame describes a
  fight that has already resolved. The drop count is reported at the end of a
  run, and it is the number to watch if the overlay looks like it is lagging.

Time only moves forward. There is no seeking, so the tracker, the roster lock
and the accumulated self-champion evidence all keep the monotonic input they
assume.

One quirk worth knowing, because it looks like a hang: Graphics Capture delivers
a frame when the window **redraws**, so a picture that is not moving produces
nothing at all. That makes no-frames-at-startup and no-frames-mid-session
different failures — the first means the wrong window was matched, the second
means the feed stalled. Only the first gets a deadline.

## Setup

Python 3.14.6 works; every dependency has a 3.14 wheel.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[capture,dev]"
```

## How stage 1 works

A champion marker is a portrait disc inside a saturated team-coloured ring. The
detector proposes circles from the image **gradient**, then confirms each one by
testing the colour of its annulus.

Two more obvious approaches were built first and both failed on real footage.
They are recorded here because they constrain anything built later.

**Match the ring as a coloured shape.** Ring brightness varies a lot around the
circumference. Any value floor high enough to exclude the background also chops
the ring into disconnected arcs.

**Find the hole the ring encloses.** This handles a champion standing in its own
base, where the ring merges into team-coloured shading and has no outer boundary
to trace. But it dies on a different case: *champion portrait art frequently
contains the team hue*. A red champion with warm art fills the red mask solid, so
there is no hole to find. Measured on a real frame, two of seven markers had a
flawless ring — colour fill 0.96 and 1.00 — and were missed anyway.

There is no threshold that fixes this. Raise the value floor and the ring breaks
into arcs; lower it and the portrait fills the core. Both ends of the knob
destroy the topology the approach depends on.

What *is* invariant is the **circular edge** at the ring. It exists whether or not
the interior matches the ring colour, and whether or not the surroundings do. So
Hough proposes circles over the gradient, and a colour test on the annulus
confirms them and assigns a team. Hough tolerates broken arcs — a partial ring
still votes for the same centre — which neutralises the first failure, and it
never looks inside the disc, which neutralises the second.

Two details are load-bearing:

- **Proposals are pooled from luminance and saturation.** Luminance alone misses
  a marker whose portrait matches the brightness of the shading it stands on.
  The ring is still far more saturated than both, so saturation recovers it.
- **Hough's radius is refit per candidate** by maximising ring colour fill.
  Hough is only accurate to a couple of pixels, and that was enough to score a
  real marker at 0.43 against a 0.45 threshold because it reported r=11.8 where
  the ring sat at r=14.0.

### Measured on real footage

Against hand-marked ground truth on a 2118×1354 capture: **7/7 markers, 0 false
positives.** Fitted ring colour fill per marker ran 0.73–1.00.

Two things about the colour bands were not what they looked like:

- The enemy ring is **magenta (H≈168–178)**, not red. A naive H 0–10 band mostly
  catches terrain.
- Ring brightness drops to **V≈70**, so the value floor has to sit far lower than
  seems reasonable.

### Known limits

- **Fitted to one clip.** The bands come from a single recording. The client's
  colour-blind setting shifts them, and they have not been checked against map
  skins. Champion skins are not a concern here — minimap icons are stock art.
- **Detection is per-frame and memoryless.** Stage 1 reports what is on the
  minimap right now. Persistence across fog — last known position, time since
  seen — is the tracker's job, not this stage's.
- **Hough is tuned for recall.** `param2=18` deliberately over-proposes and
  leans on the colour test to filter. Raising it to 30 dropped recall from 7/7
  to 4/7 while only removing candidates the colour test rejects for free.

## Performance

Measured over 321 real frames at a 325px minimap, single-threaded on CPU:

| | Mean | p95 | Max |
|---|---|---|---|
| `detect()` | 8.25 ms | 9.74 ms | 11.23 ms |

Roughly two thirds of that is the two Hough passes; the rest is per-candidate
radius refinement, which is a Python loop and the obvious thing to vectorise if
this ever needs to be faster.

That is half a 60 FPS frame budget, which sounds alarming and is not. Minimap
positions do not need 60 Hz — champions have a movement speed cap, so 10–15 Hz
plus interpolation is plenty, putting this near 12% duty. Different HUD signals
have different natural sample rates and should not share one cadence.

Capture is the cost worth watching instead. `MonitorSource` downloads the full
monitor surface out of VRAM each frame (~6 MB at 1080p, ~25 MB at 4K). If that
shows up in a profile, the fix is to crop on the GPU before the download, not to
rewrite the pipeline in another language.

## Usage

One-time setup — fetch the champion icons:

```bash
.venv/Scripts/python tools/fetch_icons.py
```

The minimap region also has to be calibrated per (resolution, minimap-scale
slider), since the panel size is not derivable from resolution alone. There is
nothing to do about that: `watch.py` handles it the first time it sees a size it
has no calibration for, by **recognising the map**, and then carries straight on
into the session.

Summoner's Rift is the same picture in every game, whatever size and shape it is
drawn at, so the panel can be found the way any known picture is — correlate a
reference against the frame across sizes and aspects and take the peak. Measured
over 80 frames spanning four clips, three of which the reference was not built
from: **every corner within one pixel** of a hand-drawn calibration.

Accuracy is not why this is trustworthy, though. A region that is merely close
is not a worse read but a confident read of the wrong pixels, and nothing
downstream could notice. What makes it safe is the *separation*: where the panel
is absent, or the window is shaped so oddly the search cannot express it,
correlation falls to 0.00–0.74, against 0.825 at worst for a true find. Nothing
has been observed in between. So it declines rather than guessing, and a decline
falls back to dragging a box by hand.

`--no-calibrate` turns the whole thing back into a hard failure, for runs with
nobody watching. To override the automatic answer, or for a second minimap scale
via `--profile`, the manual tool is still there — and like every tool here it
takes `window:name` as a source, so none of them need a screenshot on disk:

```bash
.venv/Scripts/python tools/calibrate_minimap.py --image window:kilrogg
```

The reference lives in `etc/map/reference.png` and is an average of many frames
of a calibrated panel — terrain and structures sit still and survive, while
champions, wards, pings and fog move and wash out. Rebuild it when the map art
changes, which shows up as `watch.py` starting to ask for a drag it used to skip:

```bash
.venv/Scripts/python tools/build_reference.py --input "data/your clip.mp4"
```

### The other five

Five more calibrations add game time, world coordinates, deaths, nameplates and
the player's own bars. None of them need calibrating either, and for the same
reason the minimap does not: **they are all the same HUD at one scale.** The
receiver stretches a fixed game layout to fill its window, so a frame of any
size is that layout under a scale and a shift, and finding the minimap fixes
every other rectangle with it.

Recovering that transform needs care about which numbers to trust. The
horizontal scale is the frame width over the reference width, exactly — taking
it from the panel's width instead costs a pixel of measurement error on a 325px
panel, and 0.3% over the thousand pixels to the far side of the HUD is a 15px
miss. That is not hypothetical; it is what the first version did. The vertical
scale does come from the panel, because a window's title bar is not game and
does not scale with it, and the vertical offset then absorbs that title bar
without ever measuring it — 31 pixels of chrome, 45, or none all work out.

Worst error over every HUD element at seven window sizes and three chrome
heights: 3.4 pixels, and that at the far corner. Deaths agreed with the native
calibration on every frame sampled at every size; the world transform landed
within 0.3%.

The clock is the exception, and it is the one that guards the rest. It has a
legibility floor as well as a position — shrink the window enough and the timer
is a smudge no calibration can read — so the derived reader is tried before it
is kept, on several frames rather than the one it came from. If it fails, the
run says so and continues without game time. That same check is what would catch
the assumption underneath all of this being wrong: if the streamed game's own
HUD scale changed, every derived rectangle would be misplaced by an amount no
fit could detect, and the timer not being where it was predicted is the cheapest
place to notice.

So the whole set is skipped quietly if absent, and derived automatically if it
can be. `--no-calibrate` runs with whatever exists and derives nothing.

Reach for the tools below to redo one by hand, or when the derivation declines.

Teach the clock its digits — drag a rough box around the match timer and it does
the rest, including checking itself against video time:

```bash
.venv/Scripts/python tools/calibrate_clock.py --input "data/your clip.mp4"
```

Mark the rendered map area inside the minimap panel — the terrain square, not
the ornate frame and not the border band inside it:

```bash
.venv/Scripts/python tools/calibrate_world.py --input "data/your clip.mp4"
```

```bash
.venv/Scripts/python tools/calibrate_world.py --input "data/your clip.mp4" --validate
```

Mark the friendly HUD portraits, which is what makes death readable — three
boxes: your leftmost teammate, your rightmost teammate, and yourself:

```bash
.venv/Scripts/python tools/calibrate_hud.py --input "data/your clip.mp4"
```

```bash
.venv/Scripts/python tools/calibrate_hud.py --input "data/your clip.mp4" --validate
```

That check is worth running. A box a few pixels off still reads a portrait and
still produces a baseline, so a bad calibration does not announce itself — it
just quietly stops noticing deaths.

Then watch the whole pipeline run — live, against the receiver as it plays:

```bash
.venv/Scripts/python tools/watch.py --window kilrogg
```

`--window` matches any window whose title contains the string, so the receiver
is found whatever else it has put in its title bar. `--fps` sets the rate to ask
the window for, defaulting to 10 to match the offline `--stride 3`. Ctrl+C ends
the session and closes the timeline properly rather than leaving half a file.

The startup line names whichever calibrations are missing and prints the command
for each, for the case where derivation declined and you want to supply one by
hand. All of them take `window:kilrogg` too, so none of this needs footage.

One catch on a live source: the passes that *sample* rather than ask — the
nameplate `--fit`, and the `--validate` reports — walk until the source ends,
and a window never does. They take `--limit N` for that. It defaults to 0,
meaning walk the whole thing, which is still right for a clip.

`tools/grab.py` saves a still from a window if you want one to keep, to look at,
or to hand to something else; nothing in this workflow needs it.

Or against a recorded clip, which behaves identically in every respect except
that it waits for the pipeline instead of dropping frames past it:

```bash
.venv/Scripts/python tools/watch.py --input "data/your clip.mp4"
```

Solid circles are champions currently visible; hollow dimmed circles are
champions in fog, drawn at their last known position with the seconds since they
were seen. A champion the HUD confirms is dead is crossed out rather than
dimmed. A white outer ring marks the local player. A **yellow ring** marks a
champion who has just cast, fading over two seconds — thick when the cast was
pinned to consecutive readings, thin when it was measured across a gap and so
happened somewhere in a window rather than at an instant. Q quits, SPACE pauses.

Useful flags: `--save out.mp4` to write the annotated video, and `--quiet` to
print the tracked roster per frame instead of opening a window. `--start N` to
skip into the clip and `--stride 1` to process every frame instead of 10 Hz are
`--input` only — a live window has no past to seek into, and no frames to skip
that it did not already drop.

Or extract the clip to a timeline once and ask questions of the file afterwards:

```bash
.venv/Scripts/python tools/watch.py --input "data/your clip.mp4" --quiet --export clip.jsonl
```

```python
from spectral_sight.export import read_timeline

meta, rows = read_timeline("clip.jsonl")
seen = [r for r in rows if r.champion == "Xerath" and r.visible]
print(f"{seen[0].game_time}s at {seen[0].world_x:.0f}, {seen[0].world_y:.0f}")

# An ally off the minimap who is not dead: the tracker lost them, and knowing
# that is the difference between a kill and a blink.
lost = [r for r in rows if not r.visible and r.alive is True]
```

`iter_timeline` streams the same rows without holding them all, which is what a
full match wants.

Casts are already on those rows, but the report is what says whether to believe
them:

```bash
.venv/Scripts/python tools/detect_casts.py --timeline clip.jsonl --list
```

It prints the drop sizes per champion — the clustering is the whole claim — how
much of the evidence survived a continuous follow-up, which champions were seen
but never cast, and how many casts landed on a champion the HUD said was dead,
which should be zero. `--min-drop` and `--continuity` re-derive from the raw
resource series in the file, so retuning costs a read rather than another run of
the vision.

To score it rather than describe it, against the player's own printed mana:

```bash
.venv/Scripts/python tools/validate_casts.py --input "data/your clip.mp4" --timeline clip.jsonl
```

This is the only check here with both halves. It reports precision and recall,
and for every missed cast it prints what the plate held at that moment — which
is how the misses were traced to plate association rather than to the
threshold.

For working on stage 1 specifically:

```bash
.venv/Scripts/python tools/detect_blips.py --input data/clip.mp4 --masks
```

`--masks` shows the colour masks beside the detections, which is how the HSV
bands were fitted. `--benchmark N` times the detector instead of displaying it.

## Layout

```
src/spectral_sight/
  types.py                    Blip, Frame, Team
  capture/                    frame sources: video, still, live window
  perception/minimap/
    region.py                 where the minimap sits in the frame
    locate.py                 finding that, by recognising the map art
    blips.py                  stage 1 detector
    viewport.py               camera rectangle, and so the local player
    world.py                  minimap pixels to Summoner's Rift units
  perception/identity/        champion gallery and roster locking
  perception/hud/
    portraits.py              where the friendly portraits sit
    alive.py                  who is dead, from portrait desaturation
    clock.py                  the match timer
    resources.py              the player's own health and mana, as numbers
  perception/nameplates/
    plates.py                 health, resource and level over a champion
    levels.py                 level held steady across misreads
    casts.py                  resource steps read as abilities used
    projection.py             screen to minimap, and plate to track
  tracking/                   identity accumulated across frames
  calibration.py              deriving the whole calibration set from one fit
  pipeline.py                 the stages, wired in order
  export.py                   the output: observations, and the timeline file
  debug/overlay.py            visualisation
tools/                        calibration and inspection CLIs
tests/synthetic.py            minimaps with known ground truth
etc/map/                      averaged map art, and the reference layout size
etc/regions/                  calibrated minimap regions per resolution
etc/clock/                    clock position and learned digits per resolution
etc/world/                    map area and world bounds per resolution
etc/hud/                      friendly portrait positions per resolution
etc/nameplates/               bar geometry and screen projection per resolution
etc/resources/                player health and mana text boxes per resolution
```

## Testing

```bash
.venv/Scripts/python -m pytest
```

Synthetic frames cannot tell us whether the colour bands are right, but they can
tell us whether the geometry logic is — and those two failure modes look
identical on real footage. The tests pin the geometry so that anything failing
later is a tuning problem, not a logic one.
