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

### Known limits

- **Identification is not real-time.** Roughly 17 fps of processing at 10 Hz
  sampling, dominated by the gallery pass. Fine for offline VOD analysis; not
  yet fast enough to run live alongside a game.
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
- **A timeline records `visible`, not death.** For an ally, who is never fogged,
  invisibility means death — but nothing infers that yet, so a dead ally reads
  as a champion nobody has seen for 24 seconds. The HUD portraits already carry
  health and are not wired into the pipeline; that is the next thing to fix.

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

The target input is a **screen-recorded VOD from a player's perspective** — not
a Riot `.rofl` replay, and not a spectator feed. This is a fixed property of the
problem, not a limitation of the sample footage, and it drives the design:

- **Fog of war is permanent.** Enemy champions are only on the minimap while your
  team has vision of them. Measured average is 6.4 markers per frame against a
  cap of 10.
- **Allies are not subject to it.** Your own team is always drawn on the minimap
  regardless of vision, so the five ally identities are continuously observable.
  An ally leaving the minimap means death, not fog.
- **Nothing may assume ten visible identities.** Tracking needs track birth and
  death plus re-identification on reappearance, not a fixed assignment over a
  closed set.

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

One-time setup — fetch the champion icons, and calibrate the minimap region for
your resolution and minimap-scale slider (the panel size is not derivable from
resolution alone):

```bash
.venv/Scripts/python tools/fetch_icons.py
```

```bash
.venv/Scripts/python tools/calibrate_minimap.py --image data/frame.png
```

Two optional calibrations add game time and world coordinates. Both are skipped
quietly if absent, so everything above works without them.

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

Then watch the whole pipeline run on a clip:

```bash
.venv/Scripts/python tools/watch.py --input "data/your clip.mp4"
```

Solid circles are champions currently visible; hollow dimmed circles are
champions in fog, drawn at their last known position with the seconds since they
were seen. A white outer ring marks the local player. Q quits, SPACE pauses.

Useful flags: `--start N` to skip into the clip, `--stride 1` to process every
frame instead of 10 Hz, `--save out.mp4` to write the annotated video, and
`--quiet` to print the tracked roster per frame instead of opening a window.

Or extract the clip to a timeline once and ask questions of the file afterwards:

```bash
.venv/Scripts/python tools/watch.py --input "data/your clip.mp4" --quiet --export clip.jsonl
```

```python
from spectral_sight.export import read_timeline

meta, rows = read_timeline("clip.jsonl")
seen = [r for r in rows if r.champion == "Xerath" and r.visible]
print(f"{seen[0].game_time}s at {seen[0].world_x:.0f}, {seen[0].world_y:.0f}")
```

`iter_timeline` streams the same rows without holding them all, which is what a
full match wants.

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
  capture/                    frame sources: video, still, live monitor
  perception/minimap/
    region.py                 where the minimap sits in the frame
    blips.py                  stage 1 detector
    viewport.py               camera rectangle, and so the local player
    world.py                  minimap pixels to Summoner's Rift units
  perception/identity/        champion gallery and roster locking
  perception/hud/
    portraits.py              teammate portraits, health and level
    clock.py                  the match timer
  tracking/                   identity accumulated across frames
  pipeline.py                 the stages, wired in order
  export.py                   the output: observations, and the timeline file
  debug/overlay.py            visualisation
tools/                        calibration and inspection CLIs
tests/synthetic.py            minimaps with known ground truth
etc/regions/                  calibrated minimap regions per resolution
etc/clock/                    clock position and learned digits per resolution
etc/world/                    map area and world bounds per resolution
```

## Testing

```bash
.venv/Scripts/python -m pytest
```

Synthetic frames cannot tell us whether the colour bands are right, but they can
tell us whether the geometry logic is — and those two failure modes look
identical on real footage. The tests pin the geometry so that anything failing
later is a tuning problem, not a logic one.
