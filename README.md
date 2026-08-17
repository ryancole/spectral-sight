# spectral-sight

Vision-only, real-time game state extraction from League of Legends replays. No
Riot APIs, no process interaction — pixels in, structured state out.

## Status

**Stage 1 — class-agnostic minimap marker detection.** Working. Retuned against
5.3 minutes of footage using an automatic ground truth (see below): finds at
least all five allies in 87% of frames, averaging 5.8 blue markers against 5
expected, so it over-produces slightly by design.

**Stage 2 — champion identification.** Partially working, and the current
bottleneck. Allies are identified from a gallery bootstrapped out of the HUD
itself. Three or more of the five allies are identified in 22% of frames,
typically one or two otherwise. The local player's own marker never matches and
is an open bug. Enemy identification is not started — nothing in the HUD names
them.

**Tracker.** Not built. This is where the remaining identification accuracy
should come from: evidence accumulated per track across frames, rather than a
fresh independent decision every frame.

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
  colour-blind setting shifts them, and they have not been checked across
  champion skins or map skins.
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

Calibrate once per (resolution, minimap scale) — the panel size is driven by an
in-game slider, so it cannot be derived from resolution:

```bash
.venv/Scripts/python tools/calibrate_minimap.py --image data/frame.png
```

Then run the detector:

```bash
.venv/Scripts/python tools/detect_blips.py --input data/clip.mp4 --masks
```

`--masks` shows the colour masks beside the detections, which is how you fit the
HSV bands. `--benchmark N` times the detector instead of displaying it.

## Layout

```
src/spectral_sight/
  types.py                    Blip, Frame, Team
  capture/                    frame sources: video, still, live monitor
  perception/minimap/
    region.py                 where the minimap sits in the frame
    blips.py                  stage 1 detector
  debug/overlay.py            visualisation
tools/                        calibration and inspection CLIs
tests/synthetic.py            minimaps with known ground truth
etc/regions/                  calibrated regions per resolution
```

## Testing

```bash
.venv/Scripts/python -m pytest
```

Synthetic frames cannot tell us whether the colour bands are right, but they can
tell us whether the geometry logic is — and those two failure modes look
identical on real footage. The tests pin the geometry so that anything failing
later is a tuning problem, not a logic one.
