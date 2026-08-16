# spectral-sight

Vision-only, real-time game state extraction from League of Legends replays. No
Riot APIs, no process interaction — pixels in, structured state out.

## Status

**Stage 1 — class-agnostic minimap marker detection.** Done, verified against
synthetic ground truth, unfitted to real footage.

Stage 2 (champion identification by gallery matching) and the tracker are not
built yet.

## Setup

Python 3.14.6 works; every dependency has a 3.14 wheel.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[capture,dev]"
```

## How stage 1 works

The minimap draws several things in team colours: champion markers, turret and
inhibitor glyphs, base and turret-range shading, minion dots. Only one of them
is a *portrait core enclosed by a team-coloured ring*.

The obvious implementation — find team-coloured rings — does not survive contact
with the game. A champion standing in its own base has its ring merged into the
surrounding shading, so the outer boundary of that blob is the base, not the
marker. Champions are in their base constantly; this is not an edge case. An
early version of the detector found 7 of 10 champions for exactly this reason.

So the detector inverts the problem and looks for **holes**. Build the team
colour mask, then find enclosed regions of non-team colour. A champion punches
one out of whatever it is standing on, because the portrait interrupts the
colour and the ring closes around it. A solid glyph never does.

That inversion turns the hardest case into the easy one and collapses the
discriminating filters down to two: is the hole the right size, and is it round.

It also lines up with what comes next — the hole *is* the portrait crop that
stage 2 needs to match against the champion gallery.

### Known limits

- **Colour bands are unfitted.** The HSV windows in `blips.py` are reasoned
  starting values, not measured ones. They need to be fitted against real
  frames, and the client's colour-blind setting shifts them.
- **A broken ring encloses nothing.** Compression artifacts on recorded footage
  can open a gap in a thin ring, which costs a detection outright. `close_kernel`
  is the knob for this and should be the first thing widened if recall drops on
  real video.
- **Team-coloured overlays can enclose non-coloured regions.** A jungle camp icon
  inside base shading is a circular-ish hole. The size and roundness filters
  catch most; stage 2's gallery match is the real backstop.

## Performance

Measured on synthetic minimaps, single-threaded on CPU:

| Minimap size | Mean | p95 | Throughput |
|---|---|---|---|
| 280×280 | 0.59 ms | 0.73 ms | ~1,700/sec |
| 560×560 | 1.78 ms | 2.19 ms | ~560/sec |

Roughly 3.5% of a 60 FPS frame budget at typical minimap scale. Detection is not
where the time will go — capture is. `MonitorSource` currently downloads the
full monitor surface out of VRAM each frame (~6 MB at 1080p, ~25 MB at 4K). If
that ever shows up in a profile, the fix is to crop on the GPU before the
download, not to rewrite the pipeline in another language.

Note also that minimap positions do not need 60 Hz. Champions have a movement
speed cap, so 10–15 Hz plus interpolation is plenty; different HUD signals have
different natural sample rates and should not share one cadence.

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
