"""Mark the rendered map area, and check the resulting scale against physics.

    # drag the terrain square inside the minimap panel
    python tools/calibrate_world.py --input "data/my clip.mp4"

    # check what that scale implies about how fast champions move
    python tools/calibrate_world.py --input clip.mp4 --validate

    # and compare against pretending the whole crop is the map
    python tools/calibrate_world.py --input clip.mp4 --validate --assume-crop

The drag is shown zoomed, because a few pixels here is a hundred world units.
Drag the *terrain*, not the ornate frame and not the black gutter inside it. The
box is snapped to square by default, since both the map and its render are
square and a hand-drawn box never quite is.

**This is a manual step on purpose.** Three ways to find the rectangle
automatically were tried on the sample capture and all three worked, in the
sense of returning a confident rectangle -- and they disagreed with each other
by about 5%, which is the same size as the error the calibration exists to
remove. They are recorded here so nobody spends the afternoon again:

- *Bounding box of non-black pixels.* Catches the ornate frame, which is neither
  black nor map, and comes back 20px too large.
- *Bounding box of pixels that change over time.* Finds where the game happened,
  not where the map is: it stopped 16px short at the top, where the enemy base
  sits under permanent fog and nothing ever moves, and leaked past the bottom
  edge onto HUD text drawn over the panel.
- *Columns and rows that are uniform along their length.* The panel border is
  uniform and the map is textured, so this looked like the principled one. But
  the map's outer ring is unwalkable black void, which is also uniform, so it
  lands consistently 8-9px inside the true edge on every side.

What the measurement actually rests on is the brightness profile across the
panel edge, averaged along it: frame ornament, then a flat border band, then
map. Reading the band's inner edge off that profile gave 310px on the x axis and
310px on the y axis independently -- nothing forced those to agree, so their
agreeing is real evidence. That is a five-minute measurement done once per
resolution, and it beats a detector that is confidently wrong.

**Why validation is possible at all.** Every other stage of this project could
be checked against something visible in the footage. A coordinate scale cannot:
the pixels look identical whether a map is 15,000 units across or 30,000. What
saves it is that the scale makes a *falsifiable prediction about motion*.
Champions in League move at a known few hundred units per second, so converting
tracked positions into units per second either lands in that band or does not.

**Measure over a full second, not between frames.** At roughly 48 units per
pixel and 10 Hz sampling, one pixel of position jitter reads as 480 units per
second -- more than a champion's actual speed. Frame-to-frame speeds are
therefore mostly noise, and their spread says nothing about the scale. Over a
one-second baseline the real displacement is ten times larger while the jitter
is not, and path curvature has not yet had time to matter much.

Champions stand still a great deal, so the median is near zero and meaningless.
The number to read is the **p90 over a one-second window**: the champions who
spent that second running.

**What this test can and cannot do.** It is a check on the scale, not a way to
find it. Run against the untuned whole-crop transform (`--assume-crop`) the
sample clip reports 377 u/s against the calibrated 394 -- both perfectly
plausible, because the two differ by under 5%. So a passing result means the
world bounds and the overall scale are right to within roughly ten percent. It
rules out a wrong map extent or a badly misplaced rectangle. It does not confirm
the last few pixels of the calibration, and should not be quoted as if it did.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import cv2

from spectral_sight.capture import open_source
from spectral_sight.perception.minimap import MinimapRegion, WorldTransform
from spectral_sight.perception.minimap.world import WORLD_DIR
from spectral_sight.pipeline import Pipeline

WINDOW = 1.0
"""Seconds between the two sightings a speed sample is measured across. Long
enough that pixel jitter stops dominating, short enough that a champion's path
has not curved far from a straight line."""

WINDOW_SLACK = 1.6
"""How much longer than `WINDOW` a gap may be before the sample is dropped. Fog
and missed detections leave holes, and a champion's path over three seconds
bears no relation to a straight line."""

HEADLINE = 90
"""Percentile to judge. The median is dominated by champions standing still, and
the top few percent by dashes, flashes and the odd tracker error. The p90 is the
champions who spent the whole window running."""

PLAUSIBLE = (300.0, 500.0)
"""Where the p90 should land. Base movement speeds run about 325-355, tier-one
boots put a typical mid-game figure near 370, and the alpha-beta filter damps
the peaks a little. See the module docstring: this band is wide enough to accept
a scale that is several percent off, and that is a limit of the method rather
than a threshold worth tightening."""


def _snap_square(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Centre a square of the mean side length on the drawn box."""
    side = round((w + h) / 2)
    return x + (w - side) // 2, y + (h - side) // 2, side, side


def _drag(frame, region: MinimapRegion | None, zoom: float) -> tuple[int, int, int, int] | None:
    """Zoomed drag over the minimap panel, returned in frame coordinates."""
    pad = 20
    if region is not None:
        x0, y0 = max(0, region.x - pad), max(0, region.y - pad)
        x1 = min(frame.shape[1], region.x + region.width + pad)
        y1 = min(frame.shape[0], region.y + region.height + pad)
    else:
        x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]

    view = cv2.resize(frame[y0:y1, x0:x1], None, fx=zoom, fy=zoom,
                      interpolation=cv2.INTER_NEAREST)
    box = cv2.selectROI("calibrate world", view, showCrosshair=False)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v) for v in box)
    if w == 0 or h == 0:
        return None
    return (round(x0 + x / zoom), round(y0 + y / zoom),
            round(w / zoom), round(h / zoom))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = min(len(sorted_values) - 1, int(len(sorted_values) * percentile / 100))
    return sorted_values[index]


def _window_speeds(
    history: dict[int, list[tuple[float, float, float]]], window: float
) -> list[float]:
    """Straight-line speed over `window` seconds, per track, in world units."""
    speeds: list[float] = []
    for samples in history.values():
        later = 0
        for start, (t0, x0, y0) in enumerate(samples):
            later = max(later, start)
            while later < len(samples) and samples[later][0] - t0 < window:
                later += 1
            if later >= len(samples):
                break
            t1, x1, y1 = samples[later]
            dt = t1 - t0
            if dt > window * WINDOW_SLACK:
                continue
            speeds.append(math.hypot(x1 - x0, y1 - y0) / dt)
    return sorted(speeds)


def _validate(source, pipeline: Pipeline, transform: WorldTransform,
              limit: int) -> bool:
    """Convert tracked motion into units per second and see if it is sane."""
    history: dict[int, list[tuple[float, float, float]]] = {}
    xs: list[float] = []
    ys: list[float] = []
    processed = 0

    for frame in source.frames():
        result = pipeline.process(frame.image, frame.timestamp)
        processed += 1
        for track in result.tracks:
            # Only positions actually seen this frame. A track coasting through
            # fog moves at whatever the filter last believed, which would be
            # measuring the tracker rather than the transform.
            if track.age(frame.timestamp) > 1e-6:
                continue
            wx, wy = transform.from_minimap(pipeline.region, track.x, track.y)
            xs.append(wx)
            ys.append(wy)
            history.setdefault(track.id, []).append((frame.timestamp, wx, wy))

        if limit and processed >= limit:
            break

    speeds = _window_speeds(history, WINDOW)
    if not speeds:
        print("no tracked motion to measure", file=sys.stderr)
        return False

    ux, uy = transform.units_per_pixel
    print(f"\nvalidated over {processed} frames, {len(speeds)} speed samples "
          f"across {WINDOW:.1f}s windows")
    print(f"  scale          {ux:.1f} x {uy:.1f} units/px, "
          f"squareness {transform.squareness:.4f}")
    for percentile in (50, 75, 90, 95, 99):
        marker = "  <--" if percentile == HEADLINE else ""
        print(f"  p{percentile:<12} {_percentile(speeds, percentile):7.0f} u/s{marker}")

    bounds = transform.bounds
    outside = sum(
        1 for x, y in zip(xs, ys)
        if not (bounds.min_x <= x <= bounds.max_x and bounds.min_y <= y <= bounds.max_y)
    )
    print(f"  world x span   {min(xs):8.0f} .. {max(xs):8.0f}")
    print(f"  world y span   {min(ys):8.0f} .. {max(ys):8.0f}")
    print(f"  out of bounds  {outside}/{len(xs)} ({outside / len(xs):.2%})")

    headline = _percentile(speeds, HEADLINE)
    low, high = PLAUSIBLE
    ok = low <= headline <= high
    print(f"\n  p{HEADLINE} of {headline:.0f} u/s is "
          f"{'plausible' if ok else 'NOT plausible'} "
          f"(expected {low:.0f}-{high:.0f} for champions running)")
    if not ok:
        factor = headline / statistics.mean(PLAUSIBLE)
        print(f"  the map area looks off by roughly {factor:.2f}x "
              f"({'too small' if factor > 1 else 'too large'})")
    else:
        print("  this rules out a gross scale error; it is not precise enough "
              "to confirm\n  the last few pixels of the map area.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="clip or screenshot")
    parser.add_argument("--area", help="skip the drag: x,y,width,height in frame pixels")
    parser.add_argument("--zoom", type=float, default=3.0, help="drag magnification")
    parser.add_argument("--no-snap", action="store_true",
                        help="keep the drawn box instead of snapping it square")
    parser.add_argument("--validate", action="store_true",
                        help="measure champion speeds under the saved transform")
    parser.add_argument("--assume-crop", action="store_true",
                        help="validate the untuned transform that takes the whole "
                             "minimap crop as the map area")
    parser.add_argument("--icons", help="icon set directory; defaults to newest")
    parser.add_argument("--stride", type=int, default=3,
                        help="process every Nth frame while validating")
    parser.add_argument("--limit", type=int, default=1500,
                        help="stop validating after N processed frames (0 = all)")
    parser.add_argument("--start", type=int, default=0,
                        help="skip to this source frame first")
    args = parser.parse_args()

    with open_source(args.input) as source:
        width, height = source.size
        frame = next(iter(source.frames()), None)
    if frame is None:
        print(f"no frames in {args.input}", file=sys.stderr)
        return 1

    try:
        region = MinimapRegion.for_resolution(width, height)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not args.validate:
        if args.area:
            x, y, w, h = (int(p) for p in args.area.split(","))
        else:
            print(f"frame is {width}x{height}. Drag the terrain square inside the "
                  "minimap panel -- not the ornate frame, not the black gutter.")
            drawn = _drag(frame.image, region, args.zoom)
            if drawn is None:
                print("cancelled", file=sys.stderr)
                return 1
            x, y, w, h = drawn

        if not args.no_snap:
            snapped = _snap_square(x, y, w, h)
            if snapped != (x, y, w, h):
                print(f"snapped {w}x{h} -> {snapped[2]}x{snapped[3]}")
            x, y, w, h = snapped

        transform = WorldTransform(x=float(x), y=float(y),
                                   width=float(w), height=float(h))
        path = WORLD_DIR / f"{width}x{height}.json"
        transform.save(path)
        ux, uy = transform.units_per_pixel
        print(f"map area {w}x{h} at ({x}, {y}) | {ux:.1f} units/px")
        print(f"saved {path}")
        print("\nNow check it: "
              f"python tools/calibrate_world.py --input {args.input!r} --validate")
        return 0

    if args.assume_crop:
        transform = WorldTransform.assuming_crop(region)
        print("validating the untuned whole-crop transform")
    else:
        try:
            transform = WorldTransform.for_resolution(width, height)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

    try:
        icons = Path(args.icons) if args.icons else _newest_icons()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    with open_source(args.input, stride=args.stride, start=args.start) as source:
        pipeline = Pipeline.for_resolution(width, height, icons)
        ok = _validate(source, pipeline, transform, args.limit)
    return 0 if ok else 1


def _newest_icons() -> Path:
    root = Path(__file__).resolve().parents[1] / "etc" / "icons"
    versions = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not versions:
        raise FileNotFoundError(
            f"no icon sets in {root}. Run: python tools/fetch_icons.py"
        )
    return versions[-1]


if __name__ == "__main__":
    raise SystemExit(main())
