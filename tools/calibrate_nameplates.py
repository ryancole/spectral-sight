"""Measure the champion nameplate, then fit the screen-to-minimap projection.

The bar geometry is fixed for a resolution but not derivable from it, like the
minimap panel and the clock strip, so it is marked once by hand. Find a frame
with an enemy champion on screen and drag a box around the whole plate --
level box, health bar and resource bar together:

    python tools/calibrate_nameplates.py --input "data/your clip.mp4"

The parts are then measured out of that box rather than asked for one at a
time. The resource bar is found as the blue run inside it, the health bar as
the red or green run above, and the level box as the dark plate to their left,
so a generous drag still produces tight numbers.

**The bar must be dragged on a champion, not a minion.** Minions draw a health
bar of a different width and no resource bar at all, and the fill denominator
taken from one would be wrong for every champion in the clip without ever
looking wrong.

That gives positions. Turning a plate into an *identity* also needs the map from
screen to minimap, which is fitted from footage rather than marked:

    python tools/calibrate_nameplates.py --input "data/your clip.mp4" --fit

It collects frames where exactly one enemy plate and exactly one enemy blip are
present -- which makes the pairing unambiguous without assuming the association
it is calibrating -- and least-squares fits the coefficients. A few hundred
such frames is plenty; the sample clip yielded 190 over 162 seconds.

Then check it, which reports coverage and the fit's own error:

    python tools/calibrate_nameplates.py --input "data/your clip.mp4" --validate

Worth running. A plate reader with a slightly wrong `bar_width` still returns
plausible fractions on every frame, so a bad calibration does not announce
itself -- what it does instead is put a fixed percentage error into every fill,
which is invisible until two clips disagree.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from spectral_sight.capture import open_source
from spectral_sight.perception.hud.clock import load_clock_reader
from spectral_sight.perception.minimap import (
    BlipDetector,
    BlipDetectorConfig,
    MinimapRegion,
    find_viewport,
)
from spectral_sight.perception.minimap.blips import scaled_config
from spectral_sight.perception.nameplates import (
    LAYOUT_DIR,
    NameplateConfig,
    NameplateLayout,
    NameplateReader,
    fit,
)
from spectral_sight.types import Team

DEFAULT_EXCLUDE = (
    (0.00, 0.00, 1.00, 0.035),
    (0.00, 0.00, 0.28, 0.36),
    (0.00, 0.78, 1.00, 1.00),
    (0.76, 0.60, 1.00, 1.00),
)
"""Top status bar, target frame and death recap, bottom HUD, minimap. Fractions
of the frame, and the same for every 16:9-ish layout, so they are a default
rather than something to drag."""


def measure(box: np.ndarray, config: NameplateConfig) -> NameplateLayout | None:
    """Pull the plate's geometry out of a dragged box.

    Returns None if the box does not contain the two stacked bars, which is
    what a drag on a minion or on empty ground looks like.
    """
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = config.health_hue
    red = (((h >= lo) & (h <= hi)) | (h <= 3)) & (s > config.min_saturation) & (
        v > config.min_value
    )
    lo, hi = config.ally_hue
    green = (h >= lo) & (h <= hi) & (s > config.ally_min_saturation) & (
        v > config.min_value
    )
    lo, hi = config.resource_hue
    blue = (h >= lo) & (h <= hi) & (s > config.resource_min_saturation) & (
        v > config.resource_min_value
    )

    resource_rows = np.nonzero(blue.any(axis=1))[0]
    health = red | green
    health_rows = np.nonzero(health.any(axis=1))[0]
    if resource_rows.size == 0 or health_rows.size == 0:
        return None
    if resource_rows[0] <= health_rows[0]:
        return None

    health_top = int(health_rows[0])
    health_bottom = int(health_rows[-1])
    resource_top = int(resource_rows[0])

    columns = np.nonzero((health | blue).any(axis=0))[0]
    left = int(columns[0])
    # The bar is as wide as its widest run, which is the resource bar whenever
    # the champion is nearer to full mana than to full health, and the health
    # bar otherwise. Taking the max is what makes a drag on a damaged champion
    # produce the same width as a drag on a healthy one.
    width = max(
        int(np.count_nonzero(health.any(axis=0))),
        int(np.count_nonzero(blue.any(axis=0))),
    )

    dy = resource_top - health_top
    resource_height = max(int(resource_rows[-1] - resource_rows[0] + 1), 1)

    # The level box is the dark plate immediately left of the bars. Measured
    # rather than assumed, because it is what `_anchored` samples to decide
    # whether a resource run is a real bar start -- a window that drifts off it
    # reads whatever is behind the plate and the test stops discriminating.
    band = slice(health_top, resource_top + resource_height)
    dark = (v[band, :left] < 70).mean(axis=0) >= 0.5 if left > 0 else np.zeros(0)
    box_left = left
    for column in range(left - 1, -1, -1):
        if column >= dark.size or not dark[column]:
            break
        box_left = column
    if left - box_left < 6:
        # No dark plate found: fall back to the width it has on known footage
        # rather than emitting a window of nothing.
        box_left = max(left - 25, 0)

    return NameplateLayout(
        bar_width=width,
        bar_height=max(health_bottom - health_top + 1, 1),
        resource_dy=(max(dy - 3, 1), dy + 3),
        level_dx=(-(left - box_left) - 3, -3),
        level_dy=(-(dy + 5), resource_height + 1),
        exclude=DEFAULT_EXCLUDE,
    )


def fit_projection(
    path: str, layout: NameplateLayout, stride: int
) -> NameplateLayout | None:
    """Fit the screen-to-minimap coefficients from unambiguous frames."""
    reader = NameplateReader(layout)
    region: MinimapRegion | None = None
    detector: BlipDetector | None = None
    samples: list[tuple[float, float, float, float]] = []

    with open_source(path, stride=stride) as source:
        for frame in source.frames():
            width, height = frame.size
            if region is None:
                region = MinimapRegion.for_resolution(width, height)
                detector = BlipDetector(
                    scaled_config(BlipDetectorConfig(), minimap_width=region.width)
                )
            minimap = region.crop(frame.image)
            viewport = find_viewport(minimap)
            if viewport is None:
                continue
            inside = [
                b for b in detector.detect(minimap)
                if b.team is Team.RED
                and viewport.x <= b.x <= viewport.x + viewport.width
                and viewport.y <= b.y <= viewport.y + viewport.height
            ]
            plates = [
                p for p in reader.read(frame.image) if p.hostile and not p.occluded
            ]
            if len(inside) != 1 or len(plates) != 1:
                continue
            cx, _ = plates[0].center
            samples.append((
                cx / width,
                plates[0].y / height,
                (inside[0].x - viewport.x) / viewport.width,
                (inside[0].y - viewport.y) / viewport.height,
            ))

    print(f"{len(samples)} unambiguous frames")
    if len(samples) < 8:
        print("not enough to fit; try a clip with more enemy contact",
              file=sys.stderr)
        return None

    coef_x, coef_y = fit(samples)
    fitted = NameplateLayout(
        bar_width=layout.bar_width,
        bar_height=layout.bar_height,
        resource_dy=layout.resource_dy,
        level_dx=layout.level_dx,
        level_dy=layout.level_dy,
        exclude=layout.exclude,
        projection_x=coef_x,
        projection_y=coef_y,
    )
    _report_fit(samples, fitted)
    return fitted


def _report_fit(samples, layout: NameplateLayout) -> None:
    """Residuals of the fit, in viewport fractions and in minimap pixels."""
    data = np.asarray(samples, dtype=float)
    design = np.c_[data[:, 0], data[:, 1], np.ones(len(data))]
    ex = design @ np.asarray(layout.projection_x) - data[:, 2]
    ey = design @ np.asarray(layout.projection_y) - data[:, 3]
    # A viewport measures roughly 74 x 48 minimap pixels on the sample capture;
    # reporting in pixels is what makes the number comparable to the gate.
    error = np.hypot(ex * 74.0, ey * 48.0)
    print(f"  x coefficients: {layout.projection_x[0]:+.3f} "
          f"{layout.projection_x[1]:+.3f} {layout.projection_x[2]:+.3f}")
    print(f"  y coefficients: {layout.projection_y[0]:+.3f} "
          f"{layout.projection_y[1]:+.3f} {layout.projection_y[2]:+.3f}")
    print(f"  residual: median {np.median(error):.1f}px  "
          f"p90 {np.percentile(error, 90):.1f}px  max {error.max():.1f}px")
    if np.percentile(error, 90) > 20:
        print("  p90 above 20px: association will be unreliable in fights",
              file=sys.stderr)


def validate(path: str, layout: NameplateLayout, stride: int, size) -> int:
    """Run the reader over a clip and report what it finds."""
    try:
        glyphs = load_clock_reader(*size).glyphs
    except FileNotFoundError:
        glyphs = None
        print("no clock calibration, so levels will not be read")

    reader = NameplateReader(layout, glyphs)
    frames = with_plate = readings = hostile = occluded = levelled = 0
    resources = []

    with open_source(path, stride=stride) as source:
        for frame in source.frames():
            frames += 1
            plates = reader.read(frame.image)
            with_plate += bool(plates)
            for plate in plates:
                readings += 1
                hostile += plate.hostile
                occluded += plate.occluded
                levelled += plate.level is not None
                if plate.hostile and plate.resource is not None:
                    resources.append(plate.resource)

    if frames == 0:
        print(f"no frames in {path}", file=sys.stderr)
        return 1

    print(f"\n{frames} frames at stride {stride}\n")
    print(f"  frames with a nameplate : {with_plate / frames:.1%}")
    print(f"  plate readings          : {readings}")
    print(f"  of those, hostile       : {hostile} ({hostile / frames:.2f}/frame)")
    print(f"  blanked by occlusion    : {occluded}")
    if glyphs is not None:
        print(f"  levels read             : {levelled} "
              f"({levelled / max(readings, 1):.0%} of readings)")
    if resources:
        full = sum(1 for r in resources if r >= 0.99) / len(resources)
        print(f"  resource reads at 100%  : {full:.1%}")
        print("     If a champion sat at full mana and this is near zero, "
              "bar_width is too large.")
    if with_plate == 0:
        print("\nNo nameplates at all. The box was probably dragged on a "
              "minion, or on the wrong resolution.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="clip or screenshot")
    parser.add_argument("--fit", action="store_true",
                        help="fit the screen-to-minimap projection from footage")
    parser.add_argument("--validate", action="store_true",
                        help="check the saved layout against this clip")
    parser.add_argument("--stride", type=int, default=3,
                        help="frames per sample when fitting or validating")
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()

    with open_source(args.input) as source:
        frame = next(iter(source.frames()), None)
    if frame is None:
        print(f"no frames in {args.input}", file=sys.stderr)
        return 1
    width, height = frame.size
    path = args.out or LAYOUT_DIR / f"{width}x{height}.json"

    if args.fit or args.validate:
        try:
            layout = NameplateLayout.load(path)
        except FileNotFoundError:
            print(f"no layout at {path}; run without --fit/--validate first",
                  file=sys.stderr)
            return 1
        if args.validate:
            return validate(args.input, layout, args.stride, (width, height))
        fitted = fit_projection(args.input, layout, args.stride)
        if fitted is None:
            return 1
        fitted.save(path)
        print(f"saved -> {path}")
        return 0

    print(f"frame is {width}x{height}.")
    print("Drag a box around one ENEMY CHAMPION's whole nameplate -- level box, "
          "health bar and resource bar. ENTER to accept.")
    box = cv2.selectROI("calibrate nameplates", frame.image, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in box)
    if w == 0 or h == 0:
        print("cancelled", file=sys.stderr)
        return 1

    layout = measure(frame.image[y : y + h, x : x + w], NameplateConfig())
    if layout is None:
        print("that box has no resource bar under a health bar. Drag a "
              "champion's plate, not a minion's.", file=sys.stderr)
        return 1

    layout.save(path)
    print(f"saved -> {path}")
    print(f"  bar {layout.bar_width}x{layout.bar_height}px, resource "
          f"{layout.resource_dy[0]}-{layout.resource_dy[1]}px below the health bar")
    print(f"\nNow fit the projection, then check it:\n"
          f"  python tools/calibrate_nameplates.py --input \"{args.input}\" --fit\n"
          f"  python tools/calibrate_nameplates.py --input \"{args.input}\" "
          f"--validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
