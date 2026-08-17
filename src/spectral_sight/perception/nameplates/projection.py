"""Screen positions to minimap positions, so a nameplate can find its track.

A nameplate says what a champion's resource bar is doing; a track says who that
champion is and keeps saying it across frames. Neither is useful alone, and they
live in different coordinate systems -- the plate in screen pixels, the track in
minimap-crop pixels. The viewport rectangle is the only thing relating them: it
is the region of the map currently on screen, drawn on the minimap.

**The obvious linear map is not good enough, and it is worth recording by how
much.** Scaling screen position into the viewport rectangle directly leaves a
p90 error of 29 minimap pixels in a viewport only ~74 wide, and the residual
correlates with screen height at -0.69. That correlation is the camera tilt:
League looks down at an angle rather than straight down, so a champion near the
top of the screen is further away than a flat map believes. Adding cross terms
in both axes cuts p90 to 12px, and holds up on a held-out 40% of the clip at a
median of 7.2px.

**The coefficients are fitted, not derived.** They absorb the tilt, the plate's
float above the champion's model, and whatever the viewport rectangle's own
drawing conventions are, all at once. Two consequences a caller should know.
They wobble between fits -- the x slope came out 0.49 on the whole clip and 0.40
on 60% of it -- so they are an empirical correction rather than a camera model.
And the fit only saw screen heights in the top half, because enemy plates never
appear low on screen, so the vertical extrapolation below that is unvalidated.

**So this gates rather than decides.** 12px in a 74px viewport is not enough to
assign plates by geometry when two enemies stand close. It is enough to rule out
everything across the map and let the track lineage carry the identity, which is
the same division of labour the tracker already uses for blips: geometry
proposes, continuity disposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from spectral_sight.perception.minimap.viewport import Viewport
from spectral_sight.perception.nameplates.plates import Nameplate, NameplateLayout

GATE = 15.0
"""Minimap pixels a plate may sit from a track and still be paired with it. Just
above the fitted map's p90 error, so a correct pairing is rarely gated out while
an assignment across the map stays impossible."""


@dataclass(frozen=True, slots=True)
class ScreenProjection:
    """Maps a screen position to a position inside the viewport rectangle.

    Coefficients act on (u, v, 1), where u and v are screen position as a
    fraction of frame width and height, and yield position as a fraction of the
    viewport rectangle.
    """

    x: tuple[float, float, float]
    y: tuple[float, float, float]

    @classmethod
    def from_layout(cls, layout: NameplateLayout) -> ScreenProjection | None:
        """The calibrated projection, or None if it was never fitted."""
        if layout.projection_x is None or layout.projection_y is None:
            return None
        return cls(x=layout.projection_x, y=layout.projection_y)

    def to_minimap(
        self,
        plate: Nameplate,
        viewport: Viewport,
        frame_size: tuple[int, int],
    ) -> tuple[float, float]:
        """Where on the minimap the champion wearing this plate is standing."""
        width, height = frame_size
        cx, _cy = plate.center
        u, v = cx / width, plate.y / height
        ax, bx, cx_ = self.x
        ay, by, cy_ = self.y
        return (
            viewport.x + (ax * u + bx * v + cx_) * viewport.width,
            viewport.y + (ay * u + by * v + cy_) * viewport.height,
        )


def associate(
    plates: list[Nameplate],
    tracks: list,
    viewport: Viewport | None,
    projection: ScreenProjection | None,
    frame_size: tuple[int, int],
    gate: float = GATE,
) -> dict[int, int]:
    """Plate index to track id, greedily by distance, one track per plate.

    Greedy for the reason the tracker is greedy: at most five candidates a side,
    usually well separated relative to the gate, and an assignment that can be
    explained beats one that is marginally better. Plates the gate rejects
    simply go unassigned, which is the honest outcome -- a plate matched to the
    wrong track corrupts that champion's whole series, while an unmatched one
    costs a single frame.
    """
    if viewport is None or projection is None or not plates or not tracks:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for index, plate in enumerate(plates):
        px, py = projection.to_minimap(plate, viewport, frame_size)
        for track in tracks:
            distance = track.distance_to(px, py)
            if distance <= gate:
                candidates.append((distance, index, track.id))
    candidates.sort()

    pairing: dict[int, int] = {}
    claimed: set[int] = set()
    for _distance, index, track_id in candidates:
        if index in pairing or track_id in claimed:
            continue
        pairing[index] = track_id
        claimed.add(track_id)
    return pairing


def fit(samples: list[tuple[float, float, float, float]]
        ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Least-squares fit of the coefficients from (u, v, target_u, target_v).

    Targets are the blip's position expressed as a fraction of the viewport
    rectangle. Kept here rather than in the calibration tool so that what is
    fitted and what is applied cannot drift apart.
    """
    import numpy as np

    if len(samples) < 8:
        raise ValueError(
            f"need at least 8 samples to fit a projection, got {len(samples)}"
        )
    data = np.asarray(samples, dtype=float)
    design = np.c_[data[:, 0], data[:, 1], np.ones(len(data))]
    coef_x, *_ = np.linalg.lstsq(design, data[:, 2], rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, data[:, 3], rcond=None)
    return tuple(float(v) for v in coef_x), tuple(float(v) for v in coef_y)
