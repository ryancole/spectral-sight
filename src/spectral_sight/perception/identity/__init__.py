"""Stage 2: deciding which champion a marker is."""

from spectral_sight.perception.identity.gallery import (
    CIRCLE_MASK,
    HUD_MASK,
    Gallery,
    Match,
    PatchDescriptor,
    describe,
    describe_variants,
    load_icon_gallery,
)
from spectral_sight.perception.identity.roster import TEAM_SIZE, Roster

__all__ = [
    "CIRCLE_MASK",
    "HUD_MASK",
    "TEAM_SIZE",
    "Roster",
    "Gallery",
    "Match",
    "PatchDescriptor",
    "describe",
    "describe_variants",
    "load_icon_gallery",
]
