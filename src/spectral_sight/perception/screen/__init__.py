"""Perception on the 3D world view: the part of the frame that is the game
rather than a HUD panel, read at full frame rate."""

from spectral_sight.perception.screen.motion import (
    CameraMotion,
    CameraTracker,
    MotionConfig,
    WorldView,
)
from spectral_sight.perception.screen.aim import AimConfig, AimDetector, EnemyPlate
from spectral_sight.perception.screen.threats import ThreatConfig, ThreatDetector
from spectral_sight.perception.screen.projectiles import (
    Blob,
    ProjectileConfig,
    ProjectileTrack,
    ProjectileTracker,
)

__all__ = [
    "AimConfig",
    "AimDetector",
    "Blob",
    "CameraMotion",
    "CameraTracker",
    "EnemyPlate",
    "MotionConfig",
    "ProjectileConfig",
    "ProjectileTrack",
    "ProjectileTracker",
    "ThreatConfig",
    "ThreatDetector",
    "WorldView",
]
