"""Tracking: turning per-frame detections into persistent champions."""

from spectral_sight.tracking.track import Track, TrackState
from spectral_sight.tracking.tracker import Tracker, TrackerConfig

__all__ = ["Track", "TrackState", "Tracker", "TrackerConfig"]
