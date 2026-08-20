"""Frame sources. The rest of the pipeline never knows where pixels came from.

The live sources are safe to import anywhere: `windows-capture` is an optional
extra, so it is imported when a session is actually opened rather than when the
module is, and the offline path keeps working on a machine without it.
"""

from spectral_sight.capture.base import FrameSource
from spectral_sight.capture.open import WINDOW_SCHEME, open_source
from spectral_sight.capture.video import ImageSource, VideoFileSource
from spectral_sight.capture.window import (
    FrameSizeChanged,
    MonitorSource,
    WindowClosed,
    WindowSource,
)

__all__ = [
    "WINDOW_SCHEME",
    "FrameSizeChanged",
    "FrameSource",
    "ImageSource",
    "MonitorSource",
    "VideoFileSource",
    "WindowClosed",
    "WindowSource",
    "open_source",
]
