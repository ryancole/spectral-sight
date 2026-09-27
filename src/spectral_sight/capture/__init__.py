"""Frame sources. The rest of the pipeline never knows where pixels came from.

Every source is live: a named window, normally the kilrogg receiver, or a whole
monitor. `windows-capture` is an optional extra, so it is imported when a
session is actually opened rather than when the module is.
"""

from spectral_sight.capture.base import FrameSource
from spectral_sight.capture.window import (
    DEFAULT_WINDOW,
    FrameSizeChanged,
    MonitorSource,
    WindowClosed,
    WindowSource,
    lasting,
)

__all__ = [
    "DEFAULT_WINDOW",
    "FrameSizeChanged",
    "FrameSource",
    "MonitorSource",
    "WindowClosed",
    "WindowSource",
    "lasting",
]
