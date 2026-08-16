"""Frame sources. The rest of the pipeline never knows where pixels came from."""

from spectral_sight.capture.base import FrameSource
from spectral_sight.capture.video import ImageSource, VideoFileSource, open_source

__all__ = ["FrameSource", "ImageSource", "VideoFileSource", "open_source"]
