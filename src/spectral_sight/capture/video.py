"""Offline frame sources: recorded clips and single stills.

These are the development path. Everything about the detector is easier to
iterate on when the input is byte-identical across runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from spectral_sight.capture.base import FrameSource
from spectral_sight.types import Frame

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class VideoFileSource(FrameSource):
    """Reads frames from a video file via OpenCV."""

    def __init__(self, path: str | Path, *, stride: int = 1) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.stride = stride
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")

        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            if index % self.stride == 0:
                yield Frame(image=image, index=index, timestamp=index / self._fps)
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


class ImageSource(FrameSource):
    """Yields one still image. Useful for threshold tuning and tests."""

    def __init__(self, path: str | Path, *, repeat: int = 1) -> None:
        self.path = Path(path)
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read image: {self.path}")
        self._image = image
        self._repeat = repeat

    @property
    def size(self) -> tuple[int, int]:
        height, width = self._image.shape[:2]
        return width, height

    def frames(self) -> Iterator[Frame]:
        for index in range(self._repeat):
            yield Frame(image=self._image.copy(), index=index, timestamp=0.0)


def open_source(path: str | Path, *, stride: int = 1) -> FrameSource:
    """Open a still or a clip, picking the reader from the file suffix."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageSource(path)
    return VideoFileSource(path, stride=stride)
