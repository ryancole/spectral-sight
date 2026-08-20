"""One spec string in, the right frame source out.

Every tool in `tools/` takes a source on the command line, so this is the single
place that decides what a source string means -- and therefore the only place
that has to change for a tool to accept a new kind of input. Adding the window
scheme here gave every calibration tool a live target without touching any of
them.

    data/clip.mp4       a recording
    data/frame.png      a still
    window:kilrogg      whatever that window is showing, right now

The window form is what makes calibration possible on a machine with no footage
on it. Calibrating used to mean producing a screenshot first, which for a live
review tool is a recording step in the middle of the workflow that exists to
remove recording steps.
"""

from __future__ import annotations

from pathlib import Path

from spectral_sight.capture.base import FrameSource
from spectral_sight.capture.video import IMAGE_SUFFIXES, ImageSource, VideoFileSource
from spectral_sight.capture.window import WindowSource

WINDOW_SCHEME = "window:"


def open_source(
    spec: str | Path, *, stride: int = 1, start: int = 0
) -> FrameSource:
    """Open a still, a clip or a live window, picking the reader from the spec."""
    if isinstance(spec, str) and spec.startswith(WINDOW_SCHEME):
        title = spec[len(WINDOW_SCHEME):].strip()
        if not title:
            raise ValueError(
                f"{spec!r} names no window; use e.g. {WINDOW_SCHEME}kilrogg"
            )
        if start:
            raise ValueError(
                f"--start seeks into a recording; {spec!r} is live and has no past"
            )
        return WindowSource(title, stride=stride)

    path = Path(spec)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageSource(path)
    return VideoFileSource(path, stride=stride, start=start)
