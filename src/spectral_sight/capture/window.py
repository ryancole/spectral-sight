"""Live screen capture via DXGI Desktop Duplication.

Imported lazily -- `windows-capture` is an optional extra, so the offline path
works on a machine that has never seen a League client.

Performance note: `acquire_frame().to_bgr()` copies the full monitor surface out
of VRAM into system RAM. At 1080p that is ~6 MB and costs about a millisecond,
which is fine. At 4K it is ~25 MB and starts to matter. The fix, if profiling
ever says it does, is to crop on the GPU before the download rather than to
rewrite this in another language -- see the notes in the README.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from spectral_sight.capture.base import FrameSource
from spectral_sight.types import Frame


class MonitorSource(FrameSource):
    """Pulls frames from a monitor as fast as the caller consumes them.

    Duplication only produces a frame when the screen actually changes. If
    nothing moved within `timeout_ms` the previous frame is re-emitted, so the
    downstream cadence stays stable rather than stalling.
    """

    def __init__(
        self,
        monitor_index: int = 0,
        *,
        timeout_ms: int = 16,
        max_frames: int | None = None,
    ) -> None:
        try:
            from windows_capture import DxgiDuplicationSession
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "live capture needs the 'capture' extra: pip install -e .[capture]"
            ) from exc

        self._session = DxgiDuplicationSession(monitor_index=monitor_index)
        self._timeout_ms = timeout_ms
        self._max_frames = max_frames
        self._last: np.ndarray | None = None
        self._size: tuple[int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        if self._size is None:
            raise RuntimeError("size is unknown until the first frame arrives")
        return self._size

    def frames(self) -> Iterator[Frame]:
        start = time.perf_counter()
        index = 0
        while self._max_frames is None or index < self._max_frames:
            captured = self._session.acquire_frame(timeout_ms=self._timeout_ms)
            if captured is not None:
                self._last = captured.to_bgr()
                height, width = self._last.shape[:2]
                self._size = (width, height)
            elif self._last is None:
                # Nothing has been drawn yet; wait for the first real frame.
                continue

            assert self._last is not None
            yield Frame(
                image=self._last,
                index=index,
                timestamp=time.perf_counter() - start,
            )
            index += 1

    def close(self) -> None:
        self._session = None
