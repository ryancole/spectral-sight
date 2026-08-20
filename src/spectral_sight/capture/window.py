"""Live screen capture: one window, or a whole monitor.

`windows-capture` is an optional extra, so it is imported when a session is
opened rather than when this module is: the offline path has to keep working on
a machine that has never seen a League client.

`WindowSource` is the one the real-time path uses. It captures a named window
via Windows Graphics Capture, which follows the window rather than a screen
region: the window can be moved, buried behind another window or on a second
monitor and the pixels still arrive. That is the property that makes this a VOD
review tool rather than a screenshot scraper.

Performance note: `to_bgr()` copies the full surface out of VRAM into system
RAM. At 1080p that is ~6 MB and costs about a millisecond, which is fine. At 4K
it is ~25 MB and starts to matter. The fix, if profiling ever says it does, is
to crop on the GPU before the download rather than to rewrite this in another
language -- see the notes in the README.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from spectral_sight.capture.base import FrameSource
from spectral_sight.types import Frame


class WindowClosed(RuntimeError):
    """The captured window went away while we were reading it."""


class FrameSizeChanged(RuntimeError):
    """The captured window was resized mid-run.

    Fatal on purpose. Every calibration in `etc/` -- the minimap region, the
    clock box, the portrait centres, the nameplate geometry -- is a rectangle in
    frame pixels keyed on one exact frame size. Carrying on after a resize does
    not degrade the read, it silently relocates every one of those rectangles
    onto the wrong pixels, and the pipeline has no way to notice: it would keep
    emitting confident observations of whatever now happens to sit where the
    minimap used to be.
    """


class Mailbox:
    """A single-slot, latest-wins handoff from the capture thread.

    Capture pushes frames whether or not anyone is reading them, and the
    pipeline is slower than the stream. A queue would answer that by growing a
    backlog, which for a real-time tool is the one outcome with no value at all
    -- a frame that has to wait its turn is a frame describing a fight that has
    already resolved. So a new frame overwrites the pending one and the loss is
    taken at the front, before any work is spent on it.

    `dropped` counts what that cost, since a drop rate is the honest measure of
    whether the pipeline is keeping up and nothing else reports it.
    """

    WAIT_SLICE = 0.25
    """Longest single blocking wait, so Ctrl+C is never more than this away."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._closed = False
        self._error: BaseException | None = None
        self.dropped = 0

    def put(self, image: np.ndarray) -> None:
        with self._condition:
            if self._pending is not None:
                self.dropped += 1
            self._pending = image
            self._condition.notify()

    def fail(self, error: BaseException) -> None:
        """Hand an exception raised on the capture thread to the consumer."""
        with self._condition:
            self._error = error
            self._closed = True
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> np.ndarray | None:
        """Wait for the next frame. None once the source is done.

        `timeout` of None waits indefinitely, which is what a running session
        does -- see `WindowSource.frames`. The wait is sliced rather than taken
        in one call so that Ctrl+C lands promptly, since stopping the session by
        hand is the normal way a live run ends.

        A pending frame is returned even after close, so the last frame the
        window produced is not thrown away by the race between it arriving and
        the window shutting.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._condition:
            while self._pending is None and not self._closed:
                if deadline is None:
                    self._condition.wait(self.WAIT_SLICE)
                    continue
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no frame within {timeout:.1f}s; is the window drawing?"
                    )
                self._condition.wait(min(remaining, self.WAIT_SLICE))
            if self._pending is not None:
                image, self._pending = self._pending, None
                return image
            if self._error is not None:
                raise self._error
            return None


class WindowSource(FrameSource):
    """Pulls frames from one window, newest first, dropping what it outran.

    `title` is matched as a substring, so "kilrogg" finds the receiver whatever
    else it has put in its title bar. Pass `hwnd` instead when two windows would
    both match.
    """

    def __init__(
        self,
        title: str | None = None,
        *,
        hwnd: int | None = None,
        target_fps: float | None = None,
        startup_timeout: float = 5.0,
        cursor: bool = False,
    ) -> None:
        if (title is None) == (hwnd is None):
            raise ValueError("pass exactly one of title or hwnd")
        try:
            from windows_capture import WindowsCapture
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "live capture needs the 'capture' extra: pip install -e .[capture]"
            ) from exc

        self.title = title
        self.startup_timeout = startup_timeout
        """How long to give the window to draw its first frame.

        A deadline belongs on the first frame and nowhere else. Graphics Capture
        delivers a frame when the window *redraws*, so a picture that is not
        moving produces nothing at all -- and no frames at startup means the
        wrong window was matched, while no frames mid-session means the stream
        stalled, which is a thing to wait out rather than an error. Enforcing
        this deadline for the whole run would turn every pause in the feed into
        a crash.
        """

        self._mailbox = Mailbox()
        self._size: tuple[int, int] | None = None
        self._control = None
        self._first: np.ndarray | None = None
        """Held by `size`, which has to pull a frame to learn one. Handed to
        `frames` rather than dropped, so building the pipeline does not cost the
        frame it was built from."""

        # The border is Windows' own "this window is being captured" highlight.
        # It is drawn inside the captured bounds, and the minimap sits within a
        # few pixels of the bottom-right corner, so leaving it on would paint
        # over the one region this whole project is built around.
        self._capture = WindowsCapture(
            cursor_capture=cursor,
            draw_border=False,
            window_name=title,
            window_hwnd=hwnd,
            minimum_update_interval=(
                None if target_fps is None else max(1, int(1000 / target_fps))
            ),
        )

        @self._capture.event
        def on_frame_arrived(frame, capture_control) -> None:  # noqa: ANN001
            # The buffer is a view onto a mapped native frame that is unmapped
            # when this returns, so the copy is not optional. Dropping alpha
            # here rather than downstream keeps it to a single pass.
            self._mailbox.put(np.ascontiguousarray(frame.frame_buffer[:, :, :3]))

        @self._capture.event
        def on_closed() -> None:
            self._mailbox.close()

        try:
            self._control = self._capture.start_free_threaded()
        except Exception as exc:
            what = f"window matching {title!r}" if title else f"hwnd {hwnd}"
            raise WindowClosed(f"could not capture {what}: {exc}") from exc

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the window. Blocks until the first frame lands.

        Callers build their calibration from this before iterating, and a window
        that has not painted yet has no size to give -- so this waits rather
        than making every caller order its own startup handshake.
        """
        if self._size is None:
            self._first = self._await_frame(self.startup_timeout)
        assert self._size is not None
        return self._size

    def _await_frame(self, timeout: float | None = None) -> np.ndarray:
        image = self._mailbox.take(timeout)
        if image is None:
            raise WindowClosed(f"window {self.title!r} closed before it drew a frame")
        height, width = image.shape[:2]
        if self._size is None:
            self._size = (width, height)
        elif (width, height) != self._size:
            raise FrameSizeChanged(
                f"window resized from {self._size[0]}x{self._size[1]} to "
                f"{width}x{height}; the calibration for the old size no longer "
                f"describes this frame. Restore the window size, or calibrate "
                f"for {width}x{height}."
            )
        return image

    @property
    def dropped(self) -> int:
        """Frames the window produced that the pipeline never saw."""
        return self._mailbox.dropped

    def frames(self) -> Iterator[Frame]:
        # Wall-clock timestamps, not a frame counter over an assumed rate: this
        # is a live stream, the rate is whatever the window and the pipeline
        # settle on between them, and the cast detector reasons about elapsed
        # seconds. A counted timestamp would run slow exactly when frames were
        # being dropped, which is when the timing matters most.
        start = time.perf_counter()
        index = 0
        pending, self._first = self._first, None
        while True:
            if pending is None:
                try:
                    # No deadline: a still picture is a stall in the feed, not a
                    # dead window, and the session should outlast it.
                    pending = self._await_frame()
                except WindowClosed:
                    return
            yield Frame(
                image=pending, index=index, timestamp=time.perf_counter() - start
            )
            pending = None
            index += 1

    def close(self) -> None:
        if self._control is not None:
            self._control.stop()
            self._control = None
        self._mailbox.close()


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
