"""The live capture handoff.

What is pinned here is the thing that separates a real-time source from a file
one: a clip waits for its reader, a window does not. The capture thread pushes
frames whether or not the pipeline is ready, so the interesting behaviour is all
in what happens to the frames nobody asked for, and in how the session ends.

The Windows capture backend itself is not exercised -- it needs a compositor and
a window, and neither belongs in a unit test. `Mailbox` is the whole of the
concurrency, so it is tested directly, and `WindowSource` is driven through it.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from spectral_sight.capture.window import FrameSizeChanged, Mailbox, WindowSource


def image(value: int, size: tuple[int, int] = (4, 6)) -> np.ndarray:
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


class TestMailbox:
    def test_delivers_a_frame(self) -> None:
        box = Mailbox()
        box.put(image(7))
        assert box.take(1.0)[0, 0, 0] == 7

    def test_newest_wins(self) -> None:
        """The whole point: a slow reader sees now, not the backlog."""
        box = Mailbox()
        box.put(image(1))
        box.put(image(2))
        box.put(image(3))
        assert box.take(1.0)[0, 0, 0] == 3
        assert box.dropped == 2

    def test_closing_ends_the_stream(self) -> None:
        box = Mailbox()
        box.close()
        assert box.take(1.0) is None

    def test_a_pending_frame_survives_close(self) -> None:
        """The window closing must not eat the last frame it drew."""
        box = Mailbox()
        box.put(image(9))
        box.close()
        assert box.take(1.0)[0, 0, 0] == 9
        assert box.take(1.0) is None

    def test_a_silent_source_times_out(self) -> None:
        box = Mailbox()
        with pytest.raises(TimeoutError):
            box.take(0.01)

    def test_blocks_until_a_frame_arrives(self) -> None:
        box = Mailbox()
        threading.Timer(0.02, lambda: box.put(image(5))).start()
        assert box.take(2.0)[0, 0, 0] == 5

    def test_capture_thread_errors_reach_the_consumer(self) -> None:
        box = Mailbox()
        box.fail(ValueError("the capture thread fell over"))
        with pytest.raises(ValueError, match="fell over"):
            box.take(1.0)


class TestWindowSource:
    """Driven through its mailbox, with no capture session behind it.

    `__new__` skips `__init__` deliberately -- constructing one for real opens a
    Windows capture session, and every behaviour worth pinning here is on the
    consumer side of the mailbox.
    """

    def build(self) -> WindowSource:
        source = WindowSource.__new__(WindowSource)
        source.title = "test"
        source.startup_timeout = 1.0
        source._mailbox = Mailbox()
        source._size = None
        source._control = None
        source._first = None
        return source

    def test_size_comes_from_the_first_frame(self) -> None:
        source = self.build()
        source._mailbox.put(image(1, size=(2118, 1354)))
        assert source.size == (2118, 1354)

    def test_the_frame_size_was_read_from_is_not_lost(self) -> None:
        """Building the pipeline costs a frame; the run should still see it."""
        source = self.build()
        source._mailbox.put(image(1))
        source._mailbox.close()
        assert source.size == (4, 6)
        assert len(list(source.frames())) == 1

    def test_timestamps_are_wall_clock_and_increase(self) -> None:
        source = self.build()
        stream = source.frames()
        # One at a time: a mailbox only ever holds the newest frame, so filling
        # it twice before reading would be a test of the drop, not of the clock.
        source._mailbox.put(image(1))
        first = next(stream)
        source._mailbox.put(image(2))
        second = next(stream)
        assert (first.index, second.index) == (0, 1)
        assert 0 <= first.timestamp < second.timestamp

    def test_a_window_that_never_draws_fails_at_startup(self) -> None:
        source = self.build()
        source.startup_timeout = 0.01
        with pytest.raises(TimeoutError):
            source.size  # noqa: B018

    def test_a_stalled_feed_is_waited_out_rather_than_failed(self) -> None:
        """Graphics Capture emits on redraw, so a still picture emits nothing.

        Mid-session that means the stream stalled, which a review session should
        sit through -- only the *first* frame gets a deadline, and it is the one
        that says whether the right window was found at all.
        """
        source = self.build()
        source.startup_timeout = 0.01
        source._mailbox.put(image(1))
        assert source.size == (4, 6)

        threading.Timer(0.15, lambda: source._mailbox.put(image(2))).start()
        stream = source.frames()
        assert next(stream).image[0, 0, 0] == 1
        assert next(stream).image[0, 0, 0] == 2

    def test_a_closed_window_ends_iteration(self) -> None:
        source = self.build()
        source._mailbox.close()
        assert list(source.frames()) == []

    def test_a_resize_is_fatal(self) -> None:
        """Every rectangle in etc/ is keyed on one frame size."""
        source = self.build()
        source._mailbox.put(image(1, size=(2118, 1354)))
        assert source.size == (2118, 1354)
        source._mailbox.put(image(1, size=(1920, 1080)))
        with pytest.raises(FrameSizeChanged, match="1920x1080"):
            list(source.frames())
