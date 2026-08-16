"""The frame source interface.

Keeping this narrow is what lets the same detector run against a recorded clip
during development and a live window during a game, with no branch in between.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType

from spectral_sight.types import Frame


class FrameSource(ABC):
    """Yields frames until exhausted or closed."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Iterate frames. May block waiting on the source."""

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """(width, height) of frames this source produces."""

    def close(self) -> None:
        """Release the underlying handle. Idempotent."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
