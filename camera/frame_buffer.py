"""
Frame Buffer - Thread-Safe Frame Queue

Provides thread-safe frame buffering for real-time video processing.
Prevents frame drops and maintains consistent FPS.

Author: AI Computer Vision Team
"""

import logging
import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe circular buffer for video frames.

    This buffer prevents frame drops by maintaining a queue of frames.
    When the buffer is full, oldest frames are discarded to make space for new ones.
    This ensures constant flow of frames to AI modules without blocking.

    Attributes:
        max_size: Maximum number of frames to buffer
        frame_count: Total frames added to buffer
        dropped_frames: Count of frames dropped due to buffer overflow

    Example:
        >>> buffer = FrameBuffer(max_size=10)
        >>> # Producer thread
        >>> buffer.put(frame1)
        >>> buffer.put(frame2)
        >>> # Consumer thread
        >>> ret, frame = buffer.get(timeout=0.1)
        >>> if ret:
        ...     process(frame)
    """

    def __init__(self, max_size: int = 10):
        """Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to keep in buffer
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._frame_count = 0
        self._dropped_frames = 0

        logger.info(f"Frame buffer initialized with max size {max_size}")

    def put(self, frame: np.ndarray, block: bool = False) -> bool:
        """Add frame to buffer.

        Args:
            frame: Frame to add (numpy array)
            block: If True, block until space is available (not recommended)

        Returns:
            True if frame was added, False if buffer is full and block=False
        """
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("Invalid frame received")
            return False

        with self._not_empty:
            was_full = len(self._buffer) >= self.max_size

            self._buffer.append(frame.copy())
            self._frame_count += 1

            if was_full:
                self._dropped_frames += 1

            self._not_empty.notify()

        return True

    def get(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from buffer.

        Args:
            block: If True, wait for frame if buffer is empty
            timeout: Maximum wait time in seconds (only if block=True)

        Returns:
            Tuple of (success, frame)
            - success: True if frame was retrieved
            - frame: Frame as numpy array or None if no frame available
        """
        with self._not_empty:
            if not block:
                if len(self._buffer) == 0:
                    return False, None
                return True, self._buffer.popleft()

            # Block until frame available or timeout
            if not self._not_empty.wait_for(
                lambda: len(self._buffer) > 0, timeout=timeout
            ):
                return False, None

            if len(self._buffer) > 0:
                return True, self._buffer.popleft()

            return False, None

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame without removing others.

        This is useful for getting the most recent frame while keeping
        older frames in the buffer.

        Returns:
            Tuple of (success, frame)
        """
        with self._lock:
            if len(self._buffer) == 0:
                return False, None
            return True, self._buffer[-1].copy()

    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._buffer.clear()
            logger.debug("Frame buffer cleared")

    def size(self) -> int:
        """Get current number of frames in buffer.

        Returns:
            Number of frames currently buffered
        """
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if buffer is empty.

        Returns:
            True if no frames in buffer
        """
        with self._lock:
            return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity.

        Returns:
            True if buffer is full
        """
        with self._lock:
            return len(self._buffer) >= self.max_size

    @property
    def frame_count(self) -> int:
        """Total frames added to buffer.

        Returns:
            Frame count
        """
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        """Count of dropped frames due to buffer overflow.

        Returns:
            Dropped frame count
        """
        return self._dropped_frames

    def get_stats(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "max_size": self.max_size,
                "total_frames": self._frame_count,
                "dropped_frames": self._dropped_frames,
                "utilization": f"{(len(self._buffer) / self.max_size * 100):.1f}%",
            }
