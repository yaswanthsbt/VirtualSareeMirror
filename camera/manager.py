"""
Camera Manager - Main Camera Interface

Manages camera initialization, frame capture, and preprocessing.
Supports multi-camera and real-time frame rate control.

Author: AI Computer Vision Team
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .device import CameraDevice
from .frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera capture and frame processing for the Virtual Saree Mirror.

    This class handles:
    - Multi-camera support
    - Real-time frame capture
    - FPS control and synchronization
    - Frame preprocessing (resize, rotation, color conversion)
    - Thread-safe frame buffering

    Attributes:
        device_id: Active camera device ID
        width: Frame width
        height: Frame height
        target_fps: Target frames per second
        is_running: Whether capture thread is active

    Example:
        >>> manager = CameraManager(device_id=0, width=1920, height=1080, fps=30)
        >>> if manager.is_ready:
        ...     manager.start()
        ...     time.sleep(1)  # Let buffer fill
        ...     ret, frame = manager.get_frame()
        ...     if ret:
        ...         print(f"Got frame: {frame.shape}")
        ...     manager.stop()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_size: int = 5,
        flip_h: bool = True,
        flip_v: bool = False,
    ):
        """Initialize camera manager.

        Args:
            device_id: Camera device ID (0 for default/first camera)
            width: Target frame width in pixels
            height: Target frame height in pixels
            fps: Target frames per second
            buffer_size: Frame buffer size
            flip_h: Flip frames horizontally (mirror effect)
            flip_v: Flip frames vertically

        Raises:
            RuntimeError: If camera cannot be initialized
        """
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.flip_h = flip_h
        self.flip_v = flip_v

        # Camera device
        self._device: Optional[CameraDevice] = None
        self._buffer = FrameBuffer(max_size=buffer_size)

        # Threading
        self._capture_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_event = threading.Event()

        # Performance tracking
        self._frame_times: List[float] = []
        self._actual_fps = 0.0
        self._frame_count = 0

        # Initialize device
        self._init_device()

    def _init_device(self) -> None:
        """Initialize camera device.

        Raises:
            RuntimeError: If device initialization fails
        """
        try:
            self._device = CameraDevice(
                device_id=self.device_id,
                width=self.target_width,
                height=self.target_height,
                fps=self.target_fps,
            )

            if not self._device.is_opened:
                raise RuntimeError("Camera device failed to open")

            logger.info(
                f"Camera manager initialized for device {self.device_id}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    @property
    def is_ready(self) -> bool:
        """Check if camera is ready to capture.

        Returns:
            True if camera is initialized and ready
        """
        return self._device is not None and self._device.is_opened

    @property
    def is_running(self) -> bool:
        """Check if capture thread is running.

        Returns:
            True if actively capturing
        """
        return self._is_running

    @property
    def width(self) -> int:
        """Get actual frame width.

        Returns:
            Frame width in pixels
        """
        return self._device.width if self._device else 0

    @property
    def height(self) -> int:
        """Get actual frame height.

        Returns:
            Frame height in pixels
        """
        return self._device.height if self._device else 0

    @property
    def actual_fps(self) -> float:
        """Get actual FPS being achieved.

        Returns:
            Actual frames per second
        """
        return self._actual_fps

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        This continuously reads frames from camera and adds them to buffer.
        Frame rate is controlled to match target FPS.
        """
        logger.info("Capture thread started")
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        fps_times = []

        try:
            while not self._stop_event.is_set():
                # Get frame from camera
                ret, frame = self._device.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    continue

                # Apply preprocessing
                frame = self._preprocess_frame(frame)

                # Add to buffer
                self._buffer.put(frame)
                self._frame_count += 1

                # FPS calculation
                current_time = time.time()
                elapsed = current_time - last_frame_time
                fps_times.append(elapsed)

                # Keep last 30 frame times
                if len(fps_times) > 30:
                    fps_times.pop(0)

                # Update actual FPS every 10 frames
                if self._frame_count % 10 == 0:
                    avg_time = sum(fps_times) / len(fps_times)
                    self._actual_fps = 1.0 / avg_time if avg_time > 0 else 0

                # FPS control - sleep if frame came too fast
                if elapsed < frame_interval:
                    sleep_time = frame_interval - elapsed
                    time.sleep(sleep_time)

                last_frame_time = time.time()

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            logger.info("Capture thread stopped")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame (flip, color conversion, etc).

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Flip if needed
        if self.flip_h:
            frame = cv2.flip(frame, 1)  # Horizontal flip
        if self.flip_v:
            frame = cv2.flip(frame, 0)  # Vertical flip

        return frame

    def start(self) -> bool:
        """Start capturing frames in background thread.

        Returns:
            True if capture started successfully
        """
        if not self.is_ready:
            logger.error("Camera not ready")
            return False

        if self._is_running:
            logger.warning("Capture already running")
            return True

        try:
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()
            self._is_running = True
            logger.info("Camera capture started")
            return True

        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False

    def stop(self) -> None:
        """Stop capturing frames."""
        if not self._is_running:
            return

        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)

        self._is_running = False
        logger.info("Camera capture stopped")

    def get_frame(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame from buffer.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Tuple of (success, frame)
        """
        return self._buffer.get(block=True, timeout=timeout)

    def get_frame_nowait(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame without blocking.

        Returns:
            Tuple of (success, frame) - success is False if no frame available
        """
        return self._buffer.get(block=False)

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame without removing from buffer.

        Returns:
            Tuple of (success, frame)
        """
        return self._buffer.get_latest()

    def set_brightness(self, value: float) -> None:
        """Set camera brightness.

        Args:
            value: Brightness value (0-255, camera-dependent)
        """
        if self._device:
            self._device.set_brightness(value)

    def set_contrast(self, value: float) -> None:
        """Set camera contrast.

        Args:
            value: Contrast value (0-100, camera-dependent)
        """
        if self._device:
            self._device.set_contrast(value)

    def get_buffer_stats(self) -> Dict:
        """Get frame buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        stats = self._buffer.get_stats()
        stats["actual_fps"] = f"{self._actual_fps:.1f}"
        stats["total_captured"] = self._frame_count
        return stats

    def get_camera_info(self) -> Dict:
        """Get camera information.

        Returns:
            Dictionary with camera details
        """
        if not self._device:
            return {}

        return {
            "device_id": self.device_id,
            "resolution": f"{self._device.width}x{self._device.height}",
            "fps": f"{self._device.fps:.1f}",
            "target_fps": self.target_fps,
            "actual_fps": f"{self._actual_fps:.1f}",
            "is_open": self._device.is_opened,
        }

    def release(self) -> None:
        """Release camera resources."""
        self.stop()
        if self._device:
            self._device.release()
            logger.info("Camera manager released")

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.release()
