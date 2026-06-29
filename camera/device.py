"""
Camera Device Abstraction

Provides device-level camera operations and properties.

Author: AI Computer Vision Team
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraDevice:
    """Represents a physical or virtual camera device.

    This class provides low-level access to camera properties and frame capture.
    It abstracts OpenCV's VideoCapture to provide a cleaner interface.

    Attributes:
        device_id: Camera device identifier (usually 0 for default camera)
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        is_opened: Whether the camera is successfully opened

    Example:
        >>> device = CameraDevice(device_id=0, width=1920, height=1080, fps=30)
        >>> if device.is_opened:
        ...     ret, frame = device.read()
        ...     if ret:
        ...         print(f"Frame shape: {frame.shape}")
        >>> device.release()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        backend: str = "auto",
    ):
        """Initialize camera device.

        Args:
            device_id: Camera device index (0 for default)
            width: Target frame width
            height: Target frame height
            fps: Target frames per second
            backend: OpenCV backend ('auto', 'dshow', 'v4l2', 'msmf')

        Raises:
            RuntimeError: If camera initialization fails
        """
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.backend = backend

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize camera with specified backend and settings.

        Raises:
            RuntimeError: If camera cannot be opened
        """
        try:
            # Select backend
            if self.backend == "auto":
                self._cap = cv2.VideoCapture(self.device_id)
            elif self.backend == "dshow":
                self._cap = cv2.VideoCapture(
                    self.device_id, cv2.CAP_DSHOW
                )
            elif self.backend == "v4l2":
                self._cap = cv2.VideoCapture(
                    self.device_id, cv2.CAP_V4L2
                )
            elif self.backend == "msmf":
                self._cap = cv2.VideoCapture(
                    self.device_id, cv2.CAP_MSMF
                )
            else:
                self._cap = cv2.VideoCapture(self.device_id)

            if not self._cap or not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to open camera device {self.device_id}"
                )

            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Enable hardware acceleration if available
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

            # Read actual properties
            self._actual_width = int(
                self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            self._actual_height = int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            self._is_opened = True
            logger.info(
                f"Camera {self.device_id} opened: "
                f"{self._actual_width}x{self._actual_height} @ {self._actual_fps:.1f}fps"
            )

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize camera: {e}")

    @property
    def is_opened(self) -> bool:
        """Check if camera is open and ready.

        Returns:
            True if camera is open, False otherwise
        """
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        """Get actual frame width.

        Returns:
            Frame width in pixels
        """
        return self._actual_width

    @property
    def height(self) -> int:
        """Get actual frame height.

        Returns:
            Frame height in pixels
        """
        return self._actual_height

    @property
    def fps(self) -> float:
        """Get actual frames per second.

        Returns:
            FPS value
        """
        return self._actual_fps

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera.

        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: Frame as numpy array (BGR format) or None if read failed
        """
        if not self.is_opened:
            logger.warning("Attempt to read from closed camera")
            return False, None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return False, None
            return True, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def set_brightness(self, value: float) -> None:
        """Set camera brightness.

        Args:
            value: Brightness value (camera-dependent, usually 0-255)
        """
        if self.is_opened:
            try:
                self._cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
                logger.debug(f"Brightness set to {value}")
            except Exception as e:
                logger.error(f"Failed to set brightness: {e}")

    def set_contrast(self, value: float) -> None:
        """Set camera contrast.

        Args:
            value: Contrast value (camera-dependent, usually 0-100)
        """
        if self.is_opened:
            try:
                self._cap.set(cv2.CAP_PROP_CONTRAST, value)
                logger.debug(f"Contrast set to {value}")
            except Exception as e:
                logger.error(f"Failed to set contrast: {e}")

    def set_saturation(self, value: float) -> None:
        """Set camera saturation.

        Args:
            value: Saturation value (camera-dependent)
        """
        if self.is_opened:
            try:
                self._cap.set(cv2.CAP_PROP_SATURATION, value)
                logger.debug(f"Saturation set to {value}")
            except Exception as e:
                logger.error(f"Failed to set saturation: {e}")

    def set_exposure(self, value: float) -> None:
        """Set camera exposure.

        Args:
            value: Exposure value (camera-dependent)
        """
        if self.is_opened:
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, value)
                logger.debug(f"Exposure set to {value}")
            except Exception as e:
                logger.error(f"Failed to set exposure: {e}")

    def get_property(self, prop_id: int) -> float:
        """Get camera property value.

        Args:
            prop_id: OpenCV property ID (cv2.CAP_PROP_*)

        Returns:
            Property value
        """
        if self.is_opened:
            try:
                return self._cap.get(prop_id)
            except Exception as e:
                logger.error(f"Failed to get property {prop_id}: {e}")
        return -1.0

    def set_property(self, prop_id: int, value: float) -> bool:
        """Set camera property value.

        Args:
            prop_id: OpenCV property ID (cv2.CAP_PROP_*)
            value: Property value to set

        Returns:
            True if property was set successfully
        """
        if self.is_opened:
            try:
                return self._cap.set(prop_id, value)
            except Exception as e:
                logger.error(f"Failed to set property {prop_id}: {e}")
        return False

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
                self._is_opened = False
                logger.info(f"Camera {self.device_id} released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")

    def __del__(self) -> None:
        """Ensure camera is released on object destruction."""
        self.release()
