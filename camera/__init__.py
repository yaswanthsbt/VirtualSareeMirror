"""
Camera Module

Handles webcam capture, frame processing, and camera management.
Provides real-time video input for the Virtual Saree Mirror system.

Classes:
    CameraManager: Main camera management interface
    FrameBuffer: Thread-safe frame queue for frame buffering
    CameraDevice: Camera device abstraction layer

Author: AI Computer Vision Team
Version: 1.0.0
"""

from .manager import CameraManager
from .device import CameraDevice
from .frame_buffer import FrameBuffer

__all__ = ["CameraManager", "CameraDevice", "FrameBuffer"]
