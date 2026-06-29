"""
Pose Detection Module

Provides real-time body landmark detection using MediaPipe Pose.
Detects 33 body landmarks and provides smoothing and filtering.

Classes:
    Landmark: Individual body landmark representation
    PoseDetector: MediaPipe-based pose detection engine
    PoseSmoother: Jitter reduction with Kalman filtering

Author: AI Computer Vision Team
Version: 1.0.0
"""

from .detector import PoseDetector
from .landmark import Landmark
from .smoother import PoseSmoother

__all__ = ["PoseDetector", "Landmark", "PoseSmoother"]
