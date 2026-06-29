"""
Human Segmentation Module

Provides real-time human body segmentation for background removal.
Uses MediaPipe Selfie Segmentation for fast, accurate masking.

Classes:
    BodyMask: Binary mask representation
    SegmentationEngine: Main segmentation interface

Author: AI Computer Vision Team
Version: 1.0.0
"""

from .engine import SegmentationEngine
from .mask import BodyMask

__all__ = ["SegmentationEngine", "BodyMask"]
