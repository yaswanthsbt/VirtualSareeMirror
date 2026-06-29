"""
Landmark Data Structure

Represents a single body landmark with position, confidence, and visibility.

Author: AI Computer Vision Team
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Landmark:
    """Represents a single body landmark.

    Attributes:
        x: Normalized x-coordinate (0.0-1.0)
        y: Normalized y-coordinate (0.0-1.0)
        z: Normalized z-coordinate (depth estimate)
        confidence: Detection confidence (0.0-1.0)
        visibility: Visibility score (0.0-1.0)
        name: Landmark name (e.g., "NOSE", "LEFT_SHOULDER")
        index: Landmark index in pose model (0-32)
    """

    x: float
    y: float
    z: float
    confidence: float
    visibility: float
    name: str
    index: int

    def to_pixel_coords(
        self, image_width: int, image_height: int
    ) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (x_pixel, y_pixel)
        """
        return (
            int(self.x * image_width),
            int(self.y * image_height),
        )

    def is_visible(self, threshold: float = 0.5) -> bool:
        """Check if landmark is visible above confidence threshold.

        Args:
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            True if visible and confident
        """
        return self.visibility > threshold and self.confidence > threshold

    def to_dict(self) -> dict:
        """Convert landmark to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "name": self.name,
            "index": self.index,
        }
