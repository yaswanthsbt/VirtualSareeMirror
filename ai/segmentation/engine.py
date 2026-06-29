"""
Segmentation Engine - MediaPipe Selfie Segmentation

Real-time human body segmentation for background removal.

Author: AI Computer Vision Team
"""

import logging
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .mask import BodyMask

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """Real-time human segmentation using MediaPipe Selfie Segmentation.

    This engine provides fast, accurate human body segmentation by generating
    a binary mask where 1 represents the body and 0 represents the background.

    Features:
    - Real-time processing (GPU-accelerated)
    - Two model options: light (faster) and full (more accurate)
    - Automatic mask refinement
    - Integration with pose detection
    - Confidence-based filtering

    Attributes:
        model_selection: Model type (0=light, 1=full)
        confidence_threshold: Minimum confidence for segmentation

    Example:
        >>> engine = SegmentationEngine(model_selection=1)
        >>> mask = engine.segment(frame)
        >>> if mask:
        ...     mask.dilate(3).fill_holes(100)
        ...     refined = mask.get_refined_mask()
    """

    def __init__(
        self,
        model_selection: int = 1,
        confidence_threshold: float = 0.5,
    ):
        """Initialize segmentation engine.

        Args:
            model_selection: Model type (0=light/fast, 1=full/accurate)
            confidence_threshold: Minimum confidence for valid segmentation

        Raises:
            RuntimeError: If MediaPipe initialization fails
        """
        self.model_selection = model_selection
        self.confidence_threshold = confidence_threshold

        try:
            # Initialize MediaPipe Selfie Segmentation
            self.mp_selfie_segmentation = (
                mp.solutions.selfie_segmentation
            )
            self.segmentation = (
                self.mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=model_selection
                )
            )

            model_name = "Light (Fast)" if model_selection == 0 else "Full (Accurate)"
            logger.info(
                f"SegmentationEngine initialized (model={model_name})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Segmentation: {e}")
            raise RuntimeError(f"Cannot initialize segmentation: {e}")

        # Performance tracking
        self._segmentation_count = 0
        self._last_mask: Optional[BodyMask] = None

    def segment(self, frame: np.ndarray) -> Optional[BodyMask]:
        """Segment human body from frame.

        Args:
            frame: Input frame (HxWx3, BGR)

        Returns:
            BodyMask object or None if segmentation fails
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame received")
            return None

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run segmentation
            results = self.segmentation.process(rgb_frame)

            if results.segmentation_mask is None:
                logger.warning("Segmentation returned None")
                return None

            # Convert confidence mask to binary
            seg_mask = results.segmentation_mask
            binary_mask = (seg_mask > self.confidence_threshold).astype(np.uint8) * 255

            # Create BodyMask object
            body_mask = BodyMask(
                mask=binary_mask,
                confidence=seg_mask.astype(np.float32),
                threshold=self.confidence_threshold,
            )

            self._segmentation_count += 1
            self._last_mask = body_mask

            return body_mask

        except Exception as e:
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            return None

    def segment_with_refinement(
        self,
        frame: np.ndarray,
        dilate_kernel: int = 3,
        erode_kernel: int = 3,
        fill_holes: bool = True,
    ) -> Optional[BodyMask]:
        """Segment and automatically refine mask.

        Args:
            frame: Input frame
            dilate_kernel: Dilation kernel size
            erode_kernel: Erosion kernel size
            fill_holes: Whether to fill small holes

        Returns:
            Refined BodyMask or None
        """
        mask = self.segment(frame)

        if mask is None:
            return None

        # Apply refinement pipeline
        mask.close(kernel_size=3)  # Fill holes
        if dilate_kernel > 0:
            mask.dilate(kernel_size=dilate_kernel)
        if erode_kernel > 0:
            mask.erode(kernel_size=erode_kernel)
        if fill_holes:
            mask.fill_holes(hole_size_threshold=100)

        return mask

    def segment_and_extract(
        self,
        frame: np.ndarray,
        refine: bool = True,
    ) -> Tuple[Optional[BodyMask], Optional[np.ndarray]]:
        """Segment and extract foreground from frame.

        Args:
            frame: Input frame
            refine: Whether to refine mask

        Returns:
            Tuple of (mask, foreground_image)
        """
        if refine:
            mask = self.segment_with_refinement(frame)
        else:
            mask = self.segment(frame)

        if mask is None:
            return None, None

        # Extract foreground
        foreground = mask.apply_to_image(frame, background_color=(0, 0, 0))

        return mask, foreground

    def get_last_mask(self) -> Optional[BodyMask]:
        """Get last computed mask.

        Returns:
            Last BodyMask or None
        """
        return self._last_mask

    def visualize_mask(
        self,
        mask: BodyMask,
        overlay_alpha: float = 0.5,
        body_color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Create visualization of segmentation mask.

        Args:
            mask: BodyMask object
            overlay_alpha: Alpha blending factor
            body_color: Color for body region (BGR)

        Returns:
            Visualization image
        """
        h, w = mask.height, mask.width
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw body region
        body_region = np.where(mask.mask > 0)
        vis[body_region] = body_color

        return vis

    def visualize_confidence(
        self,
        mask: BodyMask,
    ) -> np.ndarray:
        """Create heatmap visualization of confidence scores.

        Args:
            mask: BodyMask object

        Returns:
            Heatmap image (HxWx3, BGR)
        """
        confidence_normalized = (mask.get_confidence_map() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(confidence_normalized, cv2.COLORMAP_JET)
        return heatmap

    def release(self) -> None:
        """Release resources."""
        if self.segmentation:
            self.segmentation.close()
            logger.info("SegmentationEngine released")

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.release()
