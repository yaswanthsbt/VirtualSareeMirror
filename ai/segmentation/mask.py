"""
Body Mask Representation

Handles binary segmentation masks and mask operations.

Author: AI Computer Vision Team
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BodyMask:
    """Represents a binary segmentation mask for human body.

    Stores and processes binary masks where 1 = body, 0 = background.
    Provides utilities for mask refinement and feature extraction.

    Attributes:
        mask: Binary mask array (HxW, uint8)
        confidence: Segmentation confidence map (HxW, float32)
        refined: Whether mask has been refined

    Example:
        >>> mask = BodyMask(raw_mask, confidence_map)
        >>> mask.dilate(kernel_size=5)
        >>> mask.erode(kernel_size=3)
        >>> refined_mask = mask.get_refined_mask()
    """

    def __init__(
        self,
        mask: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ):
        """Initialize body mask.

        Args:
            mask: Binary mask (HxW, uint8 or float32)
            confidence: Confidence scores (HxW, float32)
            threshold: Threshold for binary conversion
        """
        # Convert to binary if needed
        if mask.dtype != np.uint8:
            mask = (mask > threshold).astype(np.uint8) * 255

        self.mask = mask
        self.confidence = confidence if confidence is not None else np.ones_like(mask, dtype=np.float32)
        self.height, self.width = mask.shape
        self.refined = False
        self._original_mask = mask.copy()

    def dilate(self, kernel_size: int = 3, iterations: int = 1) -> "BodyMask":
        """Dilate mask to expand body region.

        Args:
            kernel_size: Kernel size (must be odd)
            iterations: Number of iterations

        Returns:
            Self for chaining
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.mask = cv2.dilate(self.mask, kernel, iterations=iterations)
        self.refined = True
        logger.debug(f"Mask dilated (kernel={kernel_size}, iter={iterations})")
        return self

    def erode(self, kernel_size: int = 3, iterations: int = 1) -> "BodyMask":
        """Erode mask to shrink body region.

        Args:
            kernel_size: Kernel size (must be odd)
            iterations: Number of iterations

        Returns:
            Self for chaining
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.mask = cv2.erode(self.mask, kernel, iterations=iterations)
        self.refined = True
        logger.debug(f"Mask eroded (kernel={kernel_size}, iter={iterations})")
        return self

    def open(self, kernel_size: int = 3) -> "BodyMask":
        """Morphological opening (erode then dilate) to remove noise.

        Args:
            kernel_size: Kernel size

        Returns:
            Self for chaining
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.refined = True
        logger.debug(f"Morphological opening applied (kernel={kernel_size})")
        return self

    def close(self, kernel_size: int = 3) -> "BodyMask":
        """Morphological closing (dilate then erode) to fill holes.

        Args:
            kernel_size: Kernel size

        Returns:
            Self for chaining
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.refined = True
        logger.debug(f"Morphological closing applied (kernel={kernel_size})")
        return self

    def bilateral_filter(self, diameter: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> "BodyMask":
        """Apply bilateral filtering for edge-preserving smoothing.

        Args:
            diameter: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space

        Returns:
            Self for chaining
        """
        self.mask = cv2.bilateralFilter(
            self.mask,
            diameter,
            sigma_color,
            sigma_space
        )
        self.refined = True
        logger.debug("Bilateral filtering applied")
        return self

    def gaussian_blur(self, kernel_size: int = 5) -> "BodyMask":
        """Apply Gaussian blur for smoothing.

        Args:
            kernel_size: Kernel size (must be odd)

        Returns:
            Self for chaining
        """
        self.mask = cv2.GaussianBlur(
            self.mask, (kernel_size, kernel_size), 0
        )
        self.refined = True
        logger.debug(f"Gaussian blur applied (kernel={kernel_size})")
        return self

    def fill_holes(self, hole_size_threshold: int = 100) -> "BodyMask":
        """Fill small holes in mask using connected components.

        Args:
            hole_size_threshold: Minimum hole size to fill (pixels)

        Returns:
            Self for chaining
        """
        # Find contours and fill
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        filled_mask = self.mask.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < hole_size_threshold:
                cv2.drawContours(filled_mask, [contour], 0, 255, -1)

        self.mask = filled_mask
        self.refined = True
        logger.debug(f"Holes filled (threshold={hole_size_threshold})")
        return self

    def get_refined_mask(self) -> np.ndarray:
        """Get refined binary mask.

        Returns:
            Binary mask (HxW, uint8)
        """
        return self.mask.copy()

    def get_inverted_mask(self) -> np.ndarray:
        """Get inverted mask (background = 255, body = 0).

        Returns:
            Inverted binary mask
        """
        return cv2.bitwise_not(self.mask)

    def get_confidence_map(self) -> np.ndarray:
        """Get confidence scores.

        Returns:
            Confidence map (HxW, float32)
        """
        return self.confidence.copy()

    def apply_to_image(
        self, image: np.ndarray, background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Apply mask to image to remove background.

        Args:
            image: Input image (HxWx3, BGR)
            background_color: Color for background pixels

        Returns:
            Image with background removed
        """
        if image.shape[:2] != self.mask.shape:
            logger.warning("Image and mask size mismatch")
            return image

        # Create 3-channel mask
        mask_3ch = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        mask_3ch = mask_3ch.astype(np.float32) / 255.0

        # Apply mask
        result = (image.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Add background color where masked out
        bg_mask = (mask_3ch == 0).astype(np.uint8)
        for c in range(3):
            result[:, :, c] = np.where(
                bg_mask[:, :, c],
                background_color[c],
                result[:, :, c]
            )

        return result

    def get_foreground(self, image: np.ndarray) -> np.ndarray:
        """Extract foreground (body) from image.

        Args:
            image: Input image (HxWx3, BGR)

        Returns:
            Foreground image with transparent background
        """
        if image.shape[:2] != self.mask.shape:
            logger.warning("Image and mask size mismatch")
            return image

        # Add alpha channel
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = self.mask

        return bgra

    def get_area(self) -> int:
        """Get body area in pixels.

        Returns:
            Number of body pixels
        """
        return np.count_nonzero(self.mask)

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of body.

        Returns:
            Tuple of (x, y, w, h) or None if mask is empty
        """
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)

    def reset(self) -> "BodyMask":
        """Reset mask to original state.

        Returns:
            Self for chaining
        """
        self.mask = self._original_mask.copy()
        self.refined = False
        logger.debug("Mask reset to original")
        return self
