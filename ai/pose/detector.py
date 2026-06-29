"""
MediaPipe Pose Detector

Real-time body landmark detection using MediaPipe Pose framework.
Detects 33 body landmarks for full-body pose estimation.

Author: AI Computer Vision Team
"""

import logging
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from .landmark import Landmark
from .smoother import PoseSmoother

logger = logging.getLogger(__name__)

# MediaPipe pose landmark names (33 landmarks)
POSE_LANDMARKS = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]


class PoseDetector:
    """MediaPipe-based pose detector for body landmark detection.

    This class provides real-time body pose estimation using MediaPipe's
    pre-trained pose model. It detects 33 body landmarks and provides
    optional smoothing to reduce jitter.

    The 33 landmarks cover:
    - Face: nose, eyes, ears, mouth
    - Upper body: shoulders, elbows, wrists, hands
    - Lower body: hips, knees, ankles, feet

    Attributes:
        confidence_threshold: Minimum confidence to report landmark
        smoothing_enabled: Whether to apply Kalman smoothing
        static_image_mode: Process each frame independently

    Example:
        >>> detector = PoseDetector(confidence_threshold=0.5)
        >>> landmarks = detector.detect(frame)
        >>> for landmark in landmarks:
        ...     if landmark.is_visible():
        ...         x, y = landmark.to_pixel_coords(width, height)
        ...         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smoothing_enabled: bool = True,
        smoothing_factor: float = 0.3,
        static_image_mode: bool = False,
        model_complexity: int = 1,
    ):
        """Initialize pose detector.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0)
            smoothing_enabled: Enable jitter smoothing
            smoothing_factor: Smoothing factor for EMA (0.0-1.0)
            static_image_mode: Process each frame independently
            model_complexity: Model complexity (0=light, 1=full)

        Raises:
            RuntimeError: If MediaPipe initialization fails
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_enabled = smoothing_enabled
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity

        try:
            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold,
            )

            logger.info(
                f"PoseDetector initialized (complexity={model_complexity}, "
                f"threshold={confidence_threshold})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise RuntimeError(f"Cannot initialize pose detector: {e}")

        # Initialize smoother
        if smoothing_enabled:
            self.smoother = PoseSmoother(
                smoothing_factor=smoothing_factor, use_kalman=True
            )
        else:
            self.smoother = None

        # Performance tracking
        self._detection_count = 0
        self._last_landmarks: Optional[List[Landmark]] = None

    def detect(self, frame: np.ndarray) -> List[Landmark]:
        """Detect body pose landmarks in frame.

        Args:
            frame: Input frame (BGR format, HxWx3)

        Returns:
            List of detected landmarks
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame received")
            return []

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            results = self.pose.process(rgb_frame)

            landmarks = []

            if results.pose_landmarks:
                h, w, _ = frame.shape

                for idx, mp_landmark in enumerate(results.pose_landmarks.landmark):
                    # Check if landmark is valid
                    if (
                        mp_landmark.x < 0
                        or mp_landmark.x > 1
                        or mp_landmark.y < 0
                        or mp_landmark.y > 1
                    ):
                        continue

                    landmark = Landmark(
                        x=float(mp_landmark.x),
                        y=float(mp_landmark.y),
                        z=float(mp_landmark.z),
                        confidence=float(mp_landmark.visibility),
                        visibility=float(mp_landmark.visibility),
                        name=POSE_LANDMARKS[idx] if idx < len(POSE_LANDMARKS) else f"LANDMARK_{idx}",
                        index=idx,
                    )

                    # Only include if above confidence threshold
                    if landmark.confidence >= self.confidence_threshold:
                        landmarks.append(landmark)

            # Apply smoothing if enabled
            if self.smoothing_enabled and self.smoother:
                landmarks = self.smoother.smooth(landmarks)

            self._detection_count += 1
            self._last_landmarks = landmarks

            return landmarks

        except Exception as e:
            logger.error(f"Error during pose detection: {e}", exc_info=True)
            return []

    def get_landmark_by_name(self, name: str) -> Optional[Landmark]:
        """Get specific landmark by name from last detection.

        Args:
            name: Landmark name (e.g., "NOSE", "LEFT_SHOULDER")

        Returns:
            Landmark object or None if not found
        """
        if self._last_landmarks is None:
            return None

        for landmark in self._last_landmarks:
            if landmark.name == name:
                return landmark

        return None

    def get_landmark_by_index(self, index: int) -> Optional[Landmark]:
        """Get specific landmark by index from last detection.

        Args:
            index: Landmark index (0-32)

        Returns:
            Landmark object or None if not found
        """
        if self._last_landmarks is None:
            return None

        for landmark in self._last_landmarks:
            if landmark.index == index:
                return landmark

        return None

    def get_visible_landmarks(self, threshold: float = 0.5) -> List[Landmark]:
        """Get all visible landmarks above threshold.

        Args:
            threshold: Visibility threshold (0.0-1.0)

        Returns:
            List of visible landmarks
        """
        if self._last_landmarks is None:
            return []

        return [
            lm for lm in self._last_landmarks if lm.is_visible(threshold)
        ]

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Landmark]] = None,
        circle_radius: int = 5,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Draw landmarks and skeleton on frame.

        Args:
            frame: Frame to draw on
            landmarks: Landmarks to draw (uses last detection if None)
            circle_radius: Radius of landmark circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            landmarks = self._last_landmarks or []

        if not landmarks:
            return frame

        h, w, _ = frame.shape

        # Draw landmarks
        for landmark in landmarks:
            x, y = landmark.to_pixel_coords(w, h)
            # Color based on confidence
            intensity = int(landmark.confidence * 255)
            color = (0, intensity, 255 - intensity)
            cv2.circle(frame, (x, y), circle_radius, color, -1)

        # Draw skeleton connections
        connections = [
            (11, 12),  # Shoulders
            (11, 13),  # Left shoulder-elbow
            (13, 15),  # Left elbow-wrist
            (12, 14),  # Right shoulder-elbow
            (14, 16),  # Right elbow-wrist
            (11, 23),  # Left shoulder-hip
            (12, 24),  # Right shoulder-hip
            (23, 24),  # Hips
            (23, 25),  # Left hip-knee
            (25, 27),  # Left knee-ankle
            (24, 26),  # Right hip-knee
            (26, 28),  # Right knee-ankle
        ]

        for start_idx, end_idx in connections:
            start = self.get_landmark_by_index(start_idx)
            end = self.get_landmark_by_index(end_idx)

            if start and end and start.is_visible() and end.is_visible():
                x1, y1 = start.to_pixel_coords(w, h)
                x2, y2 = end.to_pixel_coords(w, h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

        return frame

    def reset(self) -> None:
        """Reset detector state (call on new person or camera switch)."""
        if self.smoother:
            self.smoother.reset()
        self._last_landmarks = None
        logger.debug("PoseDetector reset")

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            logger.info("PoseDetector released")

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.release()
