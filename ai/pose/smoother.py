"""
Kalman Filter-based Pose Smoother

Reduces jitter in pose detection using Kalman filtering.

Author: AI Computer Vision Team
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .landmark import Landmark

logger = logging.getLogger(__name__)


class KalmanFilterND:
    """N-dimensional Kalman filter for position tracking.

    Implements a simple Kalman filter for reducing jitter in landmark positions.
    Assumes constant velocity model.
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
        initial_state: Optional[np.ndarray] = None,
    ):
        """Initialize Kalman filter.

        Args:
            process_variance: Process noise (model uncertainty)
            measurement_variance: Measurement noise (sensor uncertainty)
            initial_state: Initial state vector
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.initial_state = initial_state
        self.state = initial_state
        self.covariance = np.eye(len(initial_state)) if initial_state is not None else None
        self.initialized = initial_state is not None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement.

        Args:
            measurement: New measurement vector

        Returns:
            Filtered state estimate
        """
        if not self.initialized:
            self.state = measurement.copy()
            self.covariance = np.eye(len(measurement)) * self.measurement_variance
            self.initialized = True
            return self.state.copy()

        # Predict step
        predicted_state = self.state.copy()
        predicted_covariance = (
            self.covariance + np.eye(len(self.state)) * self.process_variance
        )

        # Update step
        innovation = measurement - predicted_state
        innovation_covariance = (
            predicted_covariance + self.measurement_variance
        )
        kalman_gain = predicted_covariance / (innovation_covariance + 1e-6)

        self.state = predicted_state + kalman_gain * innovation
        self.covariance = (np.eye(len(self.state)) - kalman_gain) * predicted_covariance

        return self.state.copy()


class PoseSmoother:
    """Smooths pose landmarks to reduce jitter using Kalman filtering.

    Maintains separate Kalman filters for each landmark's x, y, z coordinates.
    This reduces jitter while preserving true motion.

    Attributes:
        smoothing_factor: Alpha blending factor (0.0-1.0)
        use_kalman: Whether to use Kalman filtering

    Example:
        >>> smoother = PoseSmoother(smoothing_factor=0.3)
        >>> smooth_landmarks = smoother.smooth(landmarks)
    """

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        use_kalman: bool = True,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
    ):
        """Initialize pose smoother.

        Args:
            smoothing_factor: EMA smoothing factor (0.0-1.0)
            use_kalman: Whether to use Kalman filtering
            process_variance: Kalman process noise
            measurement_variance: Kalman measurement noise
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.use_kalman = use_kalman
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # Store previous landmarks for EMA
        self._previous_landmarks: Optional[List[Landmark]] = None
        self._kalman_filters: dict = {}  # Index -> KalmanFilterND

        logger.info(
            f"PoseSmoother initialized (factor={smoothing_factor}, "
            f"kalman={use_kalman})"
        )

    def smooth(self, landmarks: List[Landmark]) -> List[Landmark]:
        """Smooth landmarks to reduce jitter.

        Args:
            landmarks: List of detected landmarks

        Returns:
            List of smoothed landmarks
        """
        if not landmarks:
            return landmarks

        # First frame - just store
        if self._previous_landmarks is None:
            self._previous_landmarks = landmarks
            # Initialize Kalman filters
            if self.use_kalman:
                for landmark in landmarks:
                    self._kalman_filters[landmark.index] = KalmanFilterND(
                        process_variance=self.process_variance,
                        measurement_variance=self.measurement_variance,
                        initial_state=np.array([landmark.x, landmark.y, landmark.z]),
                    )
            return landmarks

        smoothed_landmarks = []

        for i, landmark in enumerate(landmarks):
            if i >= len(self._previous_landmarks):
                smoothed_landmarks.append(landmark)
                continue

            prev = self._previous_landmarks[i]

            if self.use_kalman:
                # Use Kalman filter
                measurement = np.array([landmark.x, landmark.y, landmark.z])

                if landmark.index not in self._kalman_filters:
                    self._kalman_filters[landmark.index] = KalmanFilterND(
                        process_variance=self.process_variance,
                        measurement_variance=self.measurement_variance,
                        initial_state=measurement,
                    )

                filtered_state = self._kalman_filters[landmark.index].update(
                    measurement
                )
                smooth_x, smooth_y, smooth_z = filtered_state
            else:
                # Use exponential moving average (EMA)
                smooth_x = (
                    prev.x * (1 - self.smoothing_factor)
                    + landmark.x * self.smoothing_factor
                )
                smooth_y = (
                    prev.y * (1 - self.smoothing_factor)
                    + landmark.y * self.smoothing_factor
                )
                smooth_z = (
                    prev.z * (1 - self.smoothing_factor)
                    + landmark.z * self.smoothing_factor
                )

            smoothed_landmark = Landmark(
                x=float(smooth_x),
                y=float(smooth_y),
                z=float(smooth_z),
                confidence=landmark.confidence,
                visibility=landmark.visibility,
                name=landmark.name,
                index=landmark.index,
            )

            smoothed_landmarks.append(smoothed_landmark)

        self._previous_landmarks = smoothed_landmarks
        return smoothed_landmarks

    def reset(self) -> None:
        """Reset smoother state (call on new person or camera switch)."""
        self._previous_landmarks = None
        self._kalman_filters.clear()
        logger.debug("PoseSmoother reset")
