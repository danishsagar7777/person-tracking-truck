from typing import Optional, Tuple

import cv2
import numpy as np


class CameraMotionEstimator:
    """
    Estimate global camera motion between consecutive frames.

    The estimator uses ORB features and a RANSAC-based
    affine transformation.

    This is a first baseline for a moving-camera tracker.
    """

    def __init__(
        self,
        max_features: int = 500,
        match_ratio: float = 0.75,
        min_matches: int = 10,
    ):
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.min_matches = min_matches

        self.orb = cv2.ORB_create(
            nfeatures=self.max_features
        )

        self.previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the previous frame."""
        self.previous_gray = None

    def estimate(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Estimate frame-to-frame camera motion.

        Returns:
            transform:
                2x3 affine transformation matrix.

            inlier_count:
                Number of RANSAC inliers.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        if self.previous_gray is None:
            self.previous_gray = gray
            return np.eye(
                2,
                3,
                dtype=np.float32,
            ), 0

        keypoints_previous, descriptors_previous = (
            self.orb.detectAndCompute(
                self.previous_gray,
                None,
            )
        )

        keypoints_current, descriptors_current = (
            self.orb.detectAndCompute(
                gray,
                None,
            )
        )

        if (
            descriptors_previous is None
            or descriptors_current is None
        ):
            self.previous_gray = gray

            return np.eye(
                2,
                3,
                dtype=np.float32,
            ), 0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        matches = matcher.knnMatch(
            descriptors_previous,
            descriptors_current,
            k=2,
        )

        good_matches = []

        for pair in matches:
            if len(pair) != 2:
                continue

            first, second = pair

            if first.distance < self.match_ratio * second.distance:
                good_matches.append(first)

        if len(good_matches) < self.min_matches:
            self.previous_gray = gray

            return np.eye(
                2,
                3,
                dtype=np.float32,
            ), 0

        previous_points = np.float32(
            [
                keypoints_previous[m.queryIdx].pt
                for m in good_matches
            ]
        )

        current_points = np.float32(
            [
                keypoints_current[m.trainIdx].pt
                for m in good_matches
            ]
        )

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        self.previous_gray = gray

        if transform is None:
            return np.eye(
                2,
                3,
                dtype=np.float32,
            ), 0

        inlier_count = int(
            inlier_mask.sum()
        ) if inlier_mask is not None else 0

        return transform, inlier_count
