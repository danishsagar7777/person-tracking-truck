import numpy as np

from src.tracking.motion_compensation import (
    CameraMotionEstimator,
)


def test_camera_motion_estimator_initialization():
    estimator = CameraMotionEstimator()

    assert estimator.max_features == 500
    assert estimator.match_ratio == 0.75
    assert estimator.min_matches == 10


def test_first_frame_returns_identity_transform():
    estimator = CameraMotionEstimator()

    frame = np.zeros(
        (480, 848, 3),
        dtype=np.uint8,
    )

    transform, inliers = estimator.estimate(frame)

    assert transform.shape == (2, 3)
    assert inliers == 0


def test_reset_clears_previous_frame():
    estimator = CameraMotionEstimator()

    frame = np.zeros(
        (480, 848, 3),
        dtype=np.uint8,
    )

    estimator.estimate(frame)

    assert estimator.previous_gray is not None

    estimator.reset()

    assert estimator.previous_gray is None
