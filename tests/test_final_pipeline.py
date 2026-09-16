import numpy as np

from scripts.run_final_pipeline import (
    detections_to_boxes,
    transform_bbox,
    valid_camera_transform,
)


def test_detections_to_boxes_person():
    detections = [
        {
            "bbox": [10, 20, 100, 200],
            "confidence": 0.9,
            "class_id": 0,
        }
    ]

    boxes = detections_to_boxes(
        detections,
        (480, 848, 3),
    )

    assert boxes.data.shape == (1, 6)
    assert np.isclose(float(boxes.data[0, 4]), 0.9)
    assert float(boxes.data[0, 5]) == 0.0


def test_detections_to_boxes_empty():
    boxes = detections_to_boxes(
        [],
        (480, 848, 3),
    )

    assert boxes.data.shape == (0, 6)


def test_transform_bbox_identity():
    bbox = [10, 20, 100, 200]

    identity = np.eye(2, 3, dtype=np.float32)

    result = transform_bbox(
        bbox,
        identity,
        848,
        480,
    )

    assert result is not None
    assert np.allclose(result, bbox)


def test_valid_camera_transform_identity():
    identity = np.eye(2, 3, dtype=np.float32)

    assert valid_camera_transform(identity, 20) is True


def test_invalid_camera_transform_translation():
    transform = np.array(
        [
            [1.0, 0.0, 200.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert valid_camera_transform(transform, 20) is False
