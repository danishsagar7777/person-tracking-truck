from src.utils.tracking_metrics import TrackingMetrics


def test_tracking_metrics_initialization():
    metrics = TrackingMetrics()

    result = metrics.calculate()

    assert result["frames"] == 0
    assert result["total_tracked_detections"] == 0
    assert result["unique_track_ids"] == 0


def test_tracking_metrics_single_track():
    metrics = TrackingMetrics()

    metrics.update(
        0,
        [
            {
                "track_id": 1,
                "bbox": [10, 10, 50, 100],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person",
            }
        ],
    )

    metrics.update(
        1,
        [
            {
                "track_id": 1,
                "bbox": [12, 10, 52, 100],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person",
            }
        ],
    )

    metrics.update(
        2,
        [
            {
                "track_id": 1,
                "bbox": [14, 10, 54, 100],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person",
            }
        ],
    )

    result = metrics.calculate()

    assert result["frames"] == 3
    assert result["total_tracked_detections"] == 3
    assert result["unique_track_ids"] == 1
    assert result["average_track_length_frames"] == 3.0


def test_tracking_metrics_multiple_ids():
    metrics = TrackingMetrics()

    metrics.update(
        0,
        [
            {
                "track_id": 1,
                "bbox": [10, 10, 50, 100],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person",
            },
            {
                "track_id": 2,
                "bbox": [100, 10, 150, 100],
                "confidence": 0.8,
                "class_id": 0,
                "class_name": "person",
            },
        ],
    )

    result = metrics.calculate()

    assert result["unique_track_ids"] == 2
    assert result["maximum_active_tracks"] == 2
