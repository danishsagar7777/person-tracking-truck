from src.tracking.tracker import PersonTracker


def test_tracker_initialization():
    tracker = PersonTracker(
        model_name="yolo26n.pt",
        image_size=640,
        confidence=0.25,
        iou=0.45,
        tracker_config="bytetrack.yaml",
    )

    assert tracker.image_size == 640
    assert tracker.confidence == 0.25
    assert tracker.iou == 0.45
    assert tracker.tracker_config == "bytetrack.yaml"
