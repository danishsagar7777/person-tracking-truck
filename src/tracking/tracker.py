from typing import List, Dict

from ultralytics import YOLO


class PersonTracker:
    """YOLO + ByteTrack person tracker."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_name)

        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.tracker_config = tracker_config

    def track(self, frame) -> List[Dict]:
        """
        Detect and track people in one frame.

        Returns:
            [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": 0,
                    "class_name": "person",
                }
            ]
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        result = results[0]

        tracks = []

        if result.boxes is None:
            return tracks

        if result.boxes.id is None:
            return tracks

        track_ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for track_id, bbox, confidence, class_id in zip(
            track_ids,
            boxes,
            confidences,
            classes,
        ):
            if class_id != self.PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = bbox

            tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": "person",
                }
            )

        return tracks
