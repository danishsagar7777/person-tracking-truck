from typing import List, Dict

from ultralytics import YOLO


class PersonDetector:
    """YOLO-based person detector."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        self.model = YOLO(model_name)

        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame) -> List[Dict]:
        """
        Detect only people in one frame.

        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": 0,
                    "class_name": "person",
                }
            ]
        """

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())

            # IMPORTANT:
            # Ignore every class except person.
            if class_id != self.PERSON_CLASS_ID:
                continue

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": "person",
                }
            )

        return detections
