from typing import Dict, List

from ultralytics import YOLO


class OpenVINOPersonDetector:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        self.model = YOLO(model_path)
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame) -> List[Dict]:
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())

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
