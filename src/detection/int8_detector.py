from typing import Dict, List, Tuple

import cv2
import numpy as np
import openvino as ov


class INT8PersonDetector:
    """
    Native OpenVINO INT8 person detector.

    This detector directly loads the NNCF-quantized OpenVINO model
    instead of going through Ultralytics YOLO.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "CPU",
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.device = device

        self.core = ov.Core()

        self.model = self.core.read_model(model_path)

        self.compiled_model = self.core.compile_model(
            self.model,
            device,
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image while preserving aspect ratio.

        Returns:
            resized_image
            scale
            padding (pad_x, pad_y)
        """

        height, width = image.shape[:2]

        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_width = self.image_size - new_width
        pad_height = self.image_size - new_height

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return (
            padded,
            scale,
            (float(pad_left), float(pad_top)),
        )

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Convert OpenCV BGR frame into YOLO-style NCHW float32 tensor.
        """

        image, scale, padding = self.letterbox(frame)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        return image, scale, padding

    def decode(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[float, float],
    ) -> List[Dict]:
        """
        Decode YOLO output with shape:

            [1, 84, 8400]

        into person detections.
        """

        original_height, original_width = original_shape

        predictions = output[0]

        # Convert:
        #
        # [84, 8400]
        #
        # to:
        #
        # [8400, 84]
        predictions = predictions.T

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]

        person_scores = class_scores[:, self.PERSON_CLASS_ID]

        valid = person_scores >= self.confidence

        boxes = boxes[valid]
        scores = person_scores[valid]

        if len(boxes) == 0:
            return []

        detections = []

        pad_x, pad_y = padding

        for box, score in zip(boxes, scores):

            center_x, center_y, width, height = box

            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            # Remove letterbox padding.
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clamp coordinates to original frame.
            x1 = max(0.0, min(x1, original_width))
            y1 = max(0.0, min(y1, original_height))
            x2 = max(0.0, min(x2, original_width))
            y2 = max(0.0, min(y2, original_height))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                    "confidence": float(score),
                    "class_id": self.PERSON_CLASS_ID,
                    "class_name": "person",
                }
            )

        # NMS.
        if not detections:
            return []

        nms_boxes = [
            [
                detection["bbox"][0],
                detection["bbox"][1],
                detection["bbox"][2] - detection["bbox"][0],
                detection["bbox"][3] - detection["bbox"][1],
            ]
            for detection in detections
        ]

        nms_scores = [
            detection["confidence"]
            for detection in detections
        ]

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            nms_scores,
            self.confidence,
            self.iou,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).reshape(-1)

        return [
            detections[int(index)]
            for index in indices
        ]

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Dict]:
        """
        Run INT8 inference on one frame.
        """

        tensor, scale, padding = self.preprocess(frame)

        result = self.compiled_model(
            {
                self.input_layer: tensor,
            }
        )

        output = result[self.output_layer]

        return self.decode(
            output,
            frame.shape[:2],
            scale,
            padding,
        )
