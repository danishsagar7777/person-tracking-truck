import cv2


class DetectionVisualizer:
    """Draw person detections on video frames."""

    def draw(
        self,
        frame,
        detections,
        frame_number: int = 0,
    ):
        output = frame.copy()

        person_count = len(detections)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            label = f"person {confidence:.2f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            output,
            f"Frame: {frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Persons: {person_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return output
