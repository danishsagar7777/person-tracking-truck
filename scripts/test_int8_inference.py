from pathlib import Path

import cv2

from src.detection.int8_detector import INT8PersonDetector


MODEL_PATH = "models/yolo26n_int8/yolo26n_int8.xml"
VIDEO_PATH = "data/input/truck_video.mp4"


def main():
    print("=" * 60)
    print("INT8 SINGLE-FRAME INFERENCE TEST")
    print("=" * 60)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"INT8 model not found: {MODEL_PATH}"
        )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read first video frame.")

    detector = INT8PersonDetector(
        model_path=MODEL_PATH,
        image_size=640,
        confidence=0.25,
        iou=0.45,
        device="CPU",
    )

    detections = detector.detect(frame)

    print()
    print(f"Frame shape: {frame.shape}")
    print(f"Persons detected: {len(detections)}")
    print()

    for index, detection in enumerate(detections):
        print(
            f"Person {index + 1}: "
            f"bbox={detection['bbox']} "
            f"confidence={detection['confidence']:.4f}"
        )

    print()
    print("=" * 60)
    print("INT8 INFERENCE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
