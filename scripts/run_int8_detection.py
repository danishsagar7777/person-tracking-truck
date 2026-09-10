from pathlib import Path
import time

import cv2

from src.detection.int8_detector import INT8PersonDetector


MODEL_PATH = "models/yolo26n_int8/yolo26n_int8.xml"
VIDEO_PATH = "data/input/truck_video.mp4"
OUTPUT_PATH = "results/videos/person_detection_int8.mp4"


def main():
    print("=" * 60)
    print("INT8 PERSON DETECTION")
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        source_fps,
        (width, height),
    )

    detector = INT8PersonDetector(
        model_path=MODEL_PATH,
        image_size=640,
        confidence=0.25,
        iou=0.45,
        device="CPU",
    )

    processed_frames = 0
    total_detections = 0
    total_inference_time = 0.0

    while True:

        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        detections = detector.detect(frame)

        inference_time = time.perf_counter() - start

        total_inference_time += inference_time

        processed_frames += 1
        total_detections += len(detections)

        output = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                f"person {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            output,
            f"Frame: {processed_frames}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Persons: {len(detections)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(output)

    cap.release()
    writer.release()

    if processed_frames > 0:

        average_inference_ms = (
            total_inference_time
            / processed_frames
            * 1000
        )

        processing_fps = (
            processed_frames
            / total_inference_time
        )

        average_persons = (
            total_detections
            / processed_frames
        )

    else:
        average_inference_ms = 0.0
        processing_fps = 0.0
        average_persons = 0.0

    print()
    print("=" * 60)
    print("INT8 RESULTS")
    print("=" * 60)
    print(f"Frames processed       : {processed_frames}")
    print(f"Source frames          : {total_frames}")
    print(f"Total person detections: {total_detections}")
    print(f"Average persons/frame  : {average_persons:.3f}")
    print(
        f"Average inference      : "
        f"{average_inference_ms:.2f} ms"
    )
    print(f"Processing FPS         : {processing_fps:.2f}")
    print(f"Output                 : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
