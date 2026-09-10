from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from src.detection.int8_detector import INT8PersonDetector


INPUT_VIDEO = "data/input/truck_video.mp4"
INT8_MODEL = "models/yolo26n_int8/yolo26n_int8.xml"

OUTPUT_VIDEO = "results/videos/int8_bytetrack.mp4"


def main():
    input_path = Path(INPUT_VIDEO)
    model_path = Path(INT8_MODEL)
    output_path = Path(OUTPUT_VIDEO)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"INT8 model not found: {model_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INT8 + BYTE TRACK PERSON TRACKING")
    print("=" * 70)

    print(f"Input video : {input_path}")
    print(f"INT8 model  : {model_path}")
    print(f"Output      : {output_path}")
    print()

    # ---------------------------------------------------------
    # INT8 detector
    # ---------------------------------------------------------
    detector = INT8PersonDetector(
        model_path=str(model_path),
        image_size=640,
        confidence=0.25,
        iou=0.45,
        device="CPU",
    )

    # ---------------------------------------------------------
    # Video input
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video resolution : {width}x{height}")
    print(f"Video FPS        : {fps:.2f}")
    print(f"Total frames     : {total_frames}")
    print()

    # ---------------------------------------------------------
    # ByteTrack
    #
    # We use Ultralytics ByteTrack as the tracker.
    # ---------------------------------------------------------
    tracker = YOLO("yolo26n.pt")

    # ---------------------------------------------------------
    # Video writer
    # ---------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    frame_count = 0
    total_detections = 0
    unique_ids = set()

    inference_times = []

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # -----------------------------------------------------
        # INT8 detection
        # -----------------------------------------------------
        inference_start = time.perf_counter()

        detections = detector.detect(frame)

        inference_end = time.perf_counter()

        inference_time = inference_end - inference_start
        inference_times.append(inference_time)

        # -----------------------------------------------------
        # Convert detections to Ultralytics tracker format
        #
        # [x1, y1, x2, y2, confidence, class_id]
        # -----------------------------------------------------
        tracker_input = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            tracker_input.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    class_id,
                ]
            )

        # -----------------------------------------------------
        # ByteTrack
        # -----------------------------------------------------
        if tracker_input:
            results = tracker.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)

                    for track_id in ids:
                        unique_ids.add(int(track_id))

                    total_detections += len(ids)

                    # Draw tracking results
                    annotated_frame = results[0].plot()

                else:
                    annotated_frame = frame.copy()

            else:
                annotated_frame = frame.copy()

        else:
            annotated_frame = frame.copy()

        # -----------------------------------------------------
        # Add information overlay
        # -----------------------------------------------------
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Unique IDs: {len(unique_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        writer.write(annotated_frame)

        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - start_time

            processing_fps = frame_count / elapsed

            print(
                f"Processed {frame_count} frames | "
                f"Unique IDs: {len(unique_ids)} | "
                f"FPS: {processing_fps:.2f}"
            )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    cap.release()
    writer.release()

    total_time = time.perf_counter() - start_time

    processing_fps = (
        frame_count / total_time
        if total_time > 0
        else 0.0
    )

    average_inference_ms = (
        sum(inference_times) / len(inference_times) * 1000
        if inference_times
        else 0.0
    )

    print()
    print("=" * 70)
    print("INT8 + BYTE TRACK RESULTS")
    print("=" * 70)

    print(f"Frames processed       : {frame_count}")
    print(f"Total tracked detections: {total_detections}")
    print(f"Unique track IDs       : {len(unique_ids)}")
    print(
        f"Average inference     : "
        f"{average_inference_ms:.2f} ms"
    )
    print(
        f"Processing FPS        : "
        f"{processing_fps:.2f}"
    )
    print(f"Output                : {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
