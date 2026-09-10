import argparse
import time
from pathlib import Path

import cv2
import yaml

from src.tracking.tracker import PersonTracker
from src.visualization.tracking_visualizer import TrackingVisualizer


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack person tracking."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    input_path = config["video"]["input"]
    output_path = config["video"]["output_tracking"]

    model_name = config["model"]["name"]
    image_size = config["model"]["imgsz"]
    confidence = config["model"]["confidence"]
    iou = config["model"]["iou"]

    tracker_config = config["tracking"]["tracker"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if input_fps <= 0:
        input_fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        input_fps,
        (width, height),
    )

    tracker = PersonTracker(
        model_name=model_name,
        image_size=image_size,
        confidence=confidence,
        iou=iou,
        tracker_config=tracker_config,
    )

    visualizer = TrackingVisualizer()

    frame_number = 0
    total_detections = 0
    total_inference_time = 0.0

    unique_track_ids = set()

    start_time = time.perf_counter()

    while True:
        success, frame = cap.read()

        if not success:
            break

        inference_start = time.perf_counter()

        tracks = tracker.track(frame)

        inference_time = time.perf_counter() - inference_start

        total_inference_time += inference_time

        total_detections += len(tracks)

        for track in tracks:
            unique_track_ids.add(track["track_id"])

        annotated_frame = visualizer.draw(
            frame,
            tracks,
            frame_number,
        )

        writer.write(annotated_frame)

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number} frames | "
                f"Active tracks: {len(tracks)} | "
                f"Unique IDs: {len(unique_track_ids)}"
            )

    elapsed = time.perf_counter() - start_time

    cap.release()
    writer.release()

    avg_inference_ms = (
        total_inference_time / frame_number * 1000
        if frame_number > 0
        else 0.0
    )

    processing_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0.0
    )

    avg_tracks_per_frame = (
        total_detections / frame_number
        if frame_number > 0
        else 0.0
    )

    print("\n========== TRACKING RESULTS ==========")
    print(f"Frames processed: {frame_number}")
    print(f"Total tracked detections: {total_detections}")
    print(f"Unique track IDs: {len(unique_track_ids)}")
    print(
        f"Average tracked persons/frame: "
        f"{avg_tracks_per_frame:.3f}"
    )
    print(
        f"Average inference time: "
        f"{avg_inference_ms:.2f} ms"
    )
    print(
        f"Processing FPS: "
        f"{processing_fps:.2f}"
    )
    print(f"Tracker: {tracker_config}")
    print(f"Output: {output_path}")
    print("======================================\n")


if __name__ == "__main__":
    main()
