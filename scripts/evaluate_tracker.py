import argparse
import time
from pathlib import Path

import cv2
import yaml

from src.tracking.tracker import PersonTracker
from src.utils.tracking_metrics import TrackingMetrics


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_tracker(config_path, tracker_name, output_report):
    config = load_config(config_path)

    input_path = config["video"]["input"]

    model_name = config["model"]["name"]
    image_size = config["model"]["imgsz"]
    confidence = config["model"]["confidence"]
    iou = config["model"]["iou"]

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_path}"
        )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = PersonTracker(
        model_name=model_name,
        image_size=image_size,
        confidence=confidence,
        iou=iou,
        tracker_config=tracker_name,
    )

    metrics = TrackingMetrics()

    total_inference_time = 0.0
    processed_frames = 0

    start_time = time.perf_counter()

    while True:
        success, frame = cap.read()

        if not success:
            break

        inference_start = time.perf_counter()

        tracks = tracker.track(frame)

        inference_time = time.perf_counter() - inference_start

        total_inference_time += inference_time

        metrics.update(
            processed_frames,
            tracks,
        )

        processed_frames += 1

        if processed_frames % 100 == 0:
            print(
                f"Processed {processed_frames} frames | "
                f"Active tracks: {len(tracks)} | "
                f"Unique IDs: {len(metrics.unique_track_ids)}"
            )

    elapsed = time.perf_counter() - start_time

    cap.release()

    result = metrics.calculate()

    if processed_frames > 0:
        average_inference_ms = (
            total_inference_time / processed_frames
        ) * 1000
    else:
        average_inference_ms = 0.0

    processing_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0.0
    )

    result["tracker"] = tracker_name
    result["model"] = model_name
    result["average_inference_ms"] = average_inference_ms
    result["processing_fps"] = processing_fps

    output_report = Path(output_report)
    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_report,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("# Tracker Evaluation Report\n\n")

        file.write(f"## Tracker\n\n")
        file.write(f"- Tracker: `{tracker_name}`\n")
        file.write(f"- Model: `{model_name}`\n")
        file.write(f"- Frames: {result['frames']}\n\n")

        file.write("## Performance\n\n")

        file.write(
            f"- Total tracked detections: "
            f"{result['total_tracked_detections']}\n"
        )

        file.write(
            f"- Unique track IDs: "
            f"{result['unique_track_ids']}\n"
        )

        file.write(
            f"- Average tracks/frame: "
            f"{result['average_tracks_per_frame']:.3f}\n"
        )

        file.write(
            f"- Average track length: "
            f"{result['average_track_length_frames']:.2f} frames\n"
        )

        file.write(
            f"- Shortest track: "
            f"{result['shortest_track_frames']} frames\n"
        )

        file.write(
            f"- Longest track: "
            f"{result['longest_track_frames']} frames\n"
        )

        file.write(
            f"- Maximum active tracks: "
            f"{result['maximum_active_tracks']}\n"
        )

        file.write(
            f"- Average inference time: "
            f"{result['average_inference_ms']:.2f} ms\n"
        )

        file.write(
            f"- Processing FPS: "
            f"{result['processing_fps']:.2f}\n"
        )

        file.write("\n## Track Lifetimes\n\n")

        file.write(
            "| Track ID | First Frame | Last Frame | "
            "Length | Observations |\n"
        )

        file.write(
            "|---:|---:|---:|---:|---:|\n"
        )

        for track in metrics.get_track_table():
            file.write(
                f"| {track['track_id']} | "
                f"{track['first_frame']} | "
                f"{track['last_frame']} | "
                f"{track['track_length_frames']} | "
                f"{track['observations']} |\n"
            )

    print("\n========== EVALUATION RESULTS ==========")

    print(f"Tracker: {tracker_name}")
    print(f"Frames: {result['frames']}")
    print(
        f"Total tracked detections: "
        f"{result['total_tracked_detections']}"
    )
    print(
        f"Unique track IDs: "
        f"{result['unique_track_ids']}"
    )
    print(
        f"Average tracks/frame: "
        f"{result['average_tracks_per_frame']:.3f}"
    )
    print(
        f"Average track length: "
        f"{result['average_track_length_frames']:.2f} frames"
    )
    print(
        f"Shortest track: "
        f"{result['shortest_track_frames']} frames"
    )
    print(
        f"Longest track: "
        f"{result['longest_track_frames']} frames"
    )
    print(
        f"Maximum active tracks: "
        f"{result['maximum_active_tracks']}"
    )
    print(
        f"Average inference: "
        f"{result['average_inference_ms']:.2f} ms"
    )
    print(
        f"Processing FPS: "
        f"{result['processing_fps']:.2f}"
    )

    print(f"Report: {output_report}")

    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a person tracker."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--tracker",
        required=True,
        help="Tracker configuration, e.g. bytetrack.yaml",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output markdown report.",
    )

    args = parser.parse_args()

    evaluate_tracker(
        args.config,
        args.tracker,
        args.output,
    )


if __name__ == "__main__":
    main()
