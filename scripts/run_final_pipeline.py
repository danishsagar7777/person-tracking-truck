import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.int8_detector import INT8PersonDetector
class TrackerArgs:
    """Configuration required by the installed Ultralytics BYTETracker."""

    def __init__(self):
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.fuse_score = True
        self.with_reid = False
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25


from src.tracking.motion_compensation import CameraMotionEstimator


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detections_to_boxes(detections, frame_shape):
    height, width = frame_shape[:2]

    rows = []

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = float(detection["confidence"])
        class_id = float(detection["class_id"])

        rows.append(
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                confidence,
                class_id,
            ]
        )

    if not rows:
        data = np.empty((0, 6), dtype=np.float32)
    else:
        data = np.asarray(rows, dtype=np.float32)

    return Boxes(
        data,
        (height, width),
    )


def affine_to_homogeneous(matrix):
    # CameraMotionEstimator may return:
    # (affine_matrix, inlier_count)
    if isinstance(matrix, (tuple, list)):
        matrix = matrix[0]

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.shape != (2, 3):
        return np.eye(3, dtype=np.float32)

    result = np.eye(3, dtype=np.float32)
    result[:2, :] = matrix

    return result


def transform_bbox(bbox, transform, width, height):
    x1, y1, x2, y2 = bbox

    points = np.array(
        [
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ],
        dtype=np.float32,
    )

    transformed = (transform @ points.T).T

    transformed = transformed[:, :2] / np.maximum(
        transformed[:, 2:3],
        1e-6,
    )

    new_x1 = float(np.min(transformed[:, 0]))
    new_y1 = float(np.min(transformed[:, 1]))
    new_x2 = float(np.max(transformed[:, 0]))
    new_y2 = float(np.max(transformed[:, 1]))

    new_x1 = max(0.0, min(new_x1, width - 1))
    new_y1 = max(0.0, min(new_y1, height - 1))
    new_x2 = max(0.0, min(new_x2, width - 1))
    new_y2 = max(0.0, min(new_y2, height - 1))

    return (
        int(new_x1),
        int(new_y1),
        int(new_x2),
        int(new_y2),
    )


def draw_tracks(frame, tracks, inverse_transform):
    height, width = frame.shape[:2]

    for track in tracks:
        if len(track) < 5:
            continue

        x1, y1, x2, y2 = track[:4]
        track_id = int(track[4])

        bbox = transform_bbox(
            (x1, y1, x2, y2),
            inverse_transform,
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Person ID: {track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Final person tracking pipeline"
    )

    parser.add_argument(
        "--config",
        default="configs/production.yaml",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    source_path = config["source"]["path"]

    model_path = config["detector"]["model"]

    output_video = Path(config["output"]["video"])
    metrics_path = Path(config["output"]["metrics"])

    gmc_enabled = bool(
        config["camera_motion"]["enabled"]
    )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FINAL PERSON TRACKING PIPELINE")
    print("=" * 70)

    print(f"Source: {source_path}")
    print(f"Model: {model_path}")
    print(f"Tracker: {config['tracker']['type']}")
    print(f"GMC enabled: {gmc_enabled}")
    print()

    if not Path(source_path).exists():
        raise FileNotFoundError(
            f"Input video not found: {source_path}"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"INT8 model not found: {model_path}"
        )

    detector = INT8PersonDetector(
        model_path=model_path,
        image_size=config["detector"]["image_size"],
        confidence=config["detector"]["confidence"],
        iou=config["detector"]["iou"],
        device=config["detector"]["device"],
    )

    tracker = BYTETracker(
        TrackerArgs()
    )

    motion_estimator = None

    if gmc_enabled:
        motion_estimator = CameraMotionEstimator()

    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {source_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_video}"
        )

    frame_count = 0
    total_detections = 0
    total_tracked = 0

    inference_times = []
    frame_times = []

    unique_ids = set()
    track_lengths = {}

    previous_to_reference = np.eye(
        3,
        dtype=np.float32,
    )

    total_start = time.perf_counter()

    try:
        while True:
            if (
                args.max_frames is not None
                and frame_count >= args.max_frames
            ):
                break

            read_start = time.perf_counter()

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            height, width = frame.shape[:2]

            current_to_reference = np.eye(
                3,
                dtype=np.float32,
            )

            stabilized_frame = frame

            if gmc_enabled:
                transform = motion_estimator.estimate(
                    frame
                )

                if frame_count > 1:
                    previous_to_current = affine_to_homogeneous(
                        transform
                    )

                    try:
                        current_to_previous = np.linalg.inv(
                            previous_to_current
                        )
                    except np.linalg.LinAlgError:
                        current_to_previous = np.eye(
                            3,
                            dtype=np.float32,
                        )

                    current_to_reference = (
                        previous_to_reference
                        @ current_to_previous
                    )

                    stabilized_frame = cv2.warpAffine(
                        frame,
                        current_to_reference[:2],
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )

                previous_to_reference = (
                    current_to_reference.copy()
                )

            inference_start = time.perf_counter()

            detections = detector.detect(
                stabilized_frame
            )

            inference_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000.0

            inference_times.append(
                inference_ms
            )

            total_detections += len(detections)

            tracker_input = detections_to_boxes(
                detections,
                stabilized_frame.shape,
            )

            tracks = tracker.update(
                tracker_input,
                img=stabilized_frame,
            )

            total_tracked += len(tracks)

            for track in tracks:
                if len(track) < 5:
                    continue

                track_id = int(track[4])

                unique_ids.add(track_id)

                track_lengths[track_id] = (
                    track_lengths.get(track_id, 0)
                    + 1
                )

            try:
                reference_to_current = np.linalg.inv(
                    current_to_reference
                )
            except np.linalg.LinAlgError:
                reference_to_current = np.eye(
                    3,
                    dtype=np.float32,
                )

            output_frame = draw_tracks(
                frame.copy(),
                tracks,
                reference_to_current,
            )

            active_tracks = len(tracks)

            cv2.putText(
                output_frame,
                f"Frame: {frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output_frame,
                f"Persons: {len(detections)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output_frame,
                f"Active tracks: {active_tracks}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output_frame,
                f"GMC: {'ON' if gmc_enabled else 'OFF'}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(output_frame)

            frame_ms = (
                time.perf_counter()
                - read_start
            ) * 1000.0

            frame_times.append(frame_ms)

            if frame_count % 100 == 0:
                elapsed = (
                    time.perf_counter()
                    - total_start
                )

                processing_fps = (
                    frame_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"Frame {frame_count:4d} | "
                    f"Detections: {total_detections:4d} | "
                    f"Tracks: {total_tracked:4d} | "
                    f"FPS: {processing_fps:.2f}"
                )

    finally:
        cap.release()
        writer.release()

    total_time = (
        time.perf_counter()
        - total_start
    )

    processing_fps = (
        frame_count / total_time
        if total_time > 0
        else 0.0
    )

    average_inference = (
        float(np.mean(inference_times))
        if inference_times
        else 0.0
    )

    average_frame_time = (
        float(np.mean(frame_times))
        if frame_times
        else 0.0
    )

    average_track_length = (
        total_tracked / len(unique_ids)
        if unique_ids
        else 0.0
    )

    shortest_track = (
        min(track_lengths.values())
        if track_lengths
        else 0
    )

    longest_track = (
        max(track_lengths.values())
        if track_lengths
        else 0
    )

    print()
    print("=" * 70)
    print("FINAL PIPELINE RESULTS")
    print("=" * 70)
    print(f"Frames processed:       {frame_count}")
    print(f"Person detections:      {total_detections}")
    print(f"Tracked detections:     {total_tracked}")
    print(f"Unique IDs:             {len(unique_ids)}")
    print(
        f"Average persons/frame:  "
        f"{total_detections / frame_count:.3f}"
        if frame_count
        else "Average persons/frame: 0.000"
    )
    print(
        f"Average inference:      "
        f"{average_inference:.2f} ms"
    )
    print(
        f"Average frame time:     "
        f"{average_frame_time:.2f} ms"
    )
    print(
        f"Processing FPS:         "
        f"{processing_fps:.2f}"
    )
    print(
        f"Average track length:   "
        f"{average_track_length:.2f} frames"
    )
    print(f"Shortest track:         {shortest_track}")
    print(f"Longest track:          {longest_track}")
    print(f"Output video:           {output_video}")
    print(f"Metrics:                {metrics_path}")
    print("=" * 70)

    report = f"""# Day 15 — Final Person Tracking Pipeline

## Configuration

- Input video: `{source_path}`
- Detector: OpenVINO INT8
- Model: `{model_path}`
- Tracker: `{config['tracker']['type']}`
- GMC: `{"enabled" if gmc_enabled else "disabled"}`

## Results

| Metric | Result |
|---|---:|
| Frames processed | {frame_count} |
| Person detections | {total_detections} |
| Tracked detections | {total_tracked} |
| Unique track IDs | {len(unique_ids)} |
| Average persons/frame | {(total_detections / frame_count) if frame_count else 0:.3f} |
| Average inference | {average_inference:.2f} ms |
| Average frame time | {average_frame_time:.2f} ms |
| Processing FPS | {processing_fps:.2f} |
| Average track length | {average_track_length:.2f} frames |
| Shortest track | {shortest_track} frames |
| Longest track | {longest_track} frames |

## Output

`{output_video}`

## Purpose

This pipeline combines the validated INT8 person detector,
ByteTrack tracking, and optional camera-motion compensation
into a single video-processing pipeline.
"""

    metrics_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Day 15 completed successfully.")


if __name__ == "__main__":
    main()
