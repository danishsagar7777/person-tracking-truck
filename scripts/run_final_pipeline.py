import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.int8_detector import INT8PersonDetector
from src.tracking.motion_compensation import CameraMotionEstimator


# ============================================================
# BYTE TRACK CONFIGURATION
# ============================================================

class TrackerArgs:
    """
    Configuration required by the installed Ultralytics
    BYTETracker implementation.
    """

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


# ============================================================
# CONFIG
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# DETECTIONS -> ULTRALYTICS BOXES
# ============================================================

def detections_to_boxes(detections, frame_shape):
    """
    Convert detector detections into an Ultralytics Boxes object.

    Expected detection format:

        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int
        }
    """

    height, width = frame_shape[:2]

    rows = []

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        confidence = float(
            detection["confidence"]
        )

        class_id = float(
            detection["class_id"]
        )

        # Clamp coordinates to image bounds.
        x1 = max(0.0, min(float(x1), width - 1))
        y1 = max(0.0, min(float(y1), height - 1))
        x2 = max(0.0, min(float(x2), width - 1))
        y2 = max(0.0, min(float(y2), height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        rows.append(
            [
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id,
            ]
        )

    if not rows:
        data = np.empty(
            (0, 6),
            dtype=np.float32,
        )
    else:
        data = np.asarray(
            rows,
            dtype=np.float32,
        )

    return Boxes(
        data,
        (height, width),
    )


# ============================================================
# AFFINE TRANSFORM HELPERS
# ============================================================

def affine_to_homogeneous(matrix):
    """
    Convert a 2x3 affine matrix to a 3x3 homogeneous matrix.

    CameraMotionEstimator returns:

        transform, inlier_count

    or an affine matrix depending on implementation.
    """

    if isinstance(matrix, (tuple, list)):
        matrix = matrix[0]

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.shape != (2, 3):
        return np.eye(
            3,
            dtype=np.float32,
        )

    result = np.eye(
        3,
        dtype=np.float32,
    )

    result[:2, :] = matrix

    return result


def transform_bbox(
    bbox,
    transform,
    width,
    height,
):
    """
    Transform a bounding box using a 3x3 transformation matrix.
    """

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

    transformed = (
        transform @ points.T
    ).T

    denominator = np.maximum(
        transformed[:, 2:3],
        1e-6,
    )

    transformed = (
        transformed[:, :2]
        / denominator
    )

    new_x1 = float(
        np.min(transformed[:, 0])
    )

    new_y1 = float(
        np.min(transformed[:, 1])
    )

    new_x2 = float(
        np.max(transformed[:, 0])
    )

    new_y2 = float(
        np.max(transformed[:, 1])
    )

    new_x1 = max(
        0.0,
        min(new_x1, width - 1),
    )

    new_y1 = max(
        0.0,
        min(new_y1, height - 1),
    )

    new_x2 = max(
        0.0,
        min(new_x2, width - 1),
    )

    new_y2 = max(
        0.0,
        min(new_y2, height - 1),
    )

    return (
        float(new_x1),
        float(new_y1),
        float(new_x2),
        float(new_y2),
    )


# ============================================================
# GMC TRANSFORM FOR DETECTIONS
# ============================================================

def transform_detections(
    detections,
    transform,
    frame_shape,
):
    """
    Transform detector bounding boxes into the GMC reference
    coordinate system.

    IMPORTANT:

    We transform only the bounding boxes.

    We DO NOT warp the image.

    YOLO therefore always receives the original frame.
    """

    height, width = frame_shape[:2]

    transformed_detections = []

    for detection in detections:

        bbox = transform_bbox(
            detection["bbox"],
            transform,
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            continue

        transformed_detection = dict(
            detection
        )

        transformed_detection["bbox"] = [
            x1,
            y1,
            x2,
            y2,
        ]

        transformed_detections.append(
            transformed_detection
        )

    return transformed_detections


# ============================================================
# TRACK DRAWING
# ============================================================

def draw_tracks(
    frame,
    tracks,
    inverse_transform,
):
    """
    Convert tracker/reference coordinates back into original
    image coordinates and draw the tracked persons.
    """

    height, width = frame.shape[:2]

    if tracks is None:
        return frame

    for track in tracks:

        if len(track) < 5:
            continue

        x1, y1, x2, y2 = track[:4]

        track_id = int(
            track[4]
        )

        bbox = transform_bbox(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            inverse_transform,
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

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
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# GMC SANITY CHECK
# ============================================================

def valid_camera_transform(
    transform,
    inlier_count,
):
    """
    Reject obviously unstable ORB/GMC transforms.

    A moving truck camera should normally produce a reasonable
    frame-to-frame affine transform.

    If the estimated transform is obviously unstable, use
    identity for that frame instead of damaging tracking.
    """

    if inlier_count < 10:
        return False

    matrix = np.asarray(
        transform,
        dtype=np.float32,
    )

    if matrix.shape != (2, 3):
        return False

    linear = matrix[:, :2]

    determinant = float(
        np.linalg.det(linear)
    )

    if not np.isfinite(determinant):
        return False

    scale = np.sqrt(
        abs(determinant)
    )

    if scale < 0.75 or scale > 1.35:
        return False

    translation_x = abs(
        float(matrix[0, 2])
    )

    translation_y = abs(
        float(matrix[1, 2])
    )

    if translation_x > 120.0:
        return False

    if translation_y > 120.0:
        return False

    return True


# ============================================================
# MAIN
# ============================================================

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

    parser.add_argument(
        "--disable-gmc",
        action="store_true",
        help="Disable camera-motion compensation",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    source_path = config[
        "source"
    ]["path"]

    model_path = config[
        "detector"
    ]["model"]

    output_video = Path(
        config["output"]["video"]
    )

    metrics_path = Path(
        config["output"]["metrics"]
    )

    configured_gmc = bool(
        config[
            "camera_motion"
        ]["enabled"]
    )

    gmc_enabled = (
        configured_gmc
        and not args.disable_gmc
    )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print("=" * 70)
    print("FINAL PERSON TRACKING PIPELINE")
    print("=" * 70)

    print(
        f"Source: {source_path}"
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Tracker: "
        f"{config['tracker']['type']}"
    )

    print(
        f"GMC enabled: "
        f"{gmc_enabled}"
    )

    if args.disable_gmc:
        print(
            "GMC override: DISABLED"
        )

    print()

    # --------------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------------

    if not Path(
        source_path
    ).exists():

        raise FileNotFoundError(
            f"Input video not found: "
            f"{source_path}"
        )

    if not Path(
        model_path
    ).exists():

        raise FileNotFoundError(
            f"INT8 model not found: "
            f"{model_path}"
        )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    print(
        "Loading OpenVINO INT8 detector..."
    )

    detector = INT8PersonDetector(
        model_path=model_path,
        image_size=config[
            "detector"
        ]["image_size"],
        confidence=config[
            "detector"
        ]["confidence"],
        iou=config[
            "detector"
        ]["iou"],
        device=config[
            "detector"
        ]["device"],
    )

    print(
        "INT8 detector loaded."
    )

    # --------------------------------------------------------
    # TRACKER
    # --------------------------------------------------------

    print(
        "Creating ByteTrack..."
    )

    tracker = BYTETracker(
        TrackerArgs()
    )

    print(
        "ByteTrack created."
    )

    # --------------------------------------------------------
    # GMC
    # --------------------------------------------------------

    motion_estimator = None

    if gmc_enabled:

        print(
            "Creating ORB camera-motion estimator..."
        )

        motion_estimator = (
            CameraMotionEstimator()
        )

        print(
            "ORB GMC enabled."
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        source_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{source_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_source_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()

    print("Video information:")
    print(
        f"  Width        : "
        f"{frame_width}"
    )

    print(
        f"  Height       : "
        f"{frame_height}"
    )

    print(
        f"  FPS          : "
        f"{fps:.2f}"
    )

    print(
        f"  Total frames : "
        f"{total_source_frames}"
    )

    print()

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output "
            f"video: {output_video}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    frame_count = 0

    total_detections = 0

    total_tracked = 0

    inference_times = []

    frame_times = []

    unique_ids = set()

    track_lengths = {}

    gmc_valid_frames = 0

    gmc_invalid_frames = 0

    gmc_inliers = []

    # --------------------------------------------------------
    # GMC REFERENCE TRANSFORM
    # --------------------------------------------------------

    previous_to_reference = np.eye(
        3,
        dtype=np.float32,
    )

    total_start = time.perf_counter()

    print(
        "Processing video..."
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    try:

        while True:

            if (
                args.max_frames
                is not None
                and frame_count
                >= args.max_frames
            ):
                break

            frame_start = (
                time.perf_counter()
            )

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            height, width = (
                frame.shape[:2]
            )

            # ------------------------------------------------
            # DEFAULT: IDENTITY TRANSFORM
            # ------------------------------------------------

            current_to_reference = np.eye(
                3,
                dtype=np.float32,
            )

            inlier_count = 0

            # ------------------------------------------------
            # GMC
            #
            # IMPORTANT:
            # We estimate camera motion but DO NOT warp frame.
            # YOLO always sees original frame.
            # ------------------------------------------------

            if gmc_enabled:

                transform = (
                    motion_estimator.estimate(
                        frame
                    )
                )

                affine_matrix = (
                    transform[0]
                    if isinstance(
                        transform,
                        (tuple, list),
                    )
                    else transform
                )

                if frame_count > 1:

                    valid = (
                        valid_camera_transform(
                            affine_matrix,
                            (
                                transform[1]
                                if isinstance(
                                    transform,
                                    (tuple, list),
                                )
                                else 0
                            ),
                        )
                    )

                    if valid:

                        previous_to_current = (
                            affine_to_homogeneous(
                                transform
                            )
                        )

                        try:

                            current_to_previous = (
                                np.linalg.inv(
                                    previous_to_current
                                )
                            )

                        except np.linalg.LinAlgError:

                            current_to_previous = (
                                np.eye(
                                    3,
                                    dtype=np.float32,
                                )
                            )

                            valid = False

                        if valid:

                            current_to_reference = (
                                previous_to_reference
                                @ current_to_previous
                            )

                            gmc_valid_frames += 1

                            inlier_count = (
                                int(
                                    transform[1]
                                )
                                if isinstance(
                                    transform,
                                    (tuple, list),
                                )
                                else 0
                            )

                            gmc_inliers.append(
                                inlier_count
                            )

                        else:

                            current_to_reference = (
                                previous_to_reference.copy()
                            )

                            gmc_invalid_frames += 1

                    else:

                        current_to_reference = (
                            previous_to_reference.copy()
                        )

                        gmc_invalid_frames += 1

                previous_to_reference = (
                    current_to_reference.copy()
                )

            # ------------------------------------------------
            # INT8 PERSON DETECTION
            #
            # ALWAYS USE ORIGINAL FRAME
            # ------------------------------------------------

            inference_start = (
                time.perf_counter()
            )

            detections = detector.detect(
                frame
            )

            inference_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000.0

            inference_times.append(
                inference_ms
            )

            total_detections += (
                len(detections)
            )

            # ------------------------------------------------
            # TRANSFORM DETECTIONS FOR TRACKER
            # ------------------------------------------------

            tracker_detections = (
                transform_detections(
                    detections,
                    current_to_reference,
                    frame.shape,
                )
            )

            tracker_input = (
                detections_to_boxes(
                    tracker_detections,
                    frame.shape,
                )
            )

            # ------------------------------------------------
            # BYTE TRACK
            # ------------------------------------------------

            tracks = tracker.update(
                tracker_input,
                img=frame,
            )

            if tracks is None:
                tracks = np.empty(
                    (0, 7),
                    dtype=np.float32,
                )

            active_tracks = len(
                tracks
            )

            total_tracked += (
                active_tracks
            )

            # ------------------------------------------------
            # TRACK METRICS
            # ------------------------------------------------

            for track in tracks:

                if len(track) < 5:
                    continue

                track_id = int(
                    track[4]
                )

                unique_ids.add(
                    track_id
                )

                track_lengths[
                    track_id
                ] = (
                    track_lengths.get(
                        track_id,
                        0,
                    )
                    + 1
                )

            # ------------------------------------------------
            # MAP TRACKS BACK TO ORIGINAL IMAGE
            # ------------------------------------------------

            try:

                reference_to_current = (
                    np.linalg.inv(
                        current_to_reference
                    )
                )

            except np.linalg.LinAlgError:

                reference_to_current = (
                    np.eye(
                        3,
                        dtype=np.float32,
                    )
                )

            output_frame = draw_tracks(
                frame.copy(),
                tracks,
                reference_to_current,
            )

            # ------------------------------------------------
            # OVERLAY
            # ------------------------------------------------

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
                f"GMC: "
                f"{'ON' if gmc_enabled else 'OFF'}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if gmc_enabled:

                cv2.putText(
                    output_frame,
                    f"GMC inliers: "
                    f"{inlier_count}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # WRITE
            # ------------------------------------------------

            writer.write(
                output_frame
            )

            # ------------------------------------------------
            # TIMING
            # ------------------------------------------------

            frame_ms = (
                time.perf_counter()
                - frame_start
            ) * 1000.0

            frame_times.append(
                frame_ms
            )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

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
                    f"Detections: "
                    f"{total_detections:4d} | "
                    f"Tracks: "
                    f"{total_tracked:4d} | "
                    f"FPS: "
                    f"{processing_fps:.2f}"
                )

    finally:

        cap.release()
        writer.release()

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

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
        float(
            np.mean(
                inference_times
            )
        )
        if inference_times
        else 0.0
    )

    average_frame_time = (
        float(
            np.mean(
                frame_times
            )
        )
        if frame_times
        else 0.0
    )

    average_track_length = (
        total_tracked
        / len(unique_ids)
        if unique_ids
        else 0.0
    )

    shortest_track = (
        min(
            track_lengths.values()
        )
        if track_lengths
        else 0
    )

    longest_track = (
        max(
            track_lengths.values()
        )
        if track_lengths
        else 0
    )

    average_gmc_inliers = (
        float(
            np.mean(gmc_inliers)
        )
        if gmc_inliers
        else 0.0
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("FINAL PIPELINE RESULTS")
    print("=" * 70)

    print(
        f"Frames processed:       "
        f"{frame_count}"
    )

    print(
        f"Person detections:      "
        f"{total_detections}"
    )

    print(
        f"Tracked detections:     "
        f"{total_tracked}"
    )

    print(
        f"Unique IDs:             "
        f"{len(unique_ids)}"
    )

    print(
        f"Average persons/frame:  "
        f"{total_detections / frame_count:.3f}"
        if frame_count
        else
        "Average persons/frame: 0.000"
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

    print(
        f"Shortest track:         "
        f"{shortest_track}"
    )

    print(
        f"Longest track:          "
        f"{longest_track}"
    )

    if gmc_enabled:

        print(
            f"GMC valid frames:      "
            f"{gmc_valid_frames}"
        )

        print(
            f"GMC invalid frames:    "
            f"{gmc_invalid_frames}"
        )

        print(
            f"Average GMC inliers:    "
            f"{average_gmc_inliers:.2f}"
        )

    print(
        f"Output video:           "
        f"{output_video}"
    )

    print(
        f"Metrics:                "
        f"{metrics_path}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = f"""# Final Person Tracking Pipeline

## Configuration

- Input video: `{source_path}`
- Detector: OpenVINO INT8
- Model: `{model_path}`
- Tracker: `{config['tracker']['type']}`
- GMC: `{"enabled" if gmc_enabled else "disabled"}`

## Architecture

The final pipeline uses the following processing flow:

1. Read the original truck-camera frame.
2. Estimate camera motion using ORB.
3. Run the OpenVINO INT8 person detector on the original frame.
4. Transform detection bounding boxes into the GMC reference coordinate system.
5. Pass transformed person detections to ByteTrack.
6. Transform tracked boxes back to the original frame.
7. Draw tracking IDs on the original frame.
8. Save the final annotated video.

The video frame itself is not warped before detection.

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

## GMC

- Valid GMC frames: {gmc_valid_frames}
- Invalid GMC frames: {gmc_invalid_frames}
- Average GMC inliers: {average_gmc_inliers:.2f}

## Output

`{output_video}`

## Purpose

This pipeline combines:

- OpenVINO INT8 person detection
- ByteTrack multi-object tracking
- ORB-based global camera-motion estimation
- GMC-aware detection coordinate transformation
- Tracking visualization
- Performance metrics

The detector operates on the original frame to avoid image degradation caused by repeated cumulative warping.
"""

    metrics_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        "Final pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
