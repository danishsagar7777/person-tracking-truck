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
# BYTE TRACK ARGUMENTS
# ============================================================

class TrackerArgs:
    def __init__(self):
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.fuse_score = True

        # ReID disabled because this project uses ByteTrack.
        self.with_reid = False
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25


# ============================================================
# CONFIG
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# DETECTIONS -> ULTRALYTICS BOXES
# ============================================================

def detections_to_boxes(detections, frame_shape):
    """
    Convert our detector dictionaries into the Boxes object
    expected by the current Ultralytics BYTETracker API.

    Each row:

        x1, y1, x2, y2, confidence, class_id
    """

    height, width = frame_shape[:2]

    rows = []

    for detection in detections:
        bbox = detection.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        confidence = float(
            detection.get(
                "confidence",
                detection.get("conf", detection.get("score", 0.0)),
            )
        )

        class_id = int(
            detection.get(
                "class_id",
                detection.get("cls", 0),
            )
        )

        # Person-only pipeline.
        if class_id != 0:
            continue

        # Clamp coordinates.
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
                float(class_id),
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


# ============================================================
# AFFINE TRANSFORM HELPERS
# ============================================================

def affine_to_homogeneous(transform):
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :] = np.asarray(transform, dtype=np.float32)
    return matrix


def transform_bbox(
    bbox,
    transform,
    frame_width,
    frame_height,
):
    """
    Transform all four bbox corners using a 2x3 affine matrix.
    """

    x1, y1, x2, y2 = bbox

    points = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )

    matrix = affine_to_homogeneous(transform)

    homogeneous = np.concatenate(
        [
            points,
            np.ones((4, 1), dtype=np.float32),
        ],
        axis=1,
    )

    transformed = homogeneous @ matrix.T

    new_x1 = float(np.min(transformed[:, 0]))
    new_y1 = float(np.min(transformed[:, 1]))
    new_x2 = float(np.max(transformed[:, 0]))
    new_y2 = float(np.max(transformed[:, 1]))

    new_x1 = max(0.0, min(new_x1, frame_width - 1))
    new_y1 = max(0.0, min(new_y1, frame_height - 1))
    new_x2 = max(0.0, min(new_x2, frame_width - 1))
    new_y2 = max(0.0, min(new_y2, frame_height - 1))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2,
    )


def transform_detections(
    detections,
    transform,
    frame_shape,
):
    """
    Transform detection boxes only.

    Important:
    We do NOT warp the video frame before YOLO inference.

    The detector always sees the original camera image.
    """

    height, width = frame_shape[:2]

    transformed_detections = []

    for detection in detections:
        bbox = detection.get("bbox")

        if bbox is None:
            continue

        new_bbox = transform_bbox(
            bbox,
            transform,
            width,
            height,
        )

        if new_bbox is None:
            continue

        transformed_detection = dict(detection)

        transformed_detection["bbox"] = [
            int(round(new_bbox[0])),
            int(round(new_bbox[1])),
            int(round(new_bbox[2])),
            int(round(new_bbox[3])),
        ]

        transformed_detections.append(
            transformed_detection
        )

    return transformed_detections


# ============================================================
# GMC VALIDATION
# ============================================================

def valid_camera_transform(
    transform,
    inlier_count,
):
    """
    Reject unstable ORB camera-motion estimates.
    """

    if inlier_count < 10:
        return False

    matrix = np.asarray(
        transform,
        dtype=np.float32,
    )

    if matrix.shape != (2, 3):
        return False

    if not np.all(np.isfinite(matrix)):
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
# TRACK DRAWING
# ============================================================

def draw_tracks(
    frame,
    tracks,
):
    """
    Draw ByteTrack results directly in the original frame
    coordinate system.
    """

    if tracks is None:
        return frame

    if len(tracks) == 0:
        return frame

    for track in tracks:

        if len(track) < 5:
            continue

        x1 = int(round(float(track[0])))
        y1 = int(round(float(track[1])))
        x2 = int(round(float(track[2])))
        y2 = int(round(float(track[3])))

        track_id = int(track[4])

        confidence = 0.0

        if len(track) > 5:
            confidence = float(track[5])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Person ID: {track_id} {confidence:.2f}",
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Final INT8 Person Tracking Pipeline"
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
        help="Disable ORB global camera-motion compensation.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    source_path = config["source"]["path"]

    model_path = config["detector"]["model"]

    output_video = Path(
        config["output"]["video"]
    )

    metrics_path = Path(
        config["output"]["metrics"]
    )

    configured_gmc = bool(
        config["camera_motion"]["enabled"]
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

    print(f"Source: {source_path}")
    print(f"Model: {model_path}")
    print(
        f"Tracker: {config['tracker']['type']}"
    )
    print(f"GMC enabled: {gmc_enabled}")

    if args.disable_gmc:
        print("GMC override: DISABLED")

    if args.max_frames is not None:
        print(
            f"Maximum frames: {args.max_frames}"
        )

    print()

    # --------------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------------

    if not Path(source_path).exists():
        raise FileNotFoundError(
            f"Input video not found: {source_path}"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"INT8 model not found: {model_path}"
        )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    print(
        "Loading OpenVINO INT8 detector..."
    )

    detector = INT8PersonDetector(
        model_path=model_path,
        image_size=config["detector"]["image_size"],
        confidence=config["detector"]["confidence"],
        iou=config["detector"]["iou"],
        device=config["detector"]["device"],
    )

    print("INT8 detector loaded.")

    # --------------------------------------------------------
    # TRACKER
    # --------------------------------------------------------

    print("Creating ByteTrack...")

    tracker = BYTETracker(
        TrackerArgs()
    )

    print("ByteTrack created.")

    # --------------------------------------------------------
    # GMC
    # --------------------------------------------------------

    motion_estimator = None

    if gmc_enabled:

        print(
            "Creating ORB camera-motion estimator..."
        )

        motion_estimator = CameraMotionEstimator()

        print("ORB GMC enabled.")

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        source_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {source_path}"
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Video: {frame_width}x{frame_height}"
    )
    print(f"FPS: {fps:.2f}")
    print(
        f"Frames: {total_frames}"
    )
    print()

    # --------------------------------------------------------
    # OUTPUT WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_video}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    frame_count = 0

    total_detections = 0
    total_tracked_detections = 0

    unique_ids = set()

    inference_times = []
    frame_times = []

    gmc_valid_frames = 0
    gmc_invalid_frames = 0
    gmc_inliers = []

    start_total = time.perf_counter()

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    while True:

        if (
            args.max_frames is not None
            and frame_count >= args.max_frames
        ):
            break

        frame_start = time.perf_counter()

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # ----------------------------------------------------
        # 1. ESTIMATE CAMERA MOTION
        # ----------------------------------------------------

        if motion_estimator is not None:

            camera_transform, inlier_count = (
                motion_estimator.estimate(frame)
            )

            if valid_camera_transform(
                camera_transform,
                inlier_count,
            ):
                gmc_valid_frames += 1

                gmc_inliers.append(
                    inlier_count
                )

                # Camera estimator gives:
                #
                # previous -> current
                #
                # ByteTrack needs the current
                # detections expressed in the
                # previous/reference coordinate system.
                #
                # Therefore use inverse:
                #
                # current -> previous

                try:
                    compensation_transform = (
                        cv2.invertAffineTransform(
                            camera_transform
                        )
                    )
                except cv2.error:
                    compensation_transform = (
                        np.eye(
                            2,
                            3,
                            dtype=np.float32,
                        )
                    )

            else:
                gmc_invalid_frames += 1

                compensation_transform = (
                    np.eye(
                        2,
                        3,
                        dtype=np.float32,
                    )
                )

        else:

            compensation_transform = (
                np.eye(
                    2,
                    3,
                    dtype=np.float32,
                )
            )

        # ----------------------------------------------------
        # 2. INT8 YOLO ON ORIGINAL FRAME
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        detections = detector.detect(
            frame
        )

        inference_end = time.perf_counter()

        inference_ms = (
            inference_end
            - inference_start
        ) * 1000.0

        inference_times.append(
            inference_ms
        )

        total_detections += len(
            detections
        )

        # ----------------------------------------------------
        # 3. CAMERA-MOTION COMPENSATION
        # ----------------------------------------------------

        if gmc_enabled:

            tracker_detections = (
                transform_detections(
                    detections,
                    compensation_transform,
                    frame.shape,
                )
            )

        else:

            tracker_detections = detections

        # ----------------------------------------------------
        # 4. CONVERT TO ULTRALYTICS BOXES
        # ----------------------------------------------------

        boxes = detections_to_boxes(
            tracker_detections,
            frame.shape,
        )

        # ----------------------------------------------------
        # 5. BYTE TRACK
        # ----------------------------------------------------

        tracks = tracker.update(
            boxes,
            frame,
        )

        if tracks is None:
            tracks = np.empty(
                (0, 8),
                dtype=np.float32,
            )

        if len(tracks) > 0:

            total_tracked_detections += len(
                tracks
            )

            for track in tracks:

                if len(track) >= 5:

                    track_id = int(
                        track[4]
                    )

                    unique_ids.add(
                        track_id
                    )

        # ----------------------------------------------------
        # 6. DRAW
        # ----------------------------------------------------

        annotated_frame = frame.copy()

        # When GMC is enabled, ByteTrack coordinates are in
        # the previous/reference coordinate system.
        #
        # To draw them on the current frame, transform using
        # the original camera transform:
        #
        # previous -> current

        if gmc_enabled:

            draw_tracks_frame = annotated_frame.copy()

            if tracks is not None:

                for track in tracks:

                    if len(track) < 5:
                        continue

                    bbox = transform_bbox(
                        track[:4],
                        camera_transform,
                        frame_width,
                        frame_height,
                    )

                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox

                    track_id = int(
                        track[4]
                    )

                    confidence = (
                        float(track[5])
                        if len(track) > 5
                        else 0.0
                    )

                    cv2.rectangle(
                        draw_tracks_frame,
                        (
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        draw_tracks_frame,
                        (
                            f"Person ID: "
                            f"{track_id} "
                            f"{confidence:.2f}"
                        ),
                        (
                            int(x1),
                            max(
                                20,
                                int(y1) - 8,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            annotated_frame = (
                draw_tracks_frame
            )

        else:

            annotated_frame = draw_tracks(
                annotated_frame,
                tracks,
            )

        # ----------------------------------------------------
        # 7. STATUS TEXT
        # ----------------------------------------------------

        current_fps = (
            1.0
            / max(
                time.perf_counter()
                - frame_start,
                1e-9,
            )
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"Persons: {len(detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"Tracks: {len(tracks)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps:.2f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            annotated_frame
        )

        frame_time_ms = (
            time.perf_counter()
            - frame_start
        ) * 1000.0

        frame_times.append(
            frame_time_ms
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            frame_count == 1
            or frame_count % 100 == 0
        ):

            print(
                f"Frame {frame_count:4d} | "
                f"Detections: "
                f"{len(detections):3d} | "
                f"Tracks: "
                f"{len(tracks):3d} | "
                f"FPS: "
                f"{current_fps:6.2f}"
            )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()
    writer.release()

    total_time = (
        time.perf_counter()
        - start_total
    )

    processing_fps = (
        frame_count / total_time
        if total_time > 0
        else 0.0
    )

    average_inference_ms = (
        float(np.mean(inference_times))
        if inference_times
        else 0.0
    )

    average_frame_ms = (
        float(np.mean(frame_times))
        if frame_times
        else 0.0
    )

    average_gmc_inliers = (
        float(np.mean(gmc_inliers))
        if gmc_inliers
        else 0.0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {
        "frames": frame_count,
        "person_detections": total_detections,
        "tracked_detections": total_tracked_detections,
        "unique_ids": len(unique_ids),
        "average_detections_per_frame": (
            total_detections / frame_count
            if frame_count > 0
            else 0.0
        ),
        "average_tracks_per_frame": (
            total_tracked_detections / frame_count
            if frame_count > 0
            else 0.0
        ),
        "average_inference_ms": average_inference_ms,
        "average_frame_time_ms": average_frame_ms,
        "processing_fps": processing_fps,
        "gmc_enabled": gmc_enabled,
        "gmc_valid_frames": gmc_valid_frames,
        "gmc_invalid_frames": gmc_invalid_frames,
        "average_gmc_inliers": average_gmc_inliers,
        "output_video": str(output_video),
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Final INT8 Person Tracking Metrics\n\n"
        )

        for key, value in metrics.items():

            file.write(
                f"- **{key}:** {value}\n"
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Frames: {frame_count}"
    )

    print(
        f"Person detections: "
        f"{total_detections}"
    )

    print(
        f"Tracked detections: "
        f"{total_tracked_detections}"
    )

    print(
        f"Unique IDs: "
        f"{len(unique_ids)}"
    )

    print(
        f"Average inference: "
        f"{average_inference_ms:.2f} ms"
    )

    print(
        f"Average frame time: "
        f"{average_frame_ms:.2f} ms"
    )

    print(
        f"Processing FPS: "
        f"{processing_fps:.2f}"
    )

    if gmc_enabled:

        print(
            f"GMC valid frames: "
            f"{gmc_valid_frames}"
        )

        print(
            f"GMC invalid frames: "
            f"{gmc_invalid_frames}"
        )

        print(
            f"Average GMC inliers: "
            f"{average_gmc_inliers:.2f}"
        )

    print()
    print(
        f"Output video: "
        f"{output_video}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
