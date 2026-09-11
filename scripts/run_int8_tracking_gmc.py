import os
import time
from collections import defaultdict

import cv2
import numpy as np

from src.detection.int8_detector import INT8PersonDetector
from src.tracking.motion_compensation import CameraMotionEstimator

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/input/truck_video.mp4"

INT8_MODEL_PATH = "models/yolo26n_int8/yolo26n_int8.xml"

OUTPUT_PATH = "results/videos/person_tracking_int8_bytetrack_gmc.mp4"

METRICS_PATH = "results/metrics/day11_int8_bytetrack_gmc.md"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

DEVICE = "CPU"

PERSON_CLASS_ID = 0

# ORB configuration
ORB_MAX_FEATURES = 500
ORB_MATCH_RATIO = 0.75
ORB_MIN_MATCHES = 10

# Maximum frames.
# Set to None for the complete video.
MAX_FRAMES = None


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


def create_tracker():
    return BYTETracker(TrackerArgs())


# ============================================================
# DETECTIONS -> ULTRALYTICS BOXES
# ============================================================

def detections_to_boxes(detections, frame_shape):
    """
    Convert native INT8 detector detections into an
    Ultralytics Boxes object.

    Detector format:

        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int
        }
    """

    frame_height, frame_width = frame_shape[:2]

    rows = []

    for detection in detections:

        bbox = detection.get("bbox")

        if bbox is None:
            bbox = detection.get("box")

        if bbox is None:
            bbox = detection.get("xyxy")

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        confidence = detection.get("confidence")

        if confidence is None:
            confidence = detection.get("conf")

        if confidence is None:
            confidence = detection.get("score")

        if confidence is None:
            continue

        confidence = float(confidence)

        class_id = detection.get("class_id")

        if class_id is None:
            class_id = detection.get("cls")

        if class_id is None:
            class_id = PERSON_CLASS_ID

        class_id = int(class_id)

        if class_id != PERSON_CLASS_ID:
            continue

        x1 = max(0.0, min(x1, frame_width - 1))
        y1 = max(0.0, min(y1, frame_height - 1))
        x2 = max(0.0, min(x2, frame_width - 1))
        y2 = max(0.0, min(y2, frame_height - 1))

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
        empty_boxes = np.empty((0, 6), dtype=np.float32)

        return Boxes(
            empty_boxes,
            (frame_height, frame_width),
        )

    box_array = np.asarray(
        rows,
        dtype=np.float32,
    )

    return Boxes(
        box_array,
        (frame_height, frame_width),
    )


# ============================================================
# AFFINE TRANSFORM UTILITIES
# ============================================================

def affine_to_homogeneous(transform):
    """
    Convert OpenCV 2x3 affine transform into 3x3 matrix.
    """

    matrix = np.eye(3, dtype=np.float64)

    matrix[:2, :] = transform.astype(np.float64)

    return matrix


def homogeneous_to_affine(matrix):
    """
    Convert 3x3 homogeneous transform into OpenCV 2x3 affine form.
    """

    return matrix[:2, :].astype(np.float32)


def transform_points(points, matrix):
    """
    Transform Nx2 points using a 3x3 homogeneous matrix.
    """

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float64)

    homogeneous = np.concatenate(
        [
            points,
            np.ones((len(points), 1), dtype=np.float64),
        ],
        axis=1,
    )

    transformed = homogeneous @ matrix.T

    denominator = transformed[:, 2:3]

    denominator = np.where(
        np.abs(denominator) < 1e-8,
        1.0,
        denominator,
    )

    transformed = transformed[:, :2] / denominator

    return transformed.astype(np.float32)


def transform_bbox(bbox, matrix, width, height):
    """
    Transform a bounding box using its four corners.

    Returns an axis-aligned bounding box in the destination
    coordinate system.
    """

    x1, y1, x2, y2 = map(float, bbox)

    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )

    transformed = transform_points(
        corners,
        matrix,
    )

    new_x1 = float(np.min(transformed[:, 0]))
    new_y1 = float(np.min(transformed[:, 1]))
    new_x2 = float(np.max(transformed[:, 0]))
    new_y2 = float(np.max(transformed[:, 1]))

    new_x1 = max(0.0, min(new_x1, width - 1))
    new_y1 = max(0.0, min(new_y1, height - 1))
    new_x2 = max(0.0, min(new_x2, width - 1))
    new_y2 = max(0.0, min(new_y2, height - 1))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None

    return [
        new_x1,
        new_y1,
        new_x2,
        new_y2,
    ]


def transform_detections(
    detections,
    matrix,
    frame_shape,
):
    """
    Transform detector bounding boxes into stabilized coordinates.
    """

    height, width = frame_shape[:2]

    transformed = []

    for detection in detections:

        bbox = detection.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        new_bbox = transform_bbox(
            bbox,
            matrix,
            width,
            height,
        )

        if new_bbox is None:
            continue

        item = dict(detection)

        item["bbox"] = new_bbox

        transformed.append(item)

    return transformed


# ============================================================
# DRAWING
# ============================================================

def draw_tracks(frame, tracks):
    """
    Draw tracks on the original camera frame.
    """

    if tracks is None:
        return frame

    for track in tracks:

        if len(track) < 5:
            continue

        x1 = int(track[0])
        y1 = int(track[1])
        x2 = int(track[2])
        y2 = int(track[3])

        track_id = int(track[4])

        score = 0.0

        if len(track) > 5:
            score = float(track[5])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"Person ID {track_id} {score:.2f}"

        text_y = max(20, y1 - 8)

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def stabilized_tracks_to_original(
    tracks,
    stabilized_to_original,
    frame_shape,
):
    """
    Convert ByteTrack output from stabilized coordinates
    back into original camera coordinates.
    """

    height, width = frame_shape[:2]

    output = []

    for track in tracks:

        if len(track) < 5:
            continue

        bbox = track[:4]

        original_bbox = transform_bbox(
            bbox,
            stabilized_to_original,
            width,
            height,
        )

        if original_bbox is None:
            continue

        converted = list(track)

        converted[0] = original_bbox[0]
        converted[1] = original_bbox[1]
        converted[2] = original_bbox[2]
        converted[3] = original_bbox[3]

        output.append(converted)

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 11 - INT8 + BYTETRACK + ORB GMC")
    print("=" * 70)

    print()
    print(f"Video       : {VIDEO_PATH}")
    print(f"INT8 model  : {INT8_MODEL_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Device      : {DEVICE}")

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    if not os.path.isfile(INT8_MODEL_PATH):
        raise FileNotFoundError(
            f"INT8 model not found: {INT8_MODEL_PATH}"
        )

    os.makedirs(
        "results/videos",
        exist_ok=True,
    )

    os.makedirs(
        "results/metrics",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    print()
    print("Video metadata")
    print("-" * 70)
    print(f"Resolution   : {width}x{height}")
    print(f"FPS          : {fps:.2f}")
    print(f"Frame count  : {total_video_frames}")

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: {OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # INT8 detector
    # --------------------------------------------------------

    print()
    print("Loading OpenVINO INT8 detector...")

    detector = INT8PersonDetector(
        model_path=INT8_MODEL_PATH,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=DEVICE,
    )

    print("INT8 detector loaded.")

    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------

    print("Creating ByteTrack...")

    tracker = create_tracker()

    print("ByteTrack created.")

    # --------------------------------------------------------
    # ORB GMC
    # --------------------------------------------------------

    print("Creating ORB camera-motion estimator...")

    motion_estimator = CameraMotionEstimator(
        max_features=ORB_MAX_FEATURES,
        match_ratio=ORB_MATCH_RATIO,
        min_matches=ORB_MIN_MATCHES,
    )

    print("ORB GMC created.")

    # --------------------------------------------------------
    # Cumulative camera transform
    # --------------------------------------------------------

    # current_to_reference maps the current camera frame into
    # the coordinate system of the first frame.

    current_to_reference = np.eye(
        3,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    frame_count = 0

    total_detections = 0
    total_tracked_detections = 0

    inference_times = []
    frame_times = []

    orb_match_counts = []
    orb_inlier_counts = []

    tracks_by_id = defaultdict(int)

    unique_ids = set()

    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    print()
    print("Starting Day 11 processing...")
    print()

    wall_start = time.perf_counter()

    while True:

        if MAX_FRAMES is not None:
            if frame_count >= MAX_FRAMES:
                break

        frame_start = time.perf_counter()

        ok, frame = cap.read()

        if not ok:
            break

        frame_count += 1

        # ----------------------------------------------------
        # ORB camera motion
        # ----------------------------------------------------

        transform_previous_to_current, inliers = (
            motion_estimator.estimate(frame)
        )

        # ----------------------------------------------------
        # Determine cumulative transform
        #
        # previous_to_current maps:
        #
        # previous frame -> current frame
        #
        # We need:
        #
        # current frame -> reference frame
        # ----------------------------------------------------

        previous_to_reference = current_to_reference.copy()

        current_to_previous = np.linalg.inv(
            affine_to_homogeneous(
                transform_previous_to_current
            )
        )

        current_to_reference = (
            previous_to_reference
            @ current_to_previous
        )

        orb_match_counts.append(
            inliers
        )

        orb_inlier_counts.append(
            inliers
        )

        # ----------------------------------------------------
        # Warp current frame into reference coordinates
        # ----------------------------------------------------

        reference_to_current = np.linalg.inv(
            current_to_reference
        )

        current_to_reference_affine = (
            homogeneous_to_affine(
                current_to_reference
            )
        )

        stabilized_frame = cv2.warpAffine(
            frame,
            current_to_reference_affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

        # ----------------------------------------------------
        # INT8 detection
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        detections = detector.detect(
            stabilized_frame
        )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        inference_times.append(
            inference_time
        )

        total_detections += len(
            detections
        )

        # ----------------------------------------------------
        # Convert detections to Boxes
        # ----------------------------------------------------

        tracker_input = detections_to_boxes(
            detections,
            stabilized_frame.shape,
        )

        # ----------------------------------------------------
        # ByteTrack
        # ----------------------------------------------------

        tracks = tracker.update(
            tracker_input,
            img=stabilized_frame,
        )

        if tracks is None:
            tracks = []

        total_tracked_detections += len(
            tracks
        )

        # ----------------------------------------------------
        # Map stabilized tracks back to original frame
        # ----------------------------------------------------

        tracks_original = (
            stabilized_tracks_to_original(
                tracks,
                reference_to_current,
                frame.shape,
            )
        )

        for track in tracks_original:

            if len(track) < 5:
                continue

            track_id = int(track[4])

            unique_ids.add(track_id)

            tracks_by_id[track_id] += 1

        # ----------------------------------------------------
        # Draw on original camera frame
        # ----------------------------------------------------

        output_frame = frame.copy()

        output_frame = draw_tracks(
            output_frame,
            tracks_original,
        )

        # ----------------------------------------------------
        # GMC information
        # ----------------------------------------------------

        cv2.putText(
            output_frame,
            f"ORB inliers: {inliers}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"INT8 + ByteTrack + GMC",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            output_frame
        )

        frame_time = (
            time.perf_counter()
            - frame_start
        )

        frame_times.append(
            frame_time
        )

        if frame_count % 100 == 0:

            avg_inference = (
                np.mean(inference_times)
                * 1000
            )

            processing_fps = (
                1.0
                / np.mean(frame_times)
            )

            print(
                f"Frames: {frame_count:4d} | "
                f"Detections: {total_detections:4d} | "
                f"Tracks: {total_tracked_detections:4d} | "
                f"IDs: {len(unique_ids):3d} | "
                f"ORB inliers: {inliers:3d} | "
                f"INT8: {avg_inference:6.2f} ms | "
                f"FPS: {processing_fps:6.2f}"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    avg_inference_ms = (
        float(np.mean(inference_times))
        * 1000
        if inference_times
        else 0.0
    )

    avg_frame_ms = (
        float(np.mean(frame_times))
        * 1000
        if frame_times
        else 0.0
    )

    processing_fps = (
        1.0 / float(np.mean(frame_times))
        if frame_times
        else 0.0
    )

    avg_orb_inliers = (
        float(np.mean(orb_inlier_counts))
        if orb_inlier_counts
        else 0.0
    )

    max_orb_inliers = (
        max(orb_inlier_counts)
        if orb_inlier_counts
        else 0
    )

    min_orb_inliers = (
        min(orb_inlier_counts)
        if orb_inlier_counts
        else 0
    )

    average_persons_per_frame = (
        total_detections / frame_count
        if frame_count
        else 0.0
    )

    average_tracks_per_frame = (
        total_tracked_detections / frame_count
        if frame_count
        else 0.0
    )

    track_lengths = list(
        tracks_by_id.values()
    )

    average_track_length = (
        float(np.mean(track_lengths))
        if track_lengths
        else 0.0
    )

    shortest_track = (
        min(track_lengths)
        if track_lengths
        else 0
    )

    longest_track = (
        max(track_lengths)
        if track_lengths
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DAY 11 RESULTS")
    print("=" * 70)

    print(f"Frames processed       : {frame_count}")
    print(f"Total detections       : {total_detections}")
    print(
        f"Average persons/frame  : "
        f"{average_persons_per_frame:.3f}"
    )

    print(
        f"Total tracked detections: "
        f"{total_tracked_detections}"
    )

    print(
        f"Average tracks/frame   : "
        f"{average_tracks_per_frame:.3f}"
    )

    print(
        f"Unique Track IDs       : "
        f"{len(unique_ids)}"
    )

    print(
        f"Average track length   : "
        f"{average_track_length:.2f} frames"
    )

    print(
        f"Shortest track         : "
        f"{shortest_track} frames"
    )

    print(
        f"Longest track          : "
        f"{longest_track} frames"
    )

    print(
        f"Average INT8 inference : "
        f"{avg_inference_ms:.2f} ms"
    )

    print(
        f"Average frame time     : "
        f"{avg_frame_ms:.2f} ms"
    )

    print(
        f"Processing FPS         : "
        f"{processing_fps:.2f}"
    )

    print(
        f"Average ORB inliers    : "
        f"{avg_orb_inliers:.2f}"
    )

    print(
        f"Minimum ORB inliers    : "
        f"{min_orb_inliers}"
    )

    print(
        f"Maximum ORB inliers    : "
        f"{max_orb_inliers}"
    )

    print(
        f"Wall time              : "
        f"{wall_time:.2f} sec"
    )

    print()
    print(f"Output video: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Save metrics report
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "# Day 11 — OpenVINO INT8 + ByteTrack + ORB GMC\n\n"
        )

        report.write(
            "## Configuration\n\n"
        )

        report.write(
            f"- Video: `{VIDEO_PATH}`\n"
        )

        report.write(
            f"- INT8 model: `{INT8_MODEL_PATH}`\n"
        )

        report.write(
            "- Detector: OpenVINO INT8\n"
        )

        report.write(
            "- Tracker: ByteTrack\n"
        )

        report.write(
            "- GMC: ORB + RANSAC affine motion estimation\n"
        )

        report.write(
            f"- ORB features: {ORB_MAX_FEATURES}\n"
        )

        report.write(
            f"- ORB match ratio: {ORB_MATCH_RATIO}\n"
        )

        report.write(
            f"- ORB minimum matches: {ORB_MIN_MATCHES}\n\n"
        )

        report.write(
            "## Results\n\n"
        )

        report.write(
            f"- Frames processed: {frame_count}\n"
        )

        report.write(
            f"- Total person detections: {total_detections}\n"
        )

        report.write(
            f"- Average persons/frame: "
            f"{average_persons_per_frame:.3f}\n"
        )

        report.write(
            f"- Total tracked detections: "
            f"{total_tracked_detections}\n"
        )

        report.write(
            f"- Average tracks/frame: "
            f"{average_tracks_per_frame:.3f}\n"
        )

        report.write(
            f"- Unique Track IDs: {len(unique_ids)}\n"
        )

        report.write(
            f"- Average track length: "
            f"{average_track_length:.2f} frames\n"
        )

        report.write(
            f"- Shortest track: {shortest_track} frames\n"
        )

        report.write(
            f"- Longest track: {longest_track} frames\n"
        )

        report.write(
            f"- Average INT8 inference: "
            f"{avg_inference_ms:.2f} ms\n"
        )

        report.write(
            f"- Average frame time: "
            f"{avg_frame_ms:.2f} ms\n"
        )

        report.write(
            f"- Processing FPS: {processing_fps:.2f}\n"
        )

        report.write(
            f"- Average ORB inliers: "
            f"{avg_orb_inliers:.2f}\n"
        )

        report.write(
            f"- Minimum ORB inliers: {min_orb_inliers}\n"
        )

        report.write(
            f"- Maximum ORB inliers: {max_orb_inliers}\n"
        )

        report.write(
            f"- Wall time: {wall_time:.2f} sec\n"
        )

        report.write(
            f"- Output video: `{OUTPUT_PATH}`\n"
        )

    print(
        f"Metrics report: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
