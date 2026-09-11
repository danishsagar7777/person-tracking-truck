import os
import time
from collections import defaultdict

import cv2
import numpy as np

from src.detection.int8_detector import INT8PersonDetector
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/input/truck_video.mp4"

INT8_MODEL_PATH = "models/yolo26n_int8/yolo26n_int8.xml"

OUTPUT_PATH = "results/videos/person_tracking_int8_bytetrack.mp4"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

DEVICE = "CPU"

# COCO person class
PERSON_CLASS_ID = 0


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
# TRACKER
# ============================================================

def create_tracker():
    """
    Create BYTETracker using the API supported by the
    installed Ultralytics version.
    """

    return BYTETracker(TrackerArgs())


# ============================================================
# DETECTION -> ULTRALYTICS BOXES
# ============================================================

def detections_to_boxes(detections, frame_shape):
    """
    Convert the native OpenVINO INT8 detector output into
    an Ultralytics Boxes object.

    BYTETracker requires an object that provides:

        results.conf
        results.cls
        results.xywh

    and also supports:

        results[boolean_mask]

    Ultralytics Boxes provides all of these requirements.

    Expected detector detection format:

        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int
        }

    Returns:

        Ultralytics Boxes
    """

    frame_height, frame_width = frame_shape[:2]

    rows = []

    for detection in detections:

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox = detection.get("bbox")

        if bbox is None:
            bbox = detection.get("box")

        if bbox is None:
            bbox = detection.get("xyxy")

        if bbox is None:
            continue

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = detection.get("confidence")

        if confidence is None:
            confidence = detection.get("conf")

        if confidence is None:
            confidence = detection.get("score")

        if confidence is None:
            continue

        confidence = float(confidence)

        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_id = detection.get("class_id")

        if class_id is None:
            class_id = detection.get("cls")

        if class_id is None:
            class_id = PERSON_CLASS_ID

        class_id = int(class_id)

        # ----------------------------------------------------
        # Person-only filtering
        # ----------------------------------------------------

        if class_id != PERSON_CLASS_ID:
            continue

        # ----------------------------------------------------
        # Clamp coordinates
        # ----------------------------------------------------

        x1 = max(0.0, min(x1, frame_width - 1))
        y1 = max(0.0, min(y1, frame_height - 1))

        x2 = max(0.0, min(x2, frame_width - 1))
        y2 = max(0.0, min(y2, frame_height - 1))

        # Invalid box
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

    # --------------------------------------------------------
    # Empty detection case
    # --------------------------------------------------------

    if not rows:

        empty_boxes = np.empty(
            (0, 6),
            dtype=np.float32,
        )

        return Boxes(
            empty_boxes,
            (frame_height, frame_width),
        )

    # --------------------------------------------------------
    # Create Ultralytics Boxes
    # --------------------------------------------------------

    box_array = np.asarray(
        rows,
        dtype=np.float32,
    )

    return Boxes(
        box_array,
        (frame_height, frame_width),
    )


# ============================================================
# DRAW TRACKS
# ============================================================

def draw_tracks(frame, tracks):
    """
    Draw ByteTrack output.

    Expected output format:

        [x1, y1, x2, y2, track_id, score, class_id, index]
    """

    if tracks is None:
        return frame

    if len(tracks) == 0:
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

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPENVINO INT8 + BYTETRACK PERSON TRACKING")
    print("=" * 70)

    print()
    print(f"Video       : {VIDEO_PATH}")
    print(f"INT8 model  : {INT8_MODEL_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Device      : {DEVICE}")

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    if not os.path.isfile(INT8_MODEL_PATH):
        raise FileNotFoundError(
            f"INT8 model not found: {INT8_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        "results/videos",
        exist_ok=True,
    )

    os.makedirs(
        "results/metrics",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load INT8 detector
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
    # Create ByteTrack
    # --------------------------------------------------------

    print()
    print("Creating ByteTrack...")

    tracker = create_tracker()

    print("ByteTrack created.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: {VIDEO_PATH}"
        )

    source_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    source_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    total_source_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("Video information:")
    print(f"  Width        : {source_width}")
    print(f"  Height       : {source_height}")
    print(f"  FPS          : {source_fps:.2f}")
    print(f"  Total frames : {total_source_frames}")

    # --------------------------------------------------------
    # Create output writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        source_fps,
        (source_width, source_height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: {OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    frame_count = 0

    total_person_detections = 0

    total_tracked_detections = 0

    total_inference_time = 0.0

    total_wall_time = 0.0

    track_observations = defaultdict(int)

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    print()
    print("Processing video...")
    print()

    while True:

        frame_start = time.perf_counter()

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # ====================================================
        # INT8 DETECTION
        # ====================================================

        inference_start = time.perf_counter()

        detections = detector.detect(frame)

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        total_inference_time += inference_time

        # ====================================================
        # PERSON FILTERING
        # ====================================================

        person_detections = []

        for detection in detections:

            class_id = detection.get(
                "class_id"
            )

            if class_id is None:
                class_id = detection.get(
                    "cls"
                )

            if class_id is None:
                class_id = PERSON_CLASS_ID

            if int(class_id) == PERSON_CLASS_ID:

                person_detections.append(
                    detection
                )

        total_person_detections += len(
            person_detections
        )

        # ====================================================
        # CONVERT TO ULTRALYTICS BOXES
        # ====================================================

        tracker_input = detections_to_boxes(
            person_detections,
            frame.shape,
        )

        # ====================================================
        # BYTE TRACK UPDATE
        # ====================================================

        tracks = tracker.update(
            tracker_input,
            img=frame,
        )

        # ====================================================
        # TRACK METRICS
        # ====================================================

        active_tracks = 0

        if tracks is not None:

            active_tracks = len(tracks)

            total_tracked_detections += (
                active_tracks
            )

            for track in tracks:

                if len(track) >= 5:

                    track_id = int(
                        track[4]
                    )

                    track_observations[
                        track_id
                    ] += 1

        # ====================================================
        # VISUALIZATION
        # ====================================================

        output_frame = draw_tracks(
            frame.copy(),
            tracks,
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        cv2.putText(
            output_frame,
            f"Frame: {frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"Persons: {len(person_detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"Active tracks: {active_tracks}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ====================================================
        # WRITE OUTPUT
        # ====================================================

        writer.write(output_frame)

        # ====================================================
        # FRAME TIMING
        # ====================================================

        frame_time = (
            time.perf_counter()
            - frame_start
        )

        total_wall_time += frame_time

        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_count % 100 == 0:

            current_fps = (
                frame_count / total_wall_time
                if total_wall_time > 0
                else 0.0
            )

            print(
                f"Processed {frame_count}/"
                f"{total_source_frames} frames | "
                f"Persons: {total_person_detections} | "
                f"Tracks: {total_tracked_detections} | "
                f"FPS: {current_fps:.2f}"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    writer.release()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    average_persons_per_frame = (
        total_person_detections / frame_count
        if frame_count > 0
        else 0.0
    )

    average_tracks_per_frame = (
        total_tracked_detections / frame_count
        if frame_count > 0
        else 0.0
    )

    average_inference_ms = (
        total_inference_time
        / frame_count
        * 1000
        if frame_count > 0
        else 0.0
    )

    average_frame_time_ms = (
        total_wall_time
        / frame_count
        * 1000
        if frame_count > 0
        else 0.0
    )

    processing_fps = (
        frame_count / total_wall_time
        if total_wall_time > 0
        else 0.0
    )

    unique_ids = len(
        track_observations
    )

    track_lengths = list(
        track_observations.values()
    )

    if track_lengths:

        average_track_length = (
            sum(track_lengths)
            / len(track_lengths)
        )

        shortest_track = min(
            track_lengths
        )

        longest_track = max(
            track_lengths
        )

    else:

        average_track_length = 0.0

        shortest_track = 0

        longest_track = 0

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Frames processed        : "
        f"{frame_count}"
    )

    print(
        f"Source frames           : "
        f"{total_source_frames}"
    )

    print(
        f"Total person detections : "
        f"{total_person_detections}"
    )

    print(
        f"Total tracked detections: "
        f"{total_tracked_detections}"
    )

    print(
        f"Average persons/frame   : "
        f"{average_persons_per_frame:.3f}"
    )

    print(
        f"Average tracks/frame    : "
        f"{average_tracks_per_frame:.3f}"
    )

    print(
        f"Unique Track IDs        : "
        f"{unique_ids}"
    )

    print(
        f"Average track length    : "
        f"{average_track_length:.2f} frames"
    )

    print(
        f"Shortest track          : "
        f"{shortest_track} frames"
    )

    print(
        f"Longest track           : "
        f"{longest_track} frames"
    )

    print(
        f"Average inference       : "
        f"{average_inference_ms:.2f} ms"
    )

    print(
        f"Average frame time      : "
        f"{average_frame_time_ms:.2f} ms"
    )

    print(
        f"Processing FPS          : "
        f"{processing_fps:.2f}"
    )

    print(
        f"Output                  : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
