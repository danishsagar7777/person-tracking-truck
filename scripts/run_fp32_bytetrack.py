import os
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/input/truck_video.mp4"

MODEL_PATH = "yolo26n.pt"

TRACKER_CONFIG = "bytetrack.yaml"

OUTPUT_PATH = "results/videos/person_tracking_fp32_bytetrack.mp4"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

PERSON_CLASS_ID = 0


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    print("=" * 70)
    print("FP32 YOLO + ByteTrack Person Tracking")
    print("=" * 70)

    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Tracker: {TRACKER_CONFIG}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    frames_processed = 0
    total_detections = 0
    total_tracked_detections = 0

    total_inference_ms = 0.0
    total_frame_time_ms = 0.0

    unique_track_ids = set()

    track_lengths = defaultdict(int)

    wall_start = time.perf_counter()

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_start = time.perf_counter()

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        inference_ms = 0.0

        if hasattr(result, "speed") and result.speed:
            inference_ms = float(
                result.speed.get("inference", 0.0)
            )

        total_inference_ms += inference_ms

        # ----------------------------------------------------
        # Count detections
        # ----------------------------------------------------

        if result.boxes is not None:

            total_detections += len(result.boxes)

            track_ids = result.boxes.id

            if track_ids is not None:

                track_ids_list = track_ids.int().cpu().tolist()

                total_tracked_detections += len(
                    track_ids_list
                )

                for track_id in track_ids_list:

                    track_id = int(track_id)

                    unique_track_ids.add(track_id)

                    track_lengths[track_id] += 1

        # ----------------------------------------------------
        # Draw result
        # ----------------------------------------------------

        annotated_frame = result.plot()

        writer.write(annotated_frame)

        # ----------------------------------------------------
        # Frame timing
        # ----------------------------------------------------

        frame_time_ms = (
            time.perf_counter() - frame_start
        ) * 1000.0

        total_frame_time_ms += frame_time_ms

        frames_processed += 1

        if frames_processed % 100 == 0:

            active_tracks = 0

            if result.boxes is not None:
                if result.boxes.id is not None:
                    active_tracks = len(
                        result.boxes.id
                    )

            print(
                f"Processed {frames_processed} frames "
                f"| Active tracks: {active_tracks} "
                f"| Unique IDs: {len(unique_track_ids)}"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()

    wall_time = time.perf_counter() - wall_start

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if frames_processed > 0:

        avg_detections = (
            total_detections / frames_processed
        )

        avg_tracked_detections = (
            total_tracked_detections
            / frames_processed
        )

        avg_inference_ms = (
            total_inference_ms
            / frames_processed
        )

        avg_frame_time_ms = (
            total_frame_time_ms
            / frames_processed
        )

        processing_fps = (
            frames_processed / wall_time
            if wall_time > 0
            else 0.0
        )

    else:

        avg_detections = 0.0
        avg_tracked_detections = 0.0
        avg_inference_ms = 0.0
        avg_frame_time_ms = 0.0
        processing_fps = 0.0

    # --------------------------------------------------------
    # Track lifecycle metrics
    # --------------------------------------------------------

    if track_lengths:

        average_track_length = (
            sum(track_lengths.values())
            / len(track_lengths)
        )

        shortest_track = min(
            track_lengths.values()
        )

        longest_track = max(
            track_lengths.values()
        )

    else:

        average_track_length = 0.0
        shortest_track = 0
        longest_track = 0

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FP32 + ByteTrack RESULTS")
    print("=" * 70)

    print(f"Frames processed: {frames_processed}")
    print(f"Total person detections: {total_detections}")
    print(
        f"Average persons/frame: "
        f"{avg_detections:.3f}"
    )

    print(
        f"Total tracked detections: "
        f"{total_tracked_detections}"
    )

    print(
        f"Average tracked persons/frame: "
        f"{avg_tracked_detections:.3f}"
    )

    print(
        f"Unique Track IDs: "
        f"{len(unique_track_ids)}"
    )

    print(
        f"Average track length: "
        f"{average_track_length:.2f} frames"
    )

    print(
        f"Shortest track: "
        f"{shortest_track} frames"
    )

    print(
        f"Longest track: "
        f"{longest_track} frames"
    )

    print(
        f"Average inference time: "
        f"{avg_inference_ms:.2f} ms"
    )

    print(
        f"Average frame time: "
        f"{avg_frame_time_ms:.2f} ms"
    )

    print(
        f"Processing FPS: "
        f"{processing_fps:.2f}"
    )

    print(
        f"Wall time: "
        f"{wall_time:.2f} sec"
    )

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
