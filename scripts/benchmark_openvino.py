import time
from pathlib import Path

import cv2

from src.detection.openvino_detector import OpenVINOPersonDetector
from src.ingestion.video_reader import VideoReader


VIDEO_PATH = "data/input/truck_video.mp4"
MODEL_PATH = "yolo26n_openvino_model"
IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.45
WARMUP_FRAMES = 30


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"OpenVINO model directory not found: {MODEL_PATH}"
        )

    detector = OpenVINOPersonDetector(
        model_path=MODEL_PATH,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        iou=IOU,
    )

    reader = VideoReader(VIDEO_PATH)

    total_frames = 0
    total_person_detections = 0
    measured_frames = 0
    total_inference_time = 0.0

    print("Starting OpenVINO benchmark...")
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Warmup frames: {WARMUP_FRAMES}")
    print()

    for frame_number, frame in reader.frames():
        total_frames += 1

        start_time = time.perf_counter()

        detections = detector.detect(frame)

        inference_time = time.perf_counter() - start_time

        total_person_detections += len(detections)

        if frame_number >= WARMUP_FRAMES:
            total_inference_time += inference_time
            measured_frames += 1

    reader.release()

    if measured_frames == 0:
        raise RuntimeError("No frames available for benchmark.")

    average_inference_ms = (
        total_inference_time / measured_frames
    ) * 1000.0

    processing_fps = (
        measured_frames / total_inference_time
        if total_inference_time > 0
        else 0.0
    )

    average_persons_per_frame = (
        total_person_detections / total_frames
        if total_frames > 0
        else 0.0
    )

    print()
    print("=" * 50)
    print("OPENVINO BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Frames processed: {total_frames}")
    print(f"Measured frames: {measured_frames}")
    print(f"Total person detections: {total_person_detections}")
    print(
        f"Average persons/frame: "
        f"{average_persons_per_frame:.3f}"
    )
    print(
        f"Average inference time: "
        f"{average_inference_ms:.2f} ms"
    )
    print(f"Processing FPS: {processing_fps:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
