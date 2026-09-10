from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.detection.int8_detector import INT8PersonDetector


VIDEO_PATH = "data/input/truck_video.mp4"

FP32_MODEL = "yolo26n.pt"
INT8_MODEL = "models/yolo26n_int8/yolo26n_int8.xml"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.45

WARMUP_FRAMES = 10
TEST_FRAMES = 200


def benchmark_fp32(video_path):
    print("\n" + "=" * 60)
    print("BENCHMARKING YOLO FP32")
    print("=" * 60)

    model = YOLO(FP32_MODEL)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()

        if not ret:
            break

        model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            classes=[0],
            verbose=False,
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inference_times = []
    total_detections = 0
    processed_frames = 0

    wall_start = time.perf_counter()

    while processed_frames < TEST_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            classes=[0],
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed * 1000)

        if results and results[0].boxes is not None:
            total_detections += len(results[0].boxes)

        processed_frames += 1

    wall_time = time.perf_counter() - wall_start

    cap.release()

    avg_inference = np.mean(inference_times)

    fps = (
        processed_frames / wall_time
        if wall_time > 0
        else 0
    )

    avg_persons = (
        total_detections / processed_frames
        if processed_frames > 0
        else 0
    )

    return {
        "frames": processed_frames,
        "detections": total_detections,
        "avg_persons": avg_persons,
        "avg_inference": avg_inference,
        "fps": fps,
    }


def benchmark_int8(video_path):
    print("\n" + "=" * 60)
    print("BENCHMARKING OPENVINO INT8")
    print("=" * 60)

    detector = INT8PersonDetector(
        model_path=INT8_MODEL,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        iou=IOU,
        device="CPU",
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()

        if not ret:
            break

        detector.detect(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inference_times = []
    total_detections = 0
    processed_frames = 0

    wall_start = time.perf_counter()

    while processed_frames < TEST_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        start = time.perf_counter()

        detections = detector.detect(frame)

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed * 1000)

        total_detections += len(detections)

        processed_frames += 1

    wall_time = time.perf_counter() - wall_start

    cap.release()

    avg_inference = np.mean(inference_times)

    fps = (
        processed_frames / wall_time
        if wall_time > 0
        else 0
    )

    avg_persons = (
        total_detections / processed_frames
        if processed_frames > 0
        else 0
    )

    return {
        "frames": processed_frames,
        "detections": total_detections,
        "avg_persons": avg_persons,
        "avg_inference": avg_inference,
        "fps": fps,
    }


def print_results(fp32, int8):

    print("\n")
    print("=" * 70)
    print("FP32 vs OPENVINO INT8")
    print("=" * 70)

    print(
        f"{'Metric':<30}"
        f"{'YOLO FP32':>18}"
        f"{'OpenVINO INT8':>20}"
    )

    print("-" * 70)

    print(
        f"{'Frames':<30}"
        f"{fp32['frames']:>18}"
        f"{int8['frames']:>20}"
    )

    print(
        f"{'Detections':<30}"
        f"{fp32['detections']:>18}"
        f"{int8['detections']:>20}"
    )

    print(
        f"{'Average persons/frame':<30}"
        f"{fp32['avg_persons']:>18.3f}"
        f"{int8['avg_persons']:>20.3f}"
    )

    print(
        f"{'Inference time (ms)':<30}"
        f"{fp32['avg_inference']:>18.2f}"
        f"{int8['avg_inference']:>20.2f}"
    )

    print(
        f"{'Processing FPS':<30}"
        f"{fp32['fps']:>18.2f}"
        f"{int8['fps']:>20.2f}"
    )

    print("=" * 70)

    speedup = (
        fp32["avg_inference"]
        / int8["avg_inference"]
    )

    print(f"\nINT8 latency speedup: {speedup:.2f}x")

    detection_difference = (
        int8["detections"] - fp32["detections"]
    )

    print(
        f"Detection difference: "
        f"{detection_difference:+d}"
    )

    print("\nNOTE:")
    print(
        "Detection count is NOT an accuracy metric."
    )


def main():

    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(VIDEO_PATH)

    if not Path(FP32_MODEL).exists():
        raise FileNotFoundError(FP32_MODEL)

    if not Path(INT8_MODEL).exists():
        raise FileNotFoundError(INT8_MODEL)

    fp32 = benchmark_fp32(VIDEO_PATH)

    int8 = benchmark_int8(VIDEO_PATH)

    print_results(fp32, int8)


if __name__ == "__main__":
    main()
