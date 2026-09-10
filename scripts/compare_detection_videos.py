from pathlib import Path

import cv2
from ultralytics import YOLO

from src.detection.int8_detector import INT8PersonDetector


VIDEO_PATH = "data/input/truck_video.mp4"

FP32_MODEL = "yolo26n.pt"
INT8_MODEL = "models/yolo26n_int8/yolo26n_int8.xml"

OUTPUT_DIR = Path("results/videos")

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.45

MAX_FRAMES = 200


def draw_detections(frame, detections, title):

    output = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"person {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        output,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return output


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading FP32 model...")

    fp32_model = YOLO(FP32_MODEL)

    print("Loading INT8 model...")

    int8_detector = INT8PersonDetector(
        model_path=INT8_MODEL,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        iou=IOU,
        device="CPU",
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = (
        OUTPUT_DIR /
        "fp32_vs_int8_comparison.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width * 2, height),
    )

    frame_count = 0

    fp32_total = 0
    int8_total = 0

    while frame_count < MAX_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        # -------------------------------------------------
        # FP32
        # -------------------------------------------------

        results = fp32_model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            classes=[0],
            verbose=False,
        )

        fp32_detections = []

        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            for box, confidence in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):

                x1, y1, x2, y2 = box.astype(int)

                fp32_detections.append(
                    {
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        "confidence": float(
                            confidence
                        ),
                    }
                )

        # -------------------------------------------------
        # INT8
        # -------------------------------------------------

        int8_detections = (
            int8_detector.detect(frame)
        )

        # -------------------------------------------------
        # Draw
        # -------------------------------------------------

        fp32_frame = draw_detections(
            frame,
            fp32_detections,
            "YOLO FP32",
        )

        int8_frame = draw_detections(
            frame,
            int8_detections,
            "OpenVINO INT8",
        )

        comparison = cv2.hconcat(
            [
                fp32_frame,
                int8_frame,
            ]
        )

        writer.write(comparison)

        fp32_total += len(fp32_detections)
        int8_total += len(int8_detections)

        frame_count += 1

        if frame_count % 25 == 0:

            print(
                f"Processed {frame_count}/{MAX_FRAMES} frames"
            )

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

    print(f"Frames processed : {frame_count}")
    print(f"FP32 detections  : {fp32_total}")
    print(f"INT8 detections  : {int8_total}")

    print(
        f"Output            : {output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
