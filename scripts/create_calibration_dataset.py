from pathlib import Path

import cv2


VIDEO_PATH = "data/input/truck_video.mp4"
OUTPUT_DIR = Path("data/calibration")

NUM_IMAGES = 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not determine video frame count.")

    step = max(total_frames // NUM_IMAGES, 1)

    saved = 0
    frame_number = 0

    while saved < NUM_IMAGES:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % step == 0:
            output_path = OUTPUT_DIR / f"calibration_{saved:04d}.jpg"

            success_write = cv2.imwrite(str(output_path), frame)

            if success_write:
                saved += 1

        frame_number += 1

    cap.release()

    print("=" * 60)
    print("CALIBRATION DATASET")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Total video frames: {total_frames}")
    print(f"Requested images: {NUM_IMAGES}")
    print(f"Saved images: {saved}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
