import cv2
from pathlib import Path


VIDEO_PATH = "data/input/truck_video.mp4"

OUTPUT_DIR = Path("data/validation/images")

NUMBER_OF_FRAMES = 100


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = [
        int(
            i * (total_frames - 1)
            / (NUMBER_OF_FRAMES - 1)
        )
        for i in range(NUMBER_OF_FRAMES)
    ]

    saved = 0

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            index,
        )

        ret, frame = cap.read()

        if not ret:
            continue

        output_path = (
            OUTPUT_DIR /
            f"frame_{index:05d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        saved += 1

    cap.release()

    print("=" * 60)
    print("VALIDATION DATASET")
    print("=" * 60)
    print(f"Video frames : {total_frames}")
    print(f"Requested    : {NUMBER_OF_FRAMES}")
    print(f"Saved        : {saved}")
    print(f"Directory    : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
