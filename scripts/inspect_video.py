import argparse
import json
from pathlib import Path

import cv2


def inspect_video(video_path: str) -> dict:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0.0

    info = {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 3),
        "duration_minutes": round(duration / 60.0, 3),
    }

    cap.release()

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Inspect video metadata."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video",
    )

    args = parser.parse_args()

    info = inspect_video(args.input)

    print("\n========== VIDEO INFORMATION ==========")

    for key, value in info.items():
        print(f"{key}: {value}")

    print("========================================\n")


if __name__ == "__main__":
    main()
