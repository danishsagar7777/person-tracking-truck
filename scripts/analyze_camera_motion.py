import argparse
from pathlib import Path

import cv2
import yaml

from src.tracking.motion_compensation import CameraMotionEstimator


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze camera motion in the truck video."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    input_path = config["video"]["input"]

    output_path = Path(
        "results/videos/camera_motion_analysis.mp4"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    estimator = CameraMotionEstimator()

    frame_number = 0
    total_inliers = 0
    valid_motion_frames = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        transform, inlier_count = estimator.estimate(
            frame
        )

        tx = float(transform[0, 2])
        ty = float(transform[1, 2])

        rotation = float(
            __import__("math").degrees(
                __import__("math").atan2(
                    transform[1, 0],
                    transform[0, 0],
                )
            )
        )

        scale = float(
            (
                transform[0, 0] ** 2
                + transform[1, 0] ** 2
            )
            ** 0.5
        )

        if inlier_count > 0:
            total_inliers += inlier_count
            valid_motion_frames += 1

        output = frame.copy()

        cv2.putText(
            output,
            f"Frame: {frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Camera dx: {tx:.2f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Camera dy: {ty:.2f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Rotation: {rotation:.2f} deg",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Scale: {scale:.3f}",
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Inliers: {inlier_count}",
            (15, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        writer.write(output)

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number} frames | "
                f"Inliers: {inlier_count} | "
                f"dx: {tx:.2f} | "
                f"dy: {ty:.2f}"
            )

    cap.release()
    writer.release()

    average_inliers = (
        total_inliers / valid_motion_frames
        if valid_motion_frames > 0
        else 0.0
    )

    print(
        "\n========== CAMERA MOTION RESULTS =========="
    )
    print(f"Frames processed: {frame_number}")
    print(
        f"Valid motion estimates: "
        f"{valid_motion_frames}"
    )
    print(
        f"Average RANSAC inliers: "
        f"{average_inliers:.2f}"
    )
    print(f"Output: {output_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()
