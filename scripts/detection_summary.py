from pathlib import Path
import cv2


def summarize_video(video_path: str):
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frames / fps if fps > 0 else 0.0

    cap.release()

    print("\n========== VIDEO SUMMARY ==========")
    print(f"Video: {path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frames}")
    print(f"Duration: {duration:.2f} seconds")
    print("===================================\n")


def main():
    summarize_video("data/input/truck_video.mp4")


if __name__ == "__main__":
    main()
