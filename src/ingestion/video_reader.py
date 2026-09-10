from pathlib import Path
from typing import Iterator, Tuple

import cv2


class VideoReader:
    """Reusable OpenCV video reader."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {self.video_path}"
            )

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0

        return self.frame_count / self.fps

    def frames(self) -> Iterator[Tuple[int, object]]:
        """Yield (frame_number, frame) pairs."""
        frame_number = 0

        while True:
            success, frame = self.cap.read()

            if not success:
                break

            yield frame_number, frame
            frame_number += 1

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
