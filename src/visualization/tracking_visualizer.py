import cv2


class TrackingVisualizer:
    """Draw person tracking results on video frames."""

    def draw(
        self,
        frame,
        tracks,
        frame_number: int = 0,
    ):
        output = frame.copy()

        active_ids = set()

        for track in tracks:
            track_id = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]
            confidence = track["confidence"]

            active_ids.add(track_id)

            label = f"Person ID {track_id} | {confidence:.2f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

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
            f"Active tracks: {len(active_ids)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return output
