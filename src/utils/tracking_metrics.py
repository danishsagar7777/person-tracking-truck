from collections import defaultdict
from typing import Dict, List


class TrackingMetrics:
    """
    Collect and calculate basic multi-object tracking statistics.

    These metrics do not require ground-truth identity annotations.
    They are intended for baseline comparison between trackers.
    """

    def __init__(self):
        self.frame_count = 0
        self.total_tracks = 0

        self.unique_track_ids = set()

        self.track_first_frame: Dict[int, int] = {}
        self.track_last_frame: Dict[int, int] = {}

        self.track_observations: Dict[int, int] = defaultdict(int)

        self.active_tracks_per_frame: List[int] = []

    def update(self, frame_number: int, tracks: List[Dict]) -> None:
        """
        Update metrics using tracks from one frame.
        """

        self.frame_count += 1

        active_ids = set()

        for track in tracks:
            track_id = int(track["track_id"])

            active_ids.add(track_id)
            self.unique_track_ids.add(track_id)

            self.total_tracks += 1
            self.track_observations[track_id] += 1

            if track_id not in self.track_first_frame:
                self.track_first_frame[track_id] = frame_number

            self.track_last_frame[track_id] = frame_number

        self.active_tracks_per_frame.append(len(active_ids))

    def calculate(self) -> Dict:
        """
        Calculate summary tracking metrics.
        """

        unique_ids = len(self.unique_track_ids)

        if self.frame_count > 0:
            average_tracks_per_frame = (
                self.total_tracks / self.frame_count
            )
        else:
            average_tracks_per_frame = 0.0

        track_lengths = []

        for track_id in self.unique_track_ids:
            first_frame = self.track_first_frame[track_id]
            last_frame = self.track_last_frame[track_id]

            length = last_frame - first_frame + 1
            track_lengths.append(length)

        if track_lengths:
            average_track_length = sum(track_lengths) / len(track_lengths)
            longest_track = max(track_lengths)
            shortest_track = min(track_lengths)
        else:
            average_track_length = 0.0
            longest_track = 0
            shortest_track = 0

        if self.active_tracks_per_frame:
            maximum_active_tracks = max(self.active_tracks_per_frame)
        else:
            maximum_active_tracks = 0

        return {
            "frames": self.frame_count,
            "total_tracked_detections": self.total_tracks,
            "unique_track_ids": unique_ids,
            "average_tracks_per_frame": average_tracks_per_frame,
            "average_track_length_frames": average_track_length,
            "shortest_track_frames": shortest_track,
            "longest_track_frames": longest_track,
            "maximum_active_tracks": maximum_active_tracks,
        }

    def get_track_table(self) -> List[Dict]:
        """
        Return one record for every track ID.
        """

        table = []

        for track_id in sorted(self.unique_track_ids):
            first_frame = self.track_first_frame[track_id]
            last_frame = self.track_last_frame[track_id]

            table.append(
                {
                    "track_id": track_id,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "track_length_frames": last_frame - first_frame + 1,
                    "observations": self.track_observations[track_id],
                }
            )

        return table
