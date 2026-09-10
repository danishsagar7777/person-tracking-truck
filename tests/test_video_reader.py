from src.ingestion.video_reader import VideoReader


def test_video_reader_metadata():
    reader = VideoReader("data/input/truck_video.mp4")

    assert reader.width == 848
    assert reader.height == 480
    assert reader.fps > 0
    assert reader.frame_count == 1054
    assert reader.duration_seconds > 0

    reader.release()
