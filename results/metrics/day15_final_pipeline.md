# Day 15 — Final Person Tracking Pipeline

## Configuration

- Input video: `data/input/truck_video.mp4`
- Detector: OpenVINO INT8
- Model: `models/yolo26n_int8/yolo26n_int8.xml`
- Tracker: `bytetrack`
- GMC: `enabled`

## Results

| Metric | Result |
|---|---:|
| Frames processed | 100 |
| Person detections | 42 |
| Tracked detections | 0 |
| Unique track IDs | 0 |
| Average persons/frame | 0.420 |
| Average inference | 21.90 ms |
| Average frame time | 39.66 ms |
| Processing FPS | 24.49 |
| Average track length | 0.00 frames |
| Shortest track | 0 frames |
| Longest track | 0 frames |

## Output

`results/videos/final_person_tracking.mp4`

## Purpose

This pipeline combines the validated INT8 person detector,
ByteTrack tracking, and optional camera-motion compensation
into a single video-processing pipeline.
