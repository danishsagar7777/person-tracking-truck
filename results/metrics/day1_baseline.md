# Day 1 Detection Baseline

## Input Video

- Resolution: 848x480
- FPS: 30
- Frames: 1054
- Duration: 35.13 seconds
- Camera: forward-facing moving truck camera

## Model

- Model: YOLO26n
- Task: person detection
- Class: person only
- Input size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.45

## Performance

- Processing FPS: 13.14 FPS
- Average inference latency: 73.41 ms
- Total processing time: approximately 80.24 seconds

## Detection

- Total person detections: 650
- Average persons/frame: 0.617

## Observations

- Small riders: inspect output video
- Missed people: inspect output video
- False positives: inspect output video
- Partial occlusions: inspect output video
- Motion blur: inspect output video
- Camera motion: significant

## Conclusion

Day 1 establishes the pretrained YOLO person-detection baseline
on the actual moving-truck video.

The detector successfully processes all 1054 frames and detects
650 person instances.

The current baseline processes the video at approximately 13.14 FPS,
while the source video is 30 FPS.

Tracking and camera-motion compensation will be implemented next.
