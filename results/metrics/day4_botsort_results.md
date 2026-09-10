# Day 4 — BoT-SORT Tracking Baseline

## Objective

Evaluate BoT-SORT on the moving-truck camera video and compare
its tracking behavior against the Day 2 ByteTrack baseline.

## Input Video

- Resolution: 848x480
- FPS: 30
- Frames: 1054
- Duration: 35.13 seconds
- Camera: forward-facing moving truck camera

## Detector

- Model: YOLO26n
- Class: person only
- Image size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.45

## ByteTrack Baseline

- Tracker: ByteTrack
- Tracked detections: 608
- Unique track IDs: 12
- Average tracked persons/frame: 0.577
- Average inference time: 77.08 ms
- Processing FPS: 12.52

## BoT-SORT

- Tracker: BoT-SORT
- Tracked detections: 603
- Unique track IDs: 16
- Average tracked persons/frame: 0.572
- Average inference time: 82.64 ms
- Processing FPS: 11.71

## Camera Motion

- Average motion inliers:
- Valid motion frames:

## Qualitative Comparison

### ID Stability

ByteTrack:

BoT-SORT:

### Camera Motion

ByteTrack:

BoT-SORT:

### Occlusion

ByteTrack:

BoT-SORT:

### Small/Distant People

ByteTrack:

BoT-SORT:

### Track Fragmentation

ByteTrack:

BoT-SORT:

## Performance Comparison

| Metric | ByteTrack | BoT-SORT |
|---|---:|---:|
| Tracked detections | 608 | |
| Unique IDs | 12 | |
| Persons/frame | 0.577 | |
| Inference latency | 77.08 ms | |
| Processing FPS | 12.52 | |

## Conclusion

BoT-SORT was evaluated using the same YOLO model, input video,
image size, confidence threshold, and IoU threshold as ByteTrack.

The comparison will be used to determine which tracker is more
appropriate for the moving-camera person-tracking system.
