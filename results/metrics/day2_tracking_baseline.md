# Day 2 Tracking Baseline

## Input

- Video: truck_video.mp4
- Resolution: 848x480
- FPS: 30
- Frames: 1054
- Duration: 35.13 seconds
- Camera: forward-facing moving truck camera

## Detector

- Model: YOLO26n
- Class: person only
- Input size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.45

## Tracker

- Tracker: ByteTrack
- Persistent tracking: enabled

## Performance

- Processing FPS:12.76
- Average inference latency:75.59
- Total tracked detections:4
- Unique track IDs:12

## Tracking Observations

### ID Stability

- Person 1:4
- Person 2:7
- Person 3:8
- Person 4:12
### ID Switches

- Observed:
- Approximate count:

### Track Fragmentation

- Observed:
- Approximate count:

### Occlusion

- People temporarily disappear:
- IDs recovered after disappearance:

### Moving Camera

- Camera movement affects tracking:
- Large scene movement:
- Tracker behavior:

### Small/Distant People

- Detection stability:
- Track stability:

## Conclusion

ByteTrack establishes the Day 2 tracking baseline.

The next stage will investigate camera-motion compensation
and compare ByteTrack with BoT-SORT for the moving-camera scenario.
