# Day 3 — Moving Camera Compensation

## Objective

Analyze global camera motion caused by the truck-mounted camera
and establish a baseline motion-estimation module.

## Input Video

- Resolution: 848x480
- FPS: 30
- Frames: 1054
- Duration: 35.13 seconds
- Camera: forward-facing moving truck camera

## Detector

- Model: YOLO26n
- Task: person detection
- Class: person only
- Image size: 640
- Confidence: 0.25
- IoU threshold: 0.45

## Tracker

- Tracker: ByteTrack
- Persistent tracking: enabled

## Camera Motion Estimation

- Feature detector: ORB
- Matcher: BFMatcher
- Transformation: affine partial 2D
- Robust estimation: RANSAC

## Results

- Processing FPS:
- Average inference latency:
- Unique track IDs:
- Total tracked detections:
- Valid motion-estimation frames:
- Average RANSAC inliers:

## Observations

### Camera Motion

- Horizontal movement:
- Vertical movement:
- Rotation:
- Scale changes:

### Tracking

- ID stability:
- ID switches observed:
- Track fragmentation:
- Occlusions:
- Small/distant people:

### Motion Estimation Quality

- Feature matching quality:
- Number of invalid estimates:
- Static background features:
- Moving-object interference:

## Limitations

The current implementation estimates global camera motion
but does not yet modify ByteTrack's association logic.

## Conclusion

Day 3 establishes a camera-motion estimation baseline for the
moving truck camera.

The next stage is to evaluate BoT-SORT, which provides
global motion compensation and additional tracking mechanisms,
and compare it against the ByteTrack baseline.
