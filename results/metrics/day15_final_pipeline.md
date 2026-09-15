# Final Person Tracking Pipeline

## Configuration

- Input video: `data/input/truck_video.mp4`
- Detector: OpenVINO INT8
- Model: `models/yolo26n_int8/yolo26n_int8.xml`
- Tracker: `bytetrack`
- GMC: `enabled`

## Architecture

The final pipeline uses the following processing flow:

1. Read the original truck-camera frame.
2. Estimate camera motion using ORB.
3. Run the OpenVINO INT8 person detector on the original frame.
4. Transform detection bounding boxes into the GMC reference coordinate system.
5. Pass transformed person detections to ByteTrack.
6. Transform tracked boxes back to the original frame.
7. Draw tracking IDs on the original frame.
8. Save the final annotated video.

The video frame itself is not warped before detection.

## Results

| Metric | Result |
|---|---:|
| Frames processed | 1054 |
| Person detections | 699 |
| Tracked detections | 305 |
| Unique track IDs | 4 |
| Average persons/frame | 0.663 |
| Average inference | 17.50 ms |
| Average frame time | 34.55 ms |
| Processing FPS | 28.94 |
| Average track length | 76.25 frames |
| Shortest track | 30 frames |
| Longest track | 118 frames |

## GMC

- Valid GMC frames: 1053
- Invalid GMC frames: 0
- Average GMC inliers: 257.15

## Output

`results/videos/final_person_tracking.mp4`

## Purpose

This pipeline combines:

- OpenVINO INT8 person detection
- ByteTrack multi-object tracking
- ORB-based global camera-motion estimation
- GMC-aware detection coordinate transformation
- Tracking visualization
- Performance metrics

The detector operates on the original frame to avoid image degradation caused by repeated cumulative warping.
