# Person Tracking on Moving Truck

## 1. Project Objective

The objective of this project is to build a person-only multi-object tracking system for a camera mounted on a moving truck.

The system detects people in video frames, assigns persistent tracking IDs, compensates for camera motion, and uses OpenVINO INT8 optimization for efficient CPU inference.

---

## 2. Problem Statement

A camera mounted on a moving truck introduces global camera motion into the video.

Therefore, the apparent movement of a person in the image is caused by both:

- movement of the person
- movement of the camera

A tracking system must handle this camera motion while maintaining stable person identities.

---

## 3. System Components

The final system consists of:

1. Video Reader
2. YOLO Person Detector
3. OpenVINO INT8 Detector
4. ORB Feature Detection
5. RANSAC-based Camera Motion Estimation
6. Global Motion Compensation
7. ByteTrack
8. Visualization
9. Performance Metrics

---

## 4. Detection

YOLO is used for object detection.

Only the person class is retained.

COCO class:

- Class ID 0 = person

The system therefore ignores other detected object classes.

---

## 5. Tracking

ByteTrack is used for multi-object tracking.

The detector provides bounding boxes and confidence scores.

ByteTrack associates detections between frames and produces track IDs.

---

## 6. Camera Motion Compensation

The camera is mounted on a moving truck, so the camera itself continuously moves.

ORB is used to detect visual features between consecutive frames.

Feature correspondences are obtained using descriptor matching.

RANSAC is then used to estimate a robust camera transformation while rejecting incorrect matches.

The estimated transformation is used for global motion compensation.

---

## 7. INT8 Optimization

The YOLO model was converted to an OpenVINO INT8 representation.

INT8 inference was benchmarked against the FP32 baseline.

### 200-frame detector benchmark

| Metric | FP32 | INT8 |
|---|---:|---:|
| Frames | 200 | 200 |
| Detections | 84 | 83 |
| Average inference | 76.09 ms | 16.36 ms |
| FPS | 13.01 | 60.04 |

The measured INT8 latency speedup was approximately 4.65x.

The detection count difference between the two runs was 1 detection and should not by itself be interpreted as an accuracy metric.

---

## 8. Final Pipeline Benchmark

The complete pipeline was executed on the full 1054-frame truck video.

### Final result

| Metric | Result |
|---|---:|
| Frames | 1054 |
| Person detections | 699 |
| Tracked detections | 304 |
| Unique IDs | 4 |
| Average INT8 inference | 18.22 ms |
| Average frame time | 39.21 ms |
| Processing FPS | 25.50 |
| GMC valid frames | 1053 |
| GMC invalid frames | 1 |
| Average GMC inliers | 257.15 |

---

## 9. Software Validation

The project contains automated tests covering the main components.

Final validation:

- 15 tests passed
- Python compilation completed successfully

---

## 10. Final Pipeline

The final processing flow is:

Input Truck Video
    ↓
Video Reader
    ↓
Original Frame
    ├── ORB + RANSAC → Camera Motion
    │
    └── YOLO OpenVINO INT8 → Person Detections
                         ↓
                 Global Motion Compensation
                         ↓
                      ByteTrack
                         ↓
                   Persistent IDs
                         ↓
                  Annotated Video

---

## 11. Limitations

The current system is evaluated on the provided truck video.

The number of unique IDs and tracked detections should not be interpreted as standard tracking accuracy metrics without suitable ground-truth identity annotations.

Small or distant people, occlusion, motion blur, and difficult road scenes can affect detection and tracking quality.

---

## 12. Future Improvements

Possible future improvements include:

- Better handling of small and distant people
- More robust identity association
- Improved camera-motion estimation
- Re-identification features for longer-term identity persistence
- Additional tracking-quality evaluation using ground-truth annotations
- Further optimization for deployment on edge hardware

---

## 13. Final Status

The core person-tracking system is implemented and validated.

The final pipeline combines:

- YOLO person detection
- OpenVINO INT8 inference
- ORB + RANSAC camera-motion estimation
- Global Motion Compensation
- ByteTrack
- Video visualization
- Automated tests
- Performance benchmarking

The remaining work is primarily final documentation, cleanup, visual verification, and demo preparation.
