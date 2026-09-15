# Day 16 — Final Validation & Performance Benchmark

## 1. Objective

Day 16 validates the final person-tracking pipeline and compares:

1. OpenVINO INT8 + ByteTrack baseline
2. Final pipeline with GMC disabled
3. Final pipeline with GMC enabled
4. FP32 vs INT8 detector performance

The project processes an uploaded truck-camera video and detects/tracks only persons.

---

## 2. Input

- Video: `data/input/truck_video.mp4`
- Resolution: 848x480
- Source FPS: 30 FPS
- Source frames: 1054

---

## 3. INT8 + ByteTrack Baseline

| Metric | Result |
|---|---:|
| Frames processed | 1054 |
| Person detections | 699 |
| Tracked detections | 305 |
| Unique Track IDs | 4 |
| Average track length | 76.25 frames |
| Shortest track | 30 frames |
| Longest track | 118 frames |
| Average inference | 16.96 ms |
| Average frame time | 19.71 ms |
| Processing FPS | 50.74 |

---

## 4. Final Pipeline — GMC OFF

| Metric | Result |
|---|---:|
| Frames processed | REPLACE |
| Person detections | REPLACE |
| Tracked detections | REPLACE |
| Unique Track IDs | REPLACE |
| Average persons/frame | REPLACE |
| Average inference | REPLACE |
| Average frame time | REPLACE |
| Processing FPS | REPLACE |
| Average track length | REPLACE |
| Shortest track | REPLACE |
| Longest track | REPLACE |

Purpose:

The GMC-disabled run acts as a control experiment to verify that the final pipeline preserves the working INT8 + ByteTrack behavior.

---

## 5. Final Pipeline — GMC ON

| Metric | Result |
|---|---:|
| Frames processed | REPLACE |
| Person detections | REPLACE |
| Tracked detections | REPLACE |
| Unique Track IDs | REPLACE |
| Average persons/frame | REPLACE |
| Average inference | REPLACE |
| Average frame time | REPLACE |
| Processing FPS | REPLACE |
| Average track length | REPLACE |
| Shortest track | REPLACE |
| Longest track | REPLACE |
| GMC valid frames | REPLACE |
| GMC invalid frames | REPLACE |
| Average GMC inliers | REPLACE |

---

## 6. FP32 vs INT8 Detector Benchmark

| Metric | FP32 | INT8 |
|---|---:|---:|
| Frames | 200 | 200 |
| Person detections | 84 | 83 |
| Average persons/frame | 0.420 | 0.415 |
| Average inference | 76.09 ms | 16.36 ms |
| FPS | 13.01 | 60.04 |

### INT8 latency improvement

INT8 inference was approximately:

**4.65× faster in latency**

calculated as:

`76.09 / 16.36 = 4.65×`

The detection-count difference was:

`83 - 84 = -1`

This is a detection-count comparison, not a formal accuracy metric.

---

## 7. Final Architecture

```text
Truck Camera Video
        |
        v
Original Frame
        |
        +----------------------+
        |                      |
        v                      v
   ORB GMC              OpenVINO INT8
Camera Motion            Person Detector
Estimation                    |
        |                      |
        |                      v
        |                Person Boxes
        |                      |
        +---- Transform -------+
              Boxes
                |
                v
           ByteTrack
                |
                v
          Persistent IDs
                |
                v
       Transform to Original
          Coordinates
                |
                v
       Annotated Output Video
