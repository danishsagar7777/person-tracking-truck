# Day 10 — FP32 vs INT8 ByteTrack Benchmark

## Test

Same input video:

`data/input/truck_video.mp4`

Same detector:

`YOLO26n`

Same target:

`COCO person class (class 0)`

Same tracker:

`ByteTrack`

### FP32

Model:

`yolo26n.pt`

### INT8

Model:

`models/yolo26n_int8/yolo26n_int8.xml`

---

## Results

| Metric | FP32 + ByteTrack | INT8 + ByteTrack |
|---|---:|---:|
| Frames | TBD | 1054 |
| Person detections | TBD | 699 |
| Tracked detections | TBD | 305 |
| Unique Track IDs | TBD | 4 |
| Average track length | TBD | 76.25 |
| Shortest track | TBD | 30 |
| Longest track | TBD | 118 |
| Avg inference time | TBD | 17.20 ms |
| Avg frame time | TBD | 20.00 ms |
| Processing FPS | TBD | 50.00 |

---

## Important

Detection count is not an accuracy metric.

A proper accuracy comparison requires ground-truth annotations.

The main Day 10 objective is to compare:

1. inference latency
2. processing FPS
3. tracking behavior
4. number of track IDs
5. track lifecycle
6. visual consistency
7. computational efficiency

---

## Conclusion

INT8 should provide significantly lower inference latency than FP32.

The final conclusion will be written after the FP32 + ByteTrack benchmark is completed.
