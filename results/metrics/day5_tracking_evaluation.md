# Day 5 — Tracking Evaluation

## Objective

Evaluate ByteTrack and BoT-SORT on the same truck-mounted camera
video using tracking lifecycle and runtime metrics.

## Video

- Resolution: 848x480
- Source FPS: 30 FPS
- Frames: 1054
- Duration: approximately 35.13 seconds
- Detection class: person only
- YOLO model: yolo26n
- Image size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.45

## Tracker Comparison

| Metric | ByteTrack | BoT-SORT |
|---|---:|---:|
| Frames | 1054 | 1054 |
| Total tracked detections | 608 | 603 |
| Unique track IDs | 12 | 16 |
| Average tracks/frame | 0.577 | 0.572 |
| Average track length | 59.33 | 40.88 |
| Shortest track | 2 | 1 |
| Longest track | 287 | 163 |
| Maximum active tracks | 3 | 3 |
| Average inference time | 74.56 ms | 80.69 ms |
| Processing FPS | 13.31 | 12.30 |

## Track Stability

ByteTrack generated 12 unique track IDs while BoT-SORT generated
16 unique track IDs.

ByteTrack also achieved a higher average track length:

- ByteTrack: 59.33 frames
- BoT-SORT: 40.88 frames

The longest ByteTrack track lasted 287 frames, compared with
163 frames for BoT-SORT.

These results indicate that ByteTrack maintained longer track
lifecycles on this particular video.

## Runtime Performance

ByteTrack achieved:

- 74.56 ms average inference time
- 13.31 FPS processing speed

BoT-SORT achieved:

- 80.69 ms average inference time
- 12.30 FPS processing speed

Therefore, ByteTrack was faster under the current configuration.

## Preliminary Conclusion

Under the current YOLO configuration and evaluation video,
ByteTrack produced the stronger baseline according to the measured
tracking lifecycle and runtime metrics.

ByteTrack produced:

- More total tracked detections
- Fewer unique track IDs
- Longer average tracks
- A longer maximum track
- Lower average inference latency
- Higher processing FPS

However, these results do not establish that ByteTrack is
universally superior to BoT-SORT.

The evaluation is currently based on a single video and does not
contain ground-truth identity annotations.

## Important Limitation

Unique track IDs cannot be interpreted directly as the number of
physical people.

Without ground-truth identity annotations, true ID switches,
IDF1, HOTA, MOTA and MOTP cannot be reliably calculated.

The current evaluation therefore focuses on tracker lifecycle
statistics and runtime performance.

## ID Churn Indicator

A simple baseline indicator can be calculated as:

ID churn = unique track IDs / total tracked detections

ByteTrack:

12 / 608 = 0.0197

BoT-SORT:

16 / 603 = 0.0265

This is a project-specific diagnostic indicator and is not a
standard MOT metric such as IDF1 or ID-switch count.

## Next Step

Create a manually annotated ground-truth evaluation subset.

Use the ground truth to evaluate:

- Detection recall
- Detection precision
- ID switches
- Track fragmentation
- IDF1
- HOTA
- MOTA

After establishing the tracking baseline, optimize the YOLO model
using OpenVINO and evaluate the effect of FP32, FP16 and INT8
optimization on accuracy, latency and FPS.
