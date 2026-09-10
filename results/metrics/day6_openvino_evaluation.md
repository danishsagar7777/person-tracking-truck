# Day 6 — OpenVINO Optimization Evaluation

## Objective

Evaluate OpenVINO inference for the YOLO person detector and
compare it against the original PyTorch CPU baseline.

The goal is to determine whether OpenVINO improves inference
latency and processing FPS while maintaining similar detection
behavior.

## Hardware

- CPU: Snapdragon X Plus
- Operating system: Linux / WSL
- Python: 3.14.4

## Video

- Resolution: 848x480
- Source FPS: 30 FPS
- Frames: 1054
- Duration: approximately 35.13 seconds
- Detection class: person only

## YOLO Configuration

- Model: YOLO26n
- Image size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Person class ID: 0

## Results

| Metric | PyTorch | OpenVINO FP32 | OpenVINO FP16 |
|---|---:|---:|---:|
| Frames | 1054 | TODO | TODO |
| Person detections | 650 | TODO | TODO |
| Average persons/frame | 0.617 | TODO | TODO |
| Average inference time (ms) | 73.41 | TODO | TODO |
| Processing FPS | 13.14 | TODO | TODO |

## Accuracy / Detection Behavior

The number of person detections is compared between the
PyTorch and OpenVINO implementations.

A similar detection count indicates broadly similar behavior,
but detection count alone is not a complete accuracy metric.

A proper accuracy comparison should use annotated ground truth.

## Performance

Compare:

- Average inference latency
- Processing FPS
- Detection count

Calculate the relative speed improvement between PyTorch and
OpenVINO.

## Conclusion

TODO after benchmark results.

## Limitations

This experiment does not yet measure:

- Precision
- Recall
- mAP
- IDF1
- HOTA
- MOTA

Those metrics require appropriate ground-truth annotations.

## Next Step

Day 7 will evaluate INT8 quantization using representative
calibration data and compare INT8 against the FP32/FP16
baselines.
