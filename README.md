# Person Tracking on a Moving Truck

## Overview

This project implements a person detection and tracking system for video captured from a moving truck. Since the camera is moving, normal object tracking can become difficult because the movement in consecutive frames is caused by both the camera and the people in the scene.

The system uses YOLO for person detection, ByteTrack for multi-object tracking, and ORB with RANSAC for estimating camera motion. OpenVINO INT8 optimization is also used to reduce inference time and improve CPU performance.

## Objectives

* Detect people in the input video.
* Assign tracking IDs to detected people.
* Maintain person identities across consecutive frames.
* Estimate camera movement between frames.
* Compensate for camera movement during tracking.
* Optimize the detection model using OpenVINO INT8.
* Evaluate the processing speed and tracking performance.

## Approach

The main processing pipeline is:

```text
Input Video
    |
    v
YOLO Person Detection
    |
    +----------------------+
    |                      |
    v                      v
ORB Feature Matching    Detection Boxes
    |                      |
    v                      |
RANSAC Camera Motion       |
Estimation                |
    |                      |
    +----------+-----------+
               |
               v
      Camera Motion Compensation
               |
               v
           ByteTrack
               |
               v
        Person Track IDs
               |
               v
       Annotated Output
```

## Technologies

* Python
* OpenCV
* YOLO
* ByteTrack
* OpenVINO
* NNCF
* NumPy
* Pytest
* FFmpeg

## Project Structure

```text
person-tracking-truck/
|
├── data/
│   ├── input/
│   │   └── truck_video.mp4
│   └── calibration/
│
├── models/
│   └── yolo26n_int8/
│       └── yolo26n_int8.xml
│
├── src/
│   ├── detection/
│   │   └── int8_detector.py
│   └── tracking/
│       └── motion_compensation.py
│
├── scripts/
│   ├── run_detection.py
│   ├── run_int8_tracking.py
│   └── run_final_pipeline.py
│
├── tests/
├── results/
│   └── videos/
├── requirements.txt
└── README.md
```

## Person Detection

YOLO is used to detect objects in each video frame. The system is configured to process only the person class.

The detected bounding boxes and confidence scores are passed to the tracking stage.

## ByteTrack

ByteTrack is used to associate person detections between consecutive frames.

For each detection, the tracker checks whether it can be associated with an existing track. If the detection belongs to an existing track, the same tracking ID is maintained. New detections can create new tracks when they meet the tracker requirements.

The tracker output contains the bounding box and the corresponding person ID.

## Camera Motion Estimation

The camera is mounted on a moving vehicle, so the background changes between frames.

ORB is used to detect and describe visual features in consecutive frames. The feature descriptors are matched using a Hamming-distance based matcher.

RANSAC is then used to remove incorrect feature matches and estimate the dominant camera transformation.

The implementation uses OpenCV's `estimateAffinePartial2D` with RANSAC.

```python
transform, inlier_mask = cv2.estimateAffinePartial2D(
    previous_points,
    current_points,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0
)
```

The estimated transformation is used for camera motion compensation.

## OpenVINO INT8 Optimization

The YOLO model was converted to an OpenVINO INT8 model using representative calibration data.

The INT8 model is used with CPU inference to reduce detection latency.

Model path:

```text
models/yolo26n_int8/yolo26n_int8.xml
```

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd person-tracking-truck
```

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the INT8 and ByteTrack pipeline:

```bash
python -m scripts.run_int8_tracking
```

The generated video is saved to:

```text
results/videos/person_tracking_int8_bytetrack.mp4
```

## Results

The following results were obtained from the validated INT8 and ByteTrack pipeline.

| Metric                        |       Result |
| ----------------------------- | -----------: |
| Input frames                  |         1054 |
| Resolution                    |    848 x 480 |
| Video FPS                     |       30 FPS |
| Person detections             |          699 |
| Tracked detections            |          305 |
| Average persons per frame     |        0.663 |
| Unique track IDs              |            4 |
| Average track length          | 76.25 frames |
| Shortest track                |    30 frames |
| Longest track                 |   118 frames |
| Average INT8 inference        |     18.19 ms |
| Average frame processing time |     21.10 ms |
| Processing speed              |    47.40 FPS |

## INT8 and FP32 Comparison

A 200-frame benchmark was performed to compare the original FP32 inference with the INT8 model.

| Metric                       |      FP32 |      INT8 |
| ---------------------------- | --------: | --------: |
| Frames                       |       200 |       200 |
| Detections                   |        84 |        83 |
| Average detections per frame |     0.420 |     0.415 |
| Inference time               |  76.09 ms |  16.36 ms |
| Processing speed             | 13.01 FPS | 60.04 FPS |

The measured inference latency improvement was approximately 4.65x in this benchmark.

## Testing

Automated tests are included in the project.

Run the tests using:

```bash
pytest -v
```

The tests cover components such as:

* Camera motion estimator initialization
* First-frame handling
* Motion estimator reset
* Tracker initialization
* Video metadata handling

## Output

The main output of the project is:

```text
results/videos/person_tracking_int8_bytetrack.mp4
```

The output video contains the detected person bounding boxes and their corresponding tracking IDs.

## Limitations

The current system has some limitations:

* Small or distant people can be difficult to detect.
* Occlusion can cause a track to be lost.
* Motion blur can reduce detection quality.
* Significant camera movement can affect feature matching.
* A person may receive a new ID if their previous track is lost.

The number of unique track IDs should not be interpreted as the total number of people in the video. Track IDs depend on successful detection and association across frames.

## Future Improvements

Possible improvements include:

* Improving detection of small and distant people.
* Further tuning of ByteTrack parameters.
* Improving camera motion estimation for difficult frames.
* Testing larger or more accurate detection models.
* Adding appearance-based Re-ID.
* Evaluating the tracker using standard MOT metrics such as IDF1, MOTA and HOTA.
* Further optimization for edge devices.

## Project Documentation

Detailed project documentation is available separately and covers the project methodology, implementation, experiments, results, testing, limitations and future improvements.

## Project Information

Project: Person Tracking on a Moving Truck

Domain: Computer Vision and Machine Learning

Main Areas: Object Detection, Multi-Object Tracking, Camera Motion Estimation and Model Optimization
