# Person Tracking on a Moving Truck

A computer vision project for **detecting and tracking people from a moving-camera video recorded from a truck/vehicle**.

The project combines **YOLO person detection, ByteTrack multi-object tracking, ORB feature matching, RANSAC-based camera motion estimation, Global Motion Compensation (GMC), and OpenVINO INT8 optimization** to build a lightweight CPU-based person tracking pipeline.

---

## 📌 Project Overview

When a camera is mounted on a moving truck, the entire background moves between frames. This makes person tracking more challenging because the motion seen in the video is caused by both:

1. Movement of people in the scene
2. Movement of the camera itself

This project addresses the problem by combining object detection and tracking with camera-motion estimation.

### Main Pipeline

```text
Input Video
     │
     ▼
YOLO Person Detection
     │
     ▼
ORB Feature Detection & Matching
     │
     ▼
RANSAC Camera Motion Estimation
     │
     ▼
Global Motion Compensation
     │
     ▼
ByteTrack
     │
     ▼
Persistent Person IDs
     │
     ▼
Annotated Output Video
```

---

## 🎯 Objectives

* Detect people in a video using YOLO.
* Track detected people across consecutive frames.
* Assign unique tracking IDs to people.
* Handle camera movement caused by the moving truck.
* Estimate global camera motion using ORB and RANSAC.
* Apply Global Motion Compensation (GMC).
* Optimize YOLO inference using OpenVINO INT8.
* Build a CPU-friendly tracking pipeline.
* Evaluate detection, tracking, and inference performance.

---

## 🚀 Key Features

### 1. Person-Only Detection

The detector is configured to detect **class 0 — person**.

This reduces unnecessary processing of other object classes.

### 2. ByteTrack

ByteTrack is used for multi-object tracking and maintaining person IDs across frames.

The tracker uses detection information from consecutive frames to associate detections with existing tracks.

### 3. ORB Camera Motion Estimation

**ORB (Oriented FAST and Rotated BRIEF)** is used to identify and match visual features between consecutive frames.

These features help estimate how the camera itself has moved.

### 4. RANSAC

RANSAC is used to reject incorrect feature matches and estimate a reliable camera transformation from the remaining inlier matches.

The implementation uses:

```python
cv2.estimateAffinePartial2D(
    previous_points,
    current_points,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0
)
```

### 5. Global Motion Compensation

The estimated camera motion is used to compensate for movement caused by the moving camera.

This helps separate camera motion from object motion.

### 6. OpenVINO INT8 Optimization

The YOLO model is converted to an **OpenVINO INT8 model** to reduce inference latency and improve CPU performance.

---

## 🧠 Technologies Used

| Technology | Purpose                              |
| ---------- | ------------------------------------ |
| Python     | Main programming language            |
| OpenCV     | Video processing, ORB and RANSAC     |
| YOLO       | Person detection                     |
| ByteTrack  | Multi-object tracking                |
| OpenVINO   | Model optimization and CPU inference |
| NNCF       | INT8 quantization                    |
| NumPy      | Numerical operations                 |
| Pytest     | Automated testing                    |
| FFmpeg     | Video processing                     |

---

## 📁 Project Structure

```text
person-tracking-truck/
│
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
│
├── results/
│   └── videos/
│       └── person_tracking_int8_bytetrack.mp4
│
├── requirements.txt
├── README.md
└── ...
```

> The exact file structure may vary depending on the current repository version.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd person-tracking-truck
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ Running the Project

## INT8 + ByteTrack Tracking

The main validated tracking pipeline can be executed with:

```bash
python -m scripts.run_int8_tracking
```

The pipeline processes:

```text
data/input/truck_video.mp4
```

and generates:

```text
results/videos/person_tracking_int8_bytetrack.mp4
```

---

# 📊 Performance Results

The final validated **OpenVINO INT8 + ByteTrack** run produced the following results:

| Metric                        |        Result |
| ----------------------------- | ------------: |
| Video Frames                  |          1054 |
| Video FPS                     |        30 FPS |
| Resolution                    |     848 × 480 |
| Total Person Detections       |           699 |
| Tracked Detections            |           305 |
| Average Persons / Frame       |         0.663 |
| Unique Track IDs              |             4 |
| Average Track Length          |  76.25 frames |
| Shortest Track                |     30 frames |
| Longest Track                 |    118 frames |
| Average INT8 Inference        |      18.19 ms |
| Average Frame Processing Time |      21.10 ms |
| Processing Speed              | **47.40 FPS** |

### INT8 vs FP32 Benchmark

A 200-frame benchmark was performed to compare FP32 and INT8 inference.

| Metric                  |     FP32 |         INT8 |
| ----------------------- | -------: | -----------: |
| Frames                  |      200 |          200 |
| Detections              |       84 |           83 |
| Average Persons / Frame |    0.420 |        0.415 |
| Inference Time          | 76.09 ms | **16.36 ms** |
| Processing FPS          |    13.01 |    **60.04** |

The measured INT8 inference latency improvement in this benchmark was approximately **4.65×**.

The detection count differed by only one detection in this 200-frame comparison.

---

# 🎥 Output

The main validated output video is:

```text
results/videos/person_tracking_int8_bytetrack.mp4
```

The output video contains:

* Person bounding boxes
* Tracking IDs
* Detection confidence
* Frame-by-frame tracking

Example annotation format:

```text
Person ID 1 0.82
Person ID 2 0.76
```

---

# 🔬 Camera Motion Estimation

The moving-camera problem is handled using ORB and RANSAC.

### ORB

ORB performs:

```text
Feature Detection
       ↓
Orientation Estimation
       ↓
Descriptor Generation
       ↓
Feature Matching
```

ORB feature matches are obtained between consecutive frames.

### RANSAC

Some feature matches can be incorrect.

RANSAC identifies a transformation supported by the largest set of consistent matches and rejects outliers.

```text
ORB Matches
     │
     ▼
RANSAC
     │
     ├── Inliers
     │
     └── Outliers
     │
     ▼
Camera Transformation
```

The estimated transformation is then used for camera-motion compensation.

---

# 🧪 Testing

Automated tests are included in the project.

Run:

```bash
pytest -v
```

The project validation includes tests for:

* Camera motion estimator initialization
* First-frame identity transformation
* Motion estimator reset
* Tracker initialization
* Video reader metadata

---

# 📈 Development Progress

The project was developed incrementally through several stages:

```text
YOLO Detection
      ↓
ByteTrack Baseline
      ↓
ORB Camera Motion
      ↓
RANSAC Transformation
      ↓
Global Motion Compensation
      ↓
OpenVINO FP32
      ↓
INT8 Optimization
      ↓
INT8 + ByteTrack
      ↓
Final Validation
```

---

# ⚠️ Limitations

The current system has several limitations:

* Small or distant people may be difficult to detect.
* Heavy occlusion can cause tracks to be lost.
* Motion blur can reduce detection quality.
* Very fast camera movement can make feature matching more difficult.
* Tracking IDs can change when a person is lost and later detected again.
* The current system is designed and evaluated primarily on the provided truck video.

The number of unique track IDs should not be interpreted as the total number of people present in the video because people may be missed, occluded, or may not remain in a track for sufficient frames.

---

# 🔮 Future Improvements

Possible future improvements include:

* Improve detection of small/distant people.
* Tune ByteTrack parameters for the target video.
* Improve camera-motion estimation for difficult frames.
* Experiment with stronger YOLO models.
* Evaluate additional trackers such as BoT-SORT and OC-SORT.
* Add standard MOT metrics such as IDF1, MOTA and HOTA.
* Investigate appearance-based Re-ID for long-term identity preservation.
* Optimize the pipeline further for edge devices.

---

# 📚 Project Documentation

Detailed internship documentation is maintained separately and contains:

* Project background
* Problem statement
* Architecture
* Detection methodology
* Tracking methodology
* ORB and RANSAC
* Global Motion Compensation
* INT8 optimization
* Benchmark results
* Testing
* Limitations
* Future improvements

---

# 👨‍💻 Internship Project

**Project:** Person Tracking on a Moving Truck
**Domain:** Computer Vision / Machine Learning
**Focus:** Object Detection, Multi-Object Tracking, Camera Motion Estimation and Edge Optimization

---

## 📄 License

This project was developed as part of an internship project. Add the appropriate license or company-specific usage terms here if required.
