# Day 11 — OpenVINO INT8 + ByteTrack + ORB GMC

## Configuration

- Video: `data/input/truck_video.mp4`
- INT8 model: `models/yolo26n_int8/yolo26n_int8.xml`
- Detector: OpenVINO INT8
- Tracker: ByteTrack
- GMC: ORB + RANSAC affine motion estimation
- ORB features: 500
- ORB match ratio: 0.75
- ORB minimum matches: 10

## Results

- Frames processed: 1054
- Total person detections: 276
- Average persons/frame: 0.262
- Total tracked detections: 85
- Average tracks/frame: 0.081
- Unique Track IDs: 1
- Average track length: 85.00 frames
- Shortest track: 85 frames
- Longest track: 85 frames
- Average INT8 inference: 17.03 ms
- Average frame time: 34.32 ms
- Processing FPS: 29.14
- Average ORB inliers: 256.91
- Minimum ORB inliers: 0
- Maximum ORB inliers: 374
- Wall time: 36.17 sec
- Output video: `results/videos/person_tracking_int8_bytetrack_gmc.mp4`
