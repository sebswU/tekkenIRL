# tekkenIRL
Real life interactive experience

## Overview

`pipeline.py` provides a basic skeleton for real-time **3D pose estimation** and
**scene depth estimation** using [OpenCV](https://opencv.org/) and
[Ultralytics](https://docs.ultralytics.com/).

### How it works

1. **Video capture** – OpenCV reads frames from a webcam or video file.
2. **2-D pose estimation** – A YOLO pose model detects body keypoints for each
   person in the frame.
3. **Depth estimation** – A depth-anything model produces a per-pixel relative
   depth map.
4. **3-D lifting** – Each 2-D keypoint is lifted to 3-D by sampling the depth
   map at its pixel location, yielding `(x, y, z)` coordinates.
5. **Visualization** – The YOLO pose skeleton is overlaid on the frame and
   displayed in a window.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run with the default webcam (camera index 0):

```bash
python pipeline.py
```

Press **q** to quit.

### Python API

```python
from pipeline import PoseEstimationPipeline
import cv2

pipeline = PoseEstimationPipeline(
    pose_model="yolov8n-pose.pt",   # YOLO pose weights
    depth_model="depth-anything-small.pt",  # depth-anything weights
    device="cpu",                   # or "cuda:0"
)

# Process a single frame
frame = cv2.imread("image.jpg")
output = pipeline.process_frame(frame)

print(output["depth_map"].shape)      # (H, W) depth map
print(output["keypoints_3d"])         # list of (N_kps, 3) arrays per person
```

## Running tests

```bash
pip install pytest
pytest tests/
```
