
# Pose Tester: Live Human Pose Estimation

> **Important:** To run this project, you **must** use the following command in your terminal:

```bash
python -m uvicorn main:app --reload
```

> ⚠️ Also, download the **MediaPipe Pose Landmarker heavy `.tasks` file** from [here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) and place it in the same directory as this project.

This is a **real-time human pose tracking web app** that captures your pose from a live webcam feed. Great for **video game developers**, **artists practicing anatomy**, or **casual fun**.

> ⚠️ Note: This is a simple model. Leg detection and complex poses may not always work perfectly.

---

## Features

* Live webcam feed with human pose overlay.
* 5-second countdown capture: press the button to store your pose into the box below the feed.
* Captures snapshots and sends them back to the frontend for immediate preview.
* Saves captured images as JPEGs in the project directory.
* Uses **MediaPipe**, **OpenCV**, and Python packages listed below.

---

## Exact Python Packages Required

You will need the following packages installed for this project:

```bash
pip install fastapi
pip install uvicorn
pip install opencv-python
pip install jinja2
pip install numpy
pip install pillow
pip install mp3dpose
```

> Make sure you also have the `.tasks` model file for **PoseEstimator** in the project directory for it to function correctly.

---

## How to Run

1. Clone the repository.
2. Download the **Pose Landmarker heavy `.tasks` file** from [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) and place it in the project folder.
3. Install the required packages (listed above).
4. Run the app using the **exact command**:

```bash
python -m uvicorn main:app --reload
```

5. Open your browser at `http://127.0.0.1:8000/` to see the live webcam feed.
6. Press the **“Capture after 5 Second Countdown”** button to take a snapshot of your pose.
7. To restart the server, press **Command + C** (Mac) or **Control + C** (Windows/Linux) in your terminal.

---

## Usage

* **Countdown Button**: Starts a 5-second timer, then captures your pose.
* **Captured Image**: Appears below the live feed for preview.
* Save poses for **animation reference, model creation, drawing practice**, or **fun experimentation**.

---

## Limitations

* Legs or complex poses may not always be detected correctly.
* Best used with **clear lighting** and a webcam that captures your full body.
* Simple model intended for **experimentation and learning**, not production-level pose tracking.

---

## Tech Stack

* **Python** – backend logic
* **HTML** - frontend logic
* **FastAPI** – web server & API
* **OpenCV** – webcam capture & image processing
* **MediaPipe Pose Landmarker** – pose estimation
* **NumPy** – numerical processing
* **Jinja2** – HTML templating

---

## Acknowledgements

* This project was created for Hackalytics: Golden Byte 2026 at Georgia Institute of Technology.
* Thanks to [MediaPipe](https://mediapipe.dev/) for open-source pose tracking solutions.

---
