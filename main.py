from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import ssl
import urllib.request
from mp3dpose import PoseEstimator

pe = PoseEstimator()
latest_frame = None
app = FastAPI()

templates = Jinja2Templates(directory="templates") #jinja used to later send data from backend to frontend 

def generate_frames():
    camera = cv2.VideoCapture(0)
    global latest_frame

    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = pe.process_frame(frame)

        latest_frame = processed_frame

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse) #links to html file
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#get webcam feed from video in html file
@app.post("/capture")
def capture_photo():
    global latest_frame
    if latest_frame is not None:
        cv2.imwrite("captured.jpg", latest_frame)
        return {"status": "Photo saved"}
    return {"status": "No frame available"}

@app.get("/webcam")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )