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

app = FastAPI()

templates = Jinja2Templates(directory="templates") #jinja used to later send data from backend to frontend 
camera = cv2.VideoCapture(0)
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        pe.process_webcam(camera)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse) #links to html file
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#get webcam feed from video in html file
@app.get("/video")
def video_feed(): #stores induvidual frames for video processing
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
