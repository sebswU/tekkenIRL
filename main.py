from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import base64
from mp3dpose import PoseEstimator

# --- Pose estimation setup ---
pe = PoseEstimator()
latest_frame = None

# --- FastAPI setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Open camera globally ---
camera = cv2.VideoCapture(0)

def generate_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            continue  # skip if failed

        processed_frame = pe.process_frame(frame)
        latest_frame = processed_frame

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/webcam")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/capture")
def capture_photo():
    global latest_frame
    if latest_frame is None:
        return {"status": "No frame available"}
    
    # Save to disk
    cv2.imwrite("captured.jpg", latest_frame)

    # Convert to base64 to send back to frontend
    _, buffer = cv2.imencode('.jpg', latest_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"status": "Photo saved", "image": image_base64}