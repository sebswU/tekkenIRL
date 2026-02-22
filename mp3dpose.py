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

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_heavy.task'
MODEL_PATH = 'pose_landmarker_heavy.task'


class PoseEstimator:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            print(f"Downloading pose landmarker model to {model_path}...")
            # macos python does not have certifi installed by default, so we need to disable SSL verification to download the model
            # in order to make the app secure in the future, we will need to bundle the model with the app or use a secure way to download the model
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(MODEL_URL, context=ssl_ctx) as response, \
                 open(model_path, 'wb') as out_file:
                out_file.write(response.read())
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _draw_landmarks(self, image, pose_landmarks_list):
        """draw the landmarks on the image"""
        annotated = image.copy()
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                annotated,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
            )
        return annotated

    def _detect(self, bgr_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        return self.landmarker.detect(mp_image)

    def process_image(self, image_path):
        """process image for pose estimation"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        input_image = cv2.imread(image_path)
        results = self._detect(input_image)
        annotated_image = self._draw_landmarks(input_image, results.pose_landmarks)
        cv2.imshow('Pose Estimation Image', annotated_image)
        ax.clear()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            x = [lm.x for lm in landmarks]
            y = [lm.y for lm in landmarks]
            z = [lm.z for lm in landmarks]
            ax.scatter(x, y, z)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path):
        """process video for pose estimation"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self._detect(frame)
            annotated_frame = self._draw_landmarks(frame, results.pose_landmarks)
            cv2.imshow('Pose Estimation', annotated_frame)
            ax.clear()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                x = [lm.x for lm in landmarks]
                y = [lm.y for lm in landmarks]
                z = [lm.z for lm in landmarks]
                ax.scatter(x, y, z)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            plt.pause(0.001)
            plt.show(block=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        results = self._detect(frame)
        annotated = self._draw_landmarks(frame, results.pose_landmarks)
        return annotated


