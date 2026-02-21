"""Basic skeleton OpenCV/Ultralytics pipeline for 3D pose estimation and scene depth estimation.

Pipeline
--------
1. Capture video frames with OpenCV.
2. Run a YOLO pose model to detect 2D body keypoints per person.
3. Run a depth-estimation model to produce a per-pixel depth map.
4. Lift each 2D keypoint to 3D by sampling the depth map at its location.
5. Annotate and display results in real time.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

# Default model weights (downloaded automatically by Ultralytics on first use)
_DEFAULT_POSE_MODEL = "yolov8n-pose.pt"
_DEFAULT_DEPTH_MODEL = "depth-anything-small.pt"


class PoseEstimationPipeline:
    """Real-time 3D pose estimation using YOLO pose + depth estimation.

    Parameters
    ----------
    pose_model:
        Path or name of the Ultralytics YOLO pose model weights.
    depth_model:
        Path or name of the Ultralytics depth-estimation model weights.
    device:
        Inference device, e.g. ``"cpu"``, ``"cuda:0"``, or ``"mps"``.
    """

    def __init__(
        self,
        pose_model: str = _DEFAULT_POSE_MODEL,
        depth_model: str = _DEFAULT_DEPTH_MODEL,
        device: str = "cpu",
    ) -> None:
        self.pose_model = YOLO(pose_model)
        self.depth_model = YOLO(depth_model)
        self.device = device

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def estimate_pose_2d(self, frame: np.ndarray):
        """Run YOLO pose estimation on *frame* and return the results list."""
        return self.pose_model(frame, device=self.device, verbose=False)

    def estimate_depth(self, frame: np.ndarray):
        """Run depth estimation on *frame* and return the results list."""
        return self.depth_model(frame, device=self.device, verbose=False)

    # ------------------------------------------------------------------
    # 2-D → 3-D lifting
    # ------------------------------------------------------------------

    def lift_keypoints_to_3d(
        self, keypoints_2d: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        """Lift 2-D keypoints to 3-D by sampling *depth_map* at each location.

        Parameters
        ----------
        keypoints_2d:
            Array of shape ``(N, 2)`` with ``(x, y)`` pixel coordinates.
        depth_map:
            2-D array of relative depth values with shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 3)`` with ``(x, y, z)`` coordinates where
            *z* is the sampled depth value.
        """
        if keypoints_2d.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        h, w = depth_map.shape[:2]
        keypoints_3d = []
        for kp in keypoints_2d:
            x = int(np.clip(kp[0], 0, w - 1))
            y = int(np.clip(kp[1], 0, h - 1))
            z = float(depth_map[y, x])
            keypoints_3d.append([float(kp[0]), float(kp[1]), z])
        return np.array(keypoints_3d, dtype=np.float32)

    # ------------------------------------------------------------------
    # Full frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process *frame* through the complete pose + depth pipeline.

        Parameters
        ----------
        frame:
            BGR image array as returned by ``cv2.VideoCapture.read()``.

        Returns
        -------
        dict
            ``pose_results``  – raw Ultralytics pose results.
            ``depth_map``     – 2-D depth array (``None`` if unavailable).
            ``keypoints_3d``  – list of ``(N_keypoints, 3)`` arrays, one per
                               detected person.
        """
        pose_results = self.estimate_pose_2d(frame)
        depth_results = self.estimate_depth(frame)

        depth_map: np.ndarray | None = None
        if depth_results and depth_results[0].depth is not None:
            depth_map = depth_results[0].depth.cpu().numpy()

        keypoints_3d: list[np.ndarray] = []
        if depth_map is not None:
            for result in pose_results:
                if result.keypoints is not None:
                    for person_kps in result.keypoints.xy.cpu().numpy():
                        kps_3d = self.lift_keypoints_to_3d(person_kps, depth_map)
                        keypoints_3d.append(kps_3d)

        return {
            "pose_results": pose_results,
            "depth_map": depth_map,
            "keypoints_3d": keypoints_3d,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def annotate_frame(self, frame: np.ndarray, pipeline_output: dict) -> np.ndarray:
        """Draw the YOLO pose skeleton overlay on *frame*.

        Parameters
        ----------
        frame:
            Original BGR frame.
        pipeline_output:
            Dictionary returned by :meth:`process_frame`.

        Returns
        -------
        np.ndarray
            Annotated BGR frame.
        """
        annotated = frame.copy()
        for result in pipeline_output["pose_results"]:
            annotated = result.plot(img=annotated)
        return annotated

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, source: int | str = 0) -> None:
        """Run the pipeline on a video source until ``q`` is pressed.

        Parameters
        ----------
        source:
            OpenCV-compatible video source: an integer camera index or a path
            to a video file.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                output = self.process_frame(frame)
                annotated = self.annotate_frame(frame, output)

                cv2.imshow("3D Pose Estimation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = PoseEstimationPipeline()
    pipeline.run(source=0)
