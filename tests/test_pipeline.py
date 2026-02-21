"""Unit tests for pipeline.py using mocks to avoid requiring model weights."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FRAME_H, FRAME_W = 480, 640
FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
DEPTH_MAP = np.random.rand(FRAME_H, FRAME_W).astype(np.float32)

# 17 COCO keypoints, all at known pixel positions
KEYPOINTS_2D = np.array([[x * 20, y * 20] for x, y in zip(range(17), range(17))], dtype=np.float32)


def _make_pose_result(keypoints_2d: np.ndarray = KEYPOINTS_2D) -> MagicMock:
    """Build a minimal mock that resembles an Ultralytics pose result."""
    kp_tensor = MagicMock()
    kp_tensor.cpu.return_value.numpy.return_value = keypoints_2d[np.newaxis]  # (1, 17, 2)

    result = MagicMock()
    result.keypoints = MagicMock()
    result.keypoints.xy = kp_tensor
    result.plot.side_effect = lambda img: img  # return frame unchanged
    return result


def _make_depth_result(depth_map: np.ndarray = DEPTH_MAP) -> MagicMock:
    """Build a minimal mock that resembles an Ultralytics depth result."""
    depth_tensor = MagicMock()
    depth_tensor.cpu.return_value.numpy.return_value = depth_map

    result = MagicMock()
    result.depth = depth_tensor
    return result


# ---------------------------------------------------------------------------
# Import pipeline with YOLO patched so no weights are downloaded
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline():
    """Return a PoseEstimationPipeline whose YOLO models are mocked."""
    pose_mock = MagicMock()
    depth_mock = MagicMock()
    with patch("pipeline.YOLO", side_effect=[pose_mock, depth_mock]):
        from pipeline import PoseEstimationPipeline

        p = PoseEstimationPipeline(
            pose_model="yolov8n-pose.pt",
            depth_model="depth-anything-small.pt",
            device="cpu",
        )
    return p


# ---------------------------------------------------------------------------
# Tests for lift_keypoints_to_3d
# ---------------------------------------------------------------------------

class TestLiftKeypointsTo3D:
    def test_output_shape(self, pipeline):
        kps_3d = pipeline.lift_keypoints_to_3d(KEYPOINTS_2D, DEPTH_MAP)
        assert kps_3d.shape == (len(KEYPOINTS_2D), 3)

    def test_xy_preserved(self, pipeline):
        kps_3d = pipeline.lift_keypoints_to_3d(KEYPOINTS_2D, DEPTH_MAP)
        np.testing.assert_array_almost_equal(kps_3d[:, :2], KEYPOINTS_2D)

    def test_z_sampled_from_depth_map(self, pipeline):
        kps_3d = pipeline.lift_keypoints_to_3d(KEYPOINTS_2D, DEPTH_MAP)
        for i, kp in enumerate(KEYPOINTS_2D):
            x = int(np.clip(kp[0], 0, FRAME_W - 1))
            y = int(np.clip(kp[1], 0, FRAME_H - 1))
            assert kps_3d[i, 2] == pytest.approx(DEPTH_MAP[y, x])

    def test_out_of_bounds_coords_clamped(self, pipeline):
        # Coordinates outside frame dimensions must not raise an IndexError.
        oob_kps = np.array([[9999.0, 9999.0], [-1.0, -1.0]], dtype=np.float32)
        kps_3d = pipeline.lift_keypoints_to_3d(oob_kps, DEPTH_MAP)
        assert kps_3d.shape == (2, 3)

    def test_empty_keypoints(self, pipeline):
        empty = np.empty((0, 2), dtype=np.float32)
        kps_3d = pipeline.lift_keypoints_to_3d(empty, DEPTH_MAP)
        assert kps_3d.shape == (0, 3)


# ---------------------------------------------------------------------------
# Tests for process_frame
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_returns_expected_keys(self, pipeline):
        pipeline.pose_model.return_value = [_make_pose_result()]
        pipeline.depth_model.return_value = [_make_depth_result()]

        output = pipeline.process_frame(FRAME)
        assert set(output.keys()) == {"pose_results", "depth_map", "keypoints_3d"}

    def test_depth_map_shape(self, pipeline):
        pipeline.pose_model.return_value = [_make_pose_result()]
        pipeline.depth_model.return_value = [_make_depth_result()]

        output = pipeline.process_frame(FRAME)
        assert output["depth_map"].shape == DEPTH_MAP.shape

    def test_keypoints_3d_populated(self, pipeline):
        pipeline.pose_model.return_value = [_make_pose_result()]
        pipeline.depth_model.return_value = [_make_depth_result()]

        output = pipeline.process_frame(FRAME)
        assert len(output["keypoints_3d"]) == 1
        assert output["keypoints_3d"][0].shape == (17, 3)

    def test_no_depth_skips_3d_lifting(self, pipeline):
        depth_result = MagicMock()
        depth_result.depth = None

        pipeline.pose_model.return_value = [_make_pose_result()]
        pipeline.depth_model.return_value = [depth_result]

        output = pipeline.process_frame(FRAME)
        assert output["depth_map"] is None
        assert output["keypoints_3d"] == []

    def test_no_keypoints_result(self, pipeline):
        pose_result = MagicMock()
        pose_result.keypoints = None
        pose_result.plot.side_effect = lambda img: img

        pipeline.pose_model.return_value = [pose_result]
        pipeline.depth_model.return_value = [_make_depth_result()]

        output = pipeline.process_frame(FRAME)
        assert output["keypoints_3d"] == []


# ---------------------------------------------------------------------------
# Tests for annotate_frame
# ---------------------------------------------------------------------------

class TestAnnotateFrame:
    def test_returns_array_same_shape(self, pipeline):
        pose_result = _make_pose_result()
        output = {"pose_results": [pose_result], "depth_map": DEPTH_MAP, "keypoints_3d": []}
        annotated = pipeline.annotate_frame(FRAME, output)
        assert annotated.shape == FRAME.shape

    def test_plot_called_for_each_result(self, pipeline):
        pose_result = _make_pose_result()
        output = {"pose_results": [pose_result], "depth_map": None, "keypoints_3d": []}
        pipeline.annotate_frame(FRAME, output)
        pose_result.plot.assert_called_once()
