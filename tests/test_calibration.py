import numpy as np
import pytest
import yaml
import cv2

from toycar_vision.calibration import CameraUndistorter, FieldMapper, save_homography


def test_field_mapper_transforms_points():
    mapper = FieldMapper(np.asarray([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]))
    actual = mapper.transform([(1.0, 2.0), (5.0, 4.0)])
    np.testing.assert_allclose(actual, [[12.0, 26.0], [20.0, 32.0]])


def test_camera_calibration_rejects_resolution_change(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump({
        "image_size": [640, 480],
        "camera_matrix": np.eye(3).tolist(),
        "distortion_coefficients": [0, 0, 0, 0, 0],
    }), encoding="utf-8")
    undistorter = CameraUndistorter(path)
    with pytest.raises(ValueError, match="calibrated resolution"):
        undistorter.apply(np.zeros((720, 1280, 3), dtype=np.uint8))


def test_label_points_follow_undistorted_frame_coordinates(tmp_path):
    matrix = np.asarray([[510.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.asarray([0.18, -0.05, 0.001, 0.002, 0.0])
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump({
        "image_size": [640, 480],
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
    }), encoding="utf-8")
    undistorter = CameraUndistorter(path)
    undistorter.apply(np.zeros((480, 640, 3), dtype=np.uint8))
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, (640, 480), 0, (640, 480))
    expected = cv2.undistortPoints(
        np.asarray([[[550.0, 350.0]]]), matrix, distortion, P=new_matrix
    ).reshape(-1, 2)
    np.testing.assert_allclose(undistorter.transform_points([(550.0, 350.0)]), expected)


def test_homography_uses_measured_nonrectangular_world_points(tmp_path):
    image = np.asarray([[10, 20], [200, 30], [210, 180], [20, 170]], dtype=float)
    world = np.asarray([[0, 0], [900, 50], [950, 750], [-20, 700]], dtype=float)
    path = tmp_path / "field.yaml"
    save_homography(path, image, world)
    mapper = FieldMapper.from_yaml(path)
    np.testing.assert_allclose(mapper.transform(image), world, atol=1e-6)

