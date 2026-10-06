import numpy as np
import pytest
import yaml

from toycar_vision.calibration import CameraUndistorter, FieldMapper


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

