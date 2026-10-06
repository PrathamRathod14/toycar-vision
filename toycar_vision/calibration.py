from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from .models import Detection


class CameraUndistorter:
    def __init__(self, calibration_file: str | Path | None) -> None:
        self.camera_matrix: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self.image_size: tuple[int, int] | None = None
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self._map_size: tuple[int, int] | None = None
        self._new_matrix: np.ndarray | None = None
        if calibration_file:
            with Path(calibration_file).open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            self.distortion = np.asarray(data["distortion_coefficients"], dtype=np.float64)
            if "image_size" in data:
                self.image_size = tuple(int(value) for value in data["image_size"])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.distortion is None:
            return frame
        height, width = frame.shape[:2]
        size = (width, height)
        if self.image_size is not None and size != self.image_size:
            raise ValueError(
                f"Camera frame is {size}, but lens calibration used {self.image_size}; "
                "calibrate again or capture at the calibrated resolution"
            )
        if self._maps is None or self._map_size != size:
            new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.distortion, size, 0, size
            )
            self._maps = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion,
                None,
                new_matrix,
                size,
                cv2.CV_16SC2,
            )
            self._map_size = size
            self._new_matrix = new_matrix
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def transform_points(self, points: Iterable[tuple[float, float]]) -> np.ndarray:
        """Map raw image coordinates into the space returned by apply()."""
        array = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
        if self.camera_matrix is None or self.distortion is None:
            return array.reshape(-1, 2)
        if self._new_matrix is None:
            raise RuntimeError("Apply a frame before transforming its points")
        return cv2.undistortPoints(
            array, self.camera_matrix, self.distortion, P=self._new_matrix
        ).reshape(-1, 2)


class FieldMapper:
    def __init__(
        self,
        homography: np.ndarray | None = None,
        marker_world_centers: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self.homography = self._normalise(homography) if homography is not None else None
        self.marker_world_centers = marker_world_centers or {}

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if abs(float(matrix[2, 2])) < 1e-12:
            raise ValueError("Invalid homography: bottom-right value is zero")
        return matrix / matrix[2, 2]

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None,
        marker_world_centers: dict[int, tuple[float, float]] | None = None,
    ) -> "FieldMapper":
        matrix = None
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if "homography" in data:
                matrix = np.asarray(data["homography"], dtype=np.float64)
            elif "image_points" in data and "world_points" in data:
                image = np.asarray(data["image_points"], dtype=np.float64)
                world = np.asarray(data["world_points"], dtype=np.float64)
                matrix, _mask = cv2.findHomography(image, world, method=0)
            else:
                raise ValueError(f"No homography or point pairs in {path}")
        return cls(matrix, marker_world_centers)

    @property
    def ready(self) -> bool:
        return self.homography is not None

    def update_from_markers(self, detections: Iterable[Detection]) -> bool:
        pairs = [
            (det.center_uv, self.marker_world_centers[det.marker_id])
            for det in detections
            if det.marker_id in self.marker_world_centers
        ]
        if len(pairs) < 4:
            return False
        image = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        world = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        matrix, mask = cv2.findHomography(image, world, method=cv2.RANSAC, ransacReprojThreshold=10.0)
        if matrix is None or mask is None or int(mask.sum()) < 4:
            return False
        self.homography = self._normalise(matrix)
        return True

    def transform(self, points: Iterable[tuple[float, float]]) -> np.ndarray:
        if self.homography is None:
            raise RuntimeError("Field homography is not available")
        array = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(array, self.homography).reshape(-1, 2)


def save_homography(
    path: str | Path,
    image_points: np.ndarray,
    world_points: np.ndarray,
) -> None:
    matrix, _mask = cv2.findHomography(
        np.asarray(image_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        method=0,
    )
    if matrix is None:
        raise ValueError("Could not calculate a homography from the supplied points")
    payload = {
        "image_points": np.asarray(image_points, dtype=float).tolist(),
        "world_points": np.asarray(world_points, dtype=float).tolist(),
        "homography": (matrix / matrix[2, 2]).tolist(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

