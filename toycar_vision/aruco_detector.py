from __future__ import annotations

import math

import cv2
import numpy as np

from .models import Detection


_REFINEMENT = {
    "NONE": cv2.aruco.CORNER_REFINE_NONE,
    "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
    "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
}


class ArucoDetector:
    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        min_quality: float = 0.0,
        corner_refinement: str = "SUBPIX",
    ) -> None:
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        refinement = corner_refinement.upper()
        if refinement not in _REFINEMENT:
            raise ValueError(f"Unknown corner refinement: {corner_refinement}")
        parameters.cornerRefinementMethod = _REFINEMENT[refinement]
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.min_quality = float(min_quality)

    @staticmethod
    def _quality(points: np.ndarray) -> float:
        """A scale-independent geometric confidence in [0, 1]."""
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        if float(np.max(lengths)) <= 1e-6:
            return 0.0
        side_uniformity = float(np.min(lengths) / np.max(lengths))
        angle_scores = []
        for index in range(4):
            first = points[index - 1] - points[index]
            second = points[(index + 1) % 4] - points[index]
            denom = np.linalg.norm(first) * np.linalg.norm(second)
            cosine = abs(float(np.dot(first, second) / denom)) if denom > 1e-6 else 1.0
            angle_scores.append(max(0.0, 1.0 - cosine))
        return max(0.0, min(1.0, side_uniformity * float(np.mean(angle_scores))))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None:
            return []

        detections: list[Detection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
            points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            quality = self._quality(points)
            if quality < self.min_quality:
                continue
            center = points.mean(axis=0)
            top_midpoint = (points[0] + points[1]) * 0.5
            detections.append(
                Detection(
                    marker_id=int(marker_id),
                    center_uv=(float(center[0]), float(center[1])),
                    heading_point_uv=(float(top_midpoint[0]), float(top_midpoint[1])),
                    corners=points,
                    quality=quality,
                )
            )
        return detections


def draw_detection(frame: np.ndarray, detection: Detection, label: str) -> None:
    points = np.asarray(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], True, (0, 220, 0), 2, cv2.LINE_AA)
    center = tuple(int(round(value)) for value in detection.center_uv)
    heading = tuple(int(round(value)) for value in detection.heading_point_uv)
    cv2.arrowedLine(frame, center, heading, (0, 180, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        frame,
        label,
        (center[0] + 8, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

