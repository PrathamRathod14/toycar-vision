from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Detection


_HIST_BINS = (36, 32)
_HIST_RANGES = (0, 180, 0, 256)


def _histogram(hsv: np.ndarray, mask: np.ndarray, detection_scale: bool = False) -> np.ndarray:
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), _HIST_BINS, _HIST_RANGES)
    if float(hist.sum()) <= 0.0:
        return hist.astype(np.float32)
    if detection_scale:
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    else:
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist.astype(np.float32)


def save_color_model(
    path: str | Path,
    frame: np.ndarray,
    foreground_mask: np.ndarray,
    front_point: tuple[float, float],
) -> None:
    """Build a colour model and front/rear signatures from one labelled frame."""
    mask = (foreground_mask > 0).astype(np.uint8) * 255
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    useful_colour = ((hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)).astype(np.uint8) * 255
    colour_mask = cv2.bitwise_and(mask, useful_colour)
    moments = cv2.moments(mask)
    if moments["m00"] <= 0:
        raise ValueError("The selected car mask is empty")
    center = np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )
    direction = np.asarray(front_point, dtype=np.float32) - center
    norm = float(np.linalg.norm(direction))
    if norm < 5.0:
        raise ValueError("Front point is too close to the car center")
    direction /= norm

    yy, xx = np.indices(mask.shape)
    projection = (xx - center[0]) * direction[0] + (yy - center[1]) * direction[1]
    front_mask = cv2.bitwise_and(colour_mask, (projection >= 0).astype(np.uint8) * 255)
    rear_mask = cv2.bitwise_and(colour_mask, (projection < 0).astype(np.uint8) * 255)
    if cv2.countNonZero(front_mask) < 20 or cv2.countNonZero(rear_mask) < 20:
        raise ValueError("Not enough visible colour in both halves of the car")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        detection_hist=_histogram(hsv, colour_mask, detection_scale=True),
        front_hist=_histogram(hsv, front_mask),
        rear_hist=_histogram(hsv, rear_mask),
        reference_area=np.asarray([cv2.countNonZero(mask)], dtype=np.float32),
        image_size=np.asarray([frame.shape[1], frame.shape[0]], dtype=np.int32),
    )


class ColorCarDetector:
    """Detect one unmodified car using a calibrated HSV colour model."""

    def __init__(
        self,
        model_file: str | Path,
        car_id: int,
        min_quality: float = 0.25,
        backprojection_threshold: int = 35,
        min_area_fraction: float = 0.0003,
        max_area_fraction: float = 0.20,
        min_reference_area_ratio: float = 0.08,
        max_reference_area_ratio: float = 4.0,
        morphology_kernel: int = 9,
    ) -> None:
        with np.load(Path(model_file)) as model:
            self.detection_hist = model["detection_hist"].astype(np.float32)
            self.front_hist = model["front_hist"].astype(np.float32)
            self.rear_hist = model["rear_hist"].astype(np.float32)
            self.reference_area = float(model["reference_area"][0]) if "reference_area" in model else None
            self.image_size = tuple(int(v) for v in model["image_size"]) if "image_size" in model else None
        self.car_id = int(car_id)
        self.min_quality = float(min_quality)
        self.backprojection_threshold = int(backprojection_threshold)
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.min_reference_area_ratio = float(min_reference_area_ratio)
        self.max_reference_area_ratio = float(max_reference_area_ratio)
        if not 0 < self.min_reference_area_ratio <= self.max_reference_area_ratio:
            raise ValueError("Invalid reference-area ratio limits")
        kernel_size = max(3, int(morphology_kernel) | 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _orientation(self, hsv: np.ndarray, contour: np.ndarray, center: np.ndarray) -> np.ndarray:
        points = contour.reshape(-1, 2).astype(np.float32)
        covariance = np.cov(points - points.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))].astype(np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)

        x, y, width, height = cv2.boundingRect(contour)
        local_hsv = hsv[y:y + height, x:x + width]
        local_contour = contour - np.asarray([[[x, y]]], dtype=contour.dtype)
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [local_contour], -1, 255, -1)
        useful_colour = ((local_hsv[:, :, 1] >= 35) &
                         (local_hsv[:, :, 2] >= 25)).astype(np.uint8) * 255
        contour_mask = cv2.bitwise_and(contour_mask, useful_colour)
        yy, xx = np.indices((height, width))
        xx += x
        yy += y
        projection = (xx - center[0]) * axis[0] + (yy - center[1]) * axis[1]
        plus_mask = cv2.bitwise_and(contour_mask, (projection >= 0).astype(np.uint8) * 255)
        minus_mask = cv2.bitwise_and(contour_mask, (projection < 0).astype(np.uint8) * 255)
        plus_hist = _histogram(local_hsv, plus_mask)
        minus_hist = _histogram(local_hsv, minus_mask)

        plus_is_front = (
            cv2.compareHist(plus_hist, self.front_hist, cv2.HISTCMP_BHATTACHARYYA)
            + cv2.compareHist(minus_hist, self.rear_hist, cv2.HISTCMP_BHATTACHARYYA)
        )
        minus_is_front = (
            cv2.compareHist(minus_hist, self.front_hist, cv2.HISTCMP_BHATTACHARYYA)
            + cv2.compareHist(plus_hist, self.rear_hist, cv2.HISTCMP_BHATTACHARYYA)
        )
        return axis if plus_is_front <= minus_is_front else -axis

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.image_size is not None and (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError(
                f"Colour model used {self.image_size}, but frame is "
                f"{(frame.shape[1], frame.shape[0])}; recalibrate at the runtime resolution"
            )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backprojection = cv2.calcBackProject(
            [hsv], [0, 1], self.detection_hist, _HIST_RANGES, 1.0
        )
        _unused, binary = cv2.threshold(
            backprojection, self.backprojection_threshold, 255, cv2.THRESH_BINARY
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(frame.shape[0] * frame.shape[1])
        candidates: list[tuple[float, Detection]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.min_area_fraction * frame_area <= area <= self.max_area_fraction * frame_area:
                continue
            if self.reference_area is not None and not (
                self.min_reference_area_ratio * self.reference_area <= area <=
                self.max_reference_area_ratio * self.reference_area
            ):
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            if hull_area <= 0:
                continue
            rectangle = cv2.minAreaRect(contour)
            center = np.asarray(rectangle[0], dtype=np.float32)
            width, height = rectangle[1]
            if min(width, height) < 3.0:
                continue
            x, y, roi_width, roi_height = cv2.boundingRect(contour)
            contour_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
            local_contour = contour - np.asarray([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(contour_mask, [local_contour], -1, 255, -1)
            colour_strength = float(cv2.mean(
                backprojection[y:y + roi_height, x:x + roi_width], mask=contour_mask
            )[0] / 255.0)
            solidity = min(1.0, area / hull_area)
            quality = max(0.0, min(1.0, 0.75 * colour_strength + 0.25 * solidity))
            if quality < self.min_quality:
                continue
            direction = self._orientation(hsv, contour, center)
            heading_distance = max(width, height) * 0.45
            heading = center + direction * heading_distance
            corners = cv2.boxPoints(rectangle).astype(np.float64)
            candidates.append((
                quality * area,
                Detection(
                    marker_id=self.car_id,
                    center_uv=(float(center[0]), float(center[1])),
                    heading_point_uv=(float(heading[0]), float(heading[1])),
                    corners=corners,
                    quality=quality,
                ),
            ))
        if not candidates:
            return []
        return [max(candidates, key=lambda item: item[0])[1]]

