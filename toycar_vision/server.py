from __future__ import annotations

import csv
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .aruco_detector import ArucoDetector, draw_detection
from .calibration import CameraUndistorter, FieldMapper
from .color_detector import ColorCarDetector
from .config import car_configs, resolve_config_path
from .models import Measurement, Telemetry
from .protocol import serialize
from .tracking import MultiCarTracker


def _video_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _open_camera(camera_config: dict[str, Any]) -> cv2.VideoCapture:
    source = _video_source(camera_config.get("source", 0))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    properties = (
        (cv2.CAP_PROP_FRAME_WIDTH, camera_config.get("width")),
        (cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get("height")),
        (cv2.CAP_PROP_FPS, camera_config.get("fps")),
        (cv2.CAP_PROP_BUFFERSIZE, camera_config.get("buffer_size", 1)),
    )
    for prop, value in properties:
        if value is not None:
            capture.set(prop, float(value))
    if camera_config.get("exposure") is not None:
        capture.set(cv2.CAP_PROP_EXPOSURE, float(camera_config["exposure"]))
    return capture


def _measurement(detection, car, mapper: FieldMapper) -> Measurement:
    center, heading = mapper.transform([detection.center_uv, detection.heading_point_uv])
    vector = heading - center
    theta = (np.degrees(np.arctan2(vector[1], vector[0])) + car.heading_offset_deg) % 360.0
    return Measurement(
        car_id=car.marker_id,
        name=car.name,
        x_mm=float(center[0]),
        y_mm=float(center[1]),
        theta_deg=float(theta),
        u=detection.center_uv[0],
        v=detection.center_uv[1],
        quality=detection.quality,
    )


def _metrics_writer(path: str | None):
    if not path:
        return None, None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(("timestamp_us", "car_id", "name", "detected", "score",
                     "processing_ms", "x_mm", "y_mm", "theta_deg", "u", "v"))
    return handle, writer


def run(
    config: dict[str, Any],
    host_override: str | None = None,
    port_override: int | None = None,
    headless: bool = False,
    metrics_csv: str | None = None,
    max_frames: int | None = None,
) -> None:
    camera_config = config["camera"]
    cars = car_configs(config)
    detector_config = config.get("detector", {})
    detector_type = str(detector_config.get("type", "aruco")).lower()
    aruco_config = config.get("aruco", {})
    if detector_type == "color":
        if len(cars) != 1:
            raise ValueError("Colour mode currently supports exactly one configured car")
        model_file = resolve_config_path(config, detector_config.get("model_file"))
        if model_file is None or not model_file.exists():
            raise ValueError(
                "Colour model is missing. Run tools/calibrate_color_car.py first."
            )
        detector = ColorCarDetector(
            model_file=model_file,
            car_id=next(iter(cars)),
            min_quality=float(detector_config.get("min_quality", 0.25)),
            backprojection_threshold=int(detector_config.get("backprojection_threshold", 35)),
            min_area_fraction=float(detector_config.get("min_area_fraction", 0.0003)),
            max_area_fraction=float(detector_config.get("max_area_fraction", 0.20)),
            min_reference_area_ratio=float(detector_config.get("min_reference_area_ratio", 0.08)),
            max_reference_area_ratio=float(detector_config.get("max_reference_area_ratio", 4.0)),
            morphology_kernel=int(detector_config.get("morphology_kernel", 9)),
        )
    elif detector_type == "aruco":
        detector = ArucoDetector(
            dictionary_name=aruco_config.get("dictionary", "DICT_4X4_50"),
            min_quality=float(aruco_config.get("min_quality", 0.0)),
            corner_refinement=aruco_config.get("corner_refinement", "SUBPIX"),
        )
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    calibration_file = resolve_config_path(config, camera_config.get("calibration_file"))
    undistorter = CameraUndistorter(calibration_file)

    auto_config = config["field"].get("auto_homography", {})
    marker_centers = {}
    if auto_config.get("enabled", False):
        marker_centers = {
            int(marker_id): (float(point[0]), float(point[1]))
            for marker_id, point in auto_config.get("marker_world_centers", {}).items()
        }
    field_marker_detector = None
    if marker_centers and detector_type != "aruco":
        field_marker_detector = ArucoDetector(
            dictionary_name=aruco_config.get("dictionary", "DICT_4X4_50"),
            min_quality=float(aruco_config.get("min_quality", 0.0)),
            corner_refinement=aruco_config.get("corner_refinement", "SUBPIX"),
        )
    homography_file = resolve_config_path(config, config["field"].get("homography_file"))
    mapper = FieldMapper.from_yaml(homography_file, marker_centers)

    tracking_config = config.get("tracking", {})
    tracker = MultiCarTracker(
        position_time_constant_s=float(tracking_config.get("position_time_constant_s", 0.035)),
        velocity_time_constant_s=float(tracking_config.get("velocity_time_constant_s", 0.080)),
        max_gap_s=float(tracking_config.get("max_gap_s", 0.25)),
    )

    network = config["network"]
    target = (host_override or str(network.get("host", "127.0.0.1")),
              int(port_override if port_override is not None else network.get("port", 5000)))
    emit_missing = bool(network.get("emit_missing", True))
    show = bool(config.get("display", {}).get("enabled", True)) and not headless
    display_scale = float(config.get("display", {}).get("scale", 1.0))

    capture = _open_camera(camera_config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    metrics_handle, metrics_writer = _metrics_writer(metrics_csv)
    start_ns = time.perf_counter_ns()
    frame_count = 0
    last_status_ns = start_ns

    try:
        while max_frames is None or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_start_ns = time.perf_counter_ns()
            timestamp_us = (frame_start_ns - start_ns) // 1_000
            frame = undistorter.apply(frame)
            detections = detector.detect(frame)
            if marker_centers:
                field_detections = (
                    detections if field_marker_detector is None else field_marker_detector.detect(frame)
                )
                mapper.update_from_markers(field_detections)

            detected_cars: set[int] = set()
            telemetry_records: list[Telemetry] = []
            detection_by_id = {item.marker_id: item for item in detections}
            if mapper.ready:
                for marker_id, car in cars.items():
                    detection = detection_by_id.get(marker_id)
                    if detection is None:
                        continue
                    detected_cars.add(marker_id)
                    telemetry_records.append(
                        tracker.update(_measurement(detection, car, mapper), int(timestamp_us))
                    )

            if emit_missing:
                for marker_id, car in cars.items():
                    if marker_id not in detected_cars:
                        telemetry_records.append(
                            tracker.missing(int(timestamp_us), marker_id, car.name)
                        )

            for record in telemetry_records:
                sock.sendto(serialize(record), target)

            processing_ms = (time.perf_counter_ns() - frame_start_ns) / 1_000_000.0
            if metrics_writer is not None:
                records_by_id = {record.car_id: record for record in telemetry_records}
                for marker_id, car in cars.items():
                    detection = detection_by_id.get(marker_id)
                    record = records_by_id.get(marker_id)
                    metrics_writer.writerow((
                        timestamp_us,
                        marker_id,
                        car.name,
                        int(marker_id in detected_cars),
                        detection.quality if detection else 0.0,
                        f"{processing_ms:.4f}",
                        f"{record.x_mm:.3f}" if record and record.detected else "",
                        f"{record.y_mm:.3f}" if record and record.detected else "",
                        f"{record.theta_deg:.3f}" if record and record.detected else "",
                        record.u if record and record.detected else "",
                        record.v if record and record.detected else "",
                    ))

            if show:
                for detection in detections:
                    car = cars.get(detection.marker_id)
                    label = car.name if car else f"marker {detection.marker_id}"
                    draw_detection(frame, detection, f"{label} q={detection.quality:.2f}")
                status = (
                    f"{processing_ms:.1f} ms | UDP {target[0]}:{target[1]} | "
                    f"map: {'ready' if mapper.ready else 'calibration required'}"
                )
                cv2.putText(frame, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (30, 240, 240), 2, cv2.LINE_AA)
                if display_scale != 1.0:
                    frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
                cv2.imshow("Toy Car Vision (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            now_ns = time.perf_counter_ns()
            if now_ns - last_status_ns >= 2_000_000_000:
                elapsed = (now_ns - start_ns) / 1_000_000_000.0
                print(
                    f"frames={frame_count} average_fps={frame_count / elapsed:.1f} "
                    f"last_processing_ms={processing_ms:.2f} map_ready={mapper.ready}",
                    flush=True,
                )
                last_status_ns = now_ns
    finally:
        capture.release()
        sock.close()
        if metrics_handle is not None:
            metrics_handle.close()
        cv2.destroyAllWindows()

