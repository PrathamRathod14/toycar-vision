"""Score held-out labelled images for an honest detector ROC curve."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toycar_vision.aruco_detector import ArucoDetector
from toycar_vision.calibration import CameraUndistorter
from toycar_vision.color_detector import ColorCarDetector
from toycar_vision.config import car_configs, load_config, resolve_config_path


def score_frames(
    config_path: str, labels_path: str, output_path: str,
    match_radius_px: float = 25.0,
) -> int:
    config = load_config(config_path)
    cars = car_configs(config)
    detector_config = config.get("detector", {})
    mode = str(detector_config.get("type", "aruco")).lower()
    if mode == "color":
        if len(cars) != 1:
            raise ValueError("Colour mode supports exactly one configured car")
        model_file = resolve_config_path(config, detector_config.get("model_file"))
        if model_file is None or not model_file.exists():
            raise ValueError("Calibrate the colour model before scoring frames")
        detector = ColorCarDetector(
            model_file, next(iter(cars)), min_quality=0.0,
            backprojection_threshold=int(detector_config.get("backprojection_threshold", 35)),
            min_area_fraction=float(detector_config.get("min_area_fraction", 0.0003)),
            max_area_fraction=float(detector_config.get("max_area_fraction", 0.20)),
            min_reference_area_ratio=float(detector_config.get("min_reference_area_ratio", 0.08)),
            max_reference_area_ratio=float(detector_config.get("max_reference_area_ratio", 4.0)),
            morphology_kernel=int(detector_config.get("morphology_kernel", 9)),
            background_file=resolve_config_path(config, detector_config.get("background_file")),
            background_threshold=int(detector_config.get("background_threshold", 25)),
            background_kernel=int(detector_config.get("background_kernel", 3)),
            min_color_strength=float(detector_config.get("min_color_strength", 0.04)),
        )
    elif mode == "aruco":
        aruco = config.get("aruco", {})
        detector = ArucoDetector(
            dictionary_name=aruco.get("dictionary", "DICT_4X4_50"),
            min_quality=0.0,
            corner_refinement=aruco.get("corner_refinement", "SUBPIX"),
        )
    else:
        raise ValueError(f"Unsupported detector type: {mode}")

    camera_file = resolve_config_path(config, config["camera"].get("calibration_file"))
    undistorter = CameraUndistorter(camera_file)
    labels_file = Path(labels_path).resolve()
    with labels_file.open("r", newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    if not labels or any("image_path" not in row or "present" not in row for row in labels):
        raise ValueError("Labels CSV needs image_path,present and at least one row")
    if match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "present", "score", "result"))
        for row in labels:
            if row["present"] not in {"0", "1"}:
                raise ValueError(f"present must be 0 or 1: {row['image_path']}")
            image_path = (labels_file.parent / row["image_path"]).resolve()
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Cannot read image: {image_path}")
            detections = detector.detect(undistorter.apply(frame))
            target_id = int(row["car_id"]) if row.get("car_id") else None
            if target_id is not None and target_id not in cars:
                raise ValueError(f"Unknown car_id in labels: {target_id}")
            if row["present"] == "1" and len(cars) > 1 and target_id is None:
                raise ValueError("Positive labels need car_id when multiple cars are configured")
            relevant = [d for d in detections if d.marker_id in cars and
                        (target_id is None or d.marker_id == target_id)]
            candidate = max(relevant, key=lambda d: d.quality, default=None)
            score = candidate.quality if candidate is not None else 0.0
            if row["present"] == "0":
                writer.writerow((row["image_path"], 0, score, "negative"))
                continue
            if not row.get("center_u") or not row.get("center_v"):
                raise ValueError(
                    f"Positive label needs center_u,center_v: {row['image_path']}"
                )
            radius = float(row.get("radius_px") or match_radius_px)
            if radius <= 0:
                raise ValueError("radius_px must be positive")
            raw_center = (float(row["center_u"]), float(row["center_v"]))
            corrected_center = undistorter.transform_points([raw_center])[0]
            matched = candidate is not None and math.dist(
                candidate.center_uv, corrected_center,
            ) <= radius
            writer.writerow((row["image_path"], 1, score if matched else 0.0,
                             "matched" if matched else "missed"))
            if candidate is not None and not matched:
                writer.writerow((row["image_path"], 0, score, "false_location"))
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--labels", required=True, help="CSV with image_path,present columns")
    parser.add_argument("--output", default="results/scored_frames.csv")
    parser.add_argument("--match-radius-px", type=float, default=25.0)
    args = parser.parse_args()
    count = score_frames(args.config, args.labels, args.output, args.match_radius_px)
    print(f"Scored {count} labelled frames: {args.output}")


if __name__ == "__main__":
    main()
