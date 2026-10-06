from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a camera from chessboard images")
    parser.add_argument("images", nargs="+", help="image paths or one quoted glob")
    parser.add_argument("--columns", type=int, default=9, help="inner chessboard corners across")
    parser.add_argument("--rows", type=int, default=6, help="inner chessboard corners down")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--output", default="calibration/camera.yaml")
    args = parser.parse_args()

    paths: list[Path] = []
    for expression in args.images:
        candidate = Path(expression)
        if candidate.exists():
            paths.append(candidate)
        else:
            paths.extend(sorted(Path().glob(expression)))
    if not paths:
        raise SystemExit("No calibration images found")

    pattern_size = (args.columns, args.rows)
    object_template = np.zeros((args.rows * args.columns, 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_mm

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip unreadable: {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise SystemExit(
                f"Calibration images have mixed sizes: {path} is {current_size}, "
                f"expected {image_size}"
            )
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"no chessboard: {path}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_template.copy())
        image_points.append(corners)
        print(f"accepted: {path}")

    if len(object_points) < 8 or image_size is None:
        raise SystemExit(f"Need at least 8 usable views; found {len(object_points)}")
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_errors = []
    for object_view, image_view, rotation, translation in zip(
        object_points, image_points, rotations, translations, strict=True
    ):
        projected, _ = cv2.projectPoints(
            object_view, rotation, translation, camera_matrix, distortion
        )
        squared_error = float(cv2.norm(image_view, projected, cv2.NORM_L2SQR))
        per_view_errors.append(float(np.sqrt(squared_error / len(projected))))

    payload = {
        "image_size": list(image_size),
        "rms_reprojection_error": float(rms),
        "mean_per_view_error_px": float(np.mean(per_view_errors)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"Saved {output} (RMS error {rms:.4f} px, {len(object_points)} views)")


if __name__ == "__main__":
    main()

