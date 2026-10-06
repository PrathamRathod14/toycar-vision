from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Permit the documented direct invocation before the project is installed.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toycar_vision.calibration import CameraUndistorter, save_homography


def source_value(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def main() -> None:
    parser = argparse.ArgumentParser(description="Click the four field corners to save a homography")
    parser.add_argument("--source", default="0", help="camera index, video, or image path")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--width-mm", type=float, default=2500.0)
    parser.add_argument("--height-mm", type=float, default=1500.0)
    parser.add_argument("--output", default="calibration/homography.yaml")
    parser.add_argument(
        "--camera-calibration",
        help="camera calibration YAML; undistort before clicking field corners",
    )
    args = parser.parse_args()

    source = source_value(args.source)
    image = cv2.imread(source) if isinstance(source, str) else None
    capture = None
    if image is None:
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        ok, image = capture.read()
        if not ok:
            raise SystemExit(f"Could not read {source}")
    image = CameraUndistorter(args.camera_calibration).apply(image)

    points: list[tuple[int, int]] = []
    window = "Click TL, TR, BR, BL; r reset; Enter save; Esc cancel"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = image.copy()
        for index, point in enumerate(points):
            cv2.circle(view, point, 6, (0, 255, 255), -1)
            cv2.putText(view, str(index + 1), (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if len(points) > 1:
            cv2.polylines(view, [np.asarray(points, dtype=np.int32)], len(points) == 4,
                          (0, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            raise SystemExit("Cancelled")
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == 4:
            break

    world = np.asarray([
        [0.0, 0.0],
        [args.width_mm, 0.0],
        [args.width_mm, args.height_mm],
        [0.0, args.height_mm],
    ])
    save_homography(args.output, np.asarray(points, dtype=float), world)
    print(f"Saved {args.output}")
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
