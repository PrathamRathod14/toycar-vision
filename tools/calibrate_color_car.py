from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toycar_vision.calibration import CameraUndistorter
from toycar_vision.color_detector import save_color_model


def source_value(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the system the colour and front of one car")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--output", default="calibration/car_color.npz")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    capture = cv2.VideoCapture(source_value(args.source))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit("Could not open camera")
    undistorter = CameraUndistorter(args.camera_calibration)
    print("Place the car on a plain background. Press SPACE to freeze; Q to cancel.")
    frame = None
    while True:
        ok, captured = capture.read()
        if not ok:
            raise SystemExit("Could not read camera frame")
        captured = undistorter.apply(captured)
        live = captured.copy()
        cv2.putText(live, "SPACE: capture   Q: quit", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Colour calibration", live)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise SystemExit("Cancelled")
        if key == 32:
            frame = captured
            break
    capture.release()

    print("Draw a box around the whole car with a small background margin, then press ENTER.")
    x, y, width, height = (int(value) for value in cv2.selectROI("Select car", frame, False, False))
    cv2.destroyWindow("Select car")
    if width <= 0 or height <= 0:
        raise SystemExit("No car selected")
    grabcut_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(frame, grabcut_mask, (x, y, width, height), background, foreground,
                5, cv2.GC_INIT_WITH_RECT)
    car_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    clicked: list[tuple[int, int]] = []
    preview = frame.copy()
    preview[car_mask == 0] = (preview[car_mask == 0] * 0.25).astype(np.uint8)
    window = "Click the FRONT of the car, then press ENTER"

    def on_mouse(event, mouse_x, mouse_y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(mouse_x, mouse_y)]

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = preview.copy()
        if clicked:
            cv2.circle(view, clicked[0], 8, (0, 0, 255), -1)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            raise SystemExit("Cancelled")
        if key in (10, 13) and clicked:
            break

    save_color_model(args.output, frame, car_mask, clicked[0])
    cv2.destroyAllWindows()
    print(f"Saved colour model to {args.output}")


if __name__ == "__main__":
    main()

