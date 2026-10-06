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


def foreground_from_background(
    frame: np.ndarray, background: np.ndarray,
    rectangle: tuple[int, int, int, int], threshold: int,
) -> np.ndarray:
    if frame.shape != background.shape:
        raise ValueError("Empty-field image and car frame have different sizes")
    x, y, width, height = rectangle
    difference = cv2.cvtColor(cv2.absdiff(frame, background), cv2.COLOR_BGR2GRAY)
    _unused, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_mask[y:y + height, x:x + width] = 255
    binary = cv2.bitwise_and(binary, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No foreground car found inside the selected box")
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        raise ValueError("Foreground car mask is too small")
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the system the colour and front of one car")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--output", default="calibration/car_color.npz")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--background-file", help="undistorted empty-field image")
    parser.add_argument("--background-threshold", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--exposure", type=float)
    args = parser.parse_args()

    capture = cv2.VideoCapture(source_value(args.source))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if args.exposure is not None:
        capture.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
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
    if args.background_file:
        empty_field = cv2.imread(args.background_file)
        if empty_field is None:
            raise SystemExit(f"Could not read {args.background_file}")
        car_mask = foreground_from_background(
            frame, empty_field, (x, y, width, height), args.background_threshold
        )
    else:
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
    with np.load(args.output) as model:
        front_rear_distance = cv2.compareHist(
            model["front_hist"].astype(np.float32),
            model["rear_hist"].astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    cv2.destroyAllWindows()
    print(f"Saved colour model to {args.output}")
    if front_rear_distance < 0.10:
        print(
            "Warning: front and rear have similar colours; stationary heading may "
            "flip by 180 degrees. Test it physically or use a visible car tag."
        )


if __name__ == "__main__":
    main()

