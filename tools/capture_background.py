"""Capture an empty playing field for foreground-assisted car detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toycar_vision.calibration import CameraUndistorter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index, video path, or stream URL")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--camera-calibration")
    parser.add_argument("--output", default="calibration/empty_field.png")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera source: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if args.exposure is not None:
        capture.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    undistorter = CameraUndistorter(args.camera_calibration)
    window = "Empty field: remove the car; SPACE save; Q cancel"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Could not read a camera frame")
            frame = undistorter.apply(frame)
            preview = frame.copy()
            cv2.putText(preview, "EMPTY FIELD - SPACE to save", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                raise SystemExit("Cancelled")
            if key == 32:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), frame):
                    raise SystemExit(f"Could not save {output}")
                print(f"Saved empty-field image: {output}")
                return
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
