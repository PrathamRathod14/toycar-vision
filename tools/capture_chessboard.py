"""Collect varied chessboard views from the tracking camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--columns", type=int, default=9, help="inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="inner corners down")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output-dir", default="calibration/chessboard")
    args = parser.parse_args()
    if args.count < 8:
        parser.error("At least eight views are required")

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
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    window = "Chessboard: vary angle and image position; SPACE save; Q cancel"
    saved = 0
    try:
        while saved < args.count:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Could not read a camera frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (args.columns, args.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, (args.columns, args.rows), corners, found)
            status = f"{saved}/{args.count} saved | {'CORNERS FOUND' if found else 'show whole board'}"
            cv2.putText(preview, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                raise SystemExit("Cancelled")
            if key == 32 and found:
                path = output / f"view_{saved + 1:02d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise SystemExit(f"Could not save {path}")
                saved += 1
                print(f"Saved {path} ({saved}/{args.count})")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
