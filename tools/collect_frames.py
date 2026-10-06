"""Save raw camera/video frames and an empty manual-label CSV for ROC evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import yaml


def collect_frames(
    source: int | str, output_dir: str | Path, every_n: int = 0,
    max_saved: int = 200, width: int | None = 1280, height: int | None = 720,
    fps: float | None = 60.0, exposure: float | None = None,
) -> int:
    if every_n < 0 or max_saved < 1:
        raise ValueError("every_n must be nonnegative and max_saved must be positive")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open source: {source}")
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, fps)
    if exposure is not None:
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.csv"
    if labels_path.exists():
        capture.release()
        raise FileExistsError(f"Refusing to overwrite existing labels: {labels_path}")
    saved = 0
    frame_number = 0
    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image_path", "present", "center_u", "center_v", "radius_px"))
            while saved < max_saved:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                should_save = every_n > 0 and frame_number % every_n == 0
                if every_n == 0:
                    preview = frame.copy()
                    cv2.putText(preview, "SPACE save frame | Q finish", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Collect evaluation frames", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    should_save = key == 32
                if should_save:
                    saved += 1
                    filename = f"frame_{saved:04d}.png"
                    if not cv2.imwrite(str(output / filename), frame):
                        raise OSError(f"Could not save {output / filename}")
                    writer.writerow((filename, "", "", "", ""))
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} frames and blank labels to {output}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="camera index or video file; overrides config")
    parser.add_argument("--output-dir", default="captures/heldout")
    parser.add_argument("--every-n", type=int, default=0,
                        help="save every Nth frame without a preview; 0 uses Space")
    parser.add_argument("--max-saved", type=int, default=200)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--exposure", type=float)
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        camera = (yaml.safe_load(handle) or {}).get("camera", {})
    raw_source = args.source if args.source is not None else camera.get("source", 0)
    source = int(raw_source) if str(raw_source).isdigit() else str(raw_source)
    collect_frames(source, args.output_dir, args.every_n, args.max_saved,
                   args.width if args.width is not None else camera.get("width", 1280),
                   args.height if args.height is not None else camera.get("height", 720),
                   args.fps if args.fps is not None else camera.get("fps", 60.0),
                   args.exposure if args.exposure is not None else camera.get("exposure"))


if __name__ == "__main__":
    main()
