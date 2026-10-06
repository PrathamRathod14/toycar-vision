"""Set up a camera and field, then launch either supported tracker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from toycar_vision.config import load_config, resolve_config_path
from toycar_vision.server import run


ROOT = Path(__file__).resolve().parent


def _run_tool(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def _positive_number(prompt: str, default: float) -> float:
    while True:
        value = input(f"{prompt} [{default:g}]: ").strip()
        try:
            result = float(value) if value else float(default)
        except ValueError:
            print("Please enter a number in millimetres.")
            continue
        if result > 0:
            return result
        print("The value must be greater than zero.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and run Toy Car Vision")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="configuration YAML")
    parser.add_argument("--source", help="override camera index, video file, or stream URL")
    parser.add_argument("--reset-car", action="store_true", help="teach the car colours again")
    parser.add_argument("--reset-field", action="store_true", help="select the field corners again")
    parser.add_argument("--host", help="override UDP destination host")
    parser.add_argument("--port", type=int, help="override UDP destination port")
    parser.add_argument("--headless", action="store_true", help="disable camera preview")
    parser.add_argument("--metrics-csv", help="write per-frame timing and detection data")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Missing configuration: {config_path}")
    config = load_config(config_path)
    camera = config["camera"]
    if args.source is not None:
        camera["source"] = args.source
    detector = config.get("detector", {})
    detector_type = str(detector.get("type", "aruco")).lower()
    if detector_type not in {"color", "aruco"}:
        raise SystemExit(f"Unsupported detector type: {detector_type}")
    model_path = resolve_config_path(config, detector.get("model_file"))
    homography_path = resolve_config_path(config, config["field"].get("homography_file"))
    auto_field = bool(config["field"].get("auto_homography", {}).get("enabled", False))
    if detector_type == "color" and model_path is None:
        raise SystemExit("Colour mode needs detector.model_file in the configuration")
    if not auto_field and homography_path is None:
        raise SystemExit("Set field.homography_file or enable field.auto_homography")
    if args.reset_car and detector_type != "color":
        raise SystemExit("--reset-car applies only to the colour detector")
    if args.reset_field and auto_field:
        raise SystemExit("--reset-field applies only to a fixed field homography")

    source = str(camera.get("source", 0))
    width = int(camera.get("width", 1280))
    height = int(camera.get("height", 720))
    camera_calibration = resolve_config_path(config, camera.get("calibration_file"))

    if detector_type == "color" and (args.reset_car or not model_path.exists()):
        print("\nFIRST STEP: place the car alone on a plain surface.")
        tool_args = [
            "tools/calibrate_color_car.py",
            "--source", source,
            "--width", str(width),
            "--height", str(height),
            "--output", str(model_path),
        ]
        if camera_calibration:
            tool_args.extend(("--camera-calibration", str(camera_calibration)))
        _run_tool(tool_args)

    if not auto_field and (args.reset_field or not homography_path.exists()):
        print("\nSECOND STEP: measure the visible rectangular playing area.")
        default_size = config["field"].get("size_mm", [1000.0, 600.0])
        field_width = _positive_number("Playing-area width in mm", float(default_size[0]))
        field_height = _positive_number("Playing-area height in mm", float(default_size[1]))
        tool_args = [
            "tools/capture_homography.py",
            "--source", source,
            "--camera-width", str(width),
            "--camera-height", str(height),
            "--width-mm", str(field_width),
            "--height-mm", str(field_height),
            "--output", str(homography_path),
        ]
        if camera_calibration:
            tool_args.extend(("--camera-calibration", str(camera_calibration)))
        _run_tool(tool_args)

    print("\nStarting Toy Car Vision. Press Q in the camera window to stop.")
    run(config, host_override=args.host, port_override=args.port,
        headless=args.headless, metrics_csv=args.metrics_csv,
        max_frames=args.max_frames)


if __name__ == "__main__":
    main()

