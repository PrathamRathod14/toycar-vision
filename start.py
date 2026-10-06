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


def _positive_number(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default:g}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is None:
            print("Enter the measured value in millimetres.")
            continue
        try:
            result = float(value) if value else float(default)
        except ValueError:
            print("Please enter a number in millimetres.")
            continue
        if result > 0:
            return result
        print("The value must be greater than zero.")


def _camera_capture_options(camera: dict) -> list[str]:
    options: list[str] = []
    for key in ("fps", "exposure"):
        if camera.get(key) is not None:
            options.extend((f"--{key}", str(camera[key])))
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and run Toy Car Vision")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="configuration YAML")
    parser.add_argument("--source", help="override camera index, video file, or stream URL")
    parser.add_argument("--reset-car", action="store_true", help="teach the car colours again")
    parser.add_argument("--reset-camera", action="store_true", help="repeat lens calibration")
    parser.add_argument("--reset-background", action="store_true", help="capture the empty field again")
    parser.add_argument("--reset-field", action="store_true", help="select the field corners again")
    parser.add_argument("--host", help="override UDP destination host")
    parser.add_argument("--port", type=int, help="override UDP destination port")
    parser.add_argument("--headless", action="store_true", help="disable camera preview")
    parser.add_argument("--metrics-csv", help="write per-frame timing and detection data")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--chessboard-columns", type=int, default=9, help="inner corners across")
    parser.add_argument("--chessboard-rows", type=int, default=6, help="inner corners down")
    parser.add_argument("--calibration-views", type=int, default=12)
    parser.add_argument("--square-mm", type=float, help="measured chessboard square size")
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
    background_path = resolve_config_path(config, detector.get("background_file"))
    homography_path = resolve_config_path(config, config["field"].get("homography_file"))
    auto_field = bool(config["field"].get("auto_homography", {}).get("enabled", False))
    if detector_type == "color" and model_path is None:
        raise SystemExit("Colour mode needs detector.model_file in the configuration")
    if not auto_field and homography_path is None:
        raise SystemExit("Set field.homography_file or enable field.auto_homography")
    if args.reset_car and detector_type != "color":
        raise SystemExit("--reset-car applies only to the colour detector")
    if args.reset_background and (detector_type != "color" or background_path is None):
        raise SystemExit("--reset-background needs colour mode and detector.background_file")
    if args.reset_field and auto_field:
        raise SystemExit("--reset-field applies only to a fixed field homography")

    source = str(camera.get("source", 0))
    width = int(camera.get("width", 1280))
    height = int(camera.get("height", 720))
    camera_calibration = resolve_config_path(config, camera.get("calibration_file"))
    if args.reset_camera and camera_calibration is None:
        raise SystemExit("--reset-camera needs camera.calibration_file in the configuration")
    if camera_calibration is not None and (args.reset_camera or not camera_calibration.exists()):
        if args.calibration_views < 8:
            raise SystemExit("Lens calibration needs at least eight chessboard views")
        print("\nCAMERA SETUP: show a measured chessboard at varied positions and angles.")
        board_dir = camera_calibration.parent / "chessboard"
        capture_args = [
            "tools/capture_chessboard.py", "--source", source,
            "--width", str(width), "--height", str(height),
            "--columns", str(args.chessboard_columns),
            "--rows", str(args.chessboard_rows),
            "--count", str(args.calibration_views),
            "--output-dir", str(board_dir),
        ]
        _run_tool(capture_args + _camera_capture_options(camera))
        square_mm = args.square_mm
        if square_mm is None:
            square_mm = _positive_number("Measured chessboard square width in mm")
        if square_mm <= 0:
            raise SystemExit("--square-mm must be positive")
        images = [str(board_dir / f"view_{index:02d}.png")
                  for index in range(1, args.calibration_views + 1)]
        _run_tool([
            "tools/calibrate_camera.py", *images,
            "--columns", str(args.chessboard_columns),
            "--rows", str(args.chessboard_rows),
            "--square-mm", str(square_mm),
            "--output", str(camera_calibration),
        ])

    if detector_type == "color" and background_path is not None and (
        args.reset_camera or args.reset_background or not background_path.exists()
    ):
        print("\nFIRST STEP: remove the car and people from the field.")
        tool_args = [
            "tools/capture_background.py",
            "--source", source,
            "--width", str(width),
            "--height", str(height),
            "--output", str(background_path),
        ]
        if camera_calibration:
            tool_args.extend(("--camera-calibration", str(camera_calibration)))
        tool_args.extend(_camera_capture_options(camera))
        _run_tool(tool_args)

    if detector_type == "color" and (
        args.reset_camera or args.reset_background or args.reset_car
        or not model_path.exists()
    ):
        print("\nNEXT STEP: place the car in the field and teach its appearance.")
        tool_args = [
            "tools/calibrate_color_car.py",
            "--source", source,
            "--width", str(width),
            "--height", str(height),
            "--output", str(model_path),
        ]
        if camera_calibration:
            tool_args.extend(("--camera-calibration", str(camera_calibration)))
        if background_path:
            tool_args.extend(("--background-file", str(background_path),
                              "--background-threshold",
                              str(detector.get("background_threshold", 25))))
        tool_args.extend(_camera_capture_options(camera))
        _run_tool(tool_args)

    if not auto_field and (args.reset_camera or args.reset_field or not homography_path.exists()):
        print("\nNEXT STEP: measure four floor reference points.")
        tool_args = [
            "tools/capture_homography.py",
            "--source", source,
            "--camera-width", str(width),
            "--camera-height", str(height),
            "--output", str(homography_path),
        ]
        reference_points = config["field"].get("reference_points_mm")
        if reference_points is not None:
            if len(reference_points) != 4 or any(len(point) != 2 for point in reference_points):
                raise SystemExit("field.reference_points_mm needs four [x,y] pairs")
            tool_args.append("--world-points-mm")
            tool_args.extend(str(value) for point in reference_points for value in point)
        else:
            field_width = _positive_number("Measured reference rectangle width in mm")
            field_height = _positive_number("Measured reference rectangle height in mm")
            tool_args.extend(("--width-mm", str(field_width),
                              "--height-mm", str(field_height)))
        if camera_calibration:
            tool_args.extend(("--camera-calibration", str(camera_calibration)))
        tool_args.extend(_camera_capture_options(camera))
        _run_tool(tool_args)

    print("\nStarting Toy Car Vision. Press Q in the camera window to stop.")
    run(config, host_override=args.host, port_override=args.port,
        headless=args.headless, metrics_csv=args.metrics_csv,
        max_frames=args.max_frames)


if __name__ == "__main__":
    main()

