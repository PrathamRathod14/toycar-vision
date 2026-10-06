import csv
import socket
import sys

import cv2
import numpy as np
import yaml

import start
from tools.generate_chessboard import main as generate_chessboard
from tools.score_labeled_frames import score_frames
from toycar_vision.color_detector import save_color_model
from toycar_vision.config import load_config
from toycar_vision.server import run


def test_setup_requires_a_measured_dimension(monkeypatch):
    answers = iter(["", "not-a-number", "20"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert start._positive_number("Measured size") == 20.0


def test_generated_chessboard_has_requested_inner_corners(tmp_path, monkeypatch):
    output = tmp_path / "board.png"
    monkeypatch.setattr(sys, "argv", ["generate_chessboard.py", "--output", str(output)])
    generate_chessboard()
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    found, _corners = cv2.findChessboardCorners(image, (9, 6))
    assert found


def test_scoring_held_out_frames_uses_real_detector(tmp_path):
    car = np.full((240, 320, 3), 255, dtype=np.uint8)
    car[90:150, 80:140] = (180, 40, 180)
    car[90:150, 140:200] = (40, 180, 40)
    empty = np.full_like(car, 255)
    mask = np.zeros(car.shape[:2], dtype=np.uint8)
    mask[90:150, 80:200] = 255
    model = tmp_path / "car.npz"
    save_color_model(model, car, mask, (195, 120))
    cv2.imwrite(str(tmp_path / "positive.png"), car)
    cv2.imwrite(str(tmp_path / "negative.png"), empty)
    cv2.imwrite(str(tmp_path / "empty.png"), empty)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "camera": {"source": 0},
        "detector": {"type": "color", "model_file": "car.npz",
                     "background_file": "empty.png"},
        "field": {"homography_file": "unused.yaml"},
        "cars": [{"car_id": 1, "name": "Test Car"}],
        "network": {"port": 5000},
    }), encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_path,present,center_u,center_v\n"
        "positive.png,1,140,120\nnegative.png,0,,\n", encoding="utf-8"
    )
    scored = tmp_path / "scored.csv"
    assert score_frames(str(config), str(labels), str(scored)) == 2
    with scored.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["score"]) > float(rows[1]["score"])
    assert float(rows[1]["score"]) == 0.0

    wrong_labels = tmp_path / "wrong_labels.csv"
    wrong_labels.write_text(
        "image_path,present,center_u,center_v\npositive.png,1,300,200\n",
        encoding="utf-8",
    )
    score_frames(str(config), str(wrong_labels), str(scored))
    with scored.open(newline="", encoding="utf-8") as handle:
        wrong_rows = list(csv.DictReader(handle))
    assert [row["result"] for row in wrong_rows] == ["missed", "false_location"]
    assert float(wrong_rows[0]["score"]) == 0.0
    assert float(wrong_rows[1]["score"]) > 0.0


def test_launcher_accepts_auto_field_marker_mode_without_color_setup(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "camera": {"source": 0},
        "detector": {"type": "aruco"},
        "field": {"auto_homography": {"enabled": True}},
        "cars": [{"marker_id": 10, "name": "Test Car"}],
        "network": {"port": 5000},
    }), encoding="utf-8")
    captured = {}
    monkeypatch.setattr(start, "run", lambda cfg, **kwargs: captured.update(config=cfg, **kwargs))
    monkeypatch.setattr(start, "_run_tool", lambda args: (_ for _ in ()).throw(
        AssertionError("No interactive calibration expected")))
    monkeypatch.setattr(sys, "argv", ["start.py", "--config", str(config),
                                     "--source", "2", "--host", "192.0.2.10",
                                     "--port", "5010", "--headless"])
    start.main()
    assert captured["config"]["camera"]["source"] == "2"
    assert captured["host_override"] == "192.0.2.10"
    assert captured["port_override"] == 5010
    assert captured["headless"] is True


def test_first_run_guides_camera_background_car_and_field(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "camera": {"source": 0, "width": 320, "height": 240,
                   "fps": 60, "exposure": -5,
                   "calibration_file": "calibration/camera.yaml"},
        "detector": {"type": "color", "model_file": "calibration/car.npz",
                     "background_file": "calibration/empty.png"},
        "field": {"homography_file": "calibration/field.yaml",
                  "reference_points_mm": [[0, 0], [1000, 0], [1000, 500], [0, 500]]},
        "cars": [{"car_id": 1, "name": "Test Car"}],
        "network": {"port": 5000},
    }), encoding="utf-8")
    tool_calls = []
    monkeypatch.setattr(start, "_run_tool", lambda args: tool_calls.append(args))
    monkeypatch.setattr(start, "run", lambda config, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["start.py", "--config", str(config),
                                     "--headless"])
    start.main()
    assert [call[0] for call in tool_calls] == [
        "tools/capture_chessboard.py", "tools/calibrate_camera.py",
        "tools/capture_background.py", "tools/calibrate_color_car.py",
        "tools/capture_homography.py",
    ]
    assert "--background-file" not in tool_calls[2]
    assert "--background-file" in tool_calls[3]
    assert "--world-points-mm" in tool_calls[-1]
    assert tool_calls[1][tool_calls[1].index("--square-size") + 1] == "1.0"
    for call in (tool_calls[0], tool_calls[2], tool_calls[3], tool_calls[4]):
        assert "--fps" in call and "--exposure" in call


def test_color_frame_reaches_udp_receiver_with_mapped_pose(tmp_path, monkeypatch):
    background = np.full((240, 320, 3), 255, dtype=np.uint8)
    frame = background.copy()
    frame[90:150, 80:140] = (180, 40, 180)
    frame[90:150, 140:200] = (40, 180, 40)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[90:150, 80:200] = 255
    save_color_model(tmp_path / "car.npz", frame, mask, (195, 120))
    cv2.imwrite(str(tmp_path / "empty.png"), background)
    (tmp_path / "homography.yaml").write_text(yaml.safe_dump({
        "homography": np.eye(3).tolist(),
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "camera": {"source": 0},
        "detector": {"type": "color", "model_file": "car.npz",
                     "background_file": "empty.png",
                     "min_quality": 0.1, "backprojection_threshold": 10,
                     "morphology_kernel": 5},
        "field": {"homography_file": "homography.yaml"},
        "cars": [{"car_id": 1, "name": "Test Car"}],
        "network": {"host": "127.0.0.1", "port": 5000},
        "display": {"enabled": False},
    }), encoding="utf-8")

    class FakeCapture:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            return True, (frame if self.calls == 1 else background).copy()

        def release(self):
            pass

    monkeypatch.setattr("toycar_vision.server._open_camera", lambda config: FakeCapture())
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    try:
        run(load_config(config_path), port_override=receiver.getsockname()[1],
            headless=True, max_frames=2, metrics_csv=str(tmp_path / "runtime.csv"))
        payload, _address = receiver.recvfrom(1024)
        missing_payload, _address = receiver.recvfrom(1024)
    finally:
        receiver.close()
    assert b'"Test Car"' in payload
    assert b"-1000.000" not in payload
    assert b"-1000.000,-1000.000" in missing_payload
    with (tmp_path / "runtime.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["detected"] == "1"
    assert 130.0 < float(row["x_mm"]) < 150.0
