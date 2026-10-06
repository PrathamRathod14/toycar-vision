import csv
import socket
import sys

import cv2
import numpy as np
import yaml

import start
from tools.score_labeled_frames import score_frames
from toycar_vision.color_detector import save_color_model
from toycar_vision.config import load_config
from toycar_vision.server import run


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
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "camera": {"source": 0},
        "detector": {"type": "color", "model_file": "car.npz"},
        "field": {"homography_file": "unused.yaml"},
        "cars": [{"car_id": 1, "name": "Test Car"}],
        "network": {"port": 5000},
    }), encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,present\npositive.png,1\nnegative.png,0\n", encoding="utf-8")
    scored = tmp_path / "scored.csv"
    assert score_frames(str(config), str(labels), str(scored)) == 2
    with scored.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["score"]) > float(rows[1]["score"])
    assert float(rows[1]["score"]) == 0.0


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


def test_color_frame_reaches_udp_receiver_with_mapped_pose(tmp_path, monkeypatch):
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    frame[90:150, 80:140] = (180, 40, 180)
    frame[90:150, 140:200] = (40, 180, 40)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[90:150, 80:200] = 255
    save_color_model(tmp_path / "car.npz", frame, mask, (195, 120))
    (tmp_path / "homography.yaml").write_text(yaml.safe_dump({
        "homography": np.eye(3).tolist(),
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "camera": {"source": 0},
        "detector": {"type": "color", "model_file": "car.npz",
                     "min_quality": 0.1, "backprojection_threshold": 10,
                     "morphology_kernel": 5},
        "field": {"homography_file": "homography.yaml"},
        "cars": [{"car_id": 1, "name": "Test Car"}],
        "network": {"host": "127.0.0.1", "port": 5000},
        "display": {"enabled": False},
    }), encoding="utf-8")

    class FakeCapture:
        def read(self):
            return True, frame.copy()

        def release(self):
            pass

    monkeypatch.setattr("toycar_vision.server._open_camera", lambda config: FakeCapture())
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    try:
        run(load_config(config_path), port_override=receiver.getsockname()[1],
            headless=True, max_frames=1, metrics_csv=str(tmp_path / "runtime.csv"))
        payload, _address = receiver.recvfrom(1024)
    finally:
        receiver.close()
    assert b'"Test Car"' in payload
    assert b"-1000.000" not in payload
    with (tmp_path / "runtime.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["detected"] == "1"
    assert 130.0 < float(row["x_mm"]) < 150.0
