import cv2
import numpy as np

from toycar_vision.color_detector import ColorCarDetector, save_color_model


def test_colour_detector_finds_center_and_front(tmp_path):
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[190:290, 220:340] = (180, 40, 180)  # purple rear
    frame[190:290, 340:460] = (40, 180, 40)   # green front
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[190:290, 220:460] = 255
    model_path = tmp_path / "car_color.npz"
    save_color_model(model_path, frame, mask, (460, 240))

    detector = ColorCarDetector(
        model_path,
        car_id=1,
        min_quality=0.1,
        backprojection_threshold=10,
        morphology_kernel=5,
    )
    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].marker_id == 1
    assert detections[0].heading_point_uv[0] > detections[0].center_uv[0]


def test_colour_detector_rejects_oversized_matching_background(tmp_path):
    calibration = np.full((480, 640, 3), 255, dtype=np.uint8)
    calibration[190:250, 80:140] = (180, 40, 180)
    calibration[190:250, 140:200] = (40, 180, 40)
    mask = np.zeros(calibration.shape[:2], dtype=np.uint8)
    mask[190:250, 80:200] = 255
    model_path = tmp_path / "car_color.npz"
    save_color_model(model_path, calibration, mask, (195, 220))

    frame = calibration.copy()
    frame[50:230, 350:440] = (180, 40, 180)
    frame[50:230, 440:530] = (40, 180, 40)
    detector = ColorCarDetector(model_path, car_id=1, min_quality=0.1,
                                backprojection_threshold=10, morphology_kernel=5)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert 80 <= detections[0].center_uv[0] <= 200
