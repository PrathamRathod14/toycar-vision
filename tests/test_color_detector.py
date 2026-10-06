import cv2
import numpy as np

from tools.calibrate_color_car import foreground_from_background
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


def test_colour_detector_prefers_calibrated_car_size(tmp_path):
    calibration = np.full((480, 640, 3), 255, dtype=np.uint8)
    calibration[190:250, 80:140] = (180, 40, 180)
    calibration[190:250, 140:200] = (40, 180, 40)
    mask = np.zeros(calibration.shape[:2], dtype=np.uint8)
    mask[190:250, 80:200] = 255
    model_path = tmp_path / "car_color.npz"
    save_color_model(model_path, calibration, mask, (195, 220))

    frame = calibration.copy()
    frame[50:170, 350:425] = (180, 40, 180)
    frame[50:170, 425:500] = (40, 180, 40)
    detector = ColorCarDetector(model_path, car_id=1, min_quality=0.1,
                                backprojection_threshold=10, morphology_kernel=5)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert 80 <= detections[0].center_uv[0] <= 200


def test_empty_field_reference_suppresses_static_colour_match(tmp_path):
    background = np.full((240, 320, 3), 255, dtype=np.uint8)
    background[40:100, 20:60] = (180, 40, 180)
    background[40:100, 60:100] = (40, 180, 40)
    cv2.imwrite(str(tmp_path / "empty.png"), background)
    frame = background.copy()
    frame[130:190, 150:190] = (180, 40, 180)
    frame[130:190, 190:230] = (40, 180, 40)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[130:190, 150:230] = 255
    save_color_model(tmp_path / "car.npz", frame, mask, (225, 160))
    detector = ColorCarDetector(
        tmp_path / "car.npz", car_id=1, background_file=tmp_path / "empty.png",
        min_quality=0.1, background_threshold=20,
    )
    assert detector.detect(background) == []
    unrelated = background.copy()
    unrelated[130:190, 150:230] = (70, 70, 70)
    assert detector.detect(unrelated) == []
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert 180 <= detections[0].center_uv[0] <= 200


def test_car_calibration_uses_empty_field_difference():
    background = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame = background.copy()
    frame[45:75, 65:95] = (20, 100, 190)
    mask = foreground_from_background(frame, background, (55, 35, 50, 50), 25)
    assert mask[60, 80] == 255
    assert mask[10, 10] == 0
    assert cv2.countNonZero(mask) > 800
