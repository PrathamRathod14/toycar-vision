import csv

import numpy as np

from toycar_vision.evaluation import evaluate_runtime, roc_curve


def test_roc_curve_perfect_classifier():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, _ = roc_curve(labels, scores)
    assert np.trapezoid(tpr, fpr) == 1.0


def test_runtime_ignores_duplicate_rows_per_car(tmp_path):
    path = tmp_path / "runtime.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_us", "processing_ms"))
        writer.writerow((0, 10.0))
        writer.writerow((0, 10.0))
        writer.writerow((20_000, 20.0))
        writer.writerow((20_000, 20.0))
    result = evaluate_runtime(str(path), frame_budget_ms=16.0)
    assert result["frames"] == 2
    assert result["capture_loop_fps"] == 50.0
    assert result["frames_within_budget_percent"] == 50.0
