from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC needs at least one positive and one negative sample")
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    tpr, fpr = [], []
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(float(np.sum(predicted & labels) / positives))
        fpr.append(float(np.sum(predicted & ~labels) / negatives))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def evaluate_detection(csv_path: str, output_path: str, label_column: str, score_column: str) -> dict:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row[label_column]) for row in rows], dtype=bool)
    scores = np.asarray([float(row[score_column]) for row in rows], dtype=float)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(np.trapezoid(tpr, fpr))

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="0.65")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="Toy-car detector ROC", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {"samples": len(rows), "positives": int(labels.sum()),
            "negatives": int((~labels).sum()), "auc": auc}


def _angle_error(measured: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs((measured - truth + 180.0) % 360.0 - 180.0)


def evaluate_mapping(csv_path: str) -> dict:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ("measured_x_mm", "measured_y_mm", "measured_theta_deg",
                "true_x_mm", "true_y_mm", "true_theta_deg")
    if not rows:
        raise ValueError("Mapping CSV contains no rows")
    measured_x = np.asarray([float(row[required[0]]) for row in rows])
    measured_y = np.asarray([float(row[required[1]]) for row in rows])
    measured_theta = np.asarray([float(row[required[2]]) for row in rows])
    true_x = np.asarray([float(row[required[3]]) for row in rows])
    true_y = np.asarray([float(row[required[4]]) for row in rows])
    true_theta = np.asarray([float(row[required[5]]) for row in rows])
    position_error = np.hypot(measured_x - true_x, measured_y - true_y)
    angle_error = _angle_error(measured_theta, true_theta)
    return {
        "samples": len(rows),
        "position_error_mm": {
            "mean": float(np.mean(position_error)),
            "median": float(np.median(position_error)),
            "p95": float(np.percentile(position_error, 95)),
            "max": float(np.max(position_error)),
        },
        "orientation_error_deg": {
            "mean": float(np.mean(angle_error)),
            "median": float(np.median(angle_error)),
            "p95": float(np.percentile(angle_error, 95)),
            "max": float(np.max(angle_error)),
        },
    }


def evaluate_runtime(csv_path: str, frame_budget_ms: float = 1000.0 / 60.0) -> dict:
    """Summarize one metrics row per frame (duplicate car rows are ignored)."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_timestamp: dict[int, float] = {}
    for row in rows:
        by_timestamp.setdefault(int(row["timestamp_us"]), float(row["processing_ms"]))
    if not by_timestamp:
        raise ValueError("Runtime CSV contains no frames")
    timestamps = np.asarray(sorted(by_timestamp), dtype=np.int64)
    processing = np.asarray([by_timestamp[int(timestamp)] for timestamp in timestamps])
    elapsed_s = float((timestamps[-1] - timestamps[0]) / 1_000_000.0)
    capture_fps = float((len(timestamps) - 1) / elapsed_s) if elapsed_s > 0 else 0.0
    return {
        "frames": len(timestamps),
        "capture_loop_fps": capture_fps,
        "frame_budget_ms": float(frame_budget_ms),
        "frames_within_budget_percent": float(np.mean(processing <= frame_budget_ms) * 100.0),
        "processing_ms": {
            "mean": float(np.mean(processing)),
            "median": float(np.median(processing)),
            "p95": float(np.percentile(processing, 95)),
            "max": float(np.max(processing)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate toy-car detection and field mapping")
    subparsers = parser.add_subparsers(dest="command", required=True)
    detection = subparsers.add_parser("detection", help="produce a ROC curve")
    detection.add_argument("--csv", required=True)
    detection.add_argument("--output", default="roc_curve.png")
    detection.add_argument("--label-column", default="present")
    detection.add_argument("--score-column", default="score")
    mapping = subparsers.add_parser("mapping", help="calculate position/orientation error")
    mapping.add_argument("--csv", required=True)
    mapping.add_argument("--output", help="optional JSON output path")
    runtime = subparsers.add_parser("runtime", help="summarize processing time and FPS")
    runtime.add_argument("--csv", required=True)
    runtime.add_argument("--budget-ms", type=float, default=1000.0 / 60.0)
    runtime.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    if args.command == "detection":
        result = evaluate_detection(args.csv, args.output, args.label_column, args.score_column)
    elif args.command == "mapping":
        result = evaluate_mapping(args.csv)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = evaluate_runtime(args.csv, args.budget_ms)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
