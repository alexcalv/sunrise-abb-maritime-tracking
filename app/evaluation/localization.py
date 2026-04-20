from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from evaluation.mot import load_mot_frames, load_mot_rows
from evaluation.pairing import pair_prediction_and_gt


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def _center_distance(first_bbox: list[float], second_bbox: list[float]) -> float:
    first_center = _bbox_center(first_bbox)
    second_center = _bbox_center(second_bbox)
    return math.dist(first_center, second_center)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _pairing_report(pred_dir: Path, gt_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    pred_files = {path.name: path for path in pred_dir.glob("*.txt")}
    gt_files = {path.name: path for path in gt_dir.glob("*.txt")}
    pairs = pair_prediction_and_gt(str(pred_dir), str(gt_dir))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    missing_pred = sorted(set(gt_files) - set(pred_files))
    return pairs, missing_gt, missing_pred


def _sequence_localization_metrics(
    gt_path: Path,
    pred_path: Path,
    frame_offset: int = 0,
) -> dict[str, Any]:
    gt_rows = load_mot_rows(gt_path)
    pred_rows_by_frame = load_mot_frames(pred_path, frame_offset=frame_offset)

    per_frame: list[dict[str, Any]] = []
    errors: list[float] = []
    missing_prediction_frames: list[int] = []

    for gt_row in gt_rows:
        frame = int(gt_row["frame"])
        candidates = pred_rows_by_frame.get(frame, [])
        if not candidates:
            missing_prediction_frames.append(frame)
            per_frame.append(
                {
                    "frame": frame,
                    "gt_id": int(gt_row["id"]),
                    "matched": False,
                    "pred_id": None,
                    "center_error_px": None,
                }
            )
            continue

        best = min(candidates, key=lambda row: _center_distance(gt_row["bbox"], row["bbox"]))
        error = _center_distance(gt_row["bbox"], best["bbox"])
        errors.append(error)
        per_frame.append(
            {
                "frame": frame,
                "gt_id": int(gt_row["id"]),
                "matched": True,
                "pred_id": int(best["id"]),
                "center_error_px": _round(error),
            }
        )

    return {
        "sequence_name": pred_path.stem,
        "pred_file": str(pred_path),
        "gt_file": str(gt_path),
        "gt_frame_count": len(gt_rows),
        "evaluated_frame_count": len(errors),
        "missing_prediction_frame_count": len(missing_prediction_frames),
        "missing_prediction_frames": missing_prediction_frames,
        "mean_center_error_px": _round(statistics.fmean(errors)) if errors else None,
        "median_center_error_px": _round(statistics.median(errors)) if errors else None,
        "max_center_error_px": _round(max(errors)) if errors else None,
        "per_frame": per_frame,
    }


def _overall_metrics(sequence_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_errors: list[float] = []
    total_gt_frames = 0
    total_evaluated_frames = 0
    total_missing_frames = 0

    for result in sequence_results:
        total_gt_frames += int(result["gt_frame_count"])
        total_evaluated_frames += int(result["evaluated_frame_count"])
        total_missing_frames += int(result["missing_prediction_frame_count"])
        all_errors.extend(
            float(frame["center_error_px"])
            for frame in result["per_frame"]
            if frame["center_error_px"] is not None
        )

    return {
        "sequence_name": "OVERALL",
        "pred_file": None,
        "gt_file": None,
        "gt_frame_count": total_gt_frames,
        "evaluated_frame_count": total_evaluated_frames,
        "missing_prediction_frame_count": total_missing_frames,
        "mean_center_error_px": _round(statistics.fmean(all_errors)) if all_errors else None,
        "median_center_error_px": _round(statistics.median(all_errors)) if all_errors else None,
        "max_center_error_px": _round(max(all_errors)) if all_errors else None,
    }


def evaluate_sparse_localization_dir(
    pred_dir: str,
    gt_dir: str,
    output_csv: str,
    output_json: str | None = None,
    frame_offset: int = 0,
) -> dict[str, Any]:
    pred_root = Path(pred_dir)
    gt_root = Path(gt_dir)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"Sparse localization GT directory not found: {gt_root}")

    pairs, missing_gt, missing_pred = _pairing_report(pred_root, gt_root)
    if not pairs:
        raise RuntimeError(f"No comparable sparse localization txt pairs found between {pred_root} and {gt_root}")

    sequence_results = [
        _sequence_localization_metrics(gt_path=gt_path, pred_path=pred_path, frame_offset=frame_offset)
        for pred_path, gt_path in sorted(pairs, key=lambda item: item[0].name)
    ]
    overall = _overall_metrics(sequence_results)

    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sequence_name",
            "gt_frame_count",
            "evaluated_frame_count",
            "missing_prediction_frame_count",
            "mean_center_error_px",
            "median_center_error_px",
            "max_center_error_px",
            "pred_file",
            "gt_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sequence_results:
            writer.writerow({field: row.get(field) for field in fieldnames})
        writer.writerow({field: overall.get(field) for field in fieldnames})

    result: dict[str, Any] = {
        "pred_dir": str(pred_root),
        "gt_dir": str(gt_root),
        "metrics_csv": str(output_csv_path),
        "summary_json": None,
        "frame_offset": frame_offset,
        "matching_mode": "nearest_prediction_by_center_same_frame",
        "paired_sequences": [
            {
                "sequence": result["sequence_name"],
                "pred_file": result["pred_file"],
                "gt_file": result["gt_file"],
            }
            for result in sequence_results
        ],
        "missing_gt": missing_gt,
        "missing_pred": missing_pred,
        "num_pairs": len(sequence_results),
        "overall": overall,
    }

    if output_json is not None:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **result,
            "summary": {
                "sequences": sequence_results,
                "overall": overall,
            },
        }
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result["summary_json"] = str(output_json_path)

    return result
