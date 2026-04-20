from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evaluation.mot import load_mot_frames
from evaluation.pairing import pair_prediction_and_gt

DEFAULT_METRICS = [
    "num_frames",
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "num_switches",
    "num_fragmentations",
    "precision",
    "recall",
]


def _import_motmetrics():
    try:
        import motmetrics as mm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The evaluator requires the 'motmetrics' package. Rebuild the project environment after updating requirements.txt."
        ) from exc
    return mm


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def _evaluate_sequence(gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]], iou_threshold: float, mm):
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.keys()) | set(pred.keys()))

    for frame in frames:
        gt_objs = gt.get(frame, [])
        pred_objs = pred.get(frame, [])
        gt_ids = [obj["id"] for obj in gt_objs]
        pred_ids = [obj["id"] for obj in pred_objs]

        distances: list[list[float]] = []
        for gt_obj in gt_objs:
            row: list[float] = []
            for pred_obj in pred_objs:
                if gt_obj["class_id"] != pred_obj["class_id"]:
                    row.append(float("nan"))
                    continue
                iou = _iou_xywh(gt_obj["bbox"], pred_obj["bbox"])
                row.append(1.0 - iou if iou >= iou_threshold else float("nan"))
            distances.append(row)

        acc.update(gt_ids, pred_ids, distances)

    return acc


def _collect_pairing_report(pred_dir: Path, gt_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    pred_files = {path.name: path for path in pred_dir.glob("*.txt")}
    gt_files = {path.name: path for path in gt_dir.glob("*.txt")}
    pairs = pair_prediction_and_gt(str(pred_dir), str(gt_dir))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    missing_pred = sorted(set(gt_files) - set(pred_files))
    return pairs, missing_gt, missing_pred


def _summary_to_jsonable(summary) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row_name, row in summary.iterrows():
        payload[str(row_name)] = {
            column: _normalize_scalar(row[column])
            for column in summary.columns
        }
    return payload


def evaluate_mot_dir(
    pred_dir: str,
    gt_dir: str,
    output_csv: str,
    output_json: str | None = None,
    iou_threshold: float = 0.5,
    frame_offset: int = 0,
) -> dict[str, Any]:
    mm = _import_motmetrics()

    pred_root = Path(pred_dir)
    gt_root = Path(gt_dir)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_root}")

    pairs, missing_gt, missing_pred = _collect_pairing_report(pred_root, gt_root)
    if not pairs:
        raise RuntimeError(f"No comparable MOT txt pairs found between {pred_root} and {gt_root}")

    accumulators = []
    sequence_names: list[str] = []
    paired_files: list[dict[str, str]] = []
    for pred_path, gt_path in sorted(pairs, key=lambda item: item[0].name):
        gt_rows = load_mot_frames(gt_path)
        pred_rows = load_mot_frames(pred_path, frame_offset=frame_offset)
        accumulators.append(_evaluate_sequence(gt_rows, pred_rows, iou_threshold, mm))
        sequence_names.append(pred_path.stem)
        paired_files.append(
            {
                "sequence": pred_path.stem,
                "pred_file": str(pred_path),
                "gt_file": str(gt_path),
            }
        )

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=sequence_names,
        metrics=DEFAULT_METRICS,
        generate_overall=True,
    )
    summary.to_csv(output_csv_path)

    result: dict[str, Any] = {
        "pred_dir": str(pred_root),
        "gt_dir": str(gt_root),
        "metrics_csv": str(output_csv_path),
        "summary_json": None,
        "iou_threshold": iou_threshold,
        "frame_offset": frame_offset,
        "metrics": DEFAULT_METRICS,
        "paired_sequences": paired_files,
        "missing_gt": missing_gt,
        "missing_pred": missing_pred,
        "num_pairs": len(paired_files),
    }

    if output_json is not None:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_run_summary_path = pred_root.parent / "run_summary.json"
        payload = {
            "pred_dir": str(pred_root),
            "gt_dir": str(gt_root),
            "metrics_csv": str(output_csv_path),
            "iou_threshold": iou_threshold,
            "frame_offset": frame_offset,
            "metrics": DEFAULT_METRICS,
            "paired_sequences": paired_files,
            "missing_gt": missing_gt,
            "missing_pred": missing_pred,
            "tracker_run_summary_path": str(tracker_run_summary_path) if tracker_run_summary_path.exists() else None,
            "tracker_run_summary": json.loads(tracker_run_summary_path.read_text(encoding="utf-8"))
            if tracker_run_summary_path.exists()
            else None,
            "summary": _summary_to_jsonable(summary),
        }
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result["summary_json"] = str(output_json_path)

    return result
