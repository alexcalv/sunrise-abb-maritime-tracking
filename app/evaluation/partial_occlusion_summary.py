from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _local_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    normalized = str(path_value).replace("\\", "/")
    if normalized.startswith("/workspace/"):
        return Path(normalized.removeprefix("/workspace/"))
    return Path(path_value)


def _single_file(directory: Path, pattern: str) -> str | None:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    return str(files[0]) if len(files) == 1 else None


def _mot_rows_equal(path_a: str | None, path_b: str | None) -> bool | None:
    if not path_a or not path_b:
        return None
    left = _local_path(path_a)
    right = _local_path(path_b)
    if left is None or right is None:
        return None
    if not left.exists() or not right.exists():
        return None
    return left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")


def _prediction_rows(occlusion_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in (occlusion_payload.get("sequences") or {}).values():
        if isinstance(sequence, dict):
            rows.extend(row for row in sequence.get("predictions", []) if isinstance(row, dict))
    return rows


def _case_metrics(run_dir: Path) -> dict[str, Any]:
    demo_summary = _load_json(run_dir / "demo_summary.json")
    occlusion_path = _local_path(demo_summary.get("occlusion_predictions_path") or demo_summary.get("occlusion_prediction_report"))
    if occlusion_path is None:
        occlusion_path = Path("")
    elif not occlusion_path.is_absolute() and not occlusion_path.exists() and str(occlusion_path):
        occlusion_path = run_dir / occlusion_path
    occlusion_payload = _load_json(occlusion_path)
    predictions = _prediction_rows(occlusion_payload)
    continuation_rows = [row.get("visual_continuation") or {} for row in predictions if "visual_continuation" in row]
    low_confidence_count = sum(1 for row in continuation_rows if float(row.get("confidence") or 0.0) < 0.45)
    high_uncertainty_count = sum(1 for row in predictions if float(row.get("uncertainty_radius") or 0.0) >= 180.0)
    prediction_count = int(demo_summary.get("prediction_count") or len(predictions))
    recovered_count = int(demo_summary.get("recovered_prediction_count") or 0)
    return {
        "run_dir": str(run_dir),
        "source_video": demo_summary.get("source_video") or demo_summary.get("source"),
        "raw_mot_path": demo_summary.get("raw_mot_path") or _single_file(run_dir / "mot", "*.txt"),
        "canonical_mot_path": demo_summary.get("canonical_mot_path") or _single_file(run_dir / "live_reid_in_loop" / "mot", "*.txt"),
        "side_by_side_video_path": demo_summary.get("side_by_side_video_path"),
        "demo_summary_path": str(run_dir / "demo_summary.json"),
        "occlusion_predictions_path": str(occlusion_path) if str(occlusion_path) else None,
        "accepted_remaps": demo_summary.get("accepted_remaps") or [],
        "accepted_remap_count": len(demo_summary.get("accepted_remaps") or []),
        "prediction_count": prediction_count,
        "recovered_prediction_count": recovered_count,
        "unrecovered_prediction_count": max(0, prediction_count - recovered_count),
        "max_gap_frames": int(demo_summary.get("max_gap_frames") or 0),
        "mean_uncertainty_radius": demo_summary.get("mean_uncertainty_radius"),
        "visual_continuation_enabled": bool(demo_summary.get("visual_continuation_enabled")),
        "continuation_prediction_count": int(demo_summary.get("continuation_prediction_count") or 0),
        "continuation_active_count": int(demo_summary.get("continuation_active_count") or 0),
        "continuation_active_rate": (
            round(float(demo_summary.get("continuation_active_count") or 0) / int(demo_summary.get("continuation_prediction_count") or 1), 6)
            if int(demo_summary.get("continuation_prediction_count") or 0) > 0
            else 0.0
        ),
        "mean_continuation_confidence": demo_summary.get("mean_continuation_confidence"),
        "mean_best_part_score": demo_summary.get("mean_best_part_score"),
        "matched_part_distribution": demo_summary.get("matched_part_distribution") or {},
        "low_confidence_count": low_confidence_count,
        "high_uncertainty_count": high_uncertainty_count,
    }


def _labels(baseline: dict[str, Any] | None, visual: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    metrics = visual or baseline or {}
    prediction_count = int(metrics.get("prediction_count") or 0)
    recovered_count = int(metrics.get("recovered_prediction_count") or 0)
    active_rate = float(metrics.get("continuation_active_rate") or 0.0)
    mean_confidence = float(metrics.get("mean_continuation_confidence") or 0.0)
    remap_count = int(metrics.get("accepted_remap_count") or 0)
    low_confidence = int(metrics.get("low_confidence_count") or 0)
    high_uncertainty = int(metrics.get("high_uncertainty_count") or 0)

    if recovered_count > 0 and mean_confidence >= 0.50:
        labels.append("success_candidate")
    if active_rate >= 0.35 and prediction_count > 0:
        labels.append("partial_occlusion_candidate")
    if prediction_count > 0 and (mean_confidence < 0.35 or low_confidence > max(3, prediction_count // 2) or high_uncertainty > prediction_count // 2):
        labels.append("failure_candidate")
    if remap_count > 1 or prediction_count >= 120:
        labels.append("crowded_ambiguity_candidate")
    return labels or ["needs_review"]


def summarize_partial_occlusion_batches(
    batch_root: str | Path,
    *,
    output_json: str | Path | None = None,
    skipped_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(batch_root)
    cases: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(root.glob("batch_*/*")):
        if not run_dir.is_dir() or not (run_dir / "demo_summary.json").exists():
            continue
        name = run_dir.name
        if name.endswith("_baseline"):
            case_name = name.removesuffix("_baseline")
            variant = "baseline"
        elif name.endswith("_visual"):
            case_name = name.removesuffix("_visual")
            variant = "visual"
        else:
            continue
        cases.setdefault(case_name, {})[variant] = _case_metrics(run_dir)

    completed_cases: list[dict[str, Any]] = []
    for case_name, variants in sorted(cases.items()):
        baseline = variants.get("baseline")
        visual = variants.get("visual")
        if baseline and visual:
            visual["raw_mot_equality_vs_baseline"] = _mot_rows_equal(baseline.get("raw_mot_path"), visual.get("raw_mot_path"))
            visual["canonical_mot_equality_vs_baseline"] = _mot_rows_equal(
                baseline.get("canonical_mot_path"),
                visual.get("canonical_mot_path"),
            )
        completed_cases.append(
            {
                "case": case_name,
                "baseline": baseline,
                "visual": visual,
                "labels": _labels(baseline, visual),
            }
        )

    def best_by(label: str, key: str) -> dict[str, Any] | None:
        candidates = [case for case in completed_cases if label in case["labels"] and (case.get("visual") or case.get("baseline"))]
        if not candidates:
            return None
        return max(candidates, key=lambda case: float(((case.get("visual") or case.get("baseline") or {})).get(key) or 0.0))

    failure_or_limitation = best_by("failure_candidate", "low_confidence_count")
    if failure_or_limitation is None:
        failure_or_limitation = best_by("crowded_ambiguity_candidate", "unrecovered_prediction_count")

    summary = {
        "batch_root": str(root),
        "completed_case_count": len(completed_cases),
        "completed_cases": completed_cases,
        "skipped_cases": skipped_cases or [],
        "best_demo_clip": best_by("success_candidate", "recovered_prediction_count"),
        "best_partial_occlusion_clip": best_by("partial_occlusion_candidate", "continuation_active_rate"),
        "best_failure_limitation_clip": failure_or_limitation,
        "note": "Visual continuation is diagnostic/visualization-only and does not change tracking, ReID decisions, remaps, canonical IDs, or MOT output.",
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize partial-occlusion demo runs under outputs_v2.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = summarize_partial_occlusion_batches(args.batch_root, output_json=args.output_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
