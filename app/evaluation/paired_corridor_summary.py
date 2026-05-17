from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VARIANTS = ("baseline", "visual", "corridor")


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


def _rows_equal(path_a: str | None, path_b: str | None) -> bool | None:
    left = _local_path(path_a)
    right = _local_path(path_b)
    if left is None or right is None or not left.exists() or not right.exists():
        return None
    return left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")


def _prediction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in (payload.get("sequences") or {}).values():
        if isinstance(sequence, dict):
            rows.extend(row for row in sequence.get("predictions", []) if isinstance(row, dict))
    return rows


def _distribution(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _metrics(run_dir: Path) -> dict[str, Any]:
    demo_summary = _load_json(run_dir / "demo_summary.json")
    occlusion_path = _local_path(
        demo_summary.get("occlusion_predictions_path")
        or demo_summary.get("occlusion_prediction_report")
    )
    if occlusion_path is None:
        occlusion_path = Path("")
    elif not occlusion_path.is_absolute() and not occlusion_path.exists() and str(occlusion_path):
        occlusion_path = run_dir / occlusion_path
    occlusion_payload = _load_json(occlusion_path)
    predictions = _prediction_rows(occlusion_payload)
    prediction_count = int(demo_summary.get("prediction_count") or len(predictions))
    recovered_count = int(demo_summary.get("recovered_prediction_count") or 0)
    continuation_count = int(demo_summary.get("continuation_prediction_count") or 0)
    continuation_active = int(demo_summary.get("continuation_active_count") or 0)
    sides = [
        prediction.get("expected_reappearance_side")
        for prediction in predictions
        if prediction.get("expected_reappearance_side") is not None
    ]
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
        "continuation_prediction_count": continuation_count,
        "continuation_active_count": continuation_active,
        "continuation_active_rate": (
            round(float(continuation_active) / continuation_count, 6)
            if continuation_count > 0
            else 0.0
        ),
        "mean_continuation_confidence": demo_summary.get("mean_continuation_confidence"),
        "mean_best_part_score": demo_summary.get("mean_best_part_score"),
        "matched_part_distribution": demo_summary.get("matched_part_distribution") or {},
        "paired_prediction_count": int(demo_summary.get("paired_prediction_count") or 0),
        "occluder_pair_count": int(demo_summary.get("occluder_pair_count") or 0),
        "mean_occlusion_pair_confidence": demo_summary.get("mean_occlusion_pair_confidence"),
        "corridor_prediction_count": int(demo_summary.get("corridor_prediction_count") or 0),
        "expected_reappearance_side_distribution": _distribution(sides),
    }


def _labels(variants: dict[str, dict[str, Any] | None]) -> list[str]:
    corridor = variants.get("corridor") or {}
    visual = variants.get("visual") or {}
    baseline = variants.get("baseline") or {}
    main = corridor or visual or baseline
    labels: list[str] = []
    if int(main.get("recovered_prediction_count") or 0) > 0 and int(main.get("accepted_remap_count") or 0) > 0:
        labels.append("success_candidate")
    if float((visual or corridor).get("continuation_active_rate") or 0.0) >= 0.35:
        labels.append("partial_occlusion_candidate")
    if (
        int(corridor.get("corridor_prediction_count") or 0) > 0
        and int(corridor.get("occluder_pair_count") or 0) > 0
        and float(corridor.get("mean_occlusion_pair_confidence") or 0.0) >= 0.20
    ):
        labels.append("corridor_useful_candidate")
    if int(main.get("unrecovered_prediction_count") or 0) > int(main.get("recovered_prediction_count") or 0):
        labels.append("failure_candidate")
    if int(main.get("accepted_remap_count") or 0) > 1 or int(main.get("prediction_count") or 0) >= 120:
        labels.append("crowded_ambiguity_candidate")
    return labels or ["needs_review"]


def _case_name_and_variant(run_dir: Path) -> tuple[str, str] | None:
    name = run_dir.name
    for variant in VARIANTS:
        suffix = f"_{variant}"
        if name.endswith(suffix):
            return name.removesuffix(suffix), variant
    return None


def summarize_paired_corridor_hard_cases(
    batch_root: str | Path,
    *,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(batch_root)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for run_dir in sorted(root.glob("**/*")):
        if not run_dir.is_dir() or not (run_dir / "demo_summary.json").exists():
            continue
        parsed = _case_name_and_variant(run_dir)
        if parsed is None:
            continue
        case_name, variant = parsed
        grouped.setdefault(case_name, {})[variant] = _metrics(run_dir)

    completed_cases: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    for case_name, variants in sorted(grouped.items()):
        baseline = variants.get("baseline")
        if baseline:
            for variant in ("visual", "corridor"):
                if variant in variants:
                    variants[variant]["raw_mot_equality_vs_baseline"] = _rows_equal(
                        baseline.get("raw_mot_path"),
                        variants[variant].get("raw_mot_path"),
                    )
                    variants[variant]["canonical_mot_equality_vs_baseline"] = _rows_equal(
                        baseline.get("canonical_mot_path"),
                        variants[variant].get("canonical_mot_path"),
                    )
        missing = [variant for variant in VARIANTS if variant not in variants]
        if missing:
            skipped_cases.append({"case": case_name, "missing_variants": missing})
        completed_cases.append(
            {
                "case": case_name,
                "baseline": variants.get("baseline"),
                "visual": variants.get("visual"),
                "corridor": variants.get("corridor"),
                "labels": _labels(variants),
            }
        )

    def best_by(label: str, variant: str, key: str) -> dict[str, Any] | None:
        candidates = [
            case for case in completed_cases
            if label in case["labels"] and isinstance(case.get(variant), dict)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda case: float((case[variant] or {}).get(key) or 0.0))

    limitation = best_by("failure_candidate", "corridor", "unrecovered_prediction_count")
    if limitation is None:
        limitation = best_by("crowded_ambiguity_candidate", "corridor", "unrecovered_prediction_count")

    summary = {
        "batch_root": str(root),
        "completed_case_count": len([case for case in completed_cases if not any(case.get(v) is None for v in VARIANTS)]),
        "case_count": len(completed_cases),
        "completed_cases": completed_cases,
        "skipped_cases": skipped_cases,
        "best_success_clip": best_by("success_candidate", "corridor", "recovered_prediction_count"),
        "best_partial_occlusion_clip": best_by("partial_occlusion_candidate", "visual", "continuation_active_rate"),
        "best_corridor_overlay_clip": best_by("corridor_useful_candidate", "corridor", "corridor_prediction_count"),
        "best_failure_limitation_clip": limitation,
        "note": "Visual continuation and paired corridors are diagnostic/rendering only and must not change raw MOT, canonical MOT, ReID decisions, tracker state, thresholds, COLREG, or AIS.",
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize paired-corridor hard-case demo runs under outputs_v2.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_paired_corridor_hard_cases(args.batch_root, output_json=args.output_json), indent=2))


if __name__ == "__main__":
    main()
