from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evaluation.mot import load_mot_rows, write_mot_rows
from stitching.live_reid import live_reid_tracks
from stitching.online_live_reid import OnlineLiveReIDMapper, sort_mot_rows
from stitching.runner import DEFAULT_CONFIG_PATH, _load_stitch_config


def _stage_single_mot(mot_file: Path, output_dir: Path, video_file: str | None) -> tuple[Path, Path]:
    input_root = output_dir / "_input"
    input_mot_dir = input_root / "mot"
    input_mot_dir.mkdir(parents=True, exist_ok=True)
    staged_mot = input_mot_dir / mot_file.name
    shutil.copy2(mot_file, staged_mot)

    run_summary_path = input_root / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "source": video_file,
                "mot_dir": str(input_mot_dir),
                "sequences": {mot_file.stem: {"mot_path": str(staged_mot)}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return input_mot_dir, run_summary_path


def _run_post_stage(
    *,
    mot_file: Path,
    video_file: str | None,
    config_path: str,
    output_dir: Path,
    confirmation_observations: int,
    colreg_diagnostics: bool,
    colreg_scoring_experiment: bool,
) -> dict[str, Any]:
    input_mot_dir, run_summary_path = _stage_single_mot(mot_file, output_dir, video_file)
    return live_reid_tracks(
        pred_dir=str(input_mot_dir),
        output_dir=str(output_dir / "post_stage"),
        run_summary_path=str(run_summary_path),
        config_path=config_path,
        live_name="parity_post_stage",
        confirmation_observations=confirmation_observations,
        safe_mode=True,
        colreg_diagnostics=colreg_diagnostics,
        colreg_scoring_experiment=colreg_scoring_experiment,
    )


def _rows_to_detections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "raw_track_id": int(row["id"]),
            "bbox": list(row["bbox"]),
            "confidence": float(row["confidence"]),
            "class_id": int(row["class_id"]),
            "visibility": float(row["visibility"]),
        }
        for row in rows
    ]


def _run_online_mapper(
    *,
    mot_file: Path,
    video_file: str | None,
    config_path: str,
    output_dir: Path,
    confirmation_observations: int,
    colreg_diagnostics: bool,
    colreg_scoring_experiment: bool,
) -> dict[str, Any]:
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in load_mot_rows(mot_file):
        rows_by_frame.setdefault(int(row["frame"]), []).append(row)

    mapper = OnlineLiveReIDMapper(
        config=_load_stitch_config(config_path),
        confirmation_observations=confirmation_observations,
        source_video_path=video_file,
        config_path=config_path,
        sequence_name=mot_file.stem,
        safe_mode=True,
        colreg_diagnostics=colreg_diagnostics,
        colreg_scoring_experiment=colreg_scoring_experiment,
    )

    output_rows: list[dict[str, Any]] = []
    for frame in sorted(rows_by_frame):
        output_rows.extend(mapper.update(frame, _rows_to_detections(rows_by_frame[frame])))
    output_rows.extend(mapper.flush())
    output_rows = sort_mot_rows(output_rows)

    online_root = output_dir / "online_mapper"
    output_mot_path = online_root / "mot" / mot_file.name
    write_mot_rows(output_mot_path, output_rows)

    report = mapper.report()
    report["source_mot_path"] = str(mot_file)
    report["source_video_path"] = video_file
    report["input_mot_path"] = str(mot_file)
    report["output_mot_path"] = str(output_mot_path)
    report["summary"]["source_mot_path"] = str(mot_file)
    report["summary"]["source_video_path"] = video_file

    report_path = online_root / "live_reid_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mapping_path = online_root / "mapping_summary.json"
    mapping_path.write_text(
        json.dumps(
            {
                "mode": "in_loop_bounded_latency",
                "source_mot_path": str(mot_file),
                "source_video_path": video_file,
                "sequences": {mot_file.stem: mapper.mapping_summary()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "mot_dir": str(output_mot_path.parent),
        "mot_path": str(output_mot_path),
        "live_reid_report": str(report_path),
        "mapping_summary": str(mapping_path),
        "report": report,
    }


def _sequence_report(report_path: Path, sequence_name: str) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences") or {}
    if sequence_name not in sequences:
        raise KeyError(f"Sequence {sequence_name!r} not found in {report_path}")
    return sequences[sequence_name]


def _appearance_supported_remaps(report: dict[str, Any]) -> list[str]:
    if report.get("appearance_supported_remaps") is not None:
        return sorted(report.get("appearance_supported_remaps") or [])
    return sorted(
        f"{int(decision['new_source_track_id'])}->{int(decision['candidate_source_track_id'])}"
        for decision in report.get("decisions") or []
        if bool(decision.get("applied")) and bool(decision.get("appearance_assisted"))
    )


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "accepted_remaps": sorted(report.get("accepted_remaps") or []),
        "accepted_remap_count": int(report.get("accepted_remap_count") or 0),
        "unique_ids_before": len(report.get("unique_track_ids_in") or []),
        "unique_ids_after": len(report.get("unique_track_ids_out") or []),
        "appearance_supported_remaps": _appearance_supported_remaps(report),
        "chain_control_rejected_count": int(report.get("chain_control_rejected_count") or 0),
        "short_gap_gate_rejected_count": int(report.get("short_gap_gate_rejected_count") or 0),
        "decision_latency_mean": report.get("decision_latency_mean", report.get("decision_latency_frames_mean")),
        "decision_latency_max": report.get("decision_latency_max", report.get("decision_latency_frames_max")),
    }
    for key in (
        "colreg_scoring_experiment_enabled",
        "colreg_score_used_count",
        "colreg_changed_ranking_count",
        "colreg_changed_accepted_remap_count",
    ):
        if key in report:
            compact[key] = report.get(key)
    return compact


def _normalize_rows(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        bbox = tuple(round(float(value), 4) for value in row["bbox"])
        normalized.append(
            (
                int(row["frame"]),
                int(row["id"]),
                *bbox,
                round(float(row["confidence"]), 6),
                int(row["class_id"]),
            )
        )
    return sorted(normalized)


def _first_difference(first: list[tuple[Any, ...]], second: list[tuple[Any, ...]]) -> dict[str, Any] | None:
    if len(first) != len(second):
        return {"kind": "row_count", "post_stage_rows": len(first), "online_mapper_rows": len(second)}
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return {"kind": "row_content", "index": index, "post_stage": left, "online_mapper": right}
    return None


def validate_parity(
    *,
    mot_file: str | Path,
    output_dir: str | Path,
    video_file: str | None = None,
    config_path: str | Path | None = None,
    confirmation_observations: int = 10,
    colreg_diagnostics: bool = False,
    colreg_scoring_experiment: bool = False,
) -> dict[str, Any]:
    """Compare post-stage ReID with the online mapper without changing ReID decisions."""

    mot_path = Path(mot_file)
    if not mot_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = str(config_path or DEFAULT_CONFIG_PATH)

    post_result = _run_post_stage(
        mot_file=mot_path,
        video_file=video_file,
        config_path=config,
        output_dir=output_path,
        confirmation_observations=confirmation_observations,
        colreg_diagnostics=colreg_diagnostics,
        colreg_scoring_experiment=colreg_scoring_experiment,
    )
    online_result = _run_online_mapper(
        mot_file=mot_path,
        video_file=video_file,
        config_path=config,
        output_dir=output_path,
        confirmation_observations=confirmation_observations,
        colreg_diagnostics=colreg_diagnostics,
        colreg_scoring_experiment=colreg_scoring_experiment,
    )

    sequence_name = mot_path.stem
    post_report = _sequence_report(Path(post_result["live_reid_report"]), sequence_name)
    online_report = online_result["report"]
    post_mot_path = Path(post_result["mot_dir"]) / mot_path.name
    online_mot_path = Path(online_result["mot_path"])

    first_difference = _first_difference(
        _normalize_rows(load_mot_rows(post_mot_path)),
        _normalize_rows(load_mot_rows(online_mot_path)),
    )
    post_compact = _compact_report(post_report)
    online_compact = _compact_report(online_report)
    compact_fields_match = post_compact == online_compact

    summary = {
        "sequence_name": sequence_name,
        "mot_file": str(mot_path),
        "video_file": video_file,
        "config_path": config,
        "confirmation_observations": int(confirmation_observations),
        "colreg_diagnostics": bool(colreg_diagnostics),
        "colreg_scoring_experiment": bool(colreg_scoring_experiment),
        "parity_passed": first_difference is None and compact_fields_match,
        "compact_fields_match": compact_fields_match,
        "mot_rows_match": first_difference is None,
        "first_difference": first_difference,
        "post_stage": {
            "mot_path": str(post_mot_path),
            "report_path": post_result["live_reid_report"],
            "compact": post_compact,
        },
        "online_mapper": {
            "mot_path": str(online_mot_path),
            "report_path": online_result["live_reid_report"],
            "mapping_summary": online_result["mapping_summary"],
            "compact": online_compact,
        },
    }
    (output_path / "parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
