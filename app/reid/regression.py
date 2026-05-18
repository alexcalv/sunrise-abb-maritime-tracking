from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.mot import load_mot_rows
from reid.parity import validate_parity

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUT28_MOT = REPO_ROOT / "outputs" / "track" / "botsort_one" / "mot" / "cut28#1.txt"
DEFAULT_CUT28_VIDEO = REPO_ROOT / "data" / "videos" / "cut28#1.mp4"
DEFAULT_CUT28_CONFIG = REPO_ROOT / "config" / "stitching" / "reid_appearance_v2.yaml"
DEFAULT_CUT29_MOT = REPO_ROOT / "outputs" / "track" / "botsort_all_videos" / "mot" / "cut29#3.txt"
DEFAULT_CUT29_VIDEO = REPO_ROOT / "data" / "videos" / "cut29#3.mp4"
DEFAULT_CUT29_CONFIG = REPO_ROOT / "config" / "stitching" / "reid_appearance_v1.yaml"

REQUIRED_REPORT_FIELDS = [
    "accepted_remap_count",
    "chain_control_checked_count",
    "chain_control_rejected_count",
    "short_gap_gate_checked_count",
    "short_gap_gate_rejected_count",
    "decision_latency_mean",
    "decision_latency_max",
]


def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sequence_report(report_path: str | Path, sequence_name: str) -> dict[str, Any]:
    report = _load_report(report_path)
    sequences = report.get("sequences") or {}
    return sequences.get(sequence_name, report)


def _missing_fields(report: dict[str, Any]) -> list[str]:
    return [field for field in REQUIRED_REPORT_FIELDS if field not in report]


def _input_contains_ids(mot_path: Path, expected_remap: str) -> bool:
    source_id, target_id = (int(value) for value in expected_remap.split("->", maxsplit=1))
    ids = {int(row["id"]) for row in load_mot_rows(mot_path)}
    return source_id in ids and target_id in ids


def _resolve_config(global_config: str | Path | None, default_config: Path) -> str:
    return str(Path(global_config).resolve()) if global_config else str(default_config)


def _run_case(
    *,
    name: str,
    mot_path: Path,
    video_path: Path,
    config_path: str,
    output_dir: Path,
    confirmation_observations: int,
    expected_remap: str | None,
    require_expected_remap: bool,
    colreg_diagnostics: bool,
) -> dict[str, Any]:
    case_output = output_dir / name
    parity_summary = validate_parity(
        mot_file=mot_path,
        video_file=str(video_path),
        config_path=config_path,
        output_dir=case_output,
        confirmation_observations=confirmation_observations,
        colreg_diagnostics=colreg_diagnostics,
    )

    post_report = _sequence_report(parity_summary["post_stage"]["report_path"], mot_path.stem)
    online_report = _load_report(parity_summary["online_mapper"]["report_path"])
    accepted_remaps = set(parity_summary["online_mapper"]["compact"]["accepted_remaps"])
    expected_remap_checked = False
    expected_remap_present = None
    if expected_remap is not None:
        expected_remap_checked = require_expected_remap or _input_contains_ids(mot_path, expected_remap)
        expected_remap_present = expected_remap in accepted_remaps

    post_missing = _missing_fields(post_report)
    online_missing = _missing_fields(online_report)
    passed = bool(parity_summary["parity_passed"]) and not post_missing and not online_missing
    if expected_remap_checked:
        passed = passed and bool(expected_remap_present)
    passed = passed and online_report.get("mode") == "in_loop_bounded_latency"

    return {
        "case": name,
        "passed": passed,
        "mot_file": str(mot_path),
        "video_file": str(video_path),
        "config_path": config_path,
        "parity_passed": bool(parity_summary["parity_passed"]),
        "mot_rows_match": bool(parity_summary["mot_rows_match"]),
        "compact_fields_match": bool(parity_summary["compact_fields_match"]),
        "accepted_remaps": parity_summary["online_mapper"]["compact"]["accepted_remaps"],
        "accepted_remap_count": parity_summary["online_mapper"]["compact"]["accepted_remap_count"],
        "unique_ids_before": parity_summary["online_mapper"]["compact"]["unique_ids_before"],
        "unique_ids_after": parity_summary["online_mapper"]["compact"]["unique_ids_after"],
        "post_stage_missing_fields": post_missing,
        "online_mapper_missing_fields": online_missing,
        "online_mapper_mode": online_report.get("mode"),
        "expected_remap": expected_remap,
        "expected_remap_checked": expected_remap_checked,
        "expected_remap_present": expected_remap_present,
        "parity_summary_path": str(case_output / "parity_summary.json"),
    }


def check_regression(
    *,
    output_dir: str | Path,
    cut28_mot: str | Path | None = None,
    cut28_video: str | Path | None = None,
    cut29_mot: str | Path | None = None,
    cut29_video: str | Path | None = None,
    config: str | Path | None = None,
    confirmation_observations: int = 10,
    colreg_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run the saved-MOT ReID regression used for review and CI."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cut28_mot_path = Path(cut28_mot or DEFAULT_CUT28_MOT)
    cut28_video_path = Path(cut28_video or DEFAULT_CUT28_VIDEO)
    cut29_mot_path = Path(cut29_mot or DEFAULT_CUT29_MOT)
    cut29_video_path = Path(cut29_video or DEFAULT_CUT29_VIDEO)

    for path in (cut28_mot_path, cut28_video_path, cut29_mot_path, cut29_video_path):
        if not path.exists():
            raise FileNotFoundError(f"Required regression input not found: {path}")

    cases = [
        _run_case(
            name="cut28",
            mot_path=cut28_mot_path,
            video_path=cut28_video_path,
            config_path=_resolve_config(config, DEFAULT_CUT28_CONFIG),
            output_dir=output_path,
            confirmation_observations=confirmation_observations,
            expected_remap="6->2",
            require_expected_remap=False,
            colreg_diagnostics=colreg_diagnostics,
        ),
        _run_case(
            name="cut29_saved",
            mot_path=cut29_mot_path,
            video_path=cut29_video_path,
            config_path=_resolve_config(config, DEFAULT_CUT29_CONFIG),
            output_dir=output_path,
            confirmation_observations=confirmation_observations,
            expected_remap="275->244",
            require_expected_remap=True,
            colreg_diagnostics=colreg_diagnostics,
        ),
    ]
    summary = {
        "passed": all(case["passed"] for case in cases),
        "confirmation_observations": int(confirmation_observations),
        "colreg_diagnostics": bool(colreg_diagnostics),
        "note": (
            "Saved MOT regression checks historical remaps. Fresh tracking may assign different raw tracker IDs "
            "across runtime environments."
        ),
        "cases": cases,
    }
    (output_path / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
