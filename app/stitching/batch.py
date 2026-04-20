from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stitching.runner import stitch_tracks


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_stitched_run(run_root: Path, runs_root: Path, run_summary: dict[str, Any] | None) -> bool:
    try:
        relative_parts = run_root.relative_to(runs_root).parts
    except ValueError:
        relative_parts = run_root.parts

    if "stitched" in relative_parts:
        return True
    if run_summary and run_summary.get("stitching") is not None:
        return True
    return False


def _discover_tracker_runs(runs_root: Path) -> tuple[list[Path], list[str]]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    discovered: list[Path] = []
    skipped: list[str] = []
    seen: set[Path] = set()

    for mot_dir in sorted(runs_root.rglob("mot")):
        if not mot_dir.is_dir():
            continue

        run_root = mot_dir.parent
        if run_root in seen:
            continue
        seen.add(run_root)

        mot_files = sorted(mot_dir.glob("*.txt"))
        if not mot_files:
            continue

        run_summary = _load_json(run_root / "run_summary.json")
        if _is_stitched_run(run_root, runs_root, run_summary):
            skipped.append(str(run_root))
            continue

        discovered.append(run_root)

    return discovered, skipped


def _tracker_name(run_summary: dict[str, Any] | None, run_root: Path) -> str:
    tracker_yaml = None if run_summary is None else run_summary.get("tracker_yaml")
    if tracker_yaml:
        tracker_stem = Path(str(tracker_yaml)).stem.lower()
        if tracker_stem.endswith("_maritime"):
            tracker_stem = tracker_stem[: -len("_maritime")]
        return tracker_stem
    return run_root.name


def _list_to_text(values: list[Any]) -> str:
    return " ".join(str(value) for value in values)


def _accepted_remaps(decisions: list[dict[str, Any]]) -> list[str]:
    return [
        f"{int(decision['source_track_id'])}->{int(decision['target_track_id'])}"
        for decision in decisions
        if decision.get("applied")
    ]


def _accepted_scores(decisions: list[dict[str, Any]]) -> list[float]:
    return [
        round(float(decision["score"]), 6)
        for decision in decisions
        if decision.get("applied")
    ]


def _best_rejected_diagnostic(decisions: list[dict[str, Any]], matches_applied: int) -> str:
    rejected = [decision for decision in decisions if not decision.get("applied")]
    if not rejected:
        return "none" if matches_applied > 0 else "no_candidate_pairs"

    best = max(rejected, key=lambda decision: float(decision.get("score", 0.0)))
    source_track_id = int(best["source_track_id"])
    target_track_id = int(best["target_track_id"])
    score = float(best.get("score", 0.0))
    reason = str(best.get("reason", "rejected"))
    return f"{source_track_id}->{target_track_id} {reason} score={score:.6f}"


def _sequence_status(sequence_report: dict[str, Any]) -> str:
    if int(sequence_report.get("matches_applied", 0)) > 0:
        return "helped"
    return "no_change"


def _sequence_rows_from_report(
    run_root: Path,
    run_summary: dict[str, Any] | None,
    stitch_report: dict[str, Any],
) -> list[dict[str, Any]]:
    tracker = _tracker_name(run_summary, run_root)
    source = None if run_summary is None else run_summary.get("source")

    rows: list[dict[str, Any]] = []
    for sequence_name, sequence_report in sorted((stitch_report.get("sequences") or {}).items()):
        decisions = list(sequence_report.get("decisions") or [])
        accepted_remaps = _accepted_remaps(decisions)
        accepted_scores = _accepted_scores(decisions)
        matches_applied = int(sequence_report.get("matches_applied", 0))

        row = {
            "run_name": run_root.name,
            "tracker": tracker,
            "sequence_name": sequence_name,
            "source": source,
            "original_unique_ids": list(sequence_report.get("unique_track_ids_in") or []),
            "stitched_unique_ids": list(sequence_report.get("unique_track_ids_out") or []),
            "matches_applied": matches_applied,
            "accepted_remaps": accepted_remaps,
            "accepted_scores": accepted_scores,
            "status": _sequence_status(sequence_report),
            "failure_diagnostic": _best_rejected_diagnostic(decisions, matches_applied),
        }
        rows.append(row)

    return rows


def _error_row(run_root: Path, run_summary: dict[str, Any] | None, error: Exception) -> dict[str, Any]:
    return {
        "run_name": run_root.name,
        "tracker": _tracker_name(run_summary, run_root),
        "sequence_name": "",
        "source": None if run_summary is None else run_summary.get("source"),
        "original_unique_ids": [],
        "stitched_unique_ids": [],
        "matches_applied": 0,
        "accepted_remaps": [],
        "accepted_scores": [],
        "status": "error",
        "failure_diagnostic": str(error),
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": row["run_name"],
        "tracker": row["tracker"],
        "sequence_name": row["sequence_name"],
        "source": row["source"] or "",
        "original_unique_ids": _list_to_text(row["original_unique_ids"]),
        "stitched_unique_ids": _list_to_text(row["stitched_unique_ids"]),
        "matches_applied": row["matches_applied"],
        "accepted_remaps": _list_to_text(row["accepted_remaps"]),
        "accepted_scores": _list_to_text(f"{float(score):.6f}" for score in row["accepted_scores"]),
        "status": row["status"],
        "failure_diagnostic": row["failure_diagnostic"],
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "tracker",
        "sequence_name",
        "source",
        "original_unique_ids",
        "stitched_unique_ids",
        "matches_applied",
        "accepted_remaps",
        "accepted_scores",
        "status",
        "failure_diagnostic",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def stitch_batch(
    runs_root: str,
    batch_dir: str | None = None,
    config_path: str | None = None,
    stitch_name: str | None = None,
) -> dict[str, Any]:
    runs_root_path = Path(runs_root)
    run_roots, skipped_runs = _discover_tracker_runs(runs_root_path)
    if not run_roots:
        raise RuntimeError(f"No tracker runs discovered under {runs_root_path}")

    summary_rows: list[dict[str, Any]] = []
    processed_runs: list[str] = []
    failed_runs: list[str] = []
    resolved_stitch_name = stitch_name

    for run_root in run_roots:
        run_summary = _load_json(run_root / "run_summary.json")
        try:
            result = stitch_tracks(
                pred_dir=str(run_root / "mot"),
                run_summary_path=str(run_root / "run_summary.json") if (run_root / "run_summary.json").exists() else None,
                config_path=config_path,
                stitch_name=stitch_name,
            )
            stitch_report = _load_json(Path(result["stitch_report"]))
            if stitch_report is None:
                raise RuntimeError(f"Stitch report missing after processing run: {run_root}")

            resolved_stitch_name = resolved_stitch_name or str(
                ((stitch_report.get("config") or {}).get("stitch_name")) or "stitch_batch"
            )
            summary_rows.extend(_sequence_rows_from_report(run_root, run_summary, stitch_report))
            processed_runs.append(str(run_root))
        except Exception as exc:
            failed_runs.append(str(run_root))
            summary_rows.append(_error_row(run_root, run_summary, exc))

    batch_name = resolved_stitch_name or "stitch_batch"
    batch_output_root = Path(batch_dir) if batch_dir else runs_root_path.parent / "stitching" / "batches" / batch_name
    batch_output_root.mkdir(parents=True, exist_ok=True)

    summary_csv_path = batch_output_root / "summary.csv"
    _write_summary_csv(summary_csv_path, summary_rows)

    summary_json_path = batch_output_root / "summary.json"
    summary_payload = {
        "version": "stitch_batch/v1",
        "runs_root": str(runs_root_path),
        "batch_output_root": str(batch_output_root),
        "stitch_name": batch_name,
        "config_path": str(config_path) if config_path is not None else None,
        "num_discovered_runs": len(run_roots),
        "num_processed_runs": len(processed_runs),
        "num_failed_runs": len(failed_runs),
        "num_sequence_rows": len(summary_rows),
        "processed_runs": processed_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "rows": summary_rows,
        "summary_csv": str(summary_csv_path),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "runs_root": str(runs_root_path),
        "batch_output_root": str(batch_output_root),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "num_discovered_runs": len(run_roots),
        "num_processed_runs": len(processed_runs),
        "num_failed_runs": len(failed_runs),
        "num_sequence_rows": len(summary_rows),
    }
