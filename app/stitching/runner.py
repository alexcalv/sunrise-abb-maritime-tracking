from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from shutil import copy2
from typing import Any

from ais.types import AisConfig
from colreg.types import ColregConfig
from evaluation.mot import load_mot_rows, write_mot_rows
from stitching.appearance import build_tracklet_appearance
from stitching.matcher import match_tracklets
from stitching.schemas import (
    AisConfig,
    AppearanceConfig,
    BBoxConfig,
    MatchingConfig,
    MatchingWeights,
    MemoryConfig,
    MotionConfig,
    SequenceStitchReport,
    StitchConfig,
    StitchReport,
    TrackletConfig,
)
from stitching.tracklets import build_tracklets

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stitching" / "default.yaml"


def _import_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The stitcher requires PyYAML to read configuration files."
        ) from exc
    return yaml


def _load_stitch_config(config_path: str | None) -> StitchConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if config_path is None:
            return StitchConfig()
        raise FileNotFoundError(f"Stitch config not found: {path}")

    yaml = _import_yaml()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    matching_raw = dict(raw.get("matching") or {})
    weights_raw = dict((matching_raw.get("weights") or {}))
    appearance_weight = matching_raw.get("appearance_weight")
    if appearance_weight is not None:
        weights_raw["appearance"] = appearance_weight

    matching_config = MatchingConfig(
        **{
            **matching_raw,
            "weights": MatchingWeights(**weights_raw),
        }
    )
    if matching_config.appearance_weight is None:
        matching_config.appearance_weight = matching_config.weights.appearance
    else:
        matching_config.weights.appearance = float(matching_config.appearance_weight)

    ais_raw = dict(raw.get("ais") or {})
    if ais_raw.get("enabled") is None:
        ais_raw["enabled"] = False

    return StitchConfig(
        stitch_name=str(raw.get("stitch_name", "noop_v1")),
        tracklets=TrackletConfig(**(raw.get("tracklets") or {})),
        memory=MemoryConfig(**(raw.get("memory") or {})),
        appearance=AppearanceConfig(**(raw.get("appearance") or {})),
        motion=MotionConfig(**(raw.get("motion") or {})),
        bbox=BBoxConfig(**(raw.get("bbox") or {})),
        matching=matching_config,
        colreg=ColregConfig(**(raw.get("colreg") or {})),
        ais=AisConfig(**(raw.get("ais") or {})),
    )


def _stitch_mode(config: StitchConfig) -> str:
    if config.appearance.enabled and config.appearance.use_for_matching:
        return "motion_bbox_with_appearance_matching"
    if config.appearance.enabled:
        return "motion_bbox_with_appearance_diag"
    return "motion_bbox"


def _resolve_run_summary_path(pred_root: Path, run_summary_path: str | None) -> Path | None:
    if run_summary_path:
        path = Path(run_summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Run summary not found: {path}")
        return path

    default_path = pred_root.parent / "run_summary.json"
    return default_path if default_path.exists() else None


def _load_run_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids = sorted({int(row["id"]) for row in rows})
    return {str(track_id): track_id for track_id in ids}


def _unique_track_ids(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["id"]) for row in rows})


def _augment_run_summary(
    input_summary: dict[str, Any] | None,
    output_root: Path,
    mot_dir: Path,
    report_path: Path,
    config: StitchConfig,
    sequence_reports: dict[str, SequenceStitchReport],
    pred_root: Path,
    input_run_summary_path: Path | None,
) -> dict[str, Any]:
    summary = deepcopy(input_summary) if input_summary is not None else {}
    summary["output_root"] = str(output_root)
    summary["mot_dir"] = str(mot_dir)
    summary["total_rows_written"] = sum(report.num_rows_out for report in sequence_reports.values())

    if "total_frames" not in summary:
        summary["total_frames"] = sum(
            max((tracklet.frame_end for tracklet in report.tracklets), default=0)
            for report in sequence_reports.values()
        )

    sequences = dict(summary.get("sequences") or {})
    total_matches_applied = 0
    for sequence_name, report in sequence_reports.items():
        seq_summary = dict(sequences.get(sequence_name) or {})
        seq_summary["rows_written"] = report.num_rows_out
        seq_summary["mot_path"] = report.output_mot_path
        seq_summary["unique_track_ids"] = report.unique_track_ids_out
        seq_summary["num_unique_track_ids"] = len(report.unique_track_ids_out)
        seq_summary["stitch_tracklets"] = len(report.tracklets)
        seq_summary["stitch_matches_applied"] = report.matches_applied
        sequences[sequence_name] = seq_summary
        total_matches_applied += report.matches_applied
    summary["sequences"] = sequences

    appearance_summary = _aggregate_appearance_info(sequence_reports, config)
    summary["stitching"] = {
        "mode": _stitch_mode(config),
        "stitch_name": config.stitch_name,
        "input_pred_dir": str(pred_root),
        "input_run_summary_path": str(input_run_summary_path) if input_run_summary_path is not None else None,
        "report_path": str(report_path),
        "config": config.to_dict(),
        "total_tracklets": sum(len(report.tracklets) for report in sequence_reports.values()),
        "total_matches_applied": total_matches_applied,
        "appearance_backend": appearance_summary,
    }
    return summary


def _aggregate_appearance_info(
    sequence_reports: dict[str, SequenceStitchReport],
    config: StitchConfig,
) -> dict[str, Any]:
    if not sequence_reports:
        return {
            "name": config.appearance.backend,
            "enabled": config.appearance.enabled,
            "status": "not_run",
        }

    appearance_entries = [report.appearance for report in sequence_reports.values() if report.appearance]
    if not appearance_entries:
        return {
            "name": config.appearance.backend,
            "enabled": config.appearance.enabled,
            "status": "disabled" if not config.appearance.enabled else "not_run",
        }

    return {
        "name": appearance_entries[0].get("name", config.appearance.backend),
        "enabled": config.appearance.enabled,
        "status": "ready" if any(entry.get("status") == "ready" for entry in appearance_entries) else appearance_entries[0].get("status", "unknown"),
        "model_name": appearance_entries[0].get("model_name"),
        "device": appearance_entries[0].get("device"),
        "weights_source": appearance_entries[0].get("weights_source"),
        "checkpoint_path": appearance_entries[0].get("checkpoint_path"),
        "cache_dir": appearance_entries[0].get("cache_dir"),
        "embedding_dim": appearance_entries[0].get("embedding_dim"),
        "tracklets_total": sum(int(entry.get("tracklets_total", 0)) for entry in appearance_entries),
        "tracklets_with_head_embedding": sum(int(entry.get("tracklets_with_head_embedding", 0)) for entry in appearance_entries),
        "tracklets_with_tail_embedding": sum(int(entry.get("tracklets_with_tail_embedding", 0)) for entry in appearance_entries),
        "total_head_crops": sum(int(entry.get("total_head_crops", 0)) for entry in appearance_entries),
        "total_tail_crops": sum(int(entry.get("total_tail_crops", 0)) for entry in appearance_entries),
        "sample_frames": appearance_entries[0].get("sample_frames"),
        "crop_padding": appearance_entries[0].get("crop_padding"),
        "min_crop_size": appearance_entries[0].get("min_crop_size"),
        "use_for_matching": appearance_entries[0].get("use_for_matching"),
        "source_video": appearance_entries[0].get("source_video"),
    }


def stitch_tracks(
    pred_dir: str,
    output_dir: str | None = None,
    run_summary_path: str | None = None,
    config_path: str | None = None,
    stitch_name: str | None = None,
    ais_file: str | None = None,
    ais_fps: float | None = None,
) -> dict[str, Any]:
    pred_root = Path(pred_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    config = _load_stitch_config(config_path)
    if stitch_name:
        config.stitch_name = stitch_name
    if ais_file:
        config.ais.enabled = True
        config.ais.file_path = str(ais_file)
    if ais_fps is not None:
        config.ais.video_fps = float(ais_fps)

    output_root = Path(output_dir) if output_dir else pred_root.parent / "stitched" / config.stitch_name
    mot_dir = output_root / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    input_run_summary_path = _resolve_run_summary_path(pred_root, run_summary_path)
    input_run_summary = _load_run_summary(input_run_summary_path)

    if config.ais.enabled and (config.ais.video_fps is None or config.ais.video_fps <= 0):
        if input_run_summary is not None:
            inferred = input_run_summary.get("effective_fps")
            if inferred is not None and float(inferred) > 0:
                config.ais.video_fps = float(inferred)

    mot_files = sorted(pred_root.glob("*.txt"))
    if not mot_files:
        raise RuntimeError(f"No MOT txt files found in {pred_root}")

    warnings: list[str] = []
    if input_run_summary is None:
        warnings.append("Input run_summary.json was not found; stitched run summary was created from MOT outputs only.")

    sequence_reports: dict[str, SequenceStitchReport] = {}
    source_path = None
    if input_run_summary is not None:
        source_path = input_run_summary.get("source")

    global_max_frame = 0
    if config.ais.enabled:
        for mot_path in mot_files:
            rows = load_mot_rows(mot_path)
            if rows:
                global_max_frame = max(global_max_frame, max(int(r["frame"]) for r in rows))

    aligned_clip = None
    mmsi_assignments: dict[str, int | None] = {}
    if config.ais.enabled:
        from ais.stitching_bridge import load_aligned_clip_for_stitch, stamp_tracklet_mmsi

        aligned_clip = load_aligned_clip_for_stitch(
            config,
            max_frame=global_max_frame or 1,
            ais_file_override=ais_file,
        )
        if aligned_clip is None:
            warnings.append("AIS enabled but aligned clip could not be built; AIS scoring skipped.")

    for mot_path in mot_files:
        rows = load_mot_rows(mot_path)
        sequence_name = mot_path.stem
        output_mot_path = mot_dir / mot_path.name

        tracklets = build_tracklets(
            sequence_name=sequence_name,
            rows=rows,
            split_gap=config.tracklets.split_gap,
            min_length=config.tracklets.min_length,
            observation_samples=config.appearance.sample_frames if config.appearance.enabled else 0,
        )
        appearance_embeddings, sequence_appearance_info, appearance_warnings = build_tracklet_appearance(
            tracklets=tracklets,
            source_video_path=source_path,
            config=config.appearance,
        )
        warnings.extend(appearance_warnings)

        if aligned_clip is not None:
            from ais.stitching_bridge import stamp_tracklet_mmsi

            mmsi_assignments = stamp_tracklet_mmsi(
                tracklets,
                aligned_clip,
                max_distance_px=config.ais.max_distance_px,
            )

        matched_tracklets, decisions, identity_map, matches_applied = match_tracklets(
            tracklets,
            config,
            appearance_embeddings=appearance_embeddings,
            aligned_clip=aligned_clip,
            mmsi_assignments=mmsi_assignments,
        )

        remapped_rows = []
        rows_changed = False
        for row in rows:
            remapped_row = dict(row)
            original_id = int(remapped_row["id"])
            canonical_id = int(identity_map.get(str(original_id), original_id))
            if canonical_id != original_id:
                rows_changed = True
            remapped_row["id"] = canonical_id
            remapped_rows.append(remapped_row)

        if rows_changed:
            write_mot_rows(output_mot_path, remapped_rows)
        else:
            copy2(mot_path, output_mot_path)

        sequence_reports[sequence_name] = SequenceStitchReport(
            sequence_name=sequence_name,
            input_mot_path=str(mot_path),
            output_mot_path=str(output_mot_path),
            num_rows_in=len(rows),
            num_rows_out=len(remapped_rows),
            unique_track_ids_in=_unique_track_ids(rows),
            unique_track_ids_out=_unique_track_ids(remapped_rows),
            identity_map=identity_map,
            matches_applied=matches_applied,
            tracklets=matched_tracklets,
            decisions=decisions,
            appearance=sequence_appearance_info,
        )

    run_summary_payload = _augment_run_summary(
        input_summary=input_run_summary,
        output_root=output_root,
        mot_dir=mot_dir,
        report_path=output_root / "stitch_report.json",
        config=config,
        sequence_reports=sequence_reports,
        pred_root=pred_root,
        input_run_summary_path=input_run_summary_path,
    )

    run_summary_path_out = output_root / "run_summary.json"
    run_summary_path_out.write_text(json.dumps(run_summary_payload, indent=2), encoding="utf-8")

    aggregated_appearance_info = _aggregate_appearance_info(sequence_reports, config)

    report = StitchReport(
        version="stitch_tracks/v1",
        mode=_stitch_mode(config),
        pred_dir=str(pred_root),
        output_root=str(output_root),
        input_run_summary_path=str(input_run_summary_path) if input_run_summary_path is not None else None,
        output_run_summary_path=str(run_summary_path_out),
        source=source_path,
        appearance_backend=aggregated_appearance_info,
        config=config.to_dict(),
        total_sequences=len(sequence_reports),
        total_tracklets=sum(len(report.tracklets) for report in sequence_reports.values()),
        total_matches_applied=sum(report.matches_applied for report in sequence_reports.values()),
        sequences=sequence_reports,
        warnings=warnings,
    )

    report_path = output_root / "stitch_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "mode": _stitch_mode(config),
        "pred_dir": str(pred_root),
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "run_summary": str(run_summary_path_out),
        "stitch_report": str(report_path),
        "num_sequences": len(sequence_reports),
        "total_tracklets": report.total_tracklets,
        "matches_applied": report.total_matches_applied,
    }
