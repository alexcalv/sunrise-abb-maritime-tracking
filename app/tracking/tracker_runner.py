from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from evaluation.mot import write_mot_rows
from tracking.prediction import build_occlusion_prediction
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sequence_name(source: str, result_path: str | None) -> str:
    source_path = Path(source)
    default_name = source_path.stem or source_path.name or "stream"
    if result_path is None:
        return default_name

    path = Path(result_path)
    if source_path.is_dir() and path.suffix.lower() in IMAGE_EXTS:
        return default_name
    return path.stem or default_name


def _write_mot_rows(
    mot_path: Path,
    frame_index: int,
    track_ids: list[int],
    xywh_boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> int:
    rows_written = 0
    with mot_path.open("a", encoding="utf-8") as handle:
        for track_id, box, confidence, class_id in zip(track_ids, xywh_boxes, confidences, class_ids):
            xc, yc, w, h = (float(value) for value in box)
            x = xc - (w / 2.0)
            y = yc - (h / 2.0)
            handle.write(
                f"{frame_index},{track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{confidence:.6f},{class_id},1\n"
            )
            rows_written += 1
    return rows_written


def _append_mot_dict_rows(mot_path: Path, rows: list[dict]) -> int:
    rows_written = 0
    mot_path.parent.mkdir(parents=True, exist_ok=True)
    with mot_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            x, y, w, h = (float(value) for value in row["bbox"])
            handle.write(
                f"{int(row['frame'])},{int(row['id'])},{x:.4f},{y:.4f},{w:.4f},{h:.4f},"
                f"{float(row['confidence']):.6f},{int(row['class_id'])},{float(row['visibility']):.6f}\n"
            )
            rows_written += 1
    return rows_written


def _tracker_detections(
    track_ids: list[int],
    xywh_boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> list[dict]:
    detections = []
    for track_id, box, confidence, class_id in zip(track_ids, xywh_boxes, confidences, class_ids):
        xc, yc, w, h = (float(value) for value in box)
        detections.append(
            {
                "raw_track_id": int(track_id),
                "x": xc - (w / 2.0),
                "y": yc - (h / 2.0),
                "w": w,
                "h": h,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "visibility": 1.0,
            }
        )
    return detections


def _history_rows(
    frame_idx: int,
    track_ids: list[int],
    xywh_boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> list[dict]:
    rows = []
    for track_id, box, confidence, class_id in zip(track_ids, xywh_boxes, confidences, class_ids):
        xc, yc, w, h = (float(value) for value in box)
        rows.append(
            {
                "frame": int(frame_idx),
                "id": int(track_id),
                "bbox": [xc - (w / 2.0), yc - (h / 2.0), w, h],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "visibility": 1.0,
            }
        )
    return rows


def _decision_context(decision: dict, *, include_colreg: bool, include_ais: bool) -> dict:
    context: dict[str, object] = {
        "new_raw_track_id": int(decision["new_source_track_id"]),
        "matched_raw_track_id": int(decision["candidate_source_track_id"]),
        "decision_frame": int(decision.get("decision_frame") or 0),
        "reason": str(decision.get("reason") or ""),
    }
    if include_colreg:
        context["colreg"] = {
            key: decision.get(key)
            for key in (
                "colreg_encounter",
                "colreg_confidence",
                "colreg_score",
                "colreg_reason",
                "colreg_relative_bearing",
                "colreg_heading_difference",
                "colreg_proximity_passed",
                "colreg_suspicious_reason",
            )
            if key in decision
        }
    if include_ais:
        context["ais"] = {
            key: decision.get(key)
            for key in (
                "ais_source_mmsi",
                "ais_candidate_mmsi",
                "ais_assignment_score_source",
                "ais_assignment_score_candidate",
                "ais_identity_score",
                "ais_note",
            )
            if key in decision
        }
    return context


def run_tracking(
    model_path: str,
    source: str,
    tracker_yaml: str,
    device: str,
    project: str,
    name: str,
    conf: float | None = None,
    imgsz: int | None = None,
    live_reid: bool = False,
    live_reid_in_loop: bool = False,
    live_reid_config: str | None = None,
    confirmation_observations: int = 10,
    colreg_diagnostics: bool = False,
    colreg_scoring_experiment: bool = False,
    ais_diagnostics: bool = False,
    ais_file: str | None = None,
    ais_video_start_time: str | None = None,
    ais_affine_matrix: str | None = None,
):
    if live_reid and live_reid_in_loop:
        raise ValueError("--live-reid and --live-reid-in-loop are mutually exclusive")
    if colreg_scoring_experiment and not (live_reid or live_reid_in_loop):
        raise ValueError("--colreg-scoring-experiment requires --live-reid or --live-reid-in-loop")

    model = YOLO(model_path)
    started_at = time.time()

    online_mapper_cls = None
    sort_live_rows = None
    aggregate_online_reports = None
    in_loop_config = None
    if live_reid_in_loop:
        from stitching.online_live_reid import (
            OnlineLiveReIDMapper,
            aggregate_online_reports as _aggregate_online_reports,
            sort_mot_rows,
        )
        from stitching.runner import DEFAULT_CONFIG_PATH, _load_stitch_config

        online_mapper_cls = OnlineLiveReIDMapper
        sort_live_rows = sort_mot_rows
        aggregate_online_reports = _aggregate_online_reports
        in_loop_config = _load_stitch_config(live_reid_config or str(DEFAULT_CONFIG_PATH))

    track_kwargs = {
        "source": source,
        "tracker": tracker_yaml,
        "device": device,
        "project": project,
        "name": name,
        "save": True,
        "save_txt": False,
        "persist": True,
        "stream": True,
        "verbose": False,
    }
    if conf is not None:
        track_kwargs["conf"] = float(conf)
    if imgsz is not None:
        track_kwargs["imgsz"] = int(imgsz)
    results = model.track(**track_kwargs)

    output_root: Path | None = None
    mot_dir: Path | None = None
    frame_counters: dict[str, int] = {}
    rows_written: dict[str, int] = {}
    detections_seen: dict[str, int] = {}
    detections_without_ids: dict[str, int] = {}
    track_ids_seen: dict[str, set[int]] = {}
    in_loop_root: Path | None = None
    in_loop_mot_dir: Path | None = None
    in_loop_mappers: dict[str, object] = {}
    in_loop_rows: dict[str, list[dict]] = {}
    in_loop_rows_written: dict[str, int] = {}
    occlusion_histories: dict[str, dict[int, list[dict]]] = {}
    occlusion_last_seen: dict[str, dict[int, int]] = {}
    occlusion_reappearances: dict[str, dict[int, dict[str, int]]] = {}
    occlusion_predictions: dict[str, list[dict]] = {}
    max_prediction_gap = max(30, int(confirmation_observations) * 6)

    for result in results:
        if output_root is None:
            save_dir = getattr(result, "save_dir", None)
            output_root = Path(save_dir) if save_dir else Path(project) / name
            mot_dir = output_root / "mot"
            mot_dir.mkdir(parents=True, exist_ok=True)
            if live_reid_in_loop:
                in_loop_root = output_root / "live_reid_in_loop"
                in_loop_mot_dir = in_loop_root / "mot"
                in_loop_mot_dir.mkdir(parents=True, exist_ok=True)

        seq_name = _sequence_name(source, getattr(result, "path", None))
        frame_index = frame_counters.get(seq_name, 0) + 1
        frame_counters[seq_name] = frame_index

        def mapper_for_sequence():
            return in_loop_mappers.get(seq_name)

        def record_occlusion_predictions(current_rows: list[dict]) -> None:
            if not live_reid_in_loop:
                return
            histories = occlusion_histories.setdefault(seq_name, {})
            last_seen = occlusion_last_seen.setdefault(seq_name, {})
            reappearances = occlusion_reappearances.setdefault(seq_name, {})
            seen_ids = {int(row["id"]) for row in current_rows}
            mapper = mapper_for_sequence()

            for row in current_rows:
                raw_id = int(row["id"])
                previous_seen = last_seen.get(raw_id)
                if previous_seen is not None and previous_seen < frame_index - 1:
                    reappearances[raw_id] = {
                        "raw_track_id": raw_id,
                        "reappearance_frame": int(frame_index),
                        "gap_frames": int(frame_index - previous_seen - 1),
                    }
                history = histories.setdefault(raw_id, [])
                history.append(dict(row))
                del history[:-20]
                last_seen[raw_id] = int(frame_index)

            for raw_id, history in sorted(histories.items()):
                if raw_id in seen_ids:
                    continue
                last_frame = last_seen.get(raw_id)
                if last_frame is None or last_frame >= frame_index:
                    continue
                gap_frames = int(frame_index - last_frame)
                if gap_frames > max_prediction_gap:
                    continue
                canonical_id = None
                if mapper is not None:
                    canonical_id = getattr(mapper, "source_to_canonical", {}).get(raw_id)
                prediction = build_occlusion_prediction(
                    track_history=history,
                    track_id=raw_id,
                    canonical_id=canonical_id,
                    frame=frame_index,
                    gap_frames=gap_frames,
                    occluded_after_frames=confirmation_observations,
                ).to_dict()
                prediction.update(
                    {
                        "raw_track_id": int(raw_id),
                        "current_frame": int(frame_index),
                        "predicted_center": {
                            "x": prediction["predicted_x"],
                            "y": prediction["predicted_y"],
                        },
                        "sequence_name": seq_name,
                        "reporting_only": True,
                        "remap_after_reappearance": False,
                        "recovered_by_remap": False,
                        "remap_target_raw_track_id": None,
                        "remap_decision_frame": None,
                        "same_raw_id_reappeared": False,
                    }
                )
                if colreg_diagnostics:
                    prediction["colreg_context"] = {
                        "diagnostics_enabled": True,
                        "note": "candidate-level COLREG context is added when a later remap decision is available",
                    }
                if ais_diagnostics:
                    prediction["ais_context"] = {
                        "diagnostics_enabled": True,
                        "ais_file": ais_file,
                        "note": "missing AIS remains neutral; candidate-level AIS context is added when available",
                    }
                occlusion_predictions.setdefault(seq_name, []).append(prediction)

        def update_in_loop(detections: list[dict]) -> None:
            if not live_reid_in_loop:
                return
            if online_mapper_cls is None or in_loop_config is None or in_loop_mot_dir is None:
                return
            mapper = in_loop_mappers.get(seq_name)
            if mapper is None:
                mapper = online_mapper_cls(
                    config=deepcopy(in_loop_config),
                    confirmation_observations=confirmation_observations,
                    source_video_path=source,
                    config_path=live_reid_config,
                    sequence_name=seq_name,
                    safe_mode=True,
                    colreg_diagnostics=colreg_diagnostics,
                    colreg_scoring_experiment=colreg_scoring_experiment,
                    ais_diagnostics=ais_diagnostics,
                    ais_file=ais_file,
                    ais_video_start_time=ais_video_start_time,
                    ais_affine_matrix=ais_affine_matrix,
                )
                in_loop_mappers[seq_name] = mapper
                in_loop_rows.setdefault(seq_name, [])
            live_rows = mapper.update(frame_index, detections)
            if live_rows:
                in_loop_rows.setdefault(seq_name, []).extend(live_rows)
                in_loop_rows_written[seq_name] = in_loop_rows_written.get(seq_name, 0) + _append_mot_dict_rows(
                    in_loop_mot_dir / f"{seq_name}.txt",
                    live_rows,
                )

        boxes = result.boxes
        if boxes is None or boxes.xywh is None or len(boxes) == 0:
            update_in_loop([])
            record_occlusion_predictions([])
            continue

        xywh_boxes = boxes.xywh.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xywh_boxes)
        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xywh_boxes)
        detections_seen[seq_name] = detections_seen.get(seq_name, 0) + len(xywh_boxes)

        if boxes.id is None:
            detections_without_ids[seq_name] = detections_without_ids.get(seq_name, 0) + len(xywh_boxes)
            update_in_loop([])
            record_occlusion_predictions([])
            continue

        seq_track_ids = boxes.id.int().cpu().tolist()
        track_ids_seen.setdefault(seq_name, set()).update(seq_track_ids)

        mot_path = mot_dir / f"{seq_name}.txt"
        rows_written[seq_name] = rows_written.get(seq_name, 0) + _write_mot_rows(
            mot_path=mot_path,
            frame_index=frame_index,
            track_ids=seq_track_ids,
            xywh_boxes=xywh_boxes,
            confidences=confidences,
            class_ids=class_ids,
        )
        update_in_loop(_tracker_detections(seq_track_ids, xywh_boxes, confidences, class_ids))
        record_occlusion_predictions(_history_rows(frame_index, seq_track_ids, xywh_boxes, confidences, class_ids))

    finished_at = time.time()
    wall_time_seconds = max(finished_at - started_at, 0.0)
    total_frames = sum(frame_counters.values())

    if output_root is None:
        output_root = Path(project) / name
        mot_dir = output_root / "mot"
        mot_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "model_path": model_path,
        "source": source,
        "tracker_yaml": tracker_yaml,
        "device": device,
        "conf": conf,
        "imgsz": imgsz,
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "total_frames": total_frames,
        "total_rows_written": sum(rows_written.values()),
        "wall_time_seconds": round(wall_time_seconds, 6),
        "effective_fps": round((total_frames / wall_time_seconds), 6) if wall_time_seconds > 0 else None,
        "sequences": {},
    }

    sequence_names = set(frame_counters) | set(rows_written) | set(detections_seen) | set(detections_without_ids)
    for seq_name in sorted(sequence_names):
        run_summary["sequences"][seq_name] = {
            "frames_processed": frame_counters.get(seq_name, 0),
            "detections_seen": detections_seen.get(seq_name, 0),
            "detections_without_ids": detections_without_ids.get(seq_name, 0),
            "rows_written": rows_written.get(seq_name, 0),
            "unique_track_ids": sorted(track_ids_seen.get(seq_name, set())),
            "num_unique_track_ids": len(track_ids_seen.get(seq_name, set())),
            "mot_path": str(mot_dir / f"{seq_name}.txt"),
        }

    run_summary_path = output_root / "run_summary.json"

    live_reid_in_loop_result = None
    if live_reid_in_loop:
        if in_loop_root is None:
            in_loop_root = output_root / "live_reid_in_loop"
        if in_loop_mot_dir is None:
            in_loop_mot_dir = in_loop_root / "mot"
            in_loop_mot_dir.mkdir(parents=True, exist_ok=True)

        for seq_name, mapper in in_loop_mappers.items():
            flushed_rows = mapper.flush()
            if flushed_rows:
                in_loop_rows.setdefault(seq_name, []).extend(flushed_rows)
                in_loop_rows_written[seq_name] = in_loop_rows_written.get(seq_name, 0) + _append_mot_dict_rows(
                    in_loop_mot_dir / f"{seq_name}.txt",
                    flushed_rows,
                )

        if sort_live_rows is not None:
            for seq_name, rows in in_loop_rows.items():
                sorted_rows = sort_live_rows(rows)
                in_loop_rows[seq_name] = sorted_rows
                write_mot_rows(in_loop_mot_dir / f"{seq_name}.txt", sorted_rows)
                in_loop_rows_written[seq_name] = len(sorted_rows)

        sequence_reports = {}
        mapping_sequences = {}
        for seq_name, mapper in sorted(in_loop_mappers.items()):
            sequence_report = mapper.report()
            sequence_report["source_mot_path"] = str(mot_dir / f"{seq_name}.txt")
            sequence_report["source_video_path"] = source
            sequence_report["input_mot_path"] = str(mot_dir / f"{seq_name}.txt")
            sequence_report["output_mot_path"] = str(in_loop_mot_dir / f"{seq_name}.txt")
            sequence_reports[seq_name] = sequence_report
            mapping_sequences[seq_name] = mapper.mapping_summary()

        occlusion_report_path = in_loop_root / "occlusion_predictions.json"
        for seq_name, predictions in occlusion_predictions.items():
            report = sequence_reports.get(seq_name) or {}
            identity_map = {int(source_id): int(canonical_id) for source_id, canonical_id in (report.get("identity_map") or {}).items()}
            accepted_by_lost = {
                int(decision["candidate_source_track_id"]): decision
                for decision in report.get("decisions", [])
                if bool(decision.get("applied")) and decision.get("candidate_source_track_id") is not None
            }
            reappearances = occlusion_reappearances.get(seq_name, {})
            for prediction in predictions:
                raw_id = int(prediction["track_id"])
                prediction["canonical_id"] = prediction.get("canonical_id") or identity_map.get(raw_id)
                reappearance = reappearances.get(raw_id)
                if reappearance and int(reappearance["reappearance_frame"]) >= int(prediction["frame"]):
                    prediction["same_raw_id_reappeared"] = True
                    prediction["reappearance_frame"] = int(reappearance["reappearance_frame"])
                decision = accepted_by_lost.get(raw_id)
                if decision is not None:
                    prediction["remap_after_reappearance"] = True
                    prediction["recovered_by_remap"] = True
                    prediction["remap_target_raw_track_id"] = int(decision["new_source_track_id"])
                    prediction["remap_decision_frame"] = int(decision.get("decision_frame") or 0)
                    prediction["remap_context"] = _decision_context(
                        decision,
                        include_colreg=colreg_diagnostics,
                        include_ais=ais_diagnostics,
                    )
                    if colreg_diagnostics and "colreg_context" in prediction:
                        prediction["colreg_context"].update(prediction["remap_context"].get("colreg") or {})
                    if ais_diagnostics and "ais_context" in prediction:
                        prediction["ais_context"].update(prediction["remap_context"].get("ais") or {})

        all_occlusion_predictions = [
            prediction
            for predictions in occlusion_predictions.values()
            for prediction in predictions
        ]
        prediction_source_counts: dict[str, int] = {}
        for prediction in all_occlusion_predictions:
            source_name = str(prediction.get("source") or "unknown")
            prediction_source_counts[source_name] = prediction_source_counts.get(source_name, 0) + 1
        uncertainty_values = [
            float(prediction["uncertainty_radius"])
            for prediction in all_occlusion_predictions
            if prediction.get("uncertainty_radius") is not None
        ]
        occlusion_summary = {
            "mode": "in_loop_bounded_latency",
            "reporting_only": True,
            "prediction_source": "constant_velocity",
            "sequence_count": len(set(occlusion_predictions) | set(sequence_reports)),
            "prediction_count": len(all_occlusion_predictions),
            "unique_raw_tracks": sorted({int(prediction["raw_track_id"]) for prediction in all_occlusion_predictions}),
            "unique_raw_track_count": len({int(prediction["raw_track_id"]) for prediction in all_occlusion_predictions}),
            "recovered_prediction_count": sum(1 for prediction in all_occlusion_predictions if bool(prediction.get("recovered_by_remap"))),
            "remap_after_reappearance_count": sum(1 for prediction in all_occlusion_predictions if bool(prediction.get("remap_after_reappearance"))),
            "max_gap_frames": max((int(prediction["gap_frames"]) for prediction in all_occlusion_predictions), default=0),
            "mean_uncertainty_radius": (
                round(sum(uncertainty_values) / len(uncertainty_values), 6)
                if uncertainty_values
                else None
            ),
            "prediction_source_counts": dict(sorted(prediction_source_counts.items())),
            "colreg_context_enabled": bool(colreg_diagnostics),
            "ais_context_enabled": bool(ais_diagnostics),
            "colreg_diagnostics": bool(colreg_diagnostics),
            "ais_diagnostics": bool(ais_diagnostics),
        }
        occlusion_report_path.write_text(
            json.dumps(
                {
                    "summary": occlusion_summary,
                    "sequences": {
                        seq_name: {
                            "prediction_count": len(predictions),
                            "predictions": predictions,
                        }
                        for seq_name, predictions in sorted(occlusion_predictions.items())
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        report_path = in_loop_root / "live_reid_report.json"
        mapping_summary_path = in_loop_root / "mapping_summary.json"
        in_loop_run_summary_path = in_loop_root / "run_summary.json"
        report_payload = aggregate_online_reports(
            sequence_reports=sequence_reports,
            output_root=in_loop_root,
            mot_dir=in_loop_mot_dir,
            config_path=live_reid_config,
            source_video_path=source,
            safe_mode=True,
            confirmation_observations=confirmation_observations,
        )
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        mapping_summary_path.write_text(
            json.dumps(
                {
                    "mode": "in_loop_bounded_latency",
                    "source_video_path": source,
                    "sequences": mapping_sequences,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        in_loop_run_summary = {
            "mode": "in_loop_bounded_latency",
            "source": source,
            "raw_mot_dir": str(mot_dir),
            "live_reid_in_loop_mot_dir": str(in_loop_mot_dir),
            "live_reid_report": str(report_path),
            "mapping_summary": str(mapping_summary_path),
            "occlusion_predictions": str(occlusion_report_path),
            "confirmation_observations": int(confirmation_observations),
            "config_path": live_reid_config,
            "summary": report_payload["summary"],
            "sequences": {
                seq_name: {
                    "raw_mot_path": str(mot_dir / f"{seq_name}.txt"),
                    "live_reid_mot_path": str(in_loop_mot_dir / f"{seq_name}.txt"),
                    "rows_written": in_loop_rows_written.get(seq_name, 0),
                    "accepted_remap_count": int(sequence_report["accepted_remap_count"]),
                    "accepted_remaps": list(sequence_report["accepted_remaps"]),
                    "unique_ids_before": len(sequence_report["unique_track_ids_in"]),
                    "unique_ids_after": len(sequence_report["unique_track_ids_out"]),
                }
                for seq_name, sequence_report in sequence_reports.items()
            },
        }
        in_loop_run_summary_path.write_text(json.dumps(in_loop_run_summary, indent=2), encoding="utf-8")
        live_reid_in_loop_result = {
            "enabled": True,
            "experimental": True,
            "mode": "in_loop_bounded_latency",
            "colreg_scoring_experiment_enabled": bool(colreg_scoring_experiment),
            "mot_dir": str(in_loop_mot_dir),
            "run_summary": str(in_loop_run_summary_path),
            "live_reid_report": str(report_path),
            "mapping_summary": str(mapping_summary_path),
            "occlusion_predictions": str(occlusion_report_path),
            "accepted_remap_count": report_payload["summary"]["accepted_remap_count"],
            "accepted_remaps": report_payload["summary"]["accepted_remaps"],
            "rejected_candidate_count": report_payload["summary"]["rejected_candidate_count"],
            "fresh_id_count": report_payload["summary"]["fresh_id_count"],
            "chain_control_checked_count": report_payload["summary"]["chain_control_checked_count"],
            "chain_control_rejected_count": report_payload["summary"]["chain_control_rejected_count"],
            "short_gap_gate_checked_count": report_payload["summary"]["short_gap_gate_checked_count"],
            "short_gap_gate_rejected_count": report_payload["summary"]["short_gap_gate_rejected_count"],
            "appearance_supported_remap_count": report_payload["summary"]["appearance_supported_remap_count"],
            "appearance_rejection_count": report_payload["summary"]["appearance_rejection_count"],
            "decision_latency_mean": report_payload["summary"]["decision_latency_mean"],
            "decision_latency_max": report_payload["summary"]["decision_latency_max"],
            "confirmation_observations": int(confirmation_observations),
        }
        run_summary["live_reid_in_loop"] = live_reid_in_loop_result

    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    result = {
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "run_summary": str(run_summary_path),
        "frames_processed": total_frames,
        "rows_written": sum(rows_written.values()),
    }
    if live_reid_in_loop_result is not None:
        result["live_reid_in_loop"] = live_reid_in_loop_result

    if live_reid:
        from stitching.live_reid import live_reid_tracks

        live_result = live_reid_tracks(
            pred_dir=str(mot_dir),
            output_dir=str(output_root / "live_reid"),
            run_summary_path=str(run_summary_path),
            config_path=live_reid_config,
            live_name="tracker_live_reid",
            confirmation_observations=confirmation_observations,
            safe_mode=True,
            colreg_diagnostics=colreg_diagnostics,
            colreg_scoring_experiment=colreg_scoring_experiment,
            ais_diagnostics=ais_diagnostics,
            ais_file=ais_file,
            ais_video_start_time=ais_video_start_time,
            ais_affine_matrix=ais_affine_matrix,
        )
        run_summary["live_reid"] = {
            "enabled": True,
            "experimental": True,
            "mode": live_result["mode"],
            "colreg_scoring_experiment_enabled": bool(colreg_scoring_experiment),
            "mot_dir": live_result["mot_dir"],
            "run_summary": live_result["run_summary"],
            "live_reid_report": live_result["live_reid_report"],
            "accepted_remap_count": live_result["accepted_remap_count"],
            "accepted_remaps": live_result.get("accepted_remaps", []),
            "rejected_candidate_count": live_result["rejected_candidate_count"],
            "fresh_id_count": live_result.get("fresh_id_count", live_result["fresh_canonical_ids_assigned"]),
            "chain_control_checked_count": live_result["chain_control_checked_count"],
            "chain_control_rejected_count": live_result["chain_control_rejected_count"],
            "short_gap_gate_checked_count": live_result["short_gap_gate_checked_count"],
            "short_gap_gate_rejected_count": live_result["short_gap_gate_rejected_count"],
            "appearance_supported_remap_count": live_result["appearance_supported_remap_count"],
            "appearance_rejection_count": live_result.get("appearance_rejection_count", live_result["appearance_rejected_count"]),
            "confirmation_observations": live_result["confirmation_observations"],
        }
        run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        result["live_reid"] = live_result

    return result
