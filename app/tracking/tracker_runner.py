from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from evaluation.mot import write_mot_rows
from tracking.occlusion_geometry import (
    build_motion_corridor_prediction,
    estimate_motion_state,
    estimate_occlusion_relationship,
)
from tracking.prediction import build_occlusion_prediction
from tracking.visual_continuation import estimate_visual_continuation, update_visual_templates
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


def _xyxy_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _detections_from_boxes(boxes, source_name: str, confidence_threshold: float) -> list[dict]:
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return []
    xyxy_boxes = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xyxy_boxes)
    detections = []
    for xyxy, confidence in zip(xyxy_boxes, confidences):
        conf_value = float(confidence)
        if conf_value < float(confidence_threshold):
            continue
        detections.append(
            {
                "xyxy": [float(value) for value in xyxy],
                "confidence": conf_value,
                "source": source_name,
                "class_id": 0,
            }
        )
    return detections


def _union_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    ordered = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    fused: list[dict] = []
    used = [False] * len(ordered)
    for index, detection in enumerate(ordered):
        if used[index]:
            continue
        sources = {str(detection["source"])}
        duplicate_count = 0
        for other_index in range(index + 1, len(ordered)):
            if used[other_index]:
                continue
            if _xyxy_iou(detection["xyxy"], ordered[other_index]["xyxy"]) >= float(iou_threshold):
                used[other_index] = True
                duplicate_count += 1
                sources.add(str(ordered[other_index]["source"]))
        item = dict(detection)
        item["sources"] = sorted(sources)
        item["duplicate_count"] = duplicate_count
        fused.append(item)
    return fused


def _update_fusion_stats(
    stats: dict,
    primary_detections: list[dict],
    secondary_detections: list[dict],
    fused_detections: list[dict],
) -> None:
    stats["primary_detection_count"] += len(primary_detections)
    stats["secondary_detection_count"] += len(secondary_detections)
    stats["fused_detection_count"] += len(fused_detections)
    stats["duplicate_removed_count"] += max(0, len(primary_detections) + len(secondary_detections) - len(fused_detections))
    stats["secondary_only_detection_count"] += sum(1 for detection in fused_detections if detection.get("sources") == ["secondary"])
    stats["primary_only_detection_count"] += sum(1 for detection in fused_detections if detection.get("sources") == ["primary"])
    stats["mixed_detection_count"] += sum(1 for detection in fused_detections if set(detection.get("sources") or []) == {"primary", "secondary"})
    stats["_primary_confidence_sum"] += sum(float(detection["confidence"]) for detection in primary_detections)
    stats["_secondary_confidence_sum"] += sum(float(detection["confidence"]) for detection in secondary_detections)
    stats["_fused_confidence_sum"] += sum(float(detection["confidence"]) for detection in fused_detections)


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
    visual_continuation: bool = False,
    continuation_search_radius: int = 64,
    continuation_threshold: float = 0.45,
    continuation_max_gap: int = 60,
    paired_occlusion_prediction: bool = False,
    secondary_model: str | None = None,
    detector_fusion: bool = False,
    fusion_iou_threshold: float = 0.55,
    fusion_confidence_threshold: float = 0.25,
    fusion_mode: str = "union_nms",
):
    if live_reid and live_reid_in_loop:
        raise ValueError("--live-reid and --live-reid-in-loop are mutually exclusive")
    if colreg_scoring_experiment and not (live_reid or live_reid_in_loop):
        raise ValueError("--colreg-scoring-experiment requires --live-reid or --live-reid-in-loop")
    if detector_fusion and not secondary_model:
        raise ValueError("--detector-fusion requires --secondary-model")
    if detector_fusion and fusion_mode != "union_nms":
        raise ValueError("Only --fusion-mode union_nms is supported")

    model = YOLO(model_path)
    secondary_detector = YOLO(secondary_model) if detector_fusion and secondary_model else None
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
    occlusion_visual_templates: dict[str, dict[int, object]] = {}
    max_prediction_gap = max(30, int(confirmation_observations) * 6)
    fusion_stats = {
        "enabled": bool(detector_fusion),
        "primary_model": model_path,
        "secondary_model": secondary_model,
        "fusion_mode": fusion_mode,
        "fusion_iou_threshold": float(fusion_iou_threshold),
        "fusion_confidence_threshold": float(fusion_confidence_threshold),
        "applied_to_tracker": False,
        "true_pre_tracker_fusion_supported": False,
        "note": (
            "Experimental detector fusion diagnostics only. The current Ultralytics model.track loop "
            "does not safely accept externally fused detections, so MOT/ReID output remains primary-model tracking."
        ),
        "frames_evaluated": 0,
        "primary_detection_count": 0,
        "secondary_detection_count": 0,
        "fused_detection_count": 0,
        "duplicate_removed_count": 0,
        "secondary_only_detection_count": 0,
        "primary_only_detection_count": 0,
        "mixed_detection_count": 0,
        "_primary_confidence_sum": 0.0,
        "_secondary_confidence_sum": 0.0,
        "_fused_confidence_sum": 0.0,
    }

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

        def record_occlusion_predictions(current_rows: list[dict], frame_image=None) -> None:
            if not live_reid_in_loop:
                return
            histories = occlusion_histories.setdefault(seq_name, {})
            last_seen = occlusion_last_seen.setdefault(seq_name, {})
            reappearances = occlusion_reappearances.setdefault(seq_name, {})
            visual_templates = occlusion_visual_templates.setdefault(seq_name, {})
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

            if visual_continuation:
                update_visual_templates(frame_image, current_rows, visual_templates)

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
                if paired_occlusion_prediction:
                    hidden_state = estimate_motion_state(history)
                    best_pair: dict | None = None
                    best_occluder_state = None
                    best_occluder_row = None
                    if hidden_state is not None:
                        for other_row in current_rows:
                            other_id = int(other_row["id"])
                            if other_id == raw_id:
                                continue
                            other_history = histories.get(other_id, [])
                            other_state = estimate_motion_state(other_history)
                            if other_state is None:
                                continue
                            relationship = estimate_occlusion_relationship(
                                hidden_state,
                                other_state,
                                history[-1]["bbox"],
                                other_row["bbox"],
                            )
                            if best_pair is None or float(relationship["confidence"]) > float(best_pair["confidence"]):
                                best_pair = relationship
                                best_occluder_state = other_state
                                best_occluder_row = other_row

                        pair_confidence = float((best_pair or {}).get("confidence") or 0.0)
                        use_occluder = best_occluder_state is not None and pair_confidence >= 0.20
                        corridor = build_motion_corridor_prediction(
                            hidden_state=hidden_state,
                            gap_frames=gap_frames,
                            current_frame=frame_index,
                            occluder_state=best_occluder_state if use_occluder else None,
                            occlusion_pair_confidence=pair_confidence,
                        )
                        prediction.update(corridor)
                        prediction["source"] = corridor["prediction_model"]
                        prediction["predicted_center"] = {
                            "x": prediction["predicted_x"],
                            "y": prediction["predicted_y"],
                        }
                        prediction["occluder_track_id"] = int(best_occluder_row["id"]) if use_occluder and best_occluder_row else None
                        prediction["occluder_canonical_id"] = (
                            getattr(mapper, "source_to_canonical", {}).get(int(best_occluder_row["id"]))
                            if use_occluder and best_occluder_row and mapper is not None
                            else None
                        )
                        prediction["occlusion_reason"] = str((best_pair or {}).get("reason") or "no_confident_occluder")
                    else:
                        prediction["prediction_model"] = "constant_velocity"
                        prediction["occluder_track_id"] = None
                        prediction["occlusion_pair_confidence"] = 0.0
                        prediction["occlusion_reason"] = "insufficient_motion_history"
                if visual_continuation:
                    continuation = estimate_visual_continuation(
                        frame_image=frame_image,
                        template=visual_templates.get(raw_id),
                        predicted_x=float(prediction["predicted_x"]),
                        predicted_y=float(prediction["predicted_y"]),
                        gap_frames=gap_frames,
                        search_radius=int(continuation_search_radius),
                        threshold=float(continuation_threshold),
                        max_gap=int(continuation_max_gap),
                        motion_corridor=prediction if paired_occlusion_prediction else None,
                    )
                    prediction["visual_continuation"] = continuation
                    prediction["visual_continuation_active"] = bool(continuation.get("active"))
                    prediction["visual_continuation_confidence"] = continuation.get("confidence")
                    prediction["visual_continuation_status"] = continuation.get("status")
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
        if detector_fusion and secondary_detector is not None:
            frame_image = getattr(result, "orig_img", None)
            if frame_image is not None:
                primary_detections = _detections_from_boxes(boxes, "primary", fusion_confidence_threshold)
                predict_kwargs = {
                    "source": frame_image,
                    "device": device,
                    "conf": float(fusion_confidence_threshold),
                    "verbose": False,
                }
                if imgsz is not None:
                    predict_kwargs["imgsz"] = int(imgsz)
                secondary_results = secondary_detector.predict(**predict_kwargs)
                secondary_boxes = secondary_results[0].boxes if secondary_results else None
                secondary_detections = _detections_from_boxes(secondary_boxes, "secondary", fusion_confidence_threshold)
                fused_detections = _union_nms(
                    primary_detections + secondary_detections,
                    iou_threshold=float(fusion_iou_threshold),
                )
                fusion_stats["frames_evaluated"] += 1
                _update_fusion_stats(fusion_stats, primary_detections, secondary_detections, fused_detections)
        if boxes is None or boxes.xywh is None or len(boxes) == 0:
            update_in_loop([])
            record_occlusion_predictions([], getattr(result, "orig_img", None))
            continue

        xywh_boxes = boxes.xywh.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xywh_boxes)
        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xywh_boxes)
        detections_seen[seq_name] = detections_seen.get(seq_name, 0) + len(xywh_boxes)

        if boxes.id is None:
            detections_without_ids[seq_name] = detections_without_ids.get(seq_name, 0) + len(xywh_boxes)
            update_in_loop([])
            record_occlusion_predictions([], getattr(result, "orig_img", None))
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
        record_occlusion_predictions(
            _history_rows(frame_index, seq_track_ids, xywh_boxes, confidences, class_ids),
            getattr(result, "orig_img", None),
        )

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

    detector_fusion_report_path = None
    if detector_fusion:
        primary_count = int(fusion_stats["primary_detection_count"])
        secondary_count = int(fusion_stats["secondary_detection_count"])
        fused_count = int(fusion_stats["fused_detection_count"])
        detector_fusion_report = {
            key: value
            for key, value in fusion_stats.items()
            if not key.startswith("_")
        }
        detector_fusion_report.update(
            {
                "mean_primary_confidence": (
                    round(float(fusion_stats["_primary_confidence_sum"]) / primary_count, 6)
                    if primary_count
                    else None
                ),
                "mean_secondary_confidence": (
                    round(float(fusion_stats["_secondary_confidence_sum"]) / secondary_count, 6)
                    if secondary_count
                    else None
                ),
                "mean_fused_confidence": (
                    round(float(fusion_stats["_fused_confidence_sum"]) / fused_count, 6)
                    if fused_count
                    else None
                ),
                "experimental_detector_fusion": True,
            }
        )
        detector_fusion_report_path = output_root / "detector_fusion_report.json"
        detector_fusion_report_path.write_text(json.dumps(detector_fusion_report, indent=2), encoding="utf-8")
        run_summary["detector_fusion"] = {
            "enabled": True,
            "report_path": str(detector_fusion_report_path),
            "applied_to_tracker": False,
            "primary_detection_count": detector_fusion_report["primary_detection_count"],
            "secondary_detection_count": detector_fusion_report["secondary_detection_count"],
            "fused_detection_count": detector_fusion_report["fused_detection_count"],
            "duplicate_removed_count": detector_fusion_report["duplicate_removed_count"],
            "secondary_only_detection_count": detector_fusion_report["secondary_only_detection_count"],
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
        continuation_predictions = [
            prediction.get("visual_continuation") or {}
            for prediction in all_occlusion_predictions
            if "visual_continuation" in prediction
        ]
        paired_predictions = [
            prediction
            for prediction in all_occlusion_predictions
            if prediction.get("prediction_model") == "paired_motion_corridor"
        ]
        occlusion_pair_confidences = [
            float(prediction.get("occlusion_pair_confidence"))
            for prediction in all_occlusion_predictions
            if prediction.get("occlusion_pair_confidence") is not None
        ]
        continuation_confidences = [
            float(continuation.get("confidence"))
            for continuation in continuation_predictions
            if continuation.get("confidence") is not None
        ]
        continuation_best_part_scores = [
            float(continuation.get("best_part_score"))
            for continuation in continuation_predictions
            if continuation.get("best_part_score") is not None
        ]
        matched_part_distribution: dict[str, int] = {}
        for continuation in continuation_predictions:
            part_name = continuation.get("matched_part_name")
            if part_name:
                part_key = str(part_name)
                matched_part_distribution[part_key] = matched_part_distribution.get(part_key, 0) + 1
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
            "visual_continuation_enabled": bool(visual_continuation),
            "visual_continuation_prediction_count": len(continuation_predictions),
            "visual_continuation_attempted_count": sum(
                1 for continuation in continuation_predictions if bool(continuation.get("attempted"))
            ),
            "visual_continuation_active_count": sum(
                1 for continuation in continuation_predictions if bool(continuation.get("active"))
            ),
            "visual_continuation_lost_count": sum(
                1 for continuation in continuation_predictions if continuation.get("status") == "lost"
            ),
            "mean_visual_continuation_confidence": (
                round(sum(continuation_confidences) / len(continuation_confidences), 6)
                if continuation_confidences
                else None
            ),
            "continuation_part_matching_enabled": bool(visual_continuation),
            "continuation_prediction_count": len(continuation_predictions),
            "continuation_active_count": sum(
                1 for continuation in continuation_predictions if bool(continuation.get("active"))
            ),
            "mean_continuation_confidence": (
                round(sum(continuation_confidences) / len(continuation_confidences), 6)
                if continuation_confidences
                else None
            ),
            "mean_best_part_score": (
                round(sum(continuation_best_part_scores) / len(continuation_best_part_scores), 6)
                if continuation_best_part_scores
                else None
            ),
            "matched_part_distribution": dict(sorted(matched_part_distribution.items())),
            "visual_continuation_note": "diagnostic/visualization only; does not affect tracker or ReID decisions",
            "paired_occlusion_prediction_enabled": bool(paired_occlusion_prediction),
            "paired_prediction_count": len(paired_predictions),
            "occluder_pair_count": sum(1 for prediction in all_occlusion_predictions if prediction.get("occluder_track_id") is not None),
            "corridor_prediction_count": sum(1 for prediction in all_occlusion_predictions if prediction.get("motion_corridor_start")),
            "mean_occlusion_pair_confidence": (
                round(sum(occlusion_pair_confidences) / len(occlusion_pair_confidences), 6)
                if occlusion_pair_confidences
                else None
            ),
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
            "visual_continuation_enabled": bool(visual_continuation),
            "continuation_search_radius": int(continuation_search_radius),
            "continuation_threshold": float(continuation_threshold),
            "continuation_max_gap": int(continuation_max_gap),
            "paired_occlusion_prediction_enabled": bool(paired_occlusion_prediction),
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
    if detector_fusion_report_path is not None:
        result["detector_fusion"] = {
            "enabled": True,
            "report_path": str(detector_fusion_report_path),
            "applied_to_tracker": False,
            "primary_model": model_path,
            "secondary_model": secondary_model,
            "fusion_mode": fusion_mode,
            "fusion_iou_threshold": float(fusion_iou_threshold),
            "fusion_confidence_threshold": float(fusion_confidence_threshold),
            "primary_detection_count": detector_fusion_report["primary_detection_count"],
            "secondary_detection_count": detector_fusion_report["secondary_detection_count"],
            "fused_detection_count": detector_fusion_report["fused_detection_count"],
            "duplicate_removed_count": detector_fusion_report["duplicate_removed_count"],
            "secondary_only_detection_count": detector_fusion_report["secondary_only_detection_count"],
            "mean_primary_confidence": detector_fusion_report["mean_primary_confidence"],
            "mean_secondary_confidence": detector_fusion_report["mean_secondary_confidence"],
            "mean_fused_confidence": detector_fusion_report["mean_fused_confidence"],
        }

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
