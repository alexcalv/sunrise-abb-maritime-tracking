from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.mot import load_mot_rows, write_mot_rows
from stitching.appearance import TrackletAppearanceEmbeddings, build_tracklet_appearance, cosine_similarity
from stitching.runner import (
    _load_run_summary,
    _load_stitch_config,
    _resolve_run_summary_path,
)
from stitching.schemas import StitchConfig
from stitching.tracklets import build_tracklets

REPLAY_MODE = "replay_stitch_motion_bbox_v1"
REPLAY_MODE_WITH_APPEARANCE = "replay_stitch_motion_bbox_with_appearance_fallback_v1"
REPLAY_VERSION = "replay_stitch/v3"


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _center_from_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)]


def _distance(first: list[float], second: list[float]) -> float:
    return math.dist([float(first[0]), float(first[1])], [float(second[0]), float(second[1])])


def _area_ratio(first_bbox: list[float], second_bbox: list[float]) -> float:
    first_area = max(float(first_bbox[2]) * float(first_bbox[3]), 1e-9)
    second_area = max(float(second_bbox[2]) * float(second_bbox[3]), 1e-9)
    return max(first_area, second_area) / min(first_area, second_area)


def _aspect_ratio_delta(first_bbox: list[float], second_bbox: list[float]) -> float:
    first_height = max(float(first_bbox[3]), 1e-9)
    second_height = max(float(second_bbox[3]), 1e-9)
    first_ratio = float(first_bbox[2]) / first_height
    second_ratio = float(second_bbox[2]) / second_height
    return abs(first_ratio - second_ratio)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _threshold_score(value: float, max_value: float | None) -> float:
    if max_value is None or max_value <= 0:
        return 1.0
    return _clamp01(1.0 - (value / max_value))


def _ratio_score(value: float, max_value: float) -> float:
    if max_value <= 1.0:
        return 1.0 if value <= max_value else 0.0
    if value <= 1.0:
        return 1.0
    return _clamp01(1.0 - ((value - 1.0) / (max_value - 1.0)))


def _delta_score(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 1.0 if value <= 0 else 0.0
    return _clamp01(1.0 - (value / max_value))


def _gap_bucket(gap_frames: int, config: StitchConfig) -> str:
    short_gap_threshold = config.matching.short_gap_frames_threshold
    long_gap_threshold = config.matching.long_gap_frames_threshold
    if short_gap_threshold is not None and gap_frames <= int(short_gap_threshold):
        return "short"
    if long_gap_threshold is not None and gap_frames > int(long_gap_threshold):
        return "long"
    return "medium"


def _winner_margin_threshold(config: StitchConfig, gap_frames: int) -> float:
    threshold = float(config.matching.min_score_margin)
    gap_bucket = _gap_bucket(gap_frames, config)

    configured = config.matching.winner_margin_threshold
    if gap_bucket == "short" and config.matching.winner_margin_threshold_short is not None:
        configured = config.matching.winner_margin_threshold_short
    elif gap_bucket == "long" and config.matching.winner_margin_threshold_long is not None:
        configured = config.matching.winner_margin_threshold_long

    if configured is None:
        return threshold
    return max(threshold, float(configured))


def _appearance_similarity_threshold(config: StitchConfig, gap_frames: int) -> float | None:
    threshold = config.matching.min_appearance_similarity
    gap_bucket = _gap_bucket(gap_frames, config)

    if gap_bucket == "short" and config.matching.short_gap_min_appearance_similarity is not None:
        threshold = config.matching.short_gap_min_appearance_similarity
    elif gap_bucket == "long" and config.matching.long_gap_min_appearance_similarity is not None:
        threshold = config.matching.long_gap_min_appearance_similarity

    if threshold is None:
        return None
    return float(threshold)


def _appearance_margin_threshold(config: StitchConfig) -> float:
    return max(float(config.matching.min_score_margin), 0.05)


def _replay_mode(config: StitchConfig) -> str:
    if config.appearance.enabled and config.appearance.use_for_matching:
        return REPLAY_MODE_WITH_APPEARANCE
    return REPLAY_MODE


def _unique_track_ids(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["id"]) for row in rows})


def _sequence_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["frame"]), []).append(row)
    return grouped


def _initial_next_canonical_id(rows: list[dict[str, Any]]) -> int:
    source_ids = [int(row["id"]) for row in rows if int(row["id"]) >= 0]
    return (max(source_ids) + 1) if source_ids else 1


def _compact_offline_sequence_summary(sequence_report: dict[str, Any]) -> dict[str, Any]:
    unique_track_ids_out = list(sequence_report.get("unique_track_ids_out") or [])
    accepted_remaps = list(sequence_report.get("accepted_remaps") or [])
    return {
        "matches_applied": int(sequence_report.get("matches_applied") or 0),
        "accepted_remap_count": len(accepted_remaps) if accepted_remaps else int(sequence_report.get("matches_applied") or 0),
        "num_unique_track_ids_out": len(unique_track_ids_out),
        "unique_track_ids_out": unique_track_ids_out,
        "output_mot_path": sequence_report.get("output_mot_path"),
    }


def _load_offline_comparison_runs(stitched_root: Path) -> tuple[dict[str, Any], list[str]]:
    if not stitched_root.exists():
        return {}, []

    comparison_runs: dict[str, Any] = {}
    warnings: list[str] = []

    for run_root in sorted(path for path in stitched_root.iterdir() if path.is_dir()):
        report_path = run_root / "stitch_report.json"
        if not report_path.exists():
            continue

        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Failed to read offline stitch report {report_path}: {type(exc).__name__}: {exc}")
            continue

        sequence_summaries = {
            sequence_name: _compact_offline_sequence_summary(sequence_report)
            for sequence_name, sequence_report in (report_payload.get("sequences") or {}).items()
        }
        comparison_runs[run_root.name] = {
            "run_name": run_root.name,
            "mode": report_payload.get("mode"),
            "stitch_name": (report_payload.get("config") or {}).get("stitch_name") or run_root.name,
            "report_path": str(report_path),
            "total_matches_applied": int(report_payload.get("total_matches_applied") or 0),
            "num_sequences": len(sequence_summaries),
            "sequences": sequence_summaries,
        }

    return comparison_runs, warnings


def _build_sequence_summary(sequence_report: dict[str, Any], offline_comparisons: dict[str, Any]) -> dict[str, Any]:
    unique_track_ids_in = list(sequence_report["unique_track_ids_in"])
    unique_track_ids_out = list(sequence_report["unique_track_ids_out"])
    accepted_remap_count = int(sequence_report["accepted_remap_count"])
    rejected_candidate_count = int(sequence_report["rejected_candidate_count"])
    fresh_canonical_ids_assigned = int(sequence_report["fresh_canonical_ids_assigned"])
    candidate_evaluation_count = int(sequence_report["candidate_evaluation_count"])
    below_score_threshold_count = int(sequence_report["below_score_threshold_count"])
    margin_rejected_count = int(sequence_report["margin_rejected_count"])
    appearance_supported_remap_count = int(sequence_report["appearance_supported_remap_count"])
    appearance_rejected_count = int(sequence_report["appearance_rejected_count"])

    return {
        "num_rows_in": int(sequence_report["num_rows_in"]),
        "num_rows_out": int(sequence_report["num_rows_out"]),
        "num_unique_track_ids_in": len(unique_track_ids_in),
        "num_unique_track_ids_out": len(unique_track_ids_out),
        "accepted_remap_count": accepted_remap_count,
        "rejected_candidate_count": rejected_candidate_count,
        "candidate_evaluation_count": candidate_evaluation_count,
        "fresh_canonical_ids_assigned": fresh_canonical_ids_assigned,
        "below_score_threshold_count": below_score_threshold_count,
        "margin_rejected_count": margin_rejected_count,
        "appearance_supported_remap_count": appearance_supported_remap_count,
        "appearance_rejected_count": appearance_rejected_count,
        "offline_comparison_runs": sorted(offline_comparisons),
    }


def _build_replay_appearance_embeddings(
    sequence_name: str,
    rows: list[dict[str, Any]],
    source_video_path: str | None,
    config: StitchConfig,
) -> tuple[dict[int, TrackletAppearanceEmbeddings], dict[str, Any] | None, list[str]]:
    if not (config.appearance.enabled and config.appearance.use_for_matching):
        return {}, None, []

    if not rows:
        return {}, None, []

    max_frame = max(int(row["frame"]) for row in rows)
    source_tracklets = build_tracklets(
        sequence_name=sequence_name,
        rows=rows,
        split_gap=max(max_frame, int(config.memory.max_frame_gap), 1),
        min_length=1,
        observation_samples=int(config.appearance.sample_frames),
    )
    appearance_embeddings, appearance_info, warnings = build_tracklet_appearance(
        tracklets=source_tracklets,
        source_video_path=source_video_path,
        config=config.appearance,
    )
    source_track_embeddings = {
        int(tracklet.source_track_id): embedding
        for tracklet in source_tracklets
        if (embedding := appearance_embeddings.get(tracklet.tracklet_id)) is not None
    }
    return source_track_embeddings, appearance_info, warnings


def _annotate_appearance_support(
    decision: dict[str, Any],
    appearance_embeddings: dict[int, TrackletAppearanceEmbeddings] | None,
    config: StitchConfig,
) -> None:
    if not appearance_embeddings:
        decision["appearance_similarity"] = None
        decision["appearance_threshold"] = None
        decision["appearance_supported"] = False
        decision["appearance_status"] = "not_available"
        return

    source_embedding = appearance_embeddings.get(int(decision["new_source_track_id"]))
    target_embedding = appearance_embeddings.get(int(decision["candidate_source_track_id"]))
    head_embedding = None if source_embedding is None else source_embedding.head_embedding
    tail_embedding = None if target_embedding is None else target_embedding.tail_embedding
    similarity = cosine_similarity(tail_embedding, head_embedding)
    threshold = _appearance_similarity_threshold(config=config, gap_frames=int(decision["gap_frames"]))

    decision["appearance_similarity"] = _round(similarity)
    decision["appearance_threshold"] = _round(threshold)
    decision["appearance_supported"] = (
        similarity is not None and (threshold is None or float(similarity) >= float(threshold))
    )
    if head_embedding is None and tail_embedding is None:
        decision["appearance_status"] = "missing_source_and_target_embeddings"
    elif tail_embedding is None:
        decision["appearance_status"] = "missing_target_tail_embedding"
    elif head_embedding is None:
        decision["appearance_status"] = "missing_source_head_embedding"
    else:
        decision["appearance_status"] = "ready"


def _allocate_canonical_id(
    source_track_id: int,
    used_canonical_ids: set[int],
    next_canonical_id: int,
) -> tuple[int, int]:
    if source_track_id > 0 and source_track_id not in used_canonical_ids:
        used_canonical_ids.add(source_track_id)
        return source_track_id, next_canonical_id

    candidate = max(int(next_canonical_id), 1)
    while candidate in used_canonical_ids:
        candidate += 1
    used_canonical_ids.add(candidate)
    return candidate, candidate + 1


def _make_track_state(
    source_track_id: int,
    canonical_track_id: int,
    row: dict[str, Any],
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = int(row["frame"])
    bbox = [float(value) for value in row["bbox"]]
    center = _center_from_bbox(bbox)
    area = float(bbox[2]) * float(bbox[3])
    aspect_ratio = (float(bbox[2]) / float(bbox[3])) if float(bbox[3]) > 0 else 0.0

    if previous_state is None:
        velocity = [0.0, 0.0]
        mean_area = area
        mean_aspect_ratio = aspect_ratio
        seen_frames = 1
    else:
        previous_frame = int(previous_state["last_frame"])
        previous_center = list(previous_state["last_center"])
        frame_delta = max(frame - previous_frame, 1)
        velocity = [
            round((center[0] - previous_center[0]) / frame_delta, 6),
            round((center[1] - previous_center[1]) / frame_delta, 6),
        ]
        seen_frames = int(previous_state["seen_frames"]) + 1
        mean_area = (
            (float(previous_state["mean_area"]) * int(previous_state["seen_frames"])) + area
        ) / seen_frames
        mean_aspect_ratio = (
            (float(previous_state["mean_aspect_ratio"]) * int(previous_state["seen_frames"])) + aspect_ratio
        ) / seen_frames

    return {
        "source_track_id": int(source_track_id),
        "canonical_track_id": int(canonical_track_id),
        "last_frame": frame,
        "last_bbox": [round(value, 6) for value in bbox],
        "last_center": [round(value, 6) for value in center],
        "velocity": [round(float(value), 6) for value in velocity],
        "mean_area": round(mean_area, 6),
        "mean_aspect_ratio": round(mean_aspect_ratio, 6),
        "class_id": int(row["class_id"]),
        "seen_frames": seen_frames,
    }


def _evaluate_candidate(
    lost_state: dict[str, Any],
    new_row: dict[str, Any],
    config: StitchConfig,
) -> dict[str, Any]:
    frame = int(new_row["frame"])
    new_bbox = [float(value) for value in new_row["bbox"]]
    new_center = _center_from_bbox(new_bbox)
    gap_frames = max(0, frame - int(lost_state["last_frame"]) - 1)
    elapsed_frames = max(1, frame - int(lost_state["last_frame"]))
    max_gap = min(int(config.memory.max_frame_gap), int(config.motion.max_frame_gap))

    predicted_center = [
        float(lost_state["last_center"][0]) + (float(lost_state["velocity"][0]) * elapsed_frames),
        float(lost_state["last_center"][1]) + (float(lost_state["velocity"][1]) * elapsed_frames),
    ]
    direct_center_distance = _distance(list(lost_state["last_center"]), new_center)
    motion_distance = _distance(predicted_center, new_center)
    expected_radius = (
        float(config.motion.base_region_radius) + (float(config.motion.growth_per_frame) * gap_frames)
        if config.motion.enabled
        else float("inf")
    )
    area_ratio = _area_ratio(list(lost_state["last_bbox"]), new_bbox)
    aspect_ratio_delta = _aspect_ratio_delta(list(lost_state["last_bbox"]), new_bbox)

    class_match = int(lost_state["class_id"]) == int(new_row["class_id"])
    center_gate = motion_distance <= expected_radius
    if config.motion.max_center_distance is not None:
        center_gate = center_gate or direct_center_distance <= float(config.motion.max_center_distance)

    gating = {
        "temporal_gap": gap_frames <= max_gap,
        "class_consistency": class_match,
        "center_distance": center_gate,
        "area_ratio": area_ratio <= float(config.bbox.max_area_ratio),
        "aspect_ratio": aspect_ratio_delta <= float(config.bbox.max_aspect_ratio_delta),
    }
    gating["passes_all"] = all(gating.values())

    center_threshold = float(config.motion.max_center_distance) if config.motion.max_center_distance is not None else None
    if config.motion.enabled:
        center_threshold = max(center_threshold or 0.0, expected_radius)
    score = round(
        (
            _threshold_score(gap_frames, float(max_gap))
            + _threshold_score(min(direct_center_distance, motion_distance), center_threshold)
            + _ratio_score(area_ratio, float(config.bbox.max_area_ratio))
            + _delta_score(aspect_ratio_delta, float(config.bbox.max_aspect_ratio_delta))
        )
        / 4.0,
        6,
    )
    failed_gates = [name for name, passed in gating.items() if name != "passes_all" and not passed]
    reason = "eligible" if not failed_gates else f"gate_failed:{','.join(failed_gates)}"

    return {
        "frame": frame,
        "new_source_track_id": int(new_row["id"]),
        "candidate_source_track_id": int(lost_state["source_track_id"]),
        "candidate_canonical_track_id": int(lost_state["canonical_track_id"]),
        "gap_frames": gap_frames,
        "elapsed_frames": elapsed_frames,
        "direct_center_distance": _round(direct_center_distance),
        "predicted_center": [_round(value) for value in predicted_center],
        "motion_distance": _round(motion_distance),
        "expected_radius": _round(expected_radius if math.isfinite(expected_radius) else None),
        "area_ratio": _round(area_ratio),
        "aspect_ratio_delta": _round(aspect_ratio_delta),
        "score": _round(score),
        "gating": gating,
        "applied": False,
        "reason": reason,
    }


def _replay_sequence(
    sequence_name: str,
    rows: list[dict[str, Any]],
    config: StitchConfig,
    appearance_embeddings: dict[int, TrackletAppearanceEmbeddings] | None = None,
    appearance_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_frame = _sequence_rows(rows)
    active_tracks: dict[int, dict[str, Any]] = {}
    lost_tracks: dict[int, dict[str, Any]] = {}
    source_to_canonical: dict[int, int] = {}
    used_canonical_ids: set[int] = set()
    next_canonical_id = _initial_next_canonical_id(rows)

    output_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    accepted_remaps: list[str] = []
    matches_applied = 0
    fresh_canonical_ids_assigned = 0
    appearance_supported_remap_count = 0
    appearance_rejected_count = 0

    for frame in sorted(rows_by_frame):
        frame_rows = sorted(rows_by_frame[frame], key=lambda row: (int(row["id"]), float(row["bbox"][0]), float(row["bbox"][1])))
        current_source_ids = {int(row["id"]) for row in frame_rows}

        disappeared_source_ids = [source_id for source_id in list(active_tracks) if source_id not in current_source_ids]
        for source_id in disappeared_source_ids:
            lost_tracks[source_id] = active_tracks.pop(source_id)

        expired_source_ids = [
            source_id
            for source_id, state in lost_tracks.items()
            if max(0, frame - int(state["last_frame"]) - 1) > int(config.memory.max_frame_gap)
        ]
        for source_id in expired_source_ids:
            lost_tracks.pop(source_id, None)

        for row in frame_rows:
            source_track_id = int(row["id"])

            if source_track_id in active_tracks:
                canonical_track_id = int(active_tracks[source_track_id]["canonical_track_id"])
            elif source_track_id in source_to_canonical:
                canonical_track_id = int(source_to_canonical[source_track_id])
                lost_tracks.pop(source_track_id, None)
            else:
                candidate_decisions = [
                    _evaluate_candidate(lost_state=lost_state, new_row=row, config=config)
                    for lost_state in lost_tracks.values()
                ]
                for decision in candidate_decisions:
                    _annotate_appearance_support(
                        decision=decision,
                        appearance_embeddings=appearance_embeddings,
                        config=config,
                    )
                decisions.extend(candidate_decisions)
                eligible = [decision for decision in candidate_decisions if decision["gating"]["passes_all"]]

                if eligible:
                    eligible.sort(
                        key=lambda decision: (
                            -float(decision["score"] or 0.0),
                            float(decision["motion_distance"] or float("inf")),
                            float(decision["gap_frames"]),
                            float(decision["area_ratio"] or float("inf")),
                        )
                    )
                    best = eligible[0]
                    runner_up = eligible[1] if len(eligible) > 1 else None
                    accepted = False

                    if float(best["score"] or 0.0) < float(config.matching.min_match_score):
                        best["reason"] = (
                            f"below_score_threshold:{float(best['score'] or 0.0):.3f}"
                            f"<{float(config.matching.min_match_score):.3f}"
                        )
                    elif runner_up is not None:
                        winner_margin = round(float(best["score"] or 0.0) - float(runner_up["score"] or 0.0), 6)
                        winner_margin_threshold = _winner_margin_threshold(
                            config=config,
                            gap_frames=int(best["gap_frames"]),
                        )
                        best["winner_margin"] = _round(winner_margin)
                        best["winner_margin_threshold"] = _round(winner_margin_threshold)
                        if winner_margin < winner_margin_threshold:
                            best["reason"] = (
                                f"margin_too_small:{winner_margin:.3f}<{winner_margin_threshold:.3f}"
                            )
                            if config.appearance.enabled and config.appearance.use_for_matching:
                                ambiguous_candidates = [
                                    decision
                                    for decision in eligible
                                    if (
                                        float(best["score"] or 0.0) - float(decision["score"] or 0.0)
                                    ) < winner_margin_threshold
                                ]
                                appearance_candidates = [
                                    decision
                                    for decision in ambiguous_candidates
                                    if bool(decision.get("appearance_supported"))
                                    and float(decision.get("score") or 0.0) >= float(config.matching.min_match_score)
                                ]
                                if appearance_candidates:
                                    appearance_candidates.sort(
                                        key=lambda decision: (
                                            -float(decision.get("appearance_similarity") or float("-inf")),
                                            -float(decision.get("score") or 0.0),
                                            float(decision.get("motion_distance") or float("inf")),
                                        )
                                    )
                                    appearance_best = appearance_candidates[0]
                                    appearance_runner_up = (
                                        appearance_candidates[1] if len(appearance_candidates) > 1 else None
                                    )
                                    appearance_margin_threshold = _appearance_margin_threshold(config)
                                    appearance_margin = None
                                    if appearance_runner_up is not None:
                                        appearance_margin = round(
                                            float(appearance_best.get("appearance_similarity") or 0.0)
                                            - float(appearance_runner_up.get("appearance_similarity") or 0.0),
                                            6,
                                        )

                                    appearance_best["appearance_margin_threshold"] = _round(
                                        appearance_margin_threshold
                                    )
                                    appearance_best["appearance_margin"] = _round(appearance_margin)
                                    if (
                                        appearance_margin is None
                                        or appearance_margin >= appearance_margin_threshold
                                    ):
                                        appearance_best["appearance_assisted"] = True
                                        best = appearance_best
                                    else:
                                        appearance_rejected_count += 1
                                        best["appearance_margin_threshold"] = _round(
                                            appearance_margin_threshold
                                        )
                                        best["appearance_margin"] = _round(appearance_margin)
                                        best["reason"] = (
                                            f"appearance_margin_too_small:{appearance_margin:.3f}"
                                            f"<{appearance_margin_threshold:.3f}"
                                        )
                                else:
                                    appearance_rejected_count += 1

                    if best["reason"] == "eligible":
                        best["applied"] = True
                        best["reason"] = "accepted"
                        canonical_track_id = int(best["candidate_canonical_track_id"])
                        source_to_canonical[source_track_id] = canonical_track_id
                        used_canonical_ids.add(canonical_track_id)
                        matches_applied += 1
                        accepted_remaps.append(f"{source_track_id}->{int(best['candidate_source_track_id'])}")
                        lost_tracks.pop(int(best["candidate_source_track_id"]), None)
                        if bool(best.get("appearance_assisted")):
                            best["reason"] = "accepted_with_appearance_fallback"
                            appearance_supported_remap_count += 1
                        accepted = True

                        for decision in eligible[1:]:
                            if decision["reason"] == "eligible":
                                if bool(best.get("appearance_assisted")):
                                    decision["reason"] = (
                                        f"outscored_by_appearance:{source_track_id}"
                                        f"->{int(best['candidate_source_track_id'])}"
                                    )
                                else:
                                    decision["reason"] = (
                                        f"outscored_by:{source_track_id}->{int(best['candidate_source_track_id'])}"
                                    )

                    if not accepted:
                        canonical_track_id, next_canonical_id = _allocate_canonical_id(
                            source_track_id=source_track_id,
                            used_canonical_ids=used_canonical_ids,
                            next_canonical_id=next_canonical_id,
                        )
                        source_to_canonical[source_track_id] = canonical_track_id
                        fresh_canonical_ids_assigned += 1
                else:
                    canonical_track_id, next_canonical_id = _allocate_canonical_id(
                        source_track_id=source_track_id,
                        used_canonical_ids=used_canonical_ids,
                        next_canonical_id=next_canonical_id,
                    )
                    source_to_canonical[source_track_id] = canonical_track_id
                    fresh_canonical_ids_assigned += 1

            previous_state = active_tracks.get(source_track_id)
            active_tracks[source_track_id] = _make_track_state(
                source_track_id=source_track_id,
                canonical_track_id=canonical_track_id,
                row=row,
                previous_state=previous_state,
            )

            remapped_row = dict(row)
            remapped_row["id"] = canonical_track_id
            output_rows.append(remapped_row)

    accepted_remap_count = len(accepted_remaps)
    rejected_candidate_count = sum(1 for decision in decisions if not bool(decision.get("applied")))
    candidate_evaluation_count = len(decisions)
    below_score_threshold_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("below_score_threshold:")
    )
    margin_rejected_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("margin_too_small:")
    )
    sequence_report = {
        "sequence_name": sequence_name,
        "num_rows_in": len(rows),
        "num_rows_out": len(output_rows),
        "unique_track_ids_in": _unique_track_ids(rows),
        "unique_track_ids_out": _unique_track_ids(output_rows),
        "identity_map": {str(source_id): int(canonical_id) for source_id, canonical_id in sorted(source_to_canonical.items())},
        "matches_applied": matches_applied,
        "accepted_remap_count": accepted_remap_count,
        "rejected_candidate_count": rejected_candidate_count,
        "candidate_evaluation_count": candidate_evaluation_count,
        "fresh_canonical_ids_assigned": fresh_canonical_ids_assigned,
        "below_score_threshold_count": below_score_threshold_count,
        "margin_rejected_count": margin_rejected_count,
        "appearance_supported_remap_count": appearance_supported_remap_count,
        "appearance_rejected_count": appearance_rejected_count,
        "accepted_remaps": accepted_remaps,
        "decisions": decisions,
        "appearance": appearance_info or {},
    }
    return output_rows, sequence_report


def _default_replay_name(config: StitchConfig) -> str:
    stitch_name = str(config.stitch_name or "motion_bbox_v1")
    if stitch_name.startswith("replay_"):
        return stitch_name
    return f"replay_{stitch_name}"


def _augment_run_summary(
    input_summary: dict[str, Any] | None,
    output_root: Path,
    mot_dir: Path,
    report_path: Path,
    config: StitchConfig,
    sequence_reports: dict[str, dict[str, Any]],
    pred_root: Path,
    input_run_summary_path: Path | None,
) -> dict[str, Any]:
    summary = deepcopy(input_summary) if input_summary is not None else {}
    summary["output_root"] = str(output_root)
    summary["mot_dir"] = str(mot_dir)
    summary["total_rows_written"] = sum(int(report["num_rows_out"]) for report in sequence_reports.values())

    sequences = dict(summary.get("sequences") or {})
    total_matches_applied = 0
    for sequence_name, report in sequence_reports.items():
        seq_summary = dict(sequences.get(sequence_name) or {})
        seq_summary["rows_written"] = int(report["num_rows_out"])
        seq_summary["mot_path"] = report["output_mot_path"]
        seq_summary["unique_track_ids"] = list(report["unique_track_ids_out"])
        seq_summary["num_unique_track_ids"] = len(report["unique_track_ids_out"])
        seq_summary["replay_matches_applied"] = int(report["matches_applied"])
        seq_summary["replay_accepted_remap_count"] = int(report["accepted_remap_count"])
        seq_summary["replay_rejected_candidate_count"] = int(report["rejected_candidate_count"])
        seq_summary["replay_fresh_canonical_ids_assigned"] = int(report["fresh_canonical_ids_assigned"])
        seq_summary["replay_below_score_threshold_count"] = int(report["below_score_threshold_count"])
        seq_summary["replay_margin_rejected_count"] = int(report["margin_rejected_count"])
        seq_summary["replay_appearance_supported_remap_count"] = int(report["appearance_supported_remap_count"])
        seq_summary["replay_appearance_rejected_count"] = int(report["appearance_rejected_count"])
        sequences[sequence_name] = seq_summary
        total_matches_applied += int(report["matches_applied"])
    summary["sequences"] = sequences
    summary["replay_summary"] = {
        "mode": _replay_mode(config),
        "num_sequences_processed": len(sequence_reports),
        "accepted_remap_count": sum(int(report["accepted_remap_count"]) for report in sequence_reports.values()),
        "rejected_candidate_count": sum(int(report["rejected_candidate_count"]) for report in sequence_reports.values()),
        "candidate_evaluation_count": sum(int(report["candidate_evaluation_count"]) for report in sequence_reports.values()),
        "fresh_canonical_ids_assigned": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "below_score_threshold_count": sum(int(report["below_score_threshold_count"]) for report in sequence_reports.values()),
        "margin_rejected_count": sum(int(report["margin_rejected_count"]) for report in sequence_reports.values()),
        "appearance_supported_remap_count": sum(int(report["appearance_supported_remap_count"]) for report in sequence_reports.values()),
        "appearance_rejected_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
    }
    summary["stitching"] = {
        "mode": _replay_mode(config),
        "stitch_name": output_root.name,
        "input_pred_dir": str(pred_root),
        "input_run_summary_path": str(input_run_summary_path) if input_run_summary_path is not None else None,
        "report_path": str(report_path),
        "config": config.to_dict(),
        "replay_mode": True,
        "total_matches_applied": total_matches_applied,
    }
    return summary


def replay_stitch_tracks(
    pred_dir: str,
    output_dir: str | None = None,
    run_summary_path: str | None = None,
    config_path: str | None = None,
    stitch_name: str | None = None,
) -> dict[str, Any]:
    pred_root = Path(pred_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    config = _load_stitch_config(config_path)
    replay_name = stitch_name or _default_replay_name(config)
    output_root = Path(output_dir) if output_dir else pred_root.parent / "replay_stitched" / replay_name
    mot_dir = output_root / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    input_run_summary_path = _resolve_run_summary_path(pred_root, run_summary_path)
    input_run_summary = _load_run_summary(input_run_summary_path)

    mot_files = sorted(pred_root.glob("*.txt"))
    if not mot_files:
        raise RuntimeError(f"No MOT txt files found in {pred_root}")

    warnings: list[str] = []
    if input_run_summary is None:
        warnings.append("Input run_summary.json was not found; replay-stitched run summary was created from MOT outputs only.")

    offline_comparison_runs, comparison_warnings = _load_offline_comparison_runs(pred_root.parent / "stitched")
    warnings.extend(comparison_warnings)

    sequence_reports: dict[str, dict[str, Any]] = {}
    source_path = None if input_run_summary is None else input_run_summary.get("source")
    for mot_path in mot_files:
        sequence_name = mot_path.stem
        rows = load_mot_rows(mot_path)
        appearance_embeddings, appearance_info, appearance_warnings = _build_replay_appearance_embeddings(
            sequence_name=sequence_name,
            rows=rows,
            source_video_path=source_path,
            config=config,
        )
        warnings.extend(appearance_warnings)
        output_rows, sequence_report = _replay_sequence(
            sequence_name=sequence_name,
            rows=rows,
            config=config,
            appearance_embeddings=appearance_embeddings,
            appearance_info=appearance_info,
        )
        output_mot_path = mot_dir / mot_path.name
        write_mot_rows(output_mot_path, output_rows)
        sequence_report["input_mot_path"] = str(mot_path)
        sequence_report["output_mot_path"] = str(output_mot_path)
        sequence_report["comparison"] = {
            "original_num_unique_track_ids": len(sequence_report["unique_track_ids_in"]),
            "replay_num_unique_track_ids": len(sequence_report["unique_track_ids_out"]),
            "offline_runs": {
                run_name: run_report["sequences"][sequence_name]
                for run_name, run_report in offline_comparison_runs.items()
                if sequence_name in run_report["sequences"]
            },
        }
        sequence_report["summary"] = _build_sequence_summary(
            sequence_report=sequence_report,
            offline_comparisons=sequence_report["comparison"]["offline_runs"],
        )
        sequence_reports[sequence_name] = sequence_report

    report_path = output_root / "replay_report.json"
    run_summary_payload = _augment_run_summary(
        input_summary=input_run_summary,
        output_root=output_root,
        mot_dir=mot_dir,
        report_path=report_path,
        config=config,
        sequence_reports=sequence_reports,
        pred_root=pred_root,
        input_run_summary_path=input_run_summary_path,
    )

    run_summary_path_out = output_root / "run_summary.json"
    run_summary_path_out.write_text(json.dumps(run_summary_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "num_sequences_processed": len(sequence_reports),
        "accepted_remap_count": sum(int(report["accepted_remap_count"]) for report in sequence_reports.values()),
        "rejected_candidate_count": sum(int(report["rejected_candidate_count"]) for report in sequence_reports.values()),
        "candidate_evaluation_count": sum(int(report["candidate_evaluation_count"]) for report in sequence_reports.values()),
        "fresh_canonical_ids_assigned": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "below_score_threshold_count": sum(int(report["below_score_threshold_count"]) for report in sequence_reports.values()),
        "margin_rejected_count": sum(int(report["margin_rejected_count"]) for report in sequence_reports.values()),
        "appearance_supported_remap_count": sum(int(report["appearance_supported_remap_count"]) for report in sequence_reports.values()),
        "appearance_rejected_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
        "offline_comparison_runs": {
            run_name: {
                "mode": run_report["mode"],
                "stitch_name": run_report["stitch_name"],
                "report_path": run_report["report_path"],
                "total_matches_applied": run_report["total_matches_applied"],
                "num_sequences": run_report["num_sequences"],
            }
            for run_name, run_report in offline_comparison_runs.items()
        },
    }
    report_payload = {
        "version": REPLAY_VERSION,
        "mode": _replay_mode(config),
        "pred_dir": str(pred_root),
        "output_root": str(output_root),
        "input_run_summary_path": str(input_run_summary_path) if input_run_summary_path is not None else None,
        "output_run_summary_path": str(run_summary_path_out),
        "config": config.to_dict(),
        "total_sequences": len(sequence_reports),
        "total_matches_applied": sum(int(report["matches_applied"]) for report in sequence_reports.values()),
        "summary": summary_payload,
        "sequences": sequence_reports,
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return {
        "mode": _replay_mode(config),
        "pred_dir": str(pred_root),
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "run_summary": str(run_summary_path_out),
        "replay_report": str(report_path),
        "num_sequences": len(sequence_reports),
        "matches_applied": report_payload["total_matches_applied"],
        "accepted_remap_count": summary_payload["accepted_remap_count"],
        "rejected_candidate_count": summary_payload["rejected_candidate_count"],
        "fresh_canonical_ids_assigned": summary_payload["fresh_canonical_ids_assigned"],
        "appearance_supported_remap_count": summary_payload["appearance_supported_remap_count"],
        "appearance_rejected_count": summary_payload["appearance_rejected_count"],
    }
