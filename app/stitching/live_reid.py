from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from ais import align_ais_tracks_to_frames, assign_ais_to_tracks, load_ais_file_with_warnings, parse_affine_matrix
from colreg import (
    EncounterType,
    build_vessel_state,
    closest_approach_estimate,
    classify_encounter,
    distance_between_states,
    is_proximity_candidate,
    proximity_risk_score,
    score_colreg_motion_consistency,
)
from evaluation.mot import load_mot_rows, write_mot_rows
from stitching.reid_scoring import (
    _appearance_margin_threshold,
    _appearance_similarity_threshold,
    _area_ratio,
    _aspect_ratio_delta,
    _delta_score,
    _distance,
    _ratio_score,
    _round,
    _threshold_score,
    _winner_margin_threshold,
)
from stitching.runner import DEFAULT_CONFIG_PATH, _load_run_summary, _load_stitch_config, _resolve_run_summary_path
from stitching.schemas import StitchConfig

LIVE_REID_MODE = "live_reid_motion_bbox_confirmation_v1"
LIVE_REID_MODE_WITH_APPEARANCE = "live_reid_motion_bbox_with_appearance_fallback_v1"
LIVE_REID_VERSION = "live_reid/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _center_from_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)]


def _build_observation(row: dict[str, Any]) -> dict[str, Any]:
    bbox = [float(value) for value in row["bbox"]]
    return {
        "frame": int(row["frame"]),
        "bbox": bbox,
        "confidence": float(row["confidence"]),
        "class_id": int(row["class_id"]),
        "visibility": float(row["visibility"]),
        "center": _center_from_bbox(bbox),
    }


def _mean_bbox(observations: list[dict[str, Any]]) -> list[float]:
    return [
        sum(float(obs["bbox"][index]) for obs in observations) / len(observations)
        for index in range(4)
    ]


def _mean_center(observations: list[dict[str, Any]]) -> list[float]:
    return [
        sum(float(obs["center"][index]) for obs in observations) / len(observations)
        for index in range(2)
    ]


def _velocity(observations: list[dict[str, Any]]) -> list[float]:
    if len(observations) < 2:
        return [0.0, 0.0]
    start = observations[0]
    end = observations[-1]
    delta_frames = max(int(end["frame"]) - int(start["frame"]), 1)
    return [
        round((float(end["center"][0]) - float(start["center"][0])) / delta_frames, 6),
        round((float(end["center"][1]) - float(start["center"][1])) / delta_frames, 6),
    ]


def _mean_confidence(observations: list[dict[str, Any]]) -> float:
    return round(sum(float(obs["confidence"]) for obs in observations) / len(observations), 6)


def _mean_area(bbox: list[float]) -> float:
    return round(float(bbox[2]) * float(bbox[3]), 6)


def _mean_aspect_ratio(bbox: list[float]) -> float:
    return round(float(bbox[2]) / max(float(bbox[3]), 1e-9), 6)


def _initial_next_canonical_id(rows: list[dict[str, Any]]) -> int:
    source_ids = [int(row["id"]) for row in rows if int(row["id"]) >= 0]
    return (max(source_ids) + 1) if source_ids else 1


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


def _live_mode(config: StitchConfig, appearance_ready: bool) -> str:
    if config.appearance.enabled and config.appearance.use_for_matching and appearance_ready:
        return LIVE_REID_MODE_WITH_APPEARANCE
    return LIVE_REID_MODE


def _default_live_name(config: StitchConfig) -> str:
    stitch_name = str(config.stitch_name or "motion_bbox_v1")
    if stitch_name.startswith("live_reid_"):
        return stitch_name
    return f"live_reid_{stitch_name}"


def _configure_colreg_options(
    config: StitchConfig,
    *,
    diagnostics_enabled: bool = False,
    scoring_experiment: bool = False,
) -> StitchConfig:
    colreg_config = getattr(config, "colreg", None)
    if colreg_config is None:
        return config

    scoring_enabled = bool(
        scoring_experiment
        or getattr(colreg_config, "scoring_experiment", False)
        or getattr(colreg_config, "scoring_enabled", False)
    )
    diagnostics = bool(diagnostics_enabled or scoring_enabled or getattr(colreg_config, "diagnostics_enabled", False))
    if diagnostics or scoring_enabled:
        config.colreg = replace(
            colreg_config,
            diagnostics_enabled=diagnostics,
            scoring_enabled=scoring_enabled,
            scoring_experiment=scoring_enabled,
        )
    return config


def _configure_ais_options(
    config: StitchConfig,
    *,
    diagnostics_enabled: bool = False,
    ais_file: str | None = None,
    video_start_time: str | None = None,
    affine_matrix: str | list[list[float]] | None = None,
    fps: float | None = None,
) -> StitchConfig:
    ais_config = getattr(config, "ais", None)
    if ais_config is None:
        return config
    diagnostics = bool(diagnostics_enabled or getattr(ais_config, "diagnostics_enabled", False))
    if diagnostics or ais_file or video_start_time or affine_matrix or fps:
        matrix = parse_affine_matrix(affine_matrix) if affine_matrix is not None else ais_config.affine_matrix
        config.ais = replace(
            ais_config,
            diagnostics_enabled=diagnostics,
            ais_file=ais_file or ais_config.ais_file,
            video_start_time=video_start_time if video_start_time is not None else ais_config.video_start_time,
            affine_enabled=bool(matrix) or bool(ais_config.affine_enabled),
            affine_matrix=matrix,
            fps=fps if fps is not None else ais_config.fps,
        )
    return config


def _colreg_scoring_enabled(config: StitchConfig) -> bool:
    colreg_config = getattr(config, "colreg", None)
    return bool(
        colreg_config is not None
        and (
            getattr(colreg_config, "scoring_experiment", False)
            or getattr(colreg_config, "scoring_enabled", False)
        )
    )


def _candidate_sort_key(decision: dict[str, Any], score_field: str = "score") -> tuple[float, float, int, int]:
    score = decision.get(score_field)
    return (
        -float(score if score is not None else float("-inf")),
        float(decision.get("motion_distance") if decision.get("motion_distance") is not None else float("inf")),
        int(decision.get("gap_frames") or 0),
        int(decision.get("candidate_source_track_id") or 0),
    )


def _colreg_suspicious_reason(decision: dict[str, Any]) -> str:
    reasons: list[str] = []
    encounter = str(decision.get("colreg_encounter") or EncounterType.UNKNOWN.value)
    score = decision.get("colreg_score")
    confidence = float(decision.get("colreg_confidence") or 0.0)
    if encounter in {"", EncounterType.UNKNOWN.value}:
        reasons.append("unknown_encounter")
    if not bool(decision.get("colreg_proximity_passed")):
        reasons.append("outside_proximity_filter")
    if encounter == EncounterType.OVERTAKING.value and score is not None and confidence >= 0.85 and float(score) < 0.35:
        reasons.append("high_confidence_overtaking_low_motion_score")
    return ";".join(reasons)


def _set_colreg_scoring_defaults(decision: dict[str, Any], config: StitchConfig, reason: str = "not_evaluated") -> None:
    if not _colreg_scoring_enabled(config):
        return
    score = _round(decision.get("score"))
    decision.setdefault("colreg_suspicious_reason", _colreg_suspicious_reason(decision))
    decision["colreg_scoring_experiment_enabled"] = True
    decision["colreg_score_used"] = False
    decision["colreg_score_delta"] = 0.0
    decision["original_candidate_score"] = score
    decision["colreg_adjusted_candidate_score"] = score
    decision["colreg_changed_ranking"] = False
    decision["colreg_changed_accepted_remap"] = False
    decision["colreg_scoring_reason"] = reason


def _colreg_scoring_candidate_allowed(decision: dict[str, Any], config: StitchConfig) -> tuple[bool, str]:
    colreg_config = config.colreg
    if not bool((decision.get("gating") or {}).get("passes_all")):
        return False, "normal_gates_failed"
    if getattr(colreg_config, "scoring_require_proximity", True) and not bool(decision.get("colreg_proximity_passed")):
        return False, "proximity_not_passed"
    encounter = str(decision.get("colreg_encounter") or "")
    if encounter in {"", EncounterType.UNKNOWN.value, EncounterType.NO_INTERACTION.value}:
        return False, "non_interaction_or_unknown"
    confidence = float(decision.get("colreg_confidence") or 0.0)
    if confidence < float(getattr(colreg_config, "scoring_min_confidence", 0.85)):
        return False, "confidence_below_threshold"
    colreg_score = decision.get("colreg_score")
    if colreg_score is None:
        return False, "missing_colreg_score"
    if float(colreg_score) < float(getattr(colreg_config, "scoring_min_score", 0.65)):
        return False, "colreg_score_below_threshold"
    suspicious_reason = str(decision.get("colreg_suspicious_reason") or _colreg_suspicious_reason(decision))
    decision["colreg_suspicious_reason"] = suspicious_reason
    if suspicious_reason:
        return False, f"suspicious:{suspicious_reason}"
    return True, "eligible_colreg_tiebreak"


def _rank_eligible_candidates(eligible: list[dict[str, Any]], config: StitchConfig) -> list[dict[str, Any]]:
    ranked = sorted(eligible, key=_candidate_sort_key)
    if not _colreg_scoring_enabled(config):
        return ranked

    for decision in eligible:
        _set_colreg_scoring_defaults(decision, config)

    if len(ranked) < 2:
        for decision in eligible:
            decision["colreg_scoring_reason"] = "single_candidate"
        return ranked

    baseline_best = ranked[0]
    baseline_runner_up = ranked[1]
    baseline_margin = round(float(baseline_best.get("score") or 0.0) - float(baseline_runner_up.get("score") or 0.0), 6)
    margin_threshold = float(getattr(config.colreg, "scoring_ambiguous_margin_threshold", 0.05))
    for decision in eligible:
        decision["colreg_baseline_best_candidate"] = int(baseline_best["candidate_source_track_id"])
        decision["colreg_original_winner_margin"] = _round(baseline_margin)
        decision["colreg_ambiguous_margin_threshold"] = _round(margin_threshold)

    if baseline_margin > margin_threshold:
        for decision in eligible:
            decision["colreg_scoring_reason"] = f"not_ambiguous:{baseline_margin:.3f}>{margin_threshold:.3f}"
        return ranked

    used_any = False
    for decision in eligible:
        allowed, reason = _colreg_scoring_candidate_allowed(decision, config)
        decision["colreg_scoring_reason"] = reason
        if not allowed:
            continue
        delta = round(float(config.colreg.scoring_weight) * (float(decision["colreg_score"]) - 0.5), 6)
        if delta <= 0:
            decision["colreg_scoring_reason"] = "non_positive_delta"
            continue
        decision["colreg_score_used"] = True
        decision["colreg_score_delta"] = _round(delta)
        decision["colreg_adjusted_candidate_score"] = _round(float(decision.get("score") or 0.0) + delta)
        used_any = True

    if not used_any:
        return ranked

    adjusted = sorted(eligible, key=lambda decision: _candidate_sort_key(decision, "colreg_adjusted_candidate_score"))
    adjusted_best = adjusted[0]
    adjusted_runner_up = adjusted[1] if len(adjusted) > 1 else None
    changed = int(adjusted_best["candidate_source_track_id"]) != int(baseline_best["candidate_source_track_id"])
    adjusted_margin = None
    if adjusted_runner_up is not None:
        adjusted_margin = round(
            float(adjusted_best.get("colreg_adjusted_candidate_score") or 0.0)
            - float(adjusted_runner_up.get("colreg_adjusted_candidate_score") or 0.0),
            6,
        )
    for decision in eligible:
        decision["colreg_adjusted_best_candidate"] = int(adjusted_best["candidate_source_track_id"])
        decision["colreg_adjusted_winner_margin"] = _round(adjusted_margin)
    if changed:
        baseline_best["colreg_changed_ranking"] = True
        adjusted_best["colreg_changed_ranking"] = True
    return adjusted


def _mark_colreg_acceptance(decision: dict[str, Any]) -> None:
    if bool(decision.get("colreg_score_used")) and bool(decision.get("colreg_changed_ranking")):
        decision["colreg_changed_accepted_remap"] = True


def _colreg_scoring_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(bool(decision.get("colreg_scoring_experiment_enabled")) for decision in decisions):
        return {}
    return {
        "colreg_scoring_experiment_enabled": True,
        "colreg_score_used_count": sum(1 for decision in decisions if bool(decision.get("colreg_score_used"))),
        "colreg_changed_ranking_count": sum(1 for decision in decisions if bool(decision.get("colreg_changed_ranking"))),
        "colreg_changed_accepted_remap_count": sum(
            1 for decision in decisions if bool(decision.get("colreg_changed_accepted_remap"))
        ),
    }


def _build_ais_assignments(
    *,
    sequence_name: str,
    rows: list[dict[str, Any]],
    config: StitchConfig,
) -> tuple[dict[int, Any], list[str]]:
    ais_config = getattr(config, "ais", None)
    if ais_config is None or not bool(getattr(ais_config, "diagnostics_enabled", False)):
        return {}, []
    if not ais_config.ais_file:
        return {}, [f"{sequence_name}: AIS diagnostics enabled without --ais-file; candidate AIS fields are neutral."]
    try:
        ais_tracks, warnings = load_ais_file_with_warnings(ais_config.ais_file, ais_config)
        frame_count = max((int(row["frame"]) for row in rows), default=0)
        aligned = align_ais_tracks_to_frames(
            ais_tracks,
            frame_count=frame_count,
            fps=ais_config.fps or 30.0,
            video_start_time=ais_config.video_start_time,
            config=ais_config,
        )
        assignments = assign_ais_to_tracks(rows, aligned, ais_config)
    except Exception as exc:
        return {}, [f"{sequence_name}: AIS diagnostics unavailable: {type(exc).__name__}: {exc}"]
    return {int(assignment.track_id): assignment for assignment in assignments}, warnings


def _annotate_ais_diagnostics(
    decision: dict[str, Any],
    *,
    config: StitchConfig,
    ais_assignments: dict[int, Any] | None,
) -> None:
    ais_config = getattr(config, "ais", None)
    if ais_config is None or not bool(getattr(ais_config, "diagnostics_enabled", False)):
        return

    neutral = float(getattr(ais_config, "neutral_score", 0.5))
    source_assignment = (ais_assignments or {}).get(int(decision["new_source_track_id"]))
    candidate_assignment = (ais_assignments or {}).get(int(decision["candidate_source_track_id"]))
    source_mmsi = None if source_assignment is None else source_assignment.assigned_mmsi
    candidate_mmsi = None if candidate_assignment is None else candidate_assignment.assigned_mmsi
    if source_mmsi and candidate_mmsi:
        identity_score = 1.0 if str(source_mmsi) == str(candidate_mmsi) else 0.0
        note = "mmsi_match" if identity_score == 1.0 else "mmsi_mismatch_reporting_only"
    else:
        identity_score = neutral
        note = "missing_ais_assignment_neutral"

    def score_pair(attribute: str) -> float:
        values = [
            float(getattr(assignment, attribute))
            for assignment in (source_assignment, candidate_assignment)
            if assignment is not None
        ]
        return neutral if not values else round(sum(values) / len(values), 6)

    decision.update(
        {
            "ais_reporting_only": True,
            "ais_source_mmsi": source_mmsi,
            "ais_candidate_mmsi": candidate_mmsi,
            "ais_assignment_score_source": neutral if source_assignment is None else source_assignment.score,
            "ais_assignment_score_candidate": neutral if candidate_assignment is None else candidate_assignment.score,
            "ais_position_score": score_pair("position_score"),
            "ais_identity_score": identity_score,
            "ais_heading_score": score_pair("heading_score"),
            "ais_speed_score": score_pair("speed_score"),
            "ais_note": note,
        }
    )


def _normalize_workspace_path(path_value: str | None, *, require_exists: bool = True) -> str | None:
    if not path_value:
        return path_value

    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return str(candidate)

    normalized = str(path_value).replace("\\", "/")
    if normalized == "/workspace":
        mapped = REPO_ROOT
    elif normalized.startswith("/workspace/"):
        relative = normalized.removeprefix("/workspace/").strip("/")
        mapped = REPO_ROOT / Path(relative.replace("/", "\\"))
    else:
        return path_value

    if require_exists and not mapped.exists():
        return path_value
    return str(mapped)


def _resolve_source_video_path(source_video_path: str | None, sequence_name: str) -> Path | None:
    if not source_video_path:
        return None

    normalized_source = _normalize_workspace_path(source_video_path, require_exists=True)
    source_path = Path(normalized_source or source_video_path)
    if source_path.exists() and source_path.is_file():
        return source_path
    if not source_path.exists() or not source_path.is_dir():
        return None

    supported_suffixes = [".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"]
    for suffix in supported_suffixes:
        candidate = source_path / f"{sequence_name}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate
    normalized = sequence_name.lower()
    for candidate in sorted(source_path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in supported_suffixes and candidate.stem.lower() == normalized:
            return candidate
    return None


def _build_appearance_runtime(sequence_name: str, source_video_path: str | None, config: StitchConfig) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not (config.appearance.enabled and config.appearance.use_for_matching):
        return None, warnings

    resolved_source = _resolve_source_video_path(source_video_path, sequence_name)
    if resolved_source is None:
        warnings.append(f"{sequence_name}: appearance requested but source video could not be resolved.")
        return None, warnings

    try:
        from stitching.appearance import TimmAppearanceBackend, cosine_similarity
        from stitching.video import VideoFrameReader
    except Exception as exc:
        warnings.append(f"{sequence_name}: appearance backend import failed: {type(exc).__name__}: {exc}")
        return None, warnings

    runtime_config = deepcopy(config.appearance)
    runtime_config.checkpoint_path = _normalize_workspace_path(runtime_config.checkpoint_path, require_exists=True)
    runtime_config.cache_dir = _normalize_workspace_path(runtime_config.cache_dir, require_exists=False)

    try:
        backend = TimmAppearanceBackend(runtime_config)
        reader = VideoFrameReader(str(resolved_source))
    except Exception as exc:
        warnings.append(f"{sequence_name}: appearance backend unavailable: {type(exc).__name__}: {exc}")
        return None, warnings

    return {
        "backend": backend,
        "reader": reader,
        "cosine_similarity": cosine_similarity,
        "source_video": str(resolved_source),
        "info": {
            "name": "timm_dinov2",
            "status": "ready",
            "model_name": backend.model_name,
            "device": str(backend.device),
            "embedding_dim": backend.embedding_dim,
            "weights_source": backend.weights_source,
            "source_video": str(resolved_source),
            "sample_frames": int(config.appearance.sample_frames),
            "crop_padding": float(config.appearance.crop_padding),
            "min_crop_size": int(config.appearance.min_crop_size),
            "use_for_matching": True,
        },
    }, warnings


def _close_appearance_runtime(runtime: dict[str, Any] | None) -> None:
    if runtime is None:
        return
    reader = runtime.get("reader")
    if reader is not None:
        reader.close()


def _extract_embedding(
    runtime: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    config: StitchConfig,
    *,
    prefer_tail: bool = False,
) -> tuple[Any | None, str]:
    if runtime is None:
        return None, "not_available"

    reader = runtime["reader"]
    backend = runtime["backend"]
    limit = max(1, int(config.appearance.sample_frames))
    selected = observations[-limit:] if prefer_tail else observations[:limit]
    crops: list[Any] = []
    for obs in selected:
        crop = reader.extract_crop(
            frame_index=int(obs["frame"]),
            bbox=list(obs["bbox"]),
            crop_padding=float(config.appearance.crop_padding),
            min_crop_size=int(config.appearance.min_crop_size),
        )
        if crop is not None:
            crops.append(crop)
    if not crops:
        return None, "missing_crops"
    embedding = backend.prototype(crops)
    if embedding is None:
        return None, "embedding_failed"
    return embedding, "ready"


def _make_confirmed_state(
    source_track_id: int,
    canonical_track_id: int,
    observations: list[dict[str, Any]],
    head_window: int,
    tail_window: int,
) -> dict[str, Any]:
    tail_observations = observations[-tail_window:]
    head_observations = observations[:head_window]
    tail_mean_bbox = _mean_bbox(tail_observations)
    tail_mean_center = _mean_center(tail_observations)
    return {
        "source_track_id": int(source_track_id),
        "canonical_track_id": int(canonical_track_id),
        "first_frame": int(observations[0]["frame"]),
        "last_frame": int(observations[-1]["frame"]),
        "first_bbox": list(observations[0]["bbox"]),
        "last_bbox": list(observations[-1]["bbox"]),
        "last_center": list(observations[-1]["center"]),
        "class_id": int(observations[-1]["class_id"]),
        "head_observations": [deepcopy(obs) for obs in head_observations],
        "tail_observations": [deepcopy(obs) for obs in tail_observations],
        "tail_mean_bbox": [round(value, 6) for value in tail_mean_bbox],
        "tail_mean_center": [round(value, 6) for value in tail_mean_center],
        "tail_velocity": [round(value, 6) for value in _velocity(tail_observations)],
        "mean_confidence": _mean_confidence(observations),
        "mean_area": _mean_area(tail_mean_bbox),
        "mean_aspect_ratio": _mean_aspect_ratio(tail_mean_bbox),
    }


def _update_confirmed_state(
    state: dict[str, Any],
    row: dict[str, Any],
    head_window: int,
    tail_window: int,
) -> dict[str, Any]:
    observation = _build_observation(row)
    observations = [*state["tail_observations"], observation]
    observations = observations[-tail_window:]
    state["last_frame"] = int(observation["frame"])
    state["last_bbox"] = list(observation["bbox"])
    state["last_center"] = list(observation["center"])
    state["class_id"] = int(observation["class_id"])
    state["tail_observations"] = observations
    state["tail_mean_bbox"] = [round(value, 6) for value in _mean_bbox(observations)]
    state["tail_mean_center"] = [round(value, 6) for value in _mean_center(observations)]
    state["tail_velocity"] = [round(value, 6) for value in _velocity(observations)]
    state["mean_confidence"] = _mean_confidence(observations)
    state["mean_area"] = _mean_area(state["tail_mean_bbox"])
    state["mean_aspect_ratio"] = _mean_aspect_ratio(state["tail_mean_bbox"])
    if len(state["head_observations"]) < head_window:
        state["head_observations"].append(deepcopy(observation))
    return state


def _make_pending_state(source_track_id: int, row: dict[str, Any]) -> dict[str, Any]:
    observation = _build_observation(row)
    return {
        "source_track_id": int(source_track_id),
        "rows": [dict(row)],
        "observations": [observation],
        "first_frame": int(observation["frame"]),
        "last_frame": int(observation["frame"]),
        "class_id": int(observation["class_id"]),
    }


def _update_pending_state(state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    observation = _build_observation(row)
    state["rows"].append(dict(row))
    state["observations"].append(observation)
    state["last_frame"] = int(observation["frame"])
    return state


def _pending_summary(state: dict[str, Any], *, match_observation_limit: int | None = None) -> dict[str, Any]:
    observations = state["observations"]
    if match_observation_limit is None:
        summary_observations = observations
    else:
        limit = max(1, min(int(match_observation_limit), len(observations)))
        summary_observations = observations[:limit]

    head_mean_bbox = _mean_bbox(summary_observations)
    head_mean_center = _mean_center(summary_observations)
    return {
        "source_track_id": int(state["source_track_id"]),
        "rows": state["rows"],
        "observations": observations,
        "num_observations": len(observations),
        "first_frame": int(state["first_frame"]),
        "last_frame": int(state["last_frame"]),
        "class_id": int(state["class_id"]),
        "match_observations": summary_observations,
        "match_observation_limit": len(summary_observations),
        "head_mean_bbox": [round(value, 6) for value in head_mean_bbox],
        "head_mean_center": [round(value, 6) for value in head_mean_center],
        "head_velocity": [round(value, 6) for value in _velocity(summary_observations)],
        "mean_confidence": _mean_confidence(summary_observations),
        "mean_area": _mean_area(head_mean_bbox),
        "mean_aspect_ratio": _mean_aspect_ratio(head_mean_bbox),
    }


def _evaluate_live_candidate(
    lost_state: dict[str, Any],
    pending: dict[str, Any],
    config: StitchConfig,
    decision_frame: int,
) -> dict[str, Any]:
    first_frame = int(pending["first_frame"])
    gap_frames = max(0, first_frame - int(lost_state["last_frame"]) - 1)
    elapsed_frames = max(1, first_frame - int(lost_state["last_frame"]))
    max_gap = min(int(config.memory.max_frame_gap), int(config.motion.max_frame_gap))

    predicted_center = [
        float(lost_state["last_center"][0]) + (float(lost_state["tail_velocity"][0]) * elapsed_frames),
        float(lost_state["last_center"][1]) + (float(lost_state["tail_velocity"][1]) * elapsed_frames),
    ]
    direct_center_distance = _distance(list(lost_state["last_center"]), list(pending["head_mean_center"]))
    motion_distance = _distance(predicted_center, list(pending["head_mean_center"]))
    expected_radius = float(config.motion.base_region_radius) + (float(config.motion.growth_per_frame) * gap_frames)
    area_ratio = _area_ratio(list(lost_state["tail_mean_bbox"]), list(pending["head_mean_bbox"]))
    aspect_ratio_delta = _aspect_ratio_delta(list(lost_state["tail_mean_bbox"]), list(pending["head_mean_bbox"]))

    class_match = int(lost_state["class_id"]) == int(pending["class_id"])
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

    decision = {
        "decision_frame": int(decision_frame),
        "decision_latency_frames": int(decision_frame - int(pending["first_frame"])),
        "new_source_track_id": int(pending["source_track_id"]),
        "candidate_source_track_id": int(lost_state["source_track_id"]),
        "candidate_canonical_track_id": int(lost_state["canonical_track_id"]),
        "gap_frames": gap_frames,
        "elapsed_frames": elapsed_frames,
        "direct_center_distance": _round(direct_center_distance),
        "predicted_center": [_round(value) for value in predicted_center],
        "motion_distance": _round(motion_distance),
        "expected_radius": _round(expected_radius),
        "area_ratio": _round(area_ratio),
        "aspect_ratio_delta": _round(aspect_ratio_delta),
        "score": _round(score),
        "gating": gating,
        "applied": False,
        "reason": reason,
    }
    _annotate_colreg_diagnostics(decision=decision, lost_state=lost_state, pending=pending, config=config)
    return decision


def _annotate_colreg_diagnostics(
    decision: dict[str, Any],
    lost_state: dict[str, Any],
    pending: dict[str, Any],
    config: StitchConfig,
) -> None:
    colreg_config = getattr(config, "colreg", None)
    if colreg_config is None or not bool(getattr(colreg_config, "diagnostics_enabled", False)):
        return

    default_fields = {
        "colreg_reporting_only": True,
        "colreg_encounter": EncounterType.UNKNOWN.value,
        "colreg_confidence": 0.0,
        "colreg_score": None,
        "colreg_reason": "insufficient_candidate_track_history",
        "colreg_relative_bearing": None,
        "colreg_heading_difference": None,
        "colreg_distance_px": None,
        "colreg_closest_distance_px": None,
        "colreg_frames_to_closest": None,
        "colreg_relative_speed": None,
        "colreg_proximity_risk_score": None,
        "colreg_proximity_passed": False,
        "colreg_suspicious_reason": "insufficient_colreg_history",
    }
    decision.update(default_fields)

    lost_observations = list(lost_state.get("tail_observations") or [])
    pending_observations = list(pending.get("match_observations") or pending.get("observations") or [])
    own_state = build_vessel_state(
        lost_observations,
        track_id=int(lost_state["source_track_id"]),
        min_observations=int(colreg_config.min_track_observations),
    )
    other_state = build_vessel_state(
        pending_observations,
        track_id=int(pending["source_track_id"]),
        min_observations=int(colreg_config.min_track_observations),
    )
    result = classify_encounter(own_state, other_state, colreg_config)
    decision.update(
        {
            "colreg_encounter": result.encounter_type.value,
            "colreg_confidence": _round(result.confidence),
            "colreg_reason": result.reason,
            "colreg_relative_bearing": _round(result.relative_bearing_deg),
            "colreg_heading_difference": _round(result.heading_difference_deg),
        }
    )
    if own_state is not None and other_state is not None:
        closest = closest_approach_estimate(
            own_state,
            other_state,
            horizon_frames=int(colreg_config.closest_approach_horizon_frames),
        )
        decision.update(
            {
                "colreg_distance_px": _round(distance_between_states(own_state, other_state)),
                "colreg_closest_distance_px": _round(closest.get("closest_distance_px")),
                "colreg_frames_to_closest": _round(closest.get("frames_to_closest")),
                "colreg_relative_speed": _round(closest.get("relative_speed")),
                "colreg_proximity_risk_score": _round(proximity_risk_score(own_state, other_state, colreg_config)),
                "colreg_proximity_passed": bool(is_proximity_candidate(own_state, other_state, colreg_config)),
            }
        )
        decision["colreg_score"] = _round(
            score_colreg_motion_consistency(
                previous_state=own_state,
                observed_reappearing_state=other_state,
                encounter_type=result.encounter_type,
                config=colreg_config,
            )
        )
    decision["colreg_suspicious_reason"] = _colreg_suspicious_reason(decision)


def _use_preview_pending_summary(lost_state: dict[str, Any], pending: dict[str, Any], config: StitchConfig) -> bool:
    if not (config.appearance.enabled and config.appearance.use_for_matching):
        return False
    long_gap_threshold = config.matching.long_gap_frames_threshold
    if long_gap_threshold is None:
        return False
    gap_frames = max(0, int(pending["first_frame"]) - int(lost_state["last_frame"]) - 1)
    return gap_frames > int(long_gap_threshold)


def _annotate_live_appearance(
    decision: dict[str, Any],
    lost_state: dict[str, Any],
    pending: dict[str, Any],
    runtime: dict[str, Any] | None,
    config: StitchConfig,
) -> None:
    if runtime is None:
        decision["appearance_similarity"] = None
        decision["appearance_threshold"] = None
        decision["appearance_supported"] = False
        decision["appearance_status"] = "not_available"
        return

    tail_embedding = lost_state.get("tail_embedding")
    tail_status = str(lost_state.get("tail_embedding_status") or "not_available")
    head_embedding, head_status = _extract_embedding(
        runtime,
        list(pending["match_observations"]),
        config,
        prefer_tail=False,
    )
    if head_embedding is None:
        decision["appearance_similarity"] = None
        decision["appearance_threshold"] = _round(
            _appearance_similarity_threshold(config=config, gap_frames=int(decision["gap_frames"]))
        )
        decision["appearance_supported"] = False
        decision["appearance_status"] = f"pending_head_{head_status}"
        return
    if tail_embedding is None:
        decision["appearance_similarity"] = None
        decision["appearance_threshold"] = _round(
            _appearance_similarity_threshold(config=config, gap_frames=int(decision["gap_frames"]))
        )
        decision["appearance_supported"] = False
        decision["appearance_status"] = f"lost_tail_{tail_status}"
        return

    similarity = runtime["cosine_similarity"](tail_embedding, head_embedding)
    threshold = _appearance_similarity_threshold(config=config, gap_frames=int(decision["gap_frames"]))
    decision["appearance_similarity"] = _round(similarity)
    decision["appearance_threshold"] = _round(threshold)
    decision["appearance_supported"] = similarity is not None and (threshold is None or float(similarity) >= float(threshold))
    decision["appearance_status"] = "ready"


def _finalize_lost_embedding(
    state: dict[str, Any],
    runtime: dict[str, Any] | None,
    config: StitchConfig,
) -> dict[str, Any]:
    if runtime is None:
        state["tail_embedding"] = None
        state["tail_embedding_status"] = "not_available"
        return state
    embedding, status = _extract_embedding(
        runtime,
        list(state["tail_observations"]),
        config,
        prefer_tail=True,
    )
    state["tail_embedding"] = embedding
    state["tail_embedding_status"] = status
    return state


def _canonical_merge_meta(canonical_merge_state: dict[int, dict[str, Any]], canonical_track_id: int) -> dict[str, Any]:
    return canonical_merge_state.setdefault(
        int(canonical_track_id),
        {
            "absorbed_count": 0,
            "last_merge_frame": None,
        },
    )


def _apply_chain_control(
    *,
    decision: dict[str, Any],
    config: StitchConfig,
    canonical_merge_state: dict[int, dict[str, Any]],
    winner_margin_threshold: float | None,
) -> bool:
    decision["chain_control_checked"] = False

    if not bool(config.matching.chain_control_enabled):
        return False

    canonical_track_id = int(decision["candidate_canonical_track_id"])
    merge_meta = canonical_merge_state.get(canonical_track_id)
    if merge_meta is None:
        return False

    absorbed_count = int(merge_meta.get("absorbed_count") or 0)
    if absorbed_count <= 0:
        return False

    candidate_source_track_id = int(decision["candidate_source_track_id"])
    candidate_is_chained = candidate_source_track_id != canonical_track_id
    last_merge_frame = merge_meta.get("last_merge_frame")
    recent_merge_gap = None
    if last_merge_frame is not None:
        recent_merge_gap = max(0, int(decision["decision_frame"]) - int(last_merge_frame))
    recent_merge = recent_merge_gap is not None and recent_merge_gap <= int(config.matching.chain_control_recent_merge_window)
    if not (candidate_is_chained or recent_merge):
        return False

    decision["chain_control_checked"] = True
    decision["chain_control_absorbed_count"] = absorbed_count
    decision["chain_control_candidate_is_chained"] = candidate_is_chained
    decision["chain_control_recent_merge_gap"] = recent_merge_gap

    required_score = float(config.matching.min_match_score) + float(config.matching.chain_control_score_bonus)
    if candidate_is_chained:
        required_score += float(config.matching.chain_control_chained_score_bonus)
    if absorbed_count > 1:
        required_score += float(config.matching.chain_control_repeated_score_bonus) * float(absorbed_count - 1)
    decision["chain_control_required_score"] = _round(required_score)
    if float(decision.get("score") or 0.0) < required_score:
        decision["reason"] = (
            f"chain_control_score:{float(decision.get('score') or 0.0):.3f}"
            f"<{required_score:.3f}"
        )
        return True

    if candidate_is_chained and winner_margin_threshold is not None and decision.get("winner_margin") is not None:
        required_margin = float(winner_margin_threshold) + float(config.matching.chain_control_margin_bonus)
        decision["chain_control_required_margin"] = _round(required_margin)
        if float(decision.get("winner_margin") or 0.0) < required_margin:
            decision["reason"] = (
                f"chain_control_margin:{float(decision.get('winner_margin') or 0.0):.3f}"
                f"<{required_margin:.3f}"
            )
            return True

    min_appearance_similarity = config.matching.chain_control_min_appearance_similarity
    if candidate_is_chained and bool(decision.get("appearance_assisted")) and min_appearance_similarity is not None:
        decision["chain_control_required_appearance_similarity"] = _round(min_appearance_similarity)
        if float(decision.get("appearance_similarity") or 0.0) < float(min_appearance_similarity):
            decision["reason"] = (
                f"chain_control_appearance:{float(decision.get('appearance_similarity') or 0.0):.3f}"
                f"<{float(min_appearance_similarity):.3f}"
            )
            return True

    return False


def _apply_short_gap_gate(
    *,
    decision: dict[str, Any],
    config: StitchConfig,
    canonical_merge_state: dict[int, dict[str, Any]],
) -> bool:
    decision["short_gap_gate_checked"] = False

    if not bool(config.matching.short_gap_gate_enabled):
        return False
    if bool(decision.get("appearance_assisted")):
        return False

    gap_frames = int(decision["gap_frames"])
    if gap_frames > int(config.matching.short_gap_gate_max_gap_frames):
        return False

    canonical_track_id = int(decision["candidate_canonical_track_id"])
    merge_meta = canonical_merge_state.get(canonical_track_id)
    absorbed_count = int((merge_meta or {}).get("absorbed_count") or 0)
    if absorbed_count != 0:
        return False

    decision["short_gap_gate_checked"] = True
    decision["short_gap_gate_required_score"] = _round(config.matching.short_gap_gate_min_score)
    decision["short_gap_gate_required_margin"] = _round(config.matching.short_gap_gate_min_margin)

    if float(decision.get("score") or 0.0) < float(config.matching.short_gap_gate_min_score):
        decision["reason"] = (
            f"short_gap_gate_score:{float(decision.get('score') or 0.0):.3f}"
            f"<{float(config.matching.short_gap_gate_min_score):.3f}"
        )
        return True

    winner_margin = decision.get("winner_margin")
    if winner_margin is not None and float(winner_margin) < float(config.matching.short_gap_gate_min_margin):
        decision["reason"] = (
            f"short_gap_gate_margin:{float(winner_margin):.3f}"
            f"<{float(config.matching.short_gap_gate_min_margin):.3f}"
        )
        return True

    return False


def _decision_latency_summary(decisions: list[dict[str, Any]]) -> dict[str, float | int | None]:
    latencies = [int(decision["decision_latency_frames"]) for decision in decisions]
    accepted_latencies = [int(decision["decision_latency_frames"]) for decision in decisions if bool(decision.get("applied"))]
    return {
        "decision_latency_frames_mean": None if not latencies else round(sum(latencies) / len(latencies), 6),
        "decision_latency_frames_max": None if not latencies else max(latencies),
        "accepted_remap_latency_frames_mean": None if not accepted_latencies else round(sum(accepted_latencies) / len(accepted_latencies), 6),
        "accepted_remap_latency_frames_max": None if not accepted_latencies else max(accepted_latencies),
    }


def _sequence_summary(sequence_report: dict[str, Any]) -> dict[str, Any]:
    latency_summary = _decision_latency_summary(list(sequence_report["decisions"]))
    appearance_supported_remaps = [
        f"{int(decision['new_source_track_id'])}->{int(decision['candidate_source_track_id'])}"
        for decision in sequence_report["decisions"]
        if bool(decision.get("applied")) and bool(decision.get("appearance_assisted"))
    ]
    return {
        "num_rows_in": int(sequence_report["num_rows_in"]),
        "num_rows_out": int(sequence_report["num_rows_out"]),
        "source_mot_path": sequence_report.get("source_mot_path") or sequence_report.get("input_mot_path"),
        "source_video_path": sequence_report.get("source_video_path"),
        "num_unique_track_ids_in": len(sequence_report["unique_track_ids_in"]),
        "num_unique_track_ids_out": len(sequence_report["unique_track_ids_out"]),
        "unique_ids_before": len(sequence_report["unique_track_ids_in"]),
        "unique_ids_after": len(sequence_report["unique_track_ids_out"]),
        "accepted_remaps": list(sequence_report["accepted_remaps"]),
        "accepted_remap_count": int(sequence_report["accepted_remap_count"]),
        "rejected_candidate_count": int(sequence_report["rejected_candidate_count"]),
        "candidate_evaluation_count": int(sequence_report["candidate_evaluation_count"]),
        "fresh_canonical_ids_assigned": int(sequence_report["fresh_canonical_ids_assigned"]),
        "fresh_id_count": int(sequence_report["fresh_canonical_ids_assigned"]),
        "below_score_threshold_count": int(sequence_report["below_score_threshold_count"]),
        "margin_rejected_count": int(sequence_report["margin_rejected_count"]),
        "chain_control_checked_count": int(sequence_report["chain_control_checked_count"]),
        "chain_control_rejected_count": int(sequence_report["chain_control_rejected_count"]),
        "short_gap_gate_checked_count": int(sequence_report["short_gap_gate_checked_count"]),
        "short_gap_gate_rejected_count": int(sequence_report["short_gap_gate_rejected_count"]),
        "appearance_supported_remaps": appearance_supported_remaps,
        "appearance_supported_remap_count": int(sequence_report["appearance_supported_remap_count"]),
        "appearance_rejection_count": int(sequence_report["appearance_rejected_count"]),
        "appearance_rejected_count": int(sequence_report["appearance_rejected_count"]),
        "decision_latency_mean": latency_summary["decision_latency_frames_mean"],
        "decision_latency_max": latency_summary["decision_latency_frames_max"],
        **latency_summary,
        **_colreg_scoring_summary(list(sequence_report["decisions"])),
    }


def _augment_run_summary(
    input_summary: dict[str, Any] | None,
    output_root: Path,
    mot_dir: Path,
    report_path: Path,
    config: StitchConfig,
    sequence_reports: dict[str, dict[str, Any]],
    pred_root: Path,
    input_run_summary_path: Path | None,
    mode: str,
    confirmation_observations: int,
    config_path: str | None,
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
        seq_summary["live_reid_matches_applied"] = int(report["matches_applied"])
        seq_summary["live_reid_accepted_remap_count"] = int(report["accepted_remap_count"])
        seq_summary["live_reid_rejected_candidate_count"] = int(report["rejected_candidate_count"])
        seq_summary["live_reid_fresh_canonical_ids_assigned"] = int(report["fresh_canonical_ids_assigned"])
        seq_summary["live_reid_below_score_threshold_count"] = int(report["below_score_threshold_count"])
        seq_summary["live_reid_margin_rejected_count"] = int(report["margin_rejected_count"])
        seq_summary["live_reid_chain_control_checked_count"] = int(report["chain_control_checked_count"])
        seq_summary["live_reid_chain_control_rejected_count"] = int(report["chain_control_rejected_count"])
        seq_summary["live_reid_short_gap_gate_checked_count"] = int(report["short_gap_gate_checked_count"])
        seq_summary["live_reid_short_gap_gate_rejected_count"] = int(report["short_gap_gate_rejected_count"])
        seq_summary["live_reid_appearance_supported_remap_count"] = int(report["appearance_supported_remap_count"])
        seq_summary["live_reid_appearance_rejected_count"] = int(report["appearance_rejected_count"])
        sequences[sequence_name] = seq_summary
        total_matches_applied += int(report["matches_applied"])
    summary["sequences"] = sequences
    summary["live_reid_summary"] = {
        "mode": mode,
        "confirmation_observations": int(confirmation_observations),
        "config_path": config_path,
        "num_sequences_processed": len(sequence_reports),
        "accepted_remap_count": sum(int(report["accepted_remap_count"]) for report in sequence_reports.values()),
        "rejected_candidate_count": sum(int(report["rejected_candidate_count"]) for report in sequence_reports.values()),
        "candidate_evaluation_count": sum(int(report["candidate_evaluation_count"]) for report in sequence_reports.values()),
        "fresh_canonical_ids_assigned": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "fresh_id_count": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "below_score_threshold_count": sum(int(report["below_score_threshold_count"]) for report in sequence_reports.values()),
        "margin_rejected_count": sum(int(report["margin_rejected_count"]) for report in sequence_reports.values()),
        "chain_control_checked_count": sum(int(report["chain_control_checked_count"]) for report in sequence_reports.values()),
        "chain_control_rejected_count": sum(int(report["chain_control_rejected_count"]) for report in sequence_reports.values()),
        "short_gap_gate_checked_count": sum(int(report["short_gap_gate_checked_count"]) for report in sequence_reports.values()),
        "short_gap_gate_rejected_count": sum(int(report["short_gap_gate_rejected_count"]) for report in sequence_reports.values()),
        "appearance_supported_remap_count": sum(int(report["appearance_supported_remap_count"]) for report in sequence_reports.values()),
        "appearance_rejection_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
        "appearance_rejected_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
    }
    summary["stitching"] = {
        "mode": mode,
        "stitch_name": output_root.name,
        "input_pred_dir": str(pred_root),
        "input_run_summary_path": str(input_run_summary_path) if input_run_summary_path is not None else None,
        "report_path": str(report_path),
        "config_path": config_path,
        "config": config.to_dict(),
        "live_reid_mode": True,
        "total_matches_applied": total_matches_applied,
        "confirmation_observations": int(confirmation_observations),
    }
    return summary


def _live_sequence(
    sequence_name: str,
    rows: list[dict[str, Any]],
    config: StitchConfig,
    confirmation_observations: int,
    appearance_runtime: dict[str, Any] | None,
    ais_assignments: dict[int, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_rows.setdefault(int(row["frame"]), []).append(row)

    head_window = max(1, int(confirmation_observations))
    tail_window = max(2, int(confirmation_observations))
    active_tracks: dict[int, dict[str, Any]] = {}
    lost_tracks: dict[int, dict[str, Any]] = {}
    pending_tracks: dict[int, dict[str, Any]] = {}
    source_to_canonical: dict[int, int] = {}
    used_canonical_ids: set[int] = set()
    canonical_merge_state: dict[int, dict[str, Any]] = {}
    next_canonical_id = _initial_next_canonical_id(rows)

    output_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    accepted_remaps: list[str] = []
    matches_applied = 0
    fresh_canonical_ids_assigned = 0
    chain_control_checked_count = 0
    chain_control_rejected_count = 0
    appearance_supported_remap_count = 0
    appearance_rejected_count = 0

    def finalize_pending(source_track_id: int, decision_frame: int) -> None:
        nonlocal next_canonical_id, matches_applied, fresh_canonical_ids_assigned
        nonlocal chain_control_checked_count, chain_control_rejected_count
        nonlocal appearance_supported_remap_count, appearance_rejected_count

        state = pending_tracks.pop(source_track_id, None)
        if state is None:
            return

        preview_observation_limit = None
        if config.appearance.enabled and config.appearance.use_for_matching:
            preview_observation_limit = max(1, int(config.appearance.sample_frames))
        pending_full = _pending_summary(state, match_observation_limit=None)
        pending_preview = _pending_summary(state, match_observation_limit=preview_observation_limit)
        candidate_decisions: list[dict[str, Any]] = []
        for lost_state in lost_tracks.values():
            pending_for_candidate = (
                pending_preview
                if _use_preview_pending_summary(lost_state=lost_state, pending=pending_full, config=config)
                else pending_full
            )
            decision = _evaluate_live_candidate(
                lost_state=lost_state,
                pending=pending_for_candidate,
                config=config,
                decision_frame=decision_frame,
            )
            _annotate_live_appearance(
                decision=decision,
                lost_state=lost_state,
                pending=pending_preview,
                runtime=appearance_runtime,
                config=config,
            )
            _annotate_ais_diagnostics(decision, config=config, ais_assignments=ais_assignments)
            candidate_decisions.append(decision)

        decisions.extend(candidate_decisions)

        accepted = False
        canonical_track_id = None
        if candidate_decisions:
            eligible = [decision for decision in candidate_decisions if bool(decision["gating"]["passes_all"])]
            if eligible:
                eligible = _rank_eligible_candidates(eligible, config)
                best = eligible[0]
                runner_up = eligible[1] if len(eligible) > 1 else None
                if float(best["score"] or 0.0) < float(config.matching.min_match_score):
                    best["reason"] = (
                        f"below_score_threshold:{float(best['score'] or 0.0):.3f}"
                        f"<{float(config.matching.min_match_score):.3f}"
                    )
                else:
                    winner_margin_threshold = None
                    if runner_up is not None:
                        winner_margin = round(float(best["score"] or 0.0) - float(runner_up["score"] or 0.0), 6)
                        winner_margin_threshold = _winner_margin_threshold(config=config, gap_frames=int(best["gap_frames"]))
                        best["winner_margin"] = _round(winner_margin)
                        best["winner_margin_threshold"] = _round(winner_margin_threshold)
                        if winner_margin < winner_margin_threshold:
                            if bool(best.get("colreg_score_used")) and bool(best.get("colreg_changed_ranking")):
                                best["reason"] = "eligible"
                            else:
                                best["reason"] = f"margin_too_small:{winner_margin:.3f}<{winner_margin_threshold:.3f}"
                            if best["reason"] != "eligible" and appearance_runtime is not None:
                                ambiguous_candidates = [
                                    decision
                                    for decision in eligible
                                    if (float(best["score"] or 0.0) - float(decision["score"] or 0.0)) < winner_margin_threshold
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
                                    appearance_runner_up = appearance_candidates[1] if len(appearance_candidates) > 1 else None
                                    appearance_margin_threshold = _appearance_margin_threshold(config)
                                    appearance_margin = None
                                    if appearance_runner_up is not None:
                                        appearance_margin = round(
                                            float(appearance_best.get("appearance_similarity") or 0.0)
                                            - float(appearance_runner_up.get("appearance_similarity") or 0.0),
                                            6,
                                        )
                                    appearance_best["appearance_margin_threshold"] = _round(appearance_margin_threshold)
                                    appearance_best["appearance_margin"] = _round(appearance_margin)
                                    if appearance_margin is None or appearance_margin >= appearance_margin_threshold:
                                        appearance_best["appearance_assisted"] = True
                                        best = appearance_best
                                    else:
                                        appearance_rejected_count += 1
                                        best["appearance_margin_threshold"] = _round(appearance_margin_threshold)
                                        best["appearance_margin"] = _round(appearance_margin)
                                        best["reason"] = (
                                            f"appearance_margin_too_small:{appearance_margin:.3f}"
                                            f"<{appearance_margin_threshold:.3f}"
                                        )
                                else:
                                    appearance_rejected_count += 1

                    if best["reason"] == "eligible":
                        chain_control_rejected = _apply_chain_control(
                            decision=best,
                            config=config,
                            canonical_merge_state=canonical_merge_state,
                            winner_margin_threshold=winner_margin_threshold,
                        )
                        if bool(best.get("chain_control_checked")):
                            chain_control_checked_count += 1
                        if chain_control_rejected:
                            chain_control_rejected_count += 1
                    if best["reason"] == "eligible":
                        _apply_short_gap_gate(
                            decision=best,
                            config=config,
                            canonical_merge_state=canonical_merge_state,
                        )
                    if best["reason"] == "eligible":
                        best["applied"] = True
                        best["reason"] = "accepted"
                        if bool(best.get("appearance_assisted")):
                            best["reason"] = "accepted_with_appearance_fallback"
                            appearance_supported_remap_count += 1
                        _mark_colreg_acceptance(best)
                        canonical_track_id = int(best["candidate_canonical_track_id"])
                        merge_meta = _canonical_merge_meta(canonical_merge_state, canonical_track_id)
                        merge_meta["absorbed_count"] = int(merge_meta.get("absorbed_count") or 0) + 1
                        merge_meta["last_merge_frame"] = int(decision_frame)
                        source_to_canonical[source_track_id] = canonical_track_id
                        used_canonical_ids.add(canonical_track_id)
                        matches_applied += 1
                        accepted_remaps.append(f"{source_track_id}->{int(best['candidate_source_track_id'])}")
                        lost_tracks.pop(int(best["candidate_source_track_id"]), None)
                        accepted = True
                        for decision in eligible[1:]:
                            if decision["reason"] == "eligible":
                                if bool(best.get("appearance_assisted")):
                                    decision["reason"] = (
                                        f"outscored_by_appearance:{source_track_id}->{int(best['candidate_source_track_id'])}"
                                    )
                                else:
                                    decision["reason"] = f"outscored_by:{source_track_id}->{int(best['candidate_source_track_id'])}"

        if not accepted:
            canonical_track_id, next_canonical_id = _allocate_canonical_id(
                source_track_id=source_track_id,
                used_canonical_ids=used_canonical_ids,
                next_canonical_id=next_canonical_id,
            )
            source_to_canonical[source_track_id] = canonical_track_id
            _canonical_merge_meta(canonical_merge_state, canonical_track_id)
            fresh_canonical_ids_assigned += 1

        remapped_rows = []
        for buffered_row in state["rows"]:
            remapped_row = dict(buffered_row)
            remapped_row["id"] = canonical_track_id
            remapped_rows.append(remapped_row)
        output_rows.extend(remapped_rows)
        active_tracks[source_track_id] = _make_confirmed_state(
            source_track_id=source_track_id,
            canonical_track_id=canonical_track_id,
            observations=state["observations"],
            head_window=head_window,
            tail_window=tail_window,
        )

    sorted_frames = sorted(frame_rows)
    for frame in sorted_frames:
        rows_now = sorted(frame_rows[frame], key=lambda row: (int(row["id"]), float(row["bbox"][0]), float(row["bbox"][1])))
        current_source_ids = {int(row["id"]) for row in rows_now}

        pending_to_finalize = [source_id for source_id in list(pending_tracks) if source_id not in current_source_ids]
        for source_id in pending_to_finalize:
            finalize_pending(source_id, decision_frame=int(frame))

        disappeared_source_ids = [source_id for source_id in list(active_tracks) if source_id not in current_source_ids]
        for source_id in disappeared_source_ids:
            lost_tracks[source_id] = _finalize_lost_embedding(active_tracks.pop(source_id), runtime=appearance_runtime, config=config)

        expired_source_ids = [
            source_id
            for source_id, state in lost_tracks.items()
            if max(0, int(frame) - int(state["last_frame"]) - 1) > int(config.memory.max_frame_gap)
        ]
        for source_id in expired_source_ids:
            lost_tracks.pop(source_id, None)

        for row in rows_now:
            source_track_id = int(row["id"])
            if source_track_id in active_tracks:
                canonical_track_id = int(active_tracks[source_track_id]["canonical_track_id"])
                active_tracks[source_track_id] = _update_confirmed_state(
                    active_tracks[source_track_id],
                    row=row,
                    head_window=head_window,
                    tail_window=tail_window,
                )
                remapped_row = dict(row)
                remapped_row["id"] = canonical_track_id
                output_rows.append(remapped_row)
                continue

            if source_track_id in source_to_canonical:
                canonical_track_id = int(source_to_canonical[source_track_id])
                remapped_row = dict(row)
                remapped_row["id"] = canonical_track_id
                output_rows.append(remapped_row)
                lost_tracks.pop(source_track_id, None)
                active_tracks[source_track_id] = _make_confirmed_state(
                    source_track_id=source_track_id,
                    canonical_track_id=canonical_track_id,
                    observations=[_build_observation(remapped_row)],
                    head_window=head_window,
                    tail_window=tail_window,
                )
                continue

            if source_track_id not in pending_tracks:
                pending_tracks[source_track_id] = _make_pending_state(source_track_id, row)
            else:
                pending_tracks[source_track_id] = _update_pending_state(pending_tracks[source_track_id], row)

            if len(pending_tracks[source_track_id]["observations"]) >= confirmation_observations:
                finalize_pending(source_track_id, decision_frame=int(frame))

    final_frame = max(sorted_frames) if sorted_frames else 0
    for source_id in list(pending_tracks):
        finalize_pending(source_id, decision_frame=int(final_frame))

    output_rows.sort(key=lambda row: (int(row["frame"]), int(row["id"]), float(row["bbox"][0]), float(row["bbox"][1])))
    accepted_remap_count = len(accepted_remaps)
    rejected_candidate_count = sum(1 for decision in decisions if not bool(decision.get("applied")))
    candidate_evaluation_count = len(decisions)
    below_score_threshold_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("below_score_threshold:")
    )
    margin_rejected_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("margin_too_small:")
    )
    chain_control_checked_count = sum(1 for decision in decisions if bool(decision.get("chain_control_checked")))
    chain_control_rejected_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("chain_control_")
    )
    short_gap_gate_checked_count = sum(1 for decision in decisions if bool(decision.get("short_gap_gate_checked")))
    short_gap_gate_rejected_count = sum(
        1 for decision in decisions if str(decision.get("reason") or "").startswith("short_gap_gate_")
    )
    latency_summary = _decision_latency_summary(decisions)
    sequence_report = {
        "sequence_name": sequence_name,
        "num_rows_in": len(rows),
        "num_rows_out": len(output_rows),
        "unique_track_ids_in": sorted({int(row["id"]) for row in rows}),
        "unique_track_ids_out": sorted({int(row["id"]) for row in output_rows}),
        "identity_map": {str(source_id): int(canonical_id) for source_id, canonical_id in sorted(source_to_canonical.items())},
        "matches_applied": matches_applied,
        "accepted_remap_count": accepted_remap_count,
        "rejected_candidate_count": rejected_candidate_count,
        "candidate_evaluation_count": candidate_evaluation_count,
        "fresh_canonical_ids_assigned": fresh_canonical_ids_assigned,
        "below_score_threshold_count": below_score_threshold_count,
        "margin_rejected_count": margin_rejected_count,
        "chain_control_checked_count": chain_control_checked_count,
        "chain_control_rejected_count": chain_control_rejected_count,
        "short_gap_gate_checked_count": short_gap_gate_checked_count,
        "short_gap_gate_rejected_count": short_gap_gate_rejected_count,
        "appearance_supported_remap_count": appearance_supported_remap_count,
        "appearance_rejected_count": appearance_rejected_count,
        "accepted_remaps": accepted_remaps,
        "decisions": decisions,
        "appearance": {} if appearance_runtime is None else dict(appearance_runtime["info"]),
        "confirmation_observations": int(confirmation_observations),
        "decision_latency_mean": latency_summary["decision_latency_frames_mean"],
        "decision_latency_max": latency_summary["decision_latency_frames_max"],
        **latency_summary,
        **_colreg_scoring_summary(decisions),
    }
    return output_rows, sequence_report


def live_reid_tracks(
    pred_dir: str,
    output_dir: str | None = None,
    run_summary_path: str | None = None,
    config_path: str | None = None,
    live_name: str | None = None,
    confirmation_observations: int = 10,
    safe_mode: bool = True,
    colreg_diagnostics: bool = False,
    colreg_scoring_experiment: bool = False,
    ais_diagnostics: bool = False,
    ais_file: str | None = None,
    ais_video_start_time: str | None = None,
    ais_affine_matrix: str | list[list[float]] | None = None,
    ais_fps: float | None = None,
) -> dict[str, Any]:
    pred_root = Path(pred_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    confirmation_observations = max(1, int(confirmation_observations))
    config = _load_stitch_config(config_path or str(DEFAULT_CONFIG_PATH))
    if safe_mode:
        config.matching.chain_control_enabled = True
        config.matching.short_gap_gate_enabled = True
    _configure_colreg_options(
        config,
        diagnostics_enabled=colreg_diagnostics,
        scoring_experiment=colreg_scoring_experiment,
    )
    _configure_ais_options(
        config,
        diagnostics_enabled=ais_diagnostics,
        ais_file=ais_file,
        video_start_time=ais_video_start_time,
        affine_matrix=ais_affine_matrix,
        fps=ais_fps,
    )
    live_name = live_name or _default_live_name(config)
    output_root = Path(output_dir) if output_dir else pred_root.parent / "live_reid" / live_name
    mot_dir = output_root / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    input_run_summary_path = _resolve_run_summary_path(pred_root, run_summary_path)
    input_run_summary = _load_run_summary(input_run_summary_path)
    source_path = None if input_run_summary is None else input_run_summary.get("source")

    mot_files = sorted(pred_root.glob("*.txt"))
    if not mot_files:
        raise RuntimeError(f"No MOT txt files found in {pred_root}")

    warnings: list[str] = []
    if input_run_summary is None:
        warnings.append("Input run_summary.json was not found; live ReID run summary was created from MOT outputs only.")

    sequence_reports: dict[str, dict[str, Any]] = {}
    total_wall_time = 0.0
    appearance_ready_any = False
    for mot_path in mot_files:
        sequence_name = mot_path.stem
        rows = load_mot_rows(mot_path)
        appearance_runtime, appearance_warnings = _build_appearance_runtime(sequence_name, source_path, config)
        warnings.extend(appearance_warnings)
        if appearance_runtime is not None:
            appearance_ready_any = True
        ais_assignments, ais_warnings = _build_ais_assignments(sequence_name=sequence_name, rows=rows, config=config)
        warnings.extend(ais_warnings)
        started = time.perf_counter()
        try:
            output_rows, sequence_report = _live_sequence(
                sequence_name=sequence_name,
                rows=rows,
                config=config,
                confirmation_observations=confirmation_observations,
                appearance_runtime=appearance_runtime,
                ais_assignments=ais_assignments,
            )
        finally:
            _close_appearance_runtime(appearance_runtime)
        wall_time_seconds = round(time.perf_counter() - started, 6)
        total_wall_time += wall_time_seconds

        output_mot_path = mot_dir / mot_path.name
        write_mot_rows(output_mot_path, output_rows)
        resolved_source_video = _resolve_source_video_path(source_path, sequence_name)
        sequence_report["source_mot_path"] = str(mot_path)
        sequence_report["source_video_path"] = None if resolved_source_video is None else str(resolved_source_video)
        sequence_report["input_mot_path"] = str(mot_path)
        sequence_report["output_mot_path"] = str(output_mot_path)
        sequence_report["wall_time_seconds"] = wall_time_seconds
        if config.ais.diagnostics_enabled:
            sequence_report["ais"] = {
                "diagnostics_enabled": True,
                "ais_file": config.ais.ais_file,
                "assignments": {
                    str(track_id): assignment.to_dict()
                    for track_id, assignment in sorted(ais_assignments.items())
                },
            }
        sequence_report["summary"] = _sequence_summary(sequence_report)
        sequence_reports[sequence_name] = sequence_report

    mode = _live_mode(config, appearance_ready_any)
    report_path = output_root / "live_reid_report.json"
    run_summary_payload = _augment_run_summary(
        input_summary=input_run_summary,
        output_root=output_root,
        mot_dir=mot_dir,
        report_path=report_path,
        config=config,
        sequence_reports=sequence_reports,
        pred_root=pred_root,
        input_run_summary_path=input_run_summary_path,
        mode=mode,
        confirmation_observations=confirmation_observations,
        config_path=str(config_path or DEFAULT_CONFIG_PATH),
    )
    run_summary_path_out = output_root / "run_summary.json"
    run_summary_path_out.write_text(json.dumps(run_summary_payload, indent=2), encoding="utf-8")

    all_decisions = [decision for report in sequence_reports.values() for decision in report["decisions"]]
    all_accepted_remaps = [
        remap
        for report in sequence_reports.values()
        for remap in report["accepted_remaps"]
    ]
    all_appearance_supported_remaps = [
        f"{int(decision['new_source_track_id'])}->{int(decision['candidate_source_track_id'])}"
        for decision in all_decisions
        if bool(decision.get("applied")) and bool(decision.get("appearance_assisted"))
    ]
    latency_summary = _decision_latency_summary(all_decisions)
    summary_payload = {
        "num_sequences_processed": len(sequence_reports),
        "source_mot_dir": str(pred_root),
        "source_video_path": source_path,
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "accepted_remaps": all_accepted_remaps,
        "accepted_remap_count": sum(int(report["accepted_remap_count"]) for report in sequence_reports.values()),
        "rejected_candidate_count": sum(int(report["rejected_candidate_count"]) for report in sequence_reports.values()),
        "candidate_evaluation_count": sum(int(report["candidate_evaluation_count"]) for report in sequence_reports.values()),
        "fresh_canonical_ids_assigned": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "fresh_id_count": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "unique_ids_before": {
            sequence_name: len(report["unique_track_ids_in"]) for sequence_name, report in sequence_reports.items()
        },
        "unique_ids_after": {
            sequence_name: len(report["unique_track_ids_out"]) for sequence_name, report in sequence_reports.items()
        },
        "below_score_threshold_count": sum(int(report["below_score_threshold_count"]) for report in sequence_reports.values()),
        "margin_rejected_count": sum(int(report["margin_rejected_count"]) for report in sequence_reports.values()),
        "chain_control_checked_count": sum(int(report["chain_control_checked_count"]) for report in sequence_reports.values()),
        "chain_control_rejected_count": sum(int(report["chain_control_rejected_count"]) for report in sequence_reports.values()),
        "short_gap_gate_checked_count": sum(int(report["short_gap_gate_checked_count"]) for report in sequence_reports.values()),
        "short_gap_gate_rejected_count": sum(int(report["short_gap_gate_rejected_count"]) for report in sequence_reports.values()),
        "appearance_supported_remaps": all_appearance_supported_remaps,
        "appearance_supported_remap_count": sum(int(report["appearance_supported_remap_count"]) for report in sequence_reports.values()),
        "appearance_rejection_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
        "appearance_rejected_count": sum(int(report["appearance_rejected_count"]) for report in sequence_reports.values()),
        "confirmation_observations": int(confirmation_observations),
        "safe_mode": bool(safe_mode),
        "wall_time_seconds": round(total_wall_time, 6),
        "decision_latency_mean": latency_summary["decision_latency_frames_mean"],
        "decision_latency_max": latency_summary["decision_latency_frames_max"],
        **latency_summary,
        **_colreg_scoring_summary(all_decisions),
    }
    report_payload = {
        "version": LIVE_REID_VERSION,
        "mode": mode,
        "pred_dir": str(pred_root),
        "source_mot_dir": str(pred_root),
        "source_video_path": source_path,
        "output_root": str(output_root),
        "input_run_summary_path": str(input_run_summary_path) if input_run_summary_path is not None else None,
        "output_run_summary_path": str(run_summary_path_out),
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "config": config.to_dict(),
        "safe_mode": bool(safe_mode),
        "total_sequences": len(sequence_reports),
        "total_matches_applied": sum(int(report["matches_applied"]) for report in sequence_reports.values()),
        "summary": summary_payload,
        "sequences": sequence_reports,
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return {
        "mode": mode,
        "pred_dir": str(pred_root),
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "run_summary": str(run_summary_path_out),
        "live_reid_report": str(report_path),
        "num_sequences": len(sequence_reports),
        "matches_applied": report_payload["total_matches_applied"],
        "accepted_remaps": summary_payload["accepted_remaps"],
        "accepted_remap_count": summary_payload["accepted_remap_count"],
        "rejected_candidate_count": summary_payload["rejected_candidate_count"],
        "fresh_canonical_ids_assigned": summary_payload["fresh_canonical_ids_assigned"],
        "fresh_id_count": summary_payload["fresh_id_count"],
        "chain_control_checked_count": summary_payload["chain_control_checked_count"],
        "chain_control_rejected_count": summary_payload["chain_control_rejected_count"],
        "short_gap_gate_checked_count": summary_payload["short_gap_gate_checked_count"],
        "short_gap_gate_rejected_count": summary_payload["short_gap_gate_rejected_count"],
        "appearance_supported_remaps": summary_payload["appearance_supported_remaps"],
        "appearance_supported_remap_count": summary_payload["appearance_supported_remap_count"],
        "appearance_rejection_count": summary_payload["appearance_rejection_count"],
        "appearance_rejected_count": summary_payload["appearance_rejected_count"],
        "confirmation_observations": int(confirmation_observations),
        "safe_mode": bool(safe_mode),
        "config_path": summary_payload["config_path"],
        "source_mot_dir": summary_payload["source_mot_dir"],
        "source_video_path": summary_payload["source_video_path"],
        "decision_latency_mean": summary_payload["decision_latency_mean"],
        "decision_latency_max": summary_payload["decision_latency_max"],
        "decision_latency_frames_mean": summary_payload["decision_latency_frames_mean"],
        "decision_latency_frames_max": summary_payload["decision_latency_frames_max"],
    }
