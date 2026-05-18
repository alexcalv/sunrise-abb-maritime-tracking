from __future__ import annotations

import math

from ais.schemas import AIS_NEUTRAL_SCORE, AisAlignedClip
from stitching.appearance import TrackletAppearanceEmbeddings, cosine_similarity
from stitching.schemas import StitchConfig, StitchDecision, Tracklet


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _distance(a: list[float], b: list[float]) -> float:
    return math.dist([float(a[0]), float(a[1])], [float(b[0]), float(b[1])])


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_list(values: list[float], digits: int = 6) -> list[float]:
    return [_round(value, digits) for value in values]


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


def _threshold_score(value: float, max_value: float | None) -> float:
    if max_value is None or max_value <= 0:
        return 1.0
    return _clamp01(1.0 - (value / max_value))


def _weighted_score(component_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0:
        return 0.0
    return _round(
        sum(component_scores[name] * float(weight) for name, weight in weights.items()) / total_weight
    )


def _aggregated_score(component_scores: dict[str, float], weights: dict[str, float], names: list[str]) -> float | None:
    selected_weights = {name: float(weights.get(name, 0.0)) for name in names if name in component_scores}
    if sum(selected_weights.values()) <= 0:
        return None
    return _weighted_score(component_scores, selected_weights)


def _candidate_weights(
    config: StitchConfig,
    include_appearance: bool,
    include_ais: bool,
) -> dict[str, float]:
    weights = config.matching.weights.to_dict()
    if not include_appearance:
        weights["appearance"] = 0.0
    if not include_ais:
        weights["ais_position"] = 0.0
        weights["ais_identity"] = 0.0
    return weights


def _ais_enabled_in_score(config: StitchConfig) -> bool:
    if not config.ais.enabled:
        return False
    weights = config.matching.weights.to_dict()
    return float(weights.get("ais_position", 0.0)) > 0 or float(weights.get("ais_identity", 0.0)) > 0


def _appearance_threshold(config: StitchConfig) -> float | None:
    threshold = config.matching.min_appearance_similarity
    if threshold is None:
        return None
    return float(threshold)


def _uses_conditional_gap_policy(config: StitchConfig) -> bool:
    matching = config.matching
    return any(
        [
            matching.short_gap_frames_threshold is not None,
            matching.short_gap_min_appearance_similarity is not None,
            matching.long_gap_min_appearance_similarity is not None,
            matching.require_appearance_for_short_gap,
            matching.winner_margin_threshold_short is not None,
            matching.winner_margin_threshold_long is not None,
        ]
    )


def _gap_bucket(gap_frames: int, config: StitchConfig) -> str:
    short_gap_threshold = config.matching.short_gap_frames_threshold
    long_gap_threshold = config.matching.long_gap_frames_threshold
    if short_gap_threshold is not None and gap_frames <= int(short_gap_threshold):
        return "short"
    if long_gap_threshold is not None and gap_frames > int(long_gap_threshold):
        return "long"
    return "medium"


def _appearance_supports_threshold(
    appearance_similarity: float | None,
    threshold: float | None,
) -> bool:
    if threshold is None:
        return True
    return appearance_similarity is not None and float(appearance_similarity) >= float(threshold)


def _appearance_policy(
    config: StitchConfig,
    gap_frames: int,
) -> tuple[str, float | None, bool]:
    gap_bucket = _gap_bucket(gap_frames, config)
    default_threshold = _appearance_threshold(config)

    if not _uses_conditional_gap_policy(config):
        required = bool(config.appearance.use_for_matching and default_threshold is not None)
        return gap_bucket, default_threshold, required

    if gap_bucket == "short":
        threshold = config.matching.short_gap_min_appearance_similarity
        if threshold is None:
            threshold = default_threshold
        required = bool(
            config.appearance.use_for_matching
            and config.matching.require_appearance_for_short_gap
            and threshold is not None
        )
        return gap_bucket, None if threshold is None else float(threshold), required

    if gap_bucket == "long":
        threshold = config.matching.long_gap_min_appearance_similarity
        if threshold is None:
            threshold = default_threshold
        required = bool(
            config.appearance.use_for_matching
            and config.matching.require_appearance_for_long_gap
            and threshold is not None
        )
        return gap_bucket, None if threshold is None else float(threshold), required

    required = bool(config.appearance.use_for_matching and default_threshold is not None)
    return gap_bucket, default_threshold, required


def _winner_margin_threshold(config: StitchConfig, gap_frames: int) -> float:
    base_threshold = float(config.matching.min_score_margin)
    configured = config.matching.winner_margin_threshold

    if _uses_conditional_gap_policy(config):
        gap_bucket = _gap_bucket(gap_frames, config)
        if gap_bucket == "short" and config.matching.winner_margin_threshold_short is not None:
            configured = config.matching.winner_margin_threshold_short
        elif gap_bucket == "long" and config.matching.winner_margin_threshold_long is not None:
            configured = config.matching.winner_margin_threshold_long

    if configured is None:
        return base_threshold
    return max(base_threshold, float(configured))


def evaluate_candidate(
    lost: Tracklet,
    new: Tracklet,
    config: StitchConfig,
    appearance_embeddings: dict[str, TrackletAppearanceEmbeddings] | None = None,
    aligned_clip: AisAlignedClip | None = None,
    mmsi_assignments: dict[str, int | None] | None = None,
) -> StitchDecision:
    gap_frames = max(0, new.frame_start - lost.frame_end - 1)
    elapsed_frames = max(1, new.frame_start - lost.frame_end)
    max_gap = min(config.memory.max_frame_gap, config.motion.max_frame_gap)
    gap_bucket, appearance_threshold, appearance_gate_required = _appearance_policy(config, gap_frames)
    conditional_gap_policy = _uses_conditional_gap_policy(config)
    same_class = bool(set(lost.class_ids) & set(new.class_ids))
    mean_confidence_pair = min(lost.mean_confidence, new.mean_confidence)

    center_distance = _distance(lost.end_center, new.start_center)
    expected_center = [
        float(lost.end_center[0]) + (float(lost.end_velocity[0]) * elapsed_frames),
        float(lost.end_center[1]) + (float(lost.end_velocity[1]) * elapsed_frames),
    ]
    expected_radius = (
        float(config.motion.base_region_radius) + (float(config.motion.growth_per_frame) * gap_frames)
        if config.motion.enabled
        else 0.0
    )
    motion_distance = _distance(expected_center, new.start_center)

    smaller_area = max(min(lost.mean_area, new.mean_area), 1e-9)
    area_ratio = max(lost.mean_area, new.mean_area) / smaller_area
    aspect_ratio_delta = abs(lost.mean_aspect_ratio - new.mean_aspect_ratio)

    lost_embeddings = None if appearance_embeddings is None else appearance_embeddings.get(lost.tracklet_id)
    new_embeddings = None if appearance_embeddings is None else appearance_embeddings.get(new.tracklet_id)
    tail_embedding = None if lost_embeddings is None else lost_embeddings.tail_embedding
    head_embedding = None if new_embeddings is None else new_embeddings.head_embedding
    appearance_similarity = cosine_similarity(tail_embedding, head_embedding)
    appearance_score = (
        _clamp01((float(appearance_similarity) + 1.0) / 2.0)
        if appearance_similarity is not None
        else None
    )
    appearance_status = "ready"
    if not config.appearance.enabled or config.appearance.backend == "none":
        appearance_status = "disabled"
    elif tail_embedding is None and head_embedding is None:
        appearance_status = "missing_embeddings"
    elif tail_embedding is None:
        appearance_status = "missing_source_tail_embedding"
    elif head_embedding is None:
        appearance_status = "missing_target_head_embedding"

    appearance_support = _appearance_supports_threshold(appearance_similarity, appearance_threshold)
    short_gap_appearance_passes = True
    long_gap_appearance_passes = True
    appearance_similarity_passes = True

    if not conditional_gap_policy:
        appearance_similarity_passes = (
            True
            if not appearance_gate_required
            else appearance_support
        )
        long_gap_appearance_required = bool(
            config.appearance.use_for_matching
            and config.matching.require_appearance_for_long_gap
            and gap_bucket == "long"
        )
        long_gap_appearance_passes = (
            True
            if not long_gap_appearance_required
            else appearance_support
        )
    else:
        if gap_bucket == "short":
            short_gap_appearance_passes = (
                True
                if not appearance_gate_required
                else appearance_support
            )
        elif gap_bucket == "long":
            long_gap_appearance_passes = (
                True
                if not appearance_gate_required
                else appearance_support
            )
        else:
            appearance_similarity_passes = (
                True
                if not appearance_gate_required
                else appearance_support
            )

    appearance_used_in_score = bool(
        config.appearance.use_for_matching
        and (
            not conditional_gap_policy
            or appearance_score is not None
        )
    )
    winner_margin_threshold = _winner_margin_threshold(config, gap_frames)

    gating = {
        "temporal_gap": gap_frames <= max_gap,
        "class_consistency": same_class,
        "tracklet_length": (
            lost.num_frames >= config.matching.min_tracklet_length
            and new.num_frames >= config.matching.min_tracklet_length
        ),
        "confidence": mean_confidence_pair >= config.matching.min_mean_confidence,
        "motion_region": (
            (not config.motion.enabled)
            or motion_distance <= expected_radius
        ),
        "center_distance": (
            config.motion.max_center_distance is None
            or center_distance <= config.motion.max_center_distance
        ),
        "area_ratio": (
            (not config.bbox.enabled)
            or area_ratio <= config.bbox.max_area_ratio
        ),
        "aspect_ratio": (
            (not config.bbox.enabled)
            or aspect_ratio_delta <= config.bbox.max_aspect_ratio_delta
        ),
        "appearance_similarity": appearance_similarity_passes,
        "short_gap_appearance": short_gap_appearance_passes,
        "long_gap_appearance": long_gap_appearance_passes,
    }
    neutral = float(config.ais.neutral_score) if config.ais.enabled else AIS_NEUTRAL_SCORE
    ais_meta: dict[str, float | int | str | bool | None] = {
        "ais_position": neutral,
        "ais_identity": neutral,
        "ais_position_distance_px": None,
        "ais_source_mmsi": lost.assigned_mmsi,
        "ais_target_mmsi": new.assigned_mmsi,
        "ais_identity_status": "ais_disabled",
        "ais_identity_gate_pass": True,
    }
    ais_used_in_score = _ais_enabled_in_score(config)
    if config.ais.enabled and aligned_clip is not None:
        from ais.fusion import ais_pair_scores

        ais_meta = ais_pair_scores(
            lost,
            new,
            aligned_clip,
            mmsi_assignments or {},
            config.ais,
        )
        if not ais_meta.get("ais_identity_gate_pass", True):
            gating["ais_identity"] = False
        else:
            gating["ais_identity"] = True
    else:
        gating["ais_identity"] = True

    gating["passes_all"] = all(gating.values())

    component_scores = {
        "temporal": _threshold_score(gap_frames, float(max_gap)),
        "motion": _threshold_score(motion_distance, expected_radius if expected_radius > 0 else None),
        "center": _threshold_score(center_distance, config.motion.max_center_distance),
        "area": _ratio_score(area_ratio, float(config.bbox.max_area_ratio)),
        "aspect": _delta_score(aspect_ratio_delta, float(config.bbox.max_aspect_ratio_delta)),
        "confidence": _clamp01(mean_confidence_pair),
        "appearance": appearance_score if appearance_score is not None else 0.0,
        "ais_position": float(ais_meta["ais_position"]),
        "ais_identity": float(ais_meta["ais_identity"]),
    }
    score = _weighted_score(
        component_scores,
        _candidate_weights(
            config,
            include_appearance=appearance_used_in_score,
            include_ais=ais_used_in_score,
        ),
    )
    motion_score = _aggregated_score(
        component_scores,
        config.matching.weights.to_dict(),
        ["temporal", "motion", "center"],
    )
    bbox_score = _aggregated_score(
        component_scores,
        config.matching.weights.to_dict(),
        ["area", "aspect"],
    )

    failed_gates = [name for name, passed in gating.items() if name != "passes_all" and not passed]
    reason = "eligible" if not failed_gates else f"gate_failed:{','.join(failed_gates)}"

    return StitchDecision(
        sequence_name=new.sequence_name,
        source_tracklet_id=new.tracklet_id,
        target_tracklet_id=lost.tracklet_id,
        source_track_id=new.source_track_id,
        target_track_id=lost.source_track_id,
        source_canonical_track_id=new.canonical_track_id,
        target_canonical_track_id=lost.canonical_track_id,
        score=score,
        applied=False,
        reason=reason,
        gap_frames=gap_frames,
        elapsed_frames=elapsed_frames,
        same_class=same_class,
        mean_confidence_pair=_round(mean_confidence_pair),
        center_distance=_round(center_distance),
        expected_center=_round_list(expected_center),
        expected_radius=_round(expected_radius),
        motion_distance=_round(motion_distance),
        area_ratio=_round(area_ratio),
        aspect_ratio_delta=_round(aspect_ratio_delta),
        component_scores={name: _round(value) for name, value in component_scores.items()},
        gating=gating,
        appearance_similarity=_round(appearance_similarity),
        appearance_score=_round(appearance_score),
        motion_score=_round(motion_score),
        bbox_score=_round(bbox_score),
        final_score=_round(score),
        appearance_status=appearance_status,
        gap_bucket=gap_bucket,
        appearance_similarity_threshold=_round(appearance_threshold),
        appearance_gate_required=appearance_gate_required,
        appearance_used_in_score=appearance_used_in_score,
        winner_margin_threshold=_round(winner_margin_threshold),
        source_tail_embedding_ready=tail_embedding is not None,
        target_head_embedding_ready=head_embedding is not None,
        ais_source_mmsi=ais_meta.get("ais_source_mmsi"),
        ais_target_mmsi=ais_meta.get("ais_target_mmsi"),
        ais_position_score=_round(float(ais_meta["ais_position"])),
        ais_identity_score=_round(float(ais_meta["ais_identity"])),
        ais_position_distance_px=_round(ais_meta.get("ais_position_distance_px")),
        ais_identity_status=str(ais_meta.get("ais_identity_status", "ais_disabled")),
        ais_used_in_score=ais_used_in_score,
    )


def match_tracklets(
    tracklets: list[Tracklet],
    config: StitchConfig,
    appearance_embeddings: dict[str, TrackletAppearanceEmbeddings] | None = None,
    aligned_clip: AisAlignedClip | None = None,
    mmsi_assignments: dict[str, int | None] | None = None,
) -> tuple[list[Tracklet], list[StitchDecision], dict[str, int], int]:
    sorted_tracklets = sorted(
        tracklets,
        key=lambda tracklet: (tracklet.frame_start, tracklet.source_track_id, tracklet.segment_index),
    )
    decisions: list[StitchDecision] = []
    consumed_lost_tracklets: set[str] = set()
    matches_applied = 0

    for index, new_tracklet in enumerate(sorted_tracklets):
        eligible_decisions: list[StitchDecision] = []

        for lost_tracklet in sorted_tracklets[:index]:
            if lost_tracklet.tracklet_id in consumed_lost_tracklets:
                continue
            if lost_tracklet.frame_end >= new_tracklet.frame_start:
                continue

            decision = evaluate_candidate(
                lost_tracklet,
                new_tracklet,
                config,
                appearance_embeddings=appearance_embeddings,
                aligned_clip=aligned_clip,
                mmsi_assignments=mmsi_assignments,
            )
            decisions.append(decision)
            if decision.gating.get("passes_all", False):
                eligible_decisions.append(decision)

        if not eligible_decisions:
            continue

        eligible_decisions.sort(key=lambda decision: decision.score, reverse=True)
        best = eligible_decisions[0]
        runner_up = eligible_decisions[1] if len(eligible_decisions) > 1 else None

        if best.score < config.matching.min_match_score:
            best.reason = (
                f"below_score_threshold:{best.score:.3f}<{config.matching.min_match_score:.3f}"
            )
            continue

        if runner_up is not None:
            margin = best.score - runner_up.score
            winner_margin_threshold = (
                float(best.winner_margin_threshold)
                if best.winner_margin_threshold is not None
                else _winner_margin_threshold(config, best.gap_frames)
            )
            if margin < winner_margin_threshold:
                best.reason = (
                    f"margin_too_small:{margin:.3f}<{winner_margin_threshold:.3f}"
                )
                continue

        best.applied = True
        best.reason = "accepted"
        new_tracklet.canonical_track_id = best.target_canonical_track_id
        best.source_canonical_track_id = new_tracklet.canonical_track_id
        consumed_lost_tracklets.add(best.target_tracklet_id)
        matches_applied += 1

        for candidate in eligible_decisions[1:]:
            if candidate.reason == "eligible":
                candidate.reason = f"outscored_by:{best.source_track_id}->{best.target_track_id}"

    identity_map = {
        str(tracklet.source_track_id): int(tracklet.canonical_track_id)
        for tracklet in sorted_tracklets
    }
    return sorted_tracklets, decisions, identity_map, matches_applied
