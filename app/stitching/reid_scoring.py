from __future__ import annotations

import math

from stitching.schemas import StitchConfig


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


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
