from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class VisualPartTemplate:
    name: str
    image: np.ndarray
    width: int
    height: int
    offset_x: float
    offset_y: float


@dataclass
class VisualTemplate:
    track_id: int
    frame: int
    image: np.ndarray
    width: int
    height: int
    parts: dict[str, VisualPartTemplate]


def _bbox_to_ints(row: dict[str, Any], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame_shape[:2]
    x, y, w, h = (float(value) for value in row["bbox"])
    x1 = max(0, min(int(round(x)), frame_width - 1))
    y1 = max(0, min(int(round(y)), frame_height - 1))
    x2 = max(0, min(int(round(x + w)), frame_width))
    y2 = max(0, min(int(round(y + h)), frame_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _slice_part(image: np.ndarray, name: str, x1: int, y1: int, x2: int, y2: int) -> VisualPartTemplate | None:
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    template_center_x = image.shape[1] / 2.0
    template_center_y = image.shape[0] / 2.0
    part_center_x = x1 + (crop.shape[1] / 2.0)
    part_center_y = y1 + (crop.shape[0] / 2.0)
    return VisualPartTemplate(
        name=name,
        image=crop,
        width=int(crop.shape[1]),
        height=int(crop.shape[0]),
        offset_x=float(part_center_x - template_center_x),
        offset_y=float(part_center_y - template_center_y),
    )


def _build_part_templates(gray: np.ndarray) -> dict[str, VisualPartTemplate]:
    height, width = gray.shape[:2]
    mid_x = width // 2
    mid_y = height // 2
    center_x1 = max(0, width // 4)
    center_y1 = max(0, height // 4)
    center_x2 = min(width, width - center_x1)
    center_y2 = min(height, height - center_y1)
    candidates = [
        ("full", 0, 0, width, height),
        ("left", 0, 0, mid_x, height),
        ("right", mid_x, 0, width, height),
        ("upper", 0, 0, width, mid_y),
        ("lower", 0, mid_y, width, height),
        ("center", center_x1, center_y1, center_x2, center_y2),
    ]
    parts: dict[str, VisualPartTemplate] = {}
    for name, x1, y1, x2, y2 in candidates:
        part = _slice_part(gray, name, x1, y1, x2, y2)
        if part is not None:
            parts[name] = part
    return parts


def update_visual_templates(
    frame_image: np.ndarray | None,
    rows: list[dict[str, Any]],
    templates: dict[int, VisualTemplate],
) -> None:
    """Store the latest visible crop per raw track for reporting-only continuation checks."""

    if frame_image is None:
        return
    for row in rows:
        bbox = _bbox_to_ints(row, frame_image.shape)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = frame_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 4 or gray.shape[1] < 4:
            continue
        track_id = int(row["id"])
        templates[track_id] = VisualTemplate(
            track_id=track_id,
            frame=int(row["frame"]),
            image=gray,
            width=int(gray.shape[1]),
            height=int(gray.shape[0]),
            parts=_build_part_templates(gray),
        )


def _empty_result(*, enabled: bool, status: str, reason: str, threshold: float, search_radius: int) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "attempted": False,
        "status": status,
        "active": False,
        "confidence": 0.0,
        "raw_similarity": None,
        "part_matching_enabled": True,
        "best_part_score": None,
        "matched_part_name": None,
        "number_of_good_parts": 0,
        "full_template_score": None,
        "part_template_score": None,
        "continuation_confidence": 0.0,
        "matched_part_distribution": {},
        "threshold": float(threshold),
        "search_radius": int(search_radius),
        "search_window": None,
        "best_center": None,
        "template_frame": None,
        "template_size": None,
        "reason": reason,
    }


def _match_part(search_gray: np.ndarray, part: VisualPartTemplate) -> tuple[float, tuple[int, int]] | None:
    if search_gray.shape[0] < part.height or search_gray.shape[1] < part.width:
        return None
    low_texture = float(np.std(part.image)) < 1e-6 or float(np.std(search_gray)) < 1e-6
    if low_texture:
        scores = cv2.matchTemplate(search_gray, part.image, cv2.TM_SQDIFF_NORMED)
        min_score, _, min_loc, _ = cv2.minMaxLoc(scores)
        return max(0.0, min(1.0, 1.0 - float(min_score))), min_loc

    scores = cv2.matchTemplate(search_gray, part.image, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_loc = cv2.minMaxLoc(scores)
    return max(0.0, min(1.0, float(max_score))), max_loc


def estimate_visual_continuation(
    *,
    frame_image: np.ndarray | None,
    template: VisualTemplate | None,
    predicted_x: float,
    predicted_y: float,
    gap_frames: int,
    search_radius: int = 64,
    threshold: float = 0.45,
    max_gap: int = 60,
    motion_corridor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match full and partial ship crops inside a local window around the prediction."""

    if frame_image is None:
        return _empty_result(
            enabled=True,
            status="lost",
            reason="no_current_frame_available",
            threshold=threshold,
            search_radius=search_radius,
        )
    if template is None:
        return _empty_result(
            enabled=True,
            status="lost",
            reason="no_template_available",
            threshold=threshold,
            search_radius=search_radius,
        )
    if int(gap_frames) > int(max_gap):
        return _empty_result(
            enabled=True,
            status="lost",
            reason="gap_exceeds_visual_continuation_max_gap",
            threshold=threshold,
            search_radius=search_radius,
        )

    frame_height, frame_width = frame_image.shape[:2]
    half_w = max(2, template.width // 2)
    half_h = max(2, template.height // 2)
    radius = max(1, int(search_radius))
    corridor_start = (motion_corridor or {}).get("motion_corridor_start") or {}
    corridor_end = (motion_corridor or {}).get("motion_corridor_end") or {}
    major_axis = float((motion_corridor or {}).get("uncertainty_major_axis") or radius)
    minor_axis = float((motion_corridor or {}).get("uncertainty_minor_axis") or radius)
    if corridor_start and corridor_end:
        sx = float(corridor_start.get("x", predicted_x))
        sy = float(corridor_start.get("y", predicted_y))
        ex = float(corridor_end.get("x", predicted_x))
        ey = float(corridor_end.get("y", predicted_y))
        padding_x = min(max(radius, int(round(max(major_axis, minor_axis)))), radius * 3)
        padding_y = min(max(radius, int(round(max(major_axis, minor_axis)))), radius * 3)
        x1 = max(0, int(round(min(sx, ex, predicted_x))) - padding_x - half_w)
        y1 = max(0, int(round(min(sy, ey, predicted_y))) - padding_y - half_h)
        x2 = min(frame_width, int(round(max(sx, ex, predicted_x))) + padding_x + half_w)
        y2 = min(frame_height, int(round(max(sy, ey, predicted_y))) + padding_y + half_h)
        search_shape = "motion_corridor"
    else:
        x1 = max(0, int(round(predicted_x)) - radius - half_w)
        y1 = max(0, int(round(predicted_y)) - radius - half_h)
        x2 = min(frame_width, int(round(predicted_x)) + radius + half_w)
        y2 = min(frame_height, int(round(predicted_y)) + radius + half_h)
        search_shape = "local_radius"
    search = frame_image[y1:y2, x1:x2]
    min_part_height = min((part.height for part in template.parts.values()), default=template.height)
    min_part_width = min((part.width for part in template.parts.values()), default=template.width)
    if search.size == 0 or search.shape[0] < min_part_height or search.shape[1] < min_part_width:
        return _empty_result(
            enabled=True,
            status="lost",
            reason="search_window_too_small",
            threshold=threshold,
            search_radius=search_radius,
        )

    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    part_results: dict[str, dict[str, Any]] = {}
    for name, part in template.parts.items():
        match = _match_part(search_gray, part)
        if match is None:
            continue
        score, max_loc = match
        part_center_x = x1 + int(max_loc[0]) + (part.width / 2.0)
        part_center_y = y1 + int(max_loc[1]) + (part.height / 2.0)
        estimated_center_x = part_center_x - part.offset_x
        estimated_center_y = part_center_y - part.offset_y
        distance = float(np.hypot(estimated_center_x - predicted_x, estimated_center_y - predicted_y))
        part_results[name] = {
            "score": score,
            "center_x": estimated_center_x,
            "center_y": estimated_center_y,
            "distance_from_prediction": distance,
        }

    if not part_results:
        return _empty_result(
            enabled=True,
            status="lost",
            reason="no_usable_part_templates",
            threshold=threshold,
            search_radius=search_radius,
        )

    best_name, best = max(part_results.items(), key=lambda item: item[1]["score"])
    best_part_score = float(best["score"])
    full_score = float(part_results.get("full", {}).get("score", 0.0))
    good_part_threshold = max(0.5, float(threshold))
    good_parts = {
        name: result
        for name, result in part_results.items()
        if float(result["score"]) >= good_part_threshold
    }
    decay = max(0.2, 1.0 - (0.65 * (int(gap_frames) / max(int(max_gap), 1))))
    distance = float(best["distance_from_prediction"])
    distance_score = max(0.0, 1.0 - (distance / max(float(search_radius), 1.0)))
    support_score = min(1.0, len(good_parts) / 3.0)
    full_weighted_score = full_score if full_score > 0 else best_part_score * 0.75
    unclipped_window_area = float((2 * radius + template.width) * (2 * radius + template.height))
    actual_window_area = float(max(x2 - x1, 1) * max(y2 - y1, 1))
    search_reliability = min(1.0, actual_window_area / max(unclipped_window_area, 1.0))
    confidence = (
        (0.45 * best_part_score)
        + (0.25 * full_weighted_score)
        + (0.20 * support_score)
        + (0.10 * distance_score)
    ) * decay * search_reliability
    confidence = max(0.0, min(1.0, confidence))
    best_x = float(best["center_x"])
    best_y = float(best["center_y"])
    active = confidence >= float(threshold)
    return {
        "enabled": True,
        "attempted": True,
        "status": "active" if active else "lost",
        "active": bool(active),
        "confidence": round(confidence, 6),
        "raw_similarity": round(best_part_score, 6),
        "part_matching_enabled": True,
        "best_part_score": round(best_part_score, 6),
        "matched_part_name": best_name,
        "number_of_good_parts": len(good_parts),
        "full_template_score": round(full_score, 6),
        "part_template_score": round(best_part_score, 6),
        "continuation_confidence": round(confidence, 6),
        "matched_part_distribution": {
            name: round(float(result["score"]), 6)
            for name, result in sorted(part_results.items())
        },
        "distance_score": round(distance_score, 6),
        "gap_decay": round(decay, 6),
        "search_window_reliability": round(search_reliability, 6),
        "threshold": float(threshold),
        "search_radius": int(search_radius),
        "search_window": {
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "shape": search_shape,
        },
        "best_center": {
            "x": round(float(best_x), 6),
            "y": round(float(best_y), 6),
        },
        "offset_from_prediction": {
            "x": round(float(best_x - predicted_x), 6),
            "y": round(float(best_y - predicted_y), 6),
        },
        "template_frame": int(template.frame),
        "template_size": {
            "w": int(template.width),
            "h": int(template.height),
        },
        "reason": "part_template_match_above_threshold" if active else "part_template_match_below_threshold",
    }
