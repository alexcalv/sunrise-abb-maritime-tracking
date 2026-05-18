from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MotionState:
    previous_center_x: float
    previous_center_y: float
    center_x: float
    center_y: float
    vx: float
    vy: float
    speed: float
    heading_deg: float
    last_frame: int
    bbox_width: float
    bbox_height: float


def _center(row: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = row["bbox"]
    return float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)


def _bbox_center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    x, y, w, h = (float(value) for value in bbox)
    return x + (w / 2.0), y + (h / 2.0)


def _bbox_iou(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (lw * lh) + (rw * rh) - intersection
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def estimate_motion_state(track_history: list[dict[str, Any]], min_observations: int = 3) -> MotionState | None:
    """Estimate deterministic image-plane motion for reporting-only occlusion geometry."""

    if not track_history:
        return None
    ordered = sorted(track_history, key=lambda row: int(row["frame"]))
    latest = ordered[-1]
    latest_x, latest_y = _center(latest)
    _, _, latest_w, latest_h = (float(value) for value in latest["bbox"])
    if len(ordered) < 2:
        return MotionState(
            previous_center_x=latest_x,
            previous_center_y=latest_y,
            center_x=latest_x,
            center_y=latest_y,
            vx=0.0,
            vy=0.0,
            speed=0.0,
            heading_deg=0.0,
            last_frame=int(latest["frame"]),
            bbox_width=latest_w,
            bbox_height=latest_h,
        )

    previous = ordered[-2]
    previous_x, previous_y = _center(previous)
    frame_delta = max(int(latest["frame"]) - int(previous["frame"]), 1)
    vx = (latest_x - previous_x) / frame_delta
    vy = (latest_y - previous_y) / frame_delta
    speed = math.hypot(vx, vy)
    heading_deg = math.degrees(math.atan2(vy, vx)) if speed > 1e-6 else 0.0
    return MotionState(
        previous_center_x=previous_x,
        previous_center_y=previous_y,
        center_x=latest_x,
        center_y=latest_y,
        vx=vx,
        vy=vy,
        speed=speed,
        heading_deg=heading_deg,
        last_frame=int(latest["frame"]),
        bbox_width=latest_w,
        bbox_height=latest_h,
    )


def predict_position_from_motion(state: MotionState, gap_frames: int) -> tuple[float, float]:
    gap = max(0, int(gap_frames))
    return state.center_x + (state.vx * gap), state.center_y + (state.vy * gap)


def compute_directional_uncertainty(
    state: MotionState,
    gap_frames: int,
    base_uncertainty: float,
    along_track_growth: float,
    cross_track_growth: float,
) -> dict[str, float]:
    """Use a long/narrow uncertainty ellipse aligned to travel direction."""

    gap = max(0, int(gap_frames))
    major = float(base_uncertainty) + (float(along_track_growth) * gap)
    minor = float(base_uncertainty) + (float(cross_track_growth) * gap)
    if state.speed <= 1e-6:
        minor = major
    return {
        "uncertainty_major_axis": round(max(major, minor), 6),
        "uncertainty_minor_axis": round(min(major, minor), 6),
        "uncertainty_angle_deg": round(float(state.heading_deg), 6),
    }


def estimate_occlusion_relationship(
    hidden_state: MotionState,
    other_state: MotionState,
    hidden_bbox: list[float] | tuple[float, ...],
    other_bbox: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Score whether another visible vessel plausibly occludes the hidden vessel."""

    frame_gap = max(0, int(other_state.last_frame) - int(hidden_state.last_frame))
    hidden_x, hidden_y = predict_position_from_motion(hidden_state, frame_gap)
    other_x, other_y = _bbox_center(other_bbox)
    distance = math.hypot(hidden_x - other_x, hidden_y - other_y)
    scale = max(hidden_state.bbox_width, hidden_state.bbox_height, other_state.bbox_width, other_state.bbox_height, 1.0)
    proximity_score = max(0.0, 1.0 - (distance / (scale * 1.75)))

    translated_hidden = [
        hidden_x - (float(hidden_bbox[2]) / 2.0),
        hidden_y - (float(hidden_bbox[3]) / 2.0),
        float(hidden_bbox[2]),
        float(hidden_bbox[3]),
    ]
    overlap_score = _bbox_iou(translated_hidden, other_bbox)

    rel_vx = hidden_state.vx - other_state.vx
    rel_vy = hidden_state.vy - other_state.vy
    rel_speed = math.hypot(rel_vx, rel_vy)
    motion_score = max(0.0, min(1.0, rel_speed / max(hidden_state.speed + other_state.speed, 1.0)))
    confidence = (0.45 * overlap_score) + (0.40 * proximity_score) + (0.15 * motion_score)
    reason = "bbox_overlap" if overlap_score > 0 else "near_predicted_path"
    if confidence < 0.20:
        reason = "weak_spatial_relationship"
    return {
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "reason": reason,
        "distance_px": round(distance, 6),
        "overlap_score": round(overlap_score, 6),
        "proximity_score": round(proximity_score, 6),
        "relative_speed": round(rel_speed, 6),
    }


def predict_reappearance_side(hidden_state: MotionState, occluder_state: MotionState | None) -> str:
    vx = hidden_state.vx
    vy = hidden_state.vy
    if occluder_state is not None:
        vx -= occluder_state.vx
        vy -= occluder_state.vy
    if math.hypot(vx, vy) <= 1e-6:
        return "unknown"
    if abs(vx) >= abs(vy):
        return "right" if vx >= 0 else "left"
    return "bottom" if vy >= 0 else "top"


def build_motion_corridor_prediction(
    *,
    hidden_state: MotionState,
    gap_frames: int,
    current_frame: int,
    occluder_state: MotionState | None = None,
    occlusion_pair_confidence: float | None = None,
    base_uncertainty: float = 12.0,
    along_track_growth: float = 4.0,
    cross_track_growth: float = 1.35,
) -> dict[str, Any]:
    predicted_x, predicted_y = predict_position_from_motion(hidden_state, gap_frames)
    uncertainty = compute_directional_uncertainty(
        hidden_state,
        gap_frames,
        base_uncertainty=base_uncertainty,
        along_track_growth=along_track_growth,
        cross_track_growth=cross_track_growth,
    )
    reappearance_side = predict_reappearance_side(hidden_state, occluder_state)
    expected_min = int(current_frame)
    expected_max = int(current_frame + max(3, min(60, round((hidden_state.bbox_width + hidden_state.bbox_height) / max(hidden_state.speed, 1.0)))))
    return {
        "prediction_model": "paired_motion_corridor" if occluder_state is not None else "constant_velocity",
        "velocity_x": round(hidden_state.vx, 6),
        "velocity_y": round(hidden_state.vy, 6),
        "hidden_velocity_x": round(hidden_state.vx, 6),
        "hidden_velocity_y": round(hidden_state.vy, 6),
        "occluder_velocity_x": round(occluder_state.vx, 6) if occluder_state is not None else None,
        "occluder_velocity_y": round(occluder_state.vy, 6) if occluder_state is not None else None,
        "speed_px_per_frame": round(hidden_state.speed, 6),
        "heading_deg": round(hidden_state.heading_deg, 6),
        "predicted_x": round(predicted_x, 6),
        "predicted_y": round(predicted_y, 6),
        "last_visible_center": {
            "x": round(hidden_state.center_x, 6),
            "y": round(hidden_state.center_y, 6),
        },
        "previous_visible_center": {
            "x": round(hidden_state.previous_center_x, 6),
            "y": round(hidden_state.previous_center_y, 6),
        },
        "predicted_center_from_hidden_velocity": {
            "x": round(predicted_x, 6),
            "y": round(predicted_y, 6),
        },
        "uncertainty_radius": round(max(uncertainty["uncertainty_major_axis"], uncertainty["uncertainty_minor_axis"]), 6),
        **uncertainty,
        "motion_corridor_start": {
            "x": round(hidden_state.center_x, 6),
            "y": round(hidden_state.center_y, 6),
        },
        "motion_corridor_end": {
            "x": round(predicted_x, 6),
            "y": round(predicted_y, 6),
        },
        "arrow_start": {
            "x": round(hidden_state.center_x, 6),
            "y": round(hidden_state.center_y, 6),
        },
        "arrow_end": {
            "x": round(predicted_x, 6),
            "y": round(predicted_y, 6),
        },
        "motion_direction_consistency": "unknown" if hidden_state.speed <= 1e-6 else "valid",
        "corridor_direction_note": (
            "hidden track speed too small for a reliable arrow"
            if hidden_state.speed <= 1e-6
            else "arrow follows hidden-track last visible velocity"
        ),
        "expected_reappearance_side": reappearance_side,
        "expected_reappearance_frame_min": expected_min,
        "expected_reappearance_frame_max": expected_max,
        "occlusion_pair_confidence": round(float(occlusion_pair_confidence or 0.0), 6),
    }
