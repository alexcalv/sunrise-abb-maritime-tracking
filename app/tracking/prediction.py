from __future__ import annotations

import math
from typing import Any

from tracking.occlusion import OcclusionPrediction, TrackVisibilityState


def _center(row: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = row["bbox"]
    return float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)


def constant_velocity_prediction(track_history: list[dict[str, Any]], gap_frames: int) -> tuple[float, float]:
    """Predict a missing track center with deterministic image-plane velocity."""

    if not track_history:
        return 0.0, 0.0
    ordered = sorted(track_history, key=lambda row: int(row["frame"]))
    last_x, last_y = _center(ordered[-1])
    if len(ordered) < 2:
        return last_x, last_y

    previous = ordered[-2]
    latest = ordered[-1]
    previous_x, previous_y = _center(previous)
    latest_x, latest_y = _center(latest)
    frame_delta = max(int(latest["frame"]) - int(previous["frame"]), 1)
    vx = (latest_x - previous_x) / frame_delta
    vy = (latest_y - previous_y) / frame_delta
    return last_x + (vx * int(gap_frames)), last_y + (vy * int(gap_frames))


def prediction_uncertainty(
    gap_frames: int,
    base_uncertainty: float = 12.0,
    growth_rate: float = 3.5,
) -> tuple[float, float, float]:
    """Grow uncertainty with gap length so longer disappearances look less certain."""

    gap = max(0, int(gap_frames))
    uncertainty = float(base_uncertainty) + (float(growth_rate) * gap)
    return round(uncertainty, 6), round(uncertainty, 6), round(math.sqrt(2.0) * uncertainty, 6)


def build_occlusion_prediction(
    *,
    track_history: list[dict[str, Any]],
    track_id: int,
    frame: int,
    gap_frames: int,
    canonical_id: int | None = None,
    base_uncertainty: float = 12.0,
    growth_rate: float = 3.5,
    occluded_after_frames: int = 10,
) -> OcclusionPrediction:
    """Build a reporting-only prediction; this never forces a match or remap."""

    predicted_x, predicted_y = constant_velocity_prediction(track_history, gap_frames)
    uncertainty_x, uncertainty_y, uncertainty_radius = prediction_uncertainty(
        gap_frames,
        base_uncertainty=base_uncertainty,
        growth_rate=growth_rate,
    )
    state = (
        TrackVisibilityState.OCCLUDED
        if int(gap_frames) > int(occluded_after_frames)
        else TrackVisibilityState.TEMPORARILY_LOST
    )
    last_frame = int(track_history[-1]["frame"]) if track_history else None
    return OcclusionPrediction(
        track_id=int(track_id),
        canonical_id=None if canonical_id is None else int(canonical_id),
        frame=int(frame),
        last_visible_frame=last_frame,
        predicted_x=round(float(predicted_x), 6),
        predicted_y=round(float(predicted_y), 6),
        uncertainty_x=uncertainty_x,
        uncertainty_y=uncertainty_y,
        uncertainty_radius=uncertainty_radius,
        gap_frames=int(gap_frames),
        source="constant_velocity",
        reason="raw_track_temporarily_missing_reporting_only",
        visibility_state=state,
    )
