from __future__ import annotations

import math
from typing import Any

from colreg.types import ColregConfig, VesselState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance_between_states(state_a: VesselState | None, state_b: VesselState | None) -> float | None:
    if state_a is None or state_b is None:
        return None
    return math.hypot(state_b.x - state_a.x, state_b.y - state_a.y)


def closest_approach_estimate(
    state_a: VesselState | None,
    state_b: VesselState | None,
    *,
    horizon_frames: int = 30,
) -> dict[str, Any]:
    if state_a is None or state_b is None:
        return {
            "closest_distance_px": None,
            "frames_to_closest": None,
            "relative_speed": None,
            "reason": "insufficient_state",
        }

    rx = state_b.x - state_a.x
    ry = state_b.y - state_a.y
    rvx = state_b.vx - state_a.vx
    rvy = state_b.vy - state_a.vy
    relative_speed = math.hypot(rvx, rvy)
    if relative_speed <= 1e-9:
        frames_to_closest = 0.0
    else:
        frames_to_closest = -((rx * rvx) + (ry * rvy)) / max(relative_speed * relative_speed, 1e-9)
    frames_to_closest = max(0.0, min(float(horizon_frames), frames_to_closest))
    closest_x = rx + (rvx * frames_to_closest)
    closest_y = ry + (rvy * frames_to_closest)
    return {
        "closest_distance_px": round(math.hypot(closest_x, closest_y), 6),
        "frames_to_closest": round(frames_to_closest, 6),
        "relative_speed": round(relative_speed, 6),
        "reason": "ok",
    }


def proximity_risk_score(
    state_a: VesselState | None,
    state_b: VesselState | None,
    config: ColregConfig | None = None,
) -> float:
    config = config or ColregConfig()
    distance = distance_between_states(state_a, state_b)
    closest = closest_approach_estimate(
        state_a,
        state_b,
        horizon_frames=int(config.closest_approach_horizon_frames),
    )
    closest_distance = closest["closest_distance_px"]
    relative_speed = closest["relative_speed"]
    if distance is None or closest_distance is None or relative_speed is None:
        return 0.0

    threshold = max(float(config.proximity_distance_px), 1e-9)
    current_score = _clamp01(1.0 - (float(distance) / threshold))
    closest_score = _clamp01(1.0 - (float(closest_distance) / threshold))
    speed_score = _clamp01(float(relative_speed) / max(float(config.min_relative_speed), 1e-9))
    closing_bonus = 0.15 * closest_score if closest["frames_to_closest"] is not None and float(closest["frames_to_closest"]) > 0.0 else 0.0
    return round(_clamp01((0.45 * current_score) + (0.4 * closest_score) + (0.15 * speed_score) + closing_bonus), 6)


def is_proximity_candidate(
    state_a: VesselState | None,
    state_b: VesselState | None,
    config: ColregConfig | None = None,
) -> bool:
    config = config or ColregConfig()
    if not bool(config.proximity_enabled):
        return True
    closest = closest_approach_estimate(
        state_a,
        state_b,
        horizon_frames=int(config.closest_approach_horizon_frames),
    )
    distance = distance_between_states(state_a, state_b)
    closest_distance = closest["closest_distance_px"]
    relative_speed = closest["relative_speed"]
    if distance is None or closest_distance is None or relative_speed is None:
        return False

    threshold = float(config.proximity_distance_px)
    close_now = float(distance) <= threshold
    close_soon = (
        float(closest_distance) <= threshold
        and float(relative_speed) >= float(config.min_relative_speed)
    )
    return bool(close_now or close_soon or proximity_risk_score(state_a, state_b, config) >= float(config.risk_score_min))
