from __future__ import annotations

import math
from typing import Any

from colreg.types import ColregConfig, EncounterType, MotionPrediction, VesselState


def normalize_angle_deg(angle: float) -> float:
    normalized = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def angle_difference_deg(a: float, b: float) -> float:
    return normalize_angle_deg(float(a) - float(b))


def _center_from_observation(observation: dict[str, Any]) -> tuple[float, float]:
    if "center" in observation:
        center = observation["center"]
        return float(center[0]), float(center[1])
    bbox = observation["bbox"]
    return float(bbox[0]) + (float(bbox[2]) / 2.0), float(bbox[1]) + (float(bbox[3]) / 2.0)


def estimate_velocity_from_observations(observations: list[dict[str, Any]]) -> tuple[float, float]:
    if len(observations) < 2:
        return 0.0, 0.0

    ordered = sorted(observations, key=lambda row: int(row["frame"]))
    first = ordered[0]
    last = ordered[-1]
    first_x, first_y = _center_from_observation(first)
    last_x, last_y = _center_from_observation(last)
    delta_frames = max(int(last["frame"]) - int(first["frame"]), 1)
    return (last_x - first_x) / delta_frames, (last_y - first_y) / delta_frames


def estimate_heading_from_observations(observations: list[dict[str, Any]]) -> float | None:
    vx, vy = estimate_velocity_from_observations(observations)
    if math.hypot(vx, vy) <= 1e-9:
        return None
    return normalize_angle_deg(math.degrees(math.atan2(vy, vx)))


def build_vessel_state(
    observations: list[dict[str, Any]],
    *,
    track_id: int | None = None,
    min_observations: int = 2,
) -> VesselState | None:
    if len(observations) < int(min_observations):
        return None

    ordered = sorted(observations, key=lambda row: int(row["frame"]))
    latest = ordered[-1]
    x, y = _center_from_observation(latest)
    vx, vy = estimate_velocity_from_observations(ordered)
    speed = math.hypot(vx, vy)
    heading = estimate_heading_from_observations(ordered)
    if heading is None:
        heading = 0.0
    if track_id is None and latest.get("id") is not None:
        track_id = int(latest["id"])
    return VesselState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        speed=speed,
        heading_deg=heading,
        frame=int(latest["frame"]),
        track_id=track_id,
        observations=len(ordered),
    )


def relative_bearing_deg(own_state: VesselState, other_state: VesselState) -> float:
    bearing = math.degrees(math.atan2(other_state.y - own_state.y, other_state.x - own_state.x))
    return angle_difference_deg(bearing, own_state.heading_deg)


def _heading_vector(heading_deg: float) -> tuple[float, float]:
    radians = math.radians(float(heading_deg))
    return math.cos(radians), math.sin(radians)


def _state_from_heading(
    state: VesselState,
    *,
    heading_deg: float,
    speed: float,
    dt: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> VesselState:
    ux, uy = _heading_vector(heading_deg)
    vx = ux * speed
    vy = uy * speed
    return VesselState(
        x=state.x + (vx * dt) + x_offset,
        y=state.y + (vy * dt) + y_offset,
        vx=vx,
        vy=vy,
        speed=speed,
        heading_deg=normalize_angle_deg(heading_deg),
        frame=int(round(state.frame + dt)),
        track_id=state.track_id,
        observations=state.observations,
    )


def _prediction(
    state: VesselState,
    predicted: VesselState,
    config: ColregConfig,
    *,
    heading_sigma: float,
    reason: str,
) -> MotionPrediction:
    base_sigma = max(5.0, state.speed * 2.0) * float(config.uncertainty_scale)
    return MotionPrediction(
        predicted=predicted,
        sigma_x=round(base_sigma, 6),
        sigma_y=round(base_sigma, 6),
        sigma_heading=round(heading_sigma * float(config.uncertainty_scale), 6),
        reason=reason,
    )


def predict_unknown_constant_velocity(state: VesselState, config: ColregConfig, dt: float = 1.0) -> MotionPrediction:
    predicted = VesselState(
        x=state.x + (state.vx * dt),
        y=state.y + (state.vy * dt),
        vx=state.vx,
        vy=state.vy,
        speed=state.speed,
        heading_deg=state.heading_deg,
        frame=int(round(state.frame + dt)),
        track_id=state.track_id,
        observations=state.observations,
    )
    return _prediction(state, predicted, config, heading_sigma=20.0, reason="constant_velocity_unknown")


def predict_head_on(state: VesselState, config: ColregConfig, dt: float = 1.0) -> MotionPrediction:
    heading = normalize_angle_deg(state.heading_deg + float(config.starboard_yaw_bias_deg))
    predicted = _state_from_heading(state, heading_deg=heading, speed=state.speed, dt=dt)
    return _prediction(state, predicted, config, heading_sigma=12.0, reason="head_on_starboard_yaw_bias")


def predict_crossing_give_way(state: VesselState, config: ColregConfig, dt: float = 1.0) -> MotionPrediction:
    heading = normalize_angle_deg(state.heading_deg + float(config.starboard_yaw_bias_deg))
    speed = state.speed * float(config.give_way_deceleration_factor)
    predicted = _state_from_heading(state, heading_deg=heading, speed=speed, dt=dt)
    return _prediction(state, predicted, config, heading_sigma=15.0, reason="crossing_give_way_yaw_bias_deceleration")


def predict_crossing_stand_on(state: VesselState, config: ColregConfig, dt: float = 1.0) -> MotionPrediction:
    predicted = _state_from_heading(state, heading_deg=state.heading_deg, speed=state.speed, dt=dt)
    return _prediction(state, predicted, config, heading_sigma=8.0, reason="crossing_stand_on_maintain_course_speed")


def predict_overtaking(state: VesselState, config: ColregConfig, dt: float = 1.0) -> MotionPrediction:
    ux, uy = _heading_vector(state.heading_deg + 90.0)
    predicted = _state_from_heading(
        state,
        heading_deg=state.heading_deg,
        speed=state.speed,
        dt=dt,
        x_offset=ux * float(config.lateral_offset_px),
        y_offset=uy * float(config.lateral_offset_px),
    )
    return _prediction(state, predicted, config, heading_sigma=14.0, reason="overtaking_lateral_keep_clear_offset")


def predict_colreg_state(
    state: VesselState,
    encounter_type: EncounterType | str,
    config: ColregConfig,
    dt: float = 1.0,
) -> MotionPrediction:
    encounter = EncounterType(encounter_type)
    if encounter == EncounterType.HEAD_ON:
        return predict_head_on(state, config, dt=dt)
    if encounter == EncounterType.CROSSING_GIVE_WAY:
        return predict_crossing_give_way(state, config, dt=dt)
    if encounter == EncounterType.CROSSING_STAND_ON:
        return predict_crossing_stand_on(state, config, dt=dt)
    if encounter == EncounterType.OVERTAKING:
        return predict_overtaking(state, config, dt=dt)
    return predict_unknown_constant_velocity(state, config, dt=dt)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _linear_score(error: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if error <= 0 else 0.0
    return _clamp01(1.0 - (abs(error) / tolerance))


def _speed_ratio_score(observed: float, expected: float, tolerance: float = 0.35) -> float:
    if expected <= 1e-9:
        return 1.0 if observed <= 1e-9 else 0.0
    return _linear_score((observed / expected) - 1.0, tolerance)


def score_colreg_motion_consistency(
    previous_state: VesselState,
    observed_reappearing_state: VesselState,
    encounter_type: EncounterType | str,
    config: ColregConfig,
) -> float:
    encounter = EncounterType(encounter_type)
    if encounter in {EncounterType.UNKNOWN, EncounterType.NO_INTERACTION}:
        return 0.5

    dt = max(1.0, float(observed_reappearing_state.frame - previous_state.frame))
    prediction = predict_colreg_state(previous_state, encounter, config, dt=dt)
    predicted = prediction.predicted
    distance = math.hypot(observed_reappearing_state.x - predicted.x, observed_reappearing_state.y - predicted.y)
    position_score = _linear_score(distance, max(prediction.sigma_x, prediction.sigma_y) * 3.0)
    heading_delta_from_previous = angle_difference_deg(observed_reappearing_state.heading_deg, previous_state.heading_deg)
    heading_error = angle_difference_deg(observed_reappearing_state.heading_deg, predicted.heading_deg)
    heading_score = _linear_score(heading_error, max(prediction.sigma_heading * 3.0, 1.0))
    speed_score = _speed_ratio_score(observed_reappearing_state.speed, predicted.speed)

    if encounter == EncounterType.HEAD_ON:
        starboard_score = _clamp01((heading_delta_from_previous + float(config.starboard_yaw_bias_deg)) / (2.0 * float(config.starboard_yaw_bias_deg)))
        return round((position_score + heading_score + starboard_score) / 3.0, 6)

    if encounter == EncounterType.CROSSING_GIVE_WAY:
        starboard_score = _clamp01((heading_delta_from_previous + float(config.starboard_yaw_bias_deg)) / (2.0 * float(config.starboard_yaw_bias_deg)))
        deceleration_score = 1.0 if observed_reappearing_state.speed <= previous_state.speed * float(config.give_way_deceleration_threshold) else 0.5
        return round((position_score + max(starboard_score, deceleration_score) + speed_score) / 3.0, 6)

    if encounter == EncounterType.CROSSING_STAND_ON:
        maintain_heading_score = _linear_score(
            angle_difference_deg(observed_reappearing_state.heading_deg, previous_state.heading_deg),
            float(config.stand_on_heading_tolerance_deg),
        )
        maintain_speed_score = _speed_ratio_score(observed_reappearing_state.speed, previous_state.speed, tolerance=0.2)
        return round((maintain_heading_score + maintain_speed_score + position_score) / 3.0, 6)

    if encounter == EncounterType.OVERTAKING:
        ux, uy = _heading_vector(previous_state.heading_deg + 90.0)
        constant_prediction = predict_unknown_constant_velocity(previous_state, config, dt=dt).predicted
        offset_x = observed_reappearing_state.x - constant_prediction.x
        offset_y = observed_reappearing_state.y - constant_prediction.y
        lateral_offset = abs((offset_x * ux) + (offset_y * uy))
        lateral_score = _clamp01(lateral_offset / max(float(config.lateral_offset_px), 1e-9))
        return round((position_score + lateral_score + speed_score) / 3.0, 6)

    return 0.5
