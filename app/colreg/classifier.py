from __future__ import annotations

import math

from colreg.motion import angle_difference_deg, relative_bearing_deg
from colreg.types import ColregConfig, ColregResult, EncounterType, VesselState


def _approaching(own_state: VesselState, other_state: VesselState) -> bool:
    rx = other_state.x - own_state.x
    ry = other_state.y - own_state.y
    rvx = other_state.vx - own_state.vx
    rvy = other_state.vy - own_state.vy
    return (rx * rvx) + (ry * rvy) < 0.0


def _low_speed_or_uncertain(own_state: VesselState, other_state: VesselState, config: ColregConfig) -> bool:
    return (
        own_state.observations < int(config.min_track_observations)
        or other_state.observations < int(config.min_track_observations)
        or own_state.speed <= 1e-6
        or other_state.speed <= 1e-6
    )


def _result(
    encounter_type: EncounterType,
    confidence: float,
    reason: str,
    own_state: VesselState,
    other_state: VesselState,
    *,
    relative_bearing: float,
    heading_difference: float,
    config: ColregConfig,
) -> ColregResult:
    confidence = round(max(0.0, min(1.0, float(confidence))), 6)
    if encounter_type not in {EncounterType.UNKNOWN, EncounterType.NO_INTERACTION} and confidence < float(config.min_confidence):
        return ColregResult(
            encounter_type=EncounterType.UNKNOWN,
            confidence=confidence,
            reason=f"low_confidence:{reason}",
            own_track_id=own_state.track_id,
            other_track_id=other_state.track_id,
            frame=max(own_state.frame, other_state.frame),
            relative_bearing_deg=round(relative_bearing, 6),
            heading_difference_deg=round(heading_difference, 6),
        )
    return ColregResult(
        encounter_type=encounter_type,
        confidence=confidence,
        reason=reason,
        own_track_id=own_state.track_id,
        other_track_id=other_state.track_id,
        frame=max(own_state.frame, other_state.frame),
        relative_bearing_deg=round(relative_bearing, 6),
        heading_difference_deg=round(heading_difference, 6),
    )


def classify_encounter(
    own_state: VesselState | None,
    other_state: VesselState | None,
    config: ColregConfig | None = None,
) -> ColregResult:
    config = config or ColregConfig()
    if own_state is None or other_state is None:
        return ColregResult(
            encounter_type=EncounterType.UNKNOWN,
            confidence=0.0,
            reason="insufficient_state",
        )

    relative_bearing = relative_bearing_deg(own_state, other_state)
    reciprocal_bearing = relative_bearing_deg(other_state, own_state)
    heading_difference = abs(angle_difference_deg(own_state.heading_deg, other_state.heading_deg))

    if _low_speed_or_uncertain(own_state, other_state, config):
        return _result(
            EncounterType.UNKNOWN,
            0.0,
            "insufficient_observations_or_low_speed",
            own_state,
            other_state,
            relative_bearing=relative_bearing,
            heading_difference=heading_difference,
            config=config,
        )

    approaching = _approaching(own_state, other_state)
    head_on_heading_error = abs(180.0 - heading_difference)
    if (
        head_on_heading_error <= float(config.head_on_angle_threshold_deg)
        and abs(relative_bearing) <= float(config.head_on_angle_threshold_deg)
        and abs(reciprocal_bearing) <= float(config.head_on_angle_threshold_deg)
        and approaching
    ):
        confidence = 1.0 - (
            head_on_heading_error
            + abs(relative_bearing)
            + abs(reciprocal_bearing)
        ) / (3.0 * max(float(config.head_on_angle_threshold_deg), 1e-9))
        return _result(
            EncounterType.HEAD_ON,
            0.65 + (0.35 * confidence),
            "opposite_headings_and_closing",
            own_state,
            other_state,
            relative_bearing=relative_bearing,
            heading_difference=heading_difference,
            config=config,
        )

    same_course = heading_difference <= 45.0
    own_abaft_other = abs(reciprocal_bearing) >= float(config.overtaking_bearing_threshold_deg)
    if same_course and own_abaft_other and own_state.speed > other_state.speed:
        excess_speed = min((own_state.speed - other_state.speed) / max(other_state.speed, 1e-9), 1.0)
        bearing_confidence = min(
            (abs(reciprocal_bearing) - float(config.overtaking_bearing_threshold_deg))
            / max(180.0 - float(config.overtaking_bearing_threshold_deg), 1e-9),
            1.0,
        )
        return _result(
            EncounterType.OVERTAKING,
            0.65 + (0.2 * bearing_confidence) + (0.15 * excess_speed),
            "own_vessel_abaft_other_and_faster",
            own_state,
            other_state,
            relative_bearing=relative_bearing,
            heading_difference=heading_difference,
            config=config,
        )

    crossing_min = float(config.crossing_bearing_min_deg)
    crossing_max = float(config.crossing_bearing_max_deg)
    not_parallel = heading_difference > 25.0 and abs(180.0 - heading_difference) > float(config.head_on_angle_threshold_deg)
    if not_parallel and crossing_min <= relative_bearing <= crossing_max:
        bearing_center = (crossing_min + crossing_max) / 2.0
        bearing_span = (crossing_max - crossing_min) / 2.0
        confidence = 1.0 - min(abs(relative_bearing - bearing_center) / max(bearing_span, 1e-9), 1.0) * 0.25
        if approaching:
            confidence += 0.05
        return _result(
            EncounterType.CROSSING_GIVE_WAY,
            min(confidence, 1.0),
            "other_vessel_on_starboard_side",
            own_state,
            other_state,
            relative_bearing=relative_bearing,
            heading_difference=heading_difference,
            config=config,
        )

    if not_parallel and -crossing_max <= relative_bearing <= -crossing_min:
        bearing_center = -((crossing_min + crossing_max) / 2.0)
        bearing_span = (crossing_max - crossing_min) / 2.0
        confidence = 1.0 - min(abs(relative_bearing - bearing_center) / max(bearing_span, 1e-9), 1.0) * 0.25
        if approaching:
            confidence += 0.05
        return _result(
            EncounterType.CROSSING_STAND_ON,
            min(confidence, 1.0),
            "other_vessel_on_port_side",
            own_state,
            other_state,
            relative_bearing=relative_bearing,
            heading_difference=heading_difference,
            config=config,
        )

    distance = math.hypot(other_state.x - own_state.x, other_state.y - own_state.y)
    distance_confidence = max(0.1, min(0.45, 1.0 - (distance / max(float(config.proximity_distance_px), 1e-9))))
    return _result(
        EncounterType.NO_INTERACTION,
        distance_confidence,
        f"no_supported_encounter_geometry:distance={distance:.3f};low_confidence_non_interaction",
        own_state,
        other_state,
        relative_bearing=relative_bearing,
        heading_difference=heading_difference,
        config=config,
    )
