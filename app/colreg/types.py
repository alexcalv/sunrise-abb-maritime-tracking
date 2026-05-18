from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class EncounterType(str, Enum):
    UNKNOWN = "UNKNOWN"
    HEAD_ON = "HEAD_ON"
    CROSSING_GIVE_WAY = "CROSSING_GIVE_WAY"
    CROSSING_STAND_ON = "CROSSING_STAND_ON"
    OVERTAKING = "OVERTAKING"
    NO_INTERACTION = "NO_INTERACTION"


@dataclass(frozen=True)
class ColregConfig:
    """Diagnostic COLREG-inspired configuration.

    Scoring is disabled by default. The scoring fields only apply to the
    explicitly enabled tie-break experiment and must not be treated as legal
    navigation compliance.
    """

    enabled: bool = False
    diagnostics_enabled: bool = False
    scoring_enabled: bool = False
    scoring_experiment: bool = False
    scoring_weight: float = 0.02
    scoring_min_confidence: float = 0.85
    scoring_min_score: float = 0.65
    scoring_ambiguous_margin_threshold: float = 0.05
    scoring_require_proximity: bool = True
    proximity_enabled: bool = True
    proximity_distance_px: float = 250.0
    closest_approach_horizon_frames: int = 30
    min_relative_speed: float = 0.5
    risk_score_min: float = 0.2
    min_confidence: float = 0.6
    starboard_yaw_bias_deg: float = 5.0
    give_way_deceleration_factor: float = 0.85
    uncertainty_scale: float = 1.0
    head_on_angle_threshold_deg: float = 20.0
    crossing_bearing_min_deg: float = 15.0
    crossing_bearing_max_deg: float = 112.5
    overtaking_bearing_threshold_deg: float = 112.5
    min_track_observations: int = 3
    stand_on_heading_tolerance_deg: float = 10.0
    give_way_deceleration_threshold: float = 0.90
    lateral_offset_px: float = 20.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VesselState:
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading_deg: float
    frame: int
    track_id: int | None = None
    observations: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ColregResult:
    encounter_type: EncounterType
    confidence: float
    reason: str
    own_track_id: int | None = None
    other_track_id: int | None = None
    frame: int | None = None
    relative_bearing_deg: float | None = None
    heading_difference_deg: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["encounter_type"] = self.encounter_type.value
        return payload


@dataclass(frozen=True)
class MotionPrediction:
    predicted: VesselState
    sigma_x: float
    sigma_y: float
    sigma_heading: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["predicted"] = self.predicted.to_dict()
        return payload
