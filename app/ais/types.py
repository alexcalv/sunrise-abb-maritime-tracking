from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AisConfig:
    """Configuration for diagnostic AIS parsing, alignment, and soft assignment.

    AIS is neutral by default: diagnostics are off, hard identity gates are off,
    and missing AIS contributes the neutral score instead of penalizing ReID.
    """

    enabled: bool = False
    diagnostics_enabled: bool = False
    ais_file: str | None = None
    input_format: str = "auto"
    timestamp_column: str = "timestamp"
    mmsi_column: str = "mmsi"
    lat_column: str = "lat"
    lon_column: str = "lon"
    x_column: str = "x"
    y_column: str = "y"
    sog_column: str = "sog"
    cog_column: str = "cog"
    heading_column: str = "heading"
    neutral_score: float = 0.5
    max_position_distance_px: float = 250.0
    max_time_gap_seconds: float = 5.0
    min_track_overlap_frames: int = 5
    assignment_min_score: float = 0.6
    hard_identity_gate: bool = False
    affine_enabled: bool = False
    affine_matrix: list[list[float]] | None = None
    video_start_time: str | float | int | None = None
    fps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AisRecord:
    mmsi: str
    timestamp: float | None = None
    lat: float | None = None
    lon: float | None = None
    x: float | None = None
    y: float | None = None
    sog: float | None = None
    cog: float | None = None
    heading: float | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AisTrack:
    mmsi: str
    records: list[AisRecord]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True)
class AisFrameState:
    frame: int
    timestamp: float
    mmsi: str
    x: float | None = None
    y: float | None = None
    lat: float | None = None
    lon: float | None = None
    sog: float | None = None
    cog: float | None = None
    heading: float | None = None
    interpolated: bool = False
    position_available: bool = False
    reason: str = "aligned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AisAssignment:
    track_id: int
    assigned_mmsi: str | None
    score: float
    overlap_frames: int
    position_score: float
    heading_score: float
    speed_score: float
    identity_score: float
    missing_reason: str = ""
    component_notes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
