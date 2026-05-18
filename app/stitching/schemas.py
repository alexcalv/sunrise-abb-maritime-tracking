from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrackletConfig:
    split_gap: int = 1
    min_length: int = 1


@dataclass
class MemoryConfig:
    max_frame_gap: int = 900


@dataclass
class AppearanceConfig:
    backend: str = "none"
    enabled: bool = False
    use_for_matching: bool = False
    model_name: str = "vit_small_patch14_reg4_dinov2.lvd142m"
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    allow_downloads: bool = False
    device: str | None = None
    sample_frames: int = 5
    batch_size: int = 8
    crop_padding: float = 0.15
    min_crop_size: int = 48
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionConfig:
    enabled: bool = True
    max_frame_gap: int = 900
    max_center_distance: float | None = 300.0
    base_region_radius: float = 100.0
    growth_per_frame: float = 1.25


@dataclass
class BBoxConfig:
    enabled: bool = True
    max_area_ratio: float = 2.5
    max_aspect_ratio_delta: float = 0.5


@dataclass
class MatchingWeights:
    temporal: float = 0.15
    motion: float = 0.2
    center: float = 0.15
    area: float = 0.2
    aspect: float = 0.15
    confidence: float = 0.15
    appearance: float = 0.0
    ais_position: float = 0.0
    ais_identity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class AisConfig:
    """Optional AIS augmentation for offline stitching (disabled by default)."""

    enabled: bool = False
    file_path: str | None = None
    video_fps: float | None = None
    time_offset_ms: int = 0
    geo_affine: list[list[float]] | None = None
    max_distance_px: float = 400.0
    hard_identity_gate: bool = False
    neutral_score: float = 0.5


@dataclass
class MatchingConfig:
    min_match_score: float = 0.55
    min_score_margin: float = 0.05
    min_mean_confidence: float = 0.55
    min_tracklet_length: int = 20
    appearance_weight: float | None = None
    min_appearance_similarity: float | None = None
    short_gap_frames_threshold: int | None = None
    short_gap_min_appearance_similarity: float | None = None
    long_gap_frames_threshold: int | None = None
    long_gap_min_appearance_similarity: float | None = None
    require_appearance_for_short_gap: bool = False
    require_appearance_for_long_gap: bool = False
    winner_margin_threshold: float | None = None
    winner_margin_threshold_short: float | None = None
    winner_margin_threshold_long: float | None = None
    weights: MatchingWeights = field(default_factory=MatchingWeights)


@dataclass
class StitchConfig:
    stitch_name: str = "motion_bbox_v1"
    tracklets: TrackletConfig = field(default_factory=TrackletConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    bbox: BBoxConfig = field(default_factory=BBoxConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ais: AisConfig = field(default_factory=AisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackObservation:
    frame: int
    bbox: list[float]
    confidence: float
    center: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tracklet:
    sequence_name: str
    tracklet_id: str
    source_track_id: int
    canonical_track_id: int
    segment_index: int
    frame_start: int
    frame_end: int
    num_rows: int
    num_frames: int
    class_ids: list[int]
    mean_confidence: float
    first_bbox: list[float]
    last_bbox: list[float]
    mean_bbox: list[float]
    mean_area: float
    mean_aspect_ratio: float
    frame_gaps: list[int]
    start_center: list[float]
    end_center: list[float]
    start_velocity: list[float]
    end_velocity: list[float]
    head_observations: list[TrackObservation] = field(default_factory=list)
    tail_observations: list[TrackObservation] = field(default_factory=list)
    appearance: dict[str, Any] = field(default_factory=dict)
    assigned_mmsi: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StitchDecision:
    sequence_name: str
    source_tracklet_id: str
    target_tracklet_id: str
    source_track_id: int
    target_track_id: int
    source_canonical_track_id: int
    target_canonical_track_id: int
    score: float
    applied: bool
    reason: str
    gap_frames: int
    elapsed_frames: int
    same_class: bool
    mean_confidence_pair: float
    center_distance: float
    expected_center: list[float]
    expected_radius: float
    motion_distance: float
    area_ratio: float
    aspect_ratio_delta: float
    component_scores: dict[str, float] = field(default_factory=dict)
    gating: dict[str, bool] = field(default_factory=dict)
    appearance_similarity: float | None = None
    appearance_score: float | None = None
    motion_score: float | None = None
    bbox_score: float | None = None
    final_score: float | None = None
    appearance_status: str = "not_requested"
    gap_bucket: str = "medium"
    appearance_similarity_threshold: float | None = None
    appearance_gate_required: bool = False
    appearance_used_in_score: bool = False
    winner_margin_threshold: float | None = None
    source_tail_embedding_ready: bool = False
    target_head_embedding_ready: bool = False
    ais_source_mmsi: int | None = None
    ais_target_mmsi: int | None = None
    ais_position_score: float | None = None
    ais_identity_score: float | None = None
    ais_position_distance_px: float | None = None
    ais_identity_status: str = "ais_disabled"
    ais_used_in_score: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceStitchReport:
    sequence_name: str
    input_mot_path: str
    output_mot_path: str
    num_rows_in: int
    num_rows_out: int
    unique_track_ids_in: list[int]
    unique_track_ids_out: list[int]
    identity_map: dict[str, int]
    matches_applied: int = 0
    tracklets: list[Tracklet] = field(default_factory=list)
    decisions: list[StitchDecision] = field(default_factory=list)
    appearance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tracklets"] = [tracklet.to_dict() for tracklet in self.tracklets]
        payload["decisions"] = [decision.to_dict() for decision in self.decisions]
        return payload


@dataclass
class StitchReport:
    version: str
    mode: str
    pred_dir: str
    output_root: str
    input_run_summary_path: str | None
    output_run_summary_path: str
    source: str | None
    appearance_backend: dict[str, Any]
    config: dict[str, Any]
    total_sequences: int
    total_tracklets: int
    total_matches_applied: int
    sequences: dict[str, SequenceStitchReport] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sequences"] = {
            name: sequence_report.to_dict()
            for name, sequence_report in self.sequences.items()
        }
        return payload
