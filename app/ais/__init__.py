from .alignment import align_ais_tracks_to_frames, project_lonlat_to_xy
from .fusion import assign_ais_to_tracks
from .parser import load_ais_file, load_ais_file_with_warnings, parse_affine_matrix
from .types import AisAssignment, AisConfig, AisFrameState, AisRecord, AisTrack

__all__ = [
    "AisAssignment",
    "AisConfig",
    "AisFrameState",
    "AisRecord",
    "AisTrack",
    "align_ais_tracks_to_frames",
    "assign_ais_to_tracks",
    "load_ais_file",
    "load_ais_file_with_warnings",
    "parse_affine_matrix",
    "project_lonlat_to_xy",
]
"""Optional AIS augmentation: parse, align to frames, score stitching pairs."""

from ais.schemas import AIS_NEUTRAL_SCORE, AisFix, AisPosition, AisTrackLayer

__all__ = ["AIS_NEUTRAL_SCORE", "AisFix", "AisPosition", "AisTrackLayer"]
