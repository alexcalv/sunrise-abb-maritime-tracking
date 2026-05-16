from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ais.schemas import AIS_NEUTRAL_SCORE, AisAlignedClip, AisAlignedVesselFrame

if TYPE_CHECKING:
    from stitching.schemas import AisConfig, Tracklet


def _distance_xy(a: list[float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - b[0], float(a[1]) - b[1])


def _position_score(distance: float, max_distance_px: float) -> float:
    if max_distance_px <= 0:
        return AIS_NEUTRAL_SCORE
    return max(0.0, min(1.0, 1.0 - (distance / max_distance_px)))


def assign_mmsi_per_tracklet(
    tracklets: list[Tracklet],
    aligned: AisAlignedClip,
    *,
    max_distance_px: float,
) -> dict[str, int | None]:
    """
    One MMSI per tracklet (stateless upfront assignment from overlap with aligned AIS).
    """
    assignments: dict[str, int | None] = {}
    for tracklet in tracklets:
        best_mmsi: int | None = None
        best_dist = float("inf")
        sample_frames = sorted({tracklet.frame_start, tracklet.frame_end, (tracklet.frame_start + tracklet.frame_end) // 2})
        for frame in sample_frames:
            vessels = aligned.vessels_at(frame)
            if not vessels:
                continue
            center = tracklet.end_center if frame >= tracklet.frame_end else tracklet.start_center
            for mmsi, vessel in vessels.items():
                d = _distance_xy(center, (vessel.pixel_x, vessel.pixel_y))
                if d < best_dist:
                    best_dist = d
                    best_mmsi = mmsi
        if best_mmsi is not None and best_dist <= max_distance_px:
            assignments[tracklet.tracklet_id] = best_mmsi
        else:
            assignments[tracklet.tracklet_id] = None
    return assignments


def score_ais_position(
    center: list[float],
    frame: int,
    mmsi: int | None,
    aligned: AisAlignedClip | None,
    *,
    max_distance_px: float,
    neutral_score: float = AIS_NEUTRAL_SCORE,
) -> tuple[float, float | None, int | None]:
    if aligned is None or mmsi is None:
        return neutral_score, None, mmsi
    vessel = aligned.vessel_at(frame, mmsi)
    if vessel is None:
        return neutral_score, None, mmsi
    dist = _distance_xy(center, (vessel.pixel_x, vessel.pixel_y))
    return _position_score(dist, max_distance_px), dist, mmsi


def score_ais_identity(
    source_mmsi: int | None,
    target_mmsi: int | None,
    *,
    hard_gate: bool,
    neutral_score: float = AIS_NEUTRAL_SCORE,
) -> tuple[float, bool, str]:
    """
  Returns (score, gate_pass, status).
  Missing AIS on either side -> neutral 0.5, gate passes.
  Both present and equal -> 1.0; mismatch -> 0.0 (soft) or gate fail (hard).
    """
    if source_mmsi is None or target_mmsi is None:
        return neutral_score, True, "ais_missing"
    if source_mmsi == target_mmsi:
        return 1.0, True, "mmsi_match"
    if hard_gate:
        return 0.0, False, "mmsi_hard_mismatch"
    return 0.0, True, "mmsi_soft_mismatch"


def ais_pair_scores(
    lost: Tracklet,
    new: Tracklet,
    aligned: AisAlignedClip | None,
    assignments: dict[str, int | None],
    config: AisConfig,
) -> dict[str, float | int | str | bool | None]:
    src_mmsi = assignments.get(lost.tracklet_id)
    tgt_mmsi = assignments.get(new.tracklet_id)
    neutral = float(config.neutral_score)
    pos_score, pos_dist, _ = score_ais_position(
        new.start_center,
        new.frame_start,
        tgt_mmsi,
        aligned,
        max_distance_px=config.max_distance_px,
        neutral_score=neutral,
    )
    id_score, id_gate, id_status = score_ais_identity(
        src_mmsi,
        tgt_mmsi,
        hard_gate=config.hard_identity_gate,
        neutral_score=neutral,
    )
    return {
        "ais_position": pos_score,
        "ais_identity": id_score,
        "ais_position_distance_px": pos_dist,
        "ais_source_mmsi": src_mmsi,
        "ais_target_mmsi": tgt_mmsi,
        "ais_identity_status": id_status,
        "ais_identity_gate_pass": id_gate,
    }
