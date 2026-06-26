from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .types import AisAssignment, AisConfig, AisFrameState


def _center(row: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = row["bbox"]
    return float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)


def _group_mot(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["id"])].append(row)
    return {track_id: sorted(track_rows, key=lambda row: int(row["frame"])) for track_id, track_rows in grouped.items()}


def _angle_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(diff)


def _track_heading_speed(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if len(rows) < 2:
        return None, None
    start = rows[0]
    end = rows[-1]
    start_x, start_y = _center(start)
    end_x, end_y = _center(end)
    delta_frames = max(int(end["frame"]) - int(start["frame"]), 1)
    dx = (end_x - start_x) / delta_frames
    dy = (end_y - start_y) / delta_frames
    speed = math.hypot(dx, dy)
    if speed < 1e-9:
        return None, 0.0
    heading = math.degrees(math.atan2(dy, dx)) % 360.0
    return heading, speed


def _score_distance(distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (float(distance) / float(max_distance))))


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.5


def _states_by_mmsi(aligned_ais: dict[int, list[AisFrameState]]) -> dict[str, dict[int, AisFrameState]]:
    grouped: dict[str, dict[int, AisFrameState]] = defaultdict(dict)
    for frame, states in aligned_ais.items():
        for state in states:
            grouped[str(state.mmsi)][int(frame)] = state
    return grouped


def _assignment_for_pair(
    track_id: int,
    mot_rows: list[dict[str, Any]],
    mmsi: str,
    ais_by_frame: dict[int, AisFrameState],
    config: AisConfig,
) -> AisAssignment:
    overlap_frames = sorted(set(int(row["frame"]) for row in mot_rows) & set(ais_by_frame))
    if not overlap_frames:
        return AisAssignment(
            track_id=track_id,
            assigned_mmsi=None,
            score=float(config.neutral_score),
            overlap_frames=0,
            position_score=float(config.neutral_score),
            heading_score=float(config.neutral_score),
            speed_score=float(config.neutral_score),
            identity_score=float(config.neutral_score),
            missing_reason="no_temporal_overlap",
        )

    row_by_frame = {int(row["frame"]): row for row in mot_rows}
    position_scores: list[float] = []
    ais_positions = 0
    for frame in overlap_frames:
        state = ais_by_frame[frame]
        if state.x is None or state.y is None:
            continue
        mot_x, mot_y = _center(row_by_frame[frame])
        position_scores.append(_score_distance(math.hypot(mot_x - float(state.x), mot_y - float(state.y)), config.max_position_distance_px))
        ais_positions += 1
    position_score = _avg(position_scores) if position_scores else float(config.neutral_score)

    mot_heading, mot_speed = _track_heading_speed([row_by_frame[frame] for frame in overlap_frames])
    heading_values = [state.heading if state.heading is not None else state.cog for frame in overlap_frames for state in [ais_by_frame[frame]]]
    heading_values = [float(value) for value in heading_values if value is not None]
    heading_score = float(config.neutral_score)
    if mot_heading is not None and heading_values:
        heading_diff = _angle_diff(mot_heading, sum(heading_values) / len(heading_values))
        heading_score = 0.5 if heading_diff is None else max(0.0, min(1.0, 1.0 - (heading_diff / 180.0)))

    speed_score = float(config.neutral_score)
    if mot_speed is not None and ais_positions >= 2:
        first_state = ais_by_frame[overlap_frames[0]]
        last_state = ais_by_frame[overlap_frames[-1]]
        if first_state.x is not None and first_state.y is not None and last_state.x is not None and last_state.y is not None:
            frame_delta = max(overlap_frames[-1] - overlap_frames[0], 1)
            ais_speed_px = math.hypot(float(last_state.x) - float(first_state.x), float(last_state.y) - float(first_state.y)) / frame_delta
            denom = max(float(mot_speed), float(ais_speed_px), 1e-9)
            speed_score = max(0.0, min(1.0, 1.0 - (abs(float(mot_speed) - float(ais_speed_px)) / denom)))

    overlap_score = max(0.0, min(1.0, len(overlap_frames) / max(int(config.min_track_overlap_frames), 1)))
    score = _avg([position_score, heading_score, speed_score, overlap_score])
    missing: list[str] = []
    if not position_scores:
        missing.append("ais_pixel_position_unavailable")
    if not heading_values:
        missing.append("ais_heading_unavailable")
    return AisAssignment(
        track_id=track_id,
        assigned_mmsi=mmsi if score >= float(config.assignment_min_score) else None,
        score=score,
        overlap_frames=len(overlap_frames),
        position_score=round(position_score, 6),
        heading_score=round(heading_score, 6),
        speed_score=round(speed_score, 6),
        identity_score=1.0 if score >= float(config.assignment_min_score) else float(config.neutral_score),
        missing_reason=";".join(missing) if missing else "",
        component_notes={
            "raw_mmsi": mmsi,
            "overlap_score": round(overlap_score, 6),
            "assignment_threshold": float(config.assignment_min_score),
        },
    )


def assign_ais_to_tracks(
    mot_tracks: dict[int, list[dict[str, Any]]] | list[dict[str, Any]],
    aligned_ais: dict[int, list[AisFrameState]],
    config: AisConfig,
) -> list[AisAssignment]:
    tracks = _group_mot(mot_tracks) if isinstance(mot_tracks, list) else mot_tracks
    ais_by_mmsi = _states_by_mmsi(aligned_ais)
    assignments: list[AisAssignment] = []
    for track_id, rows in sorted(tracks.items()):
        if not ais_by_mmsi:
            assignments.append(
                AisAssignment(
                    track_id=int(track_id),
                    assigned_mmsi=None,
                    score=float(config.neutral_score),
                    overlap_frames=0,
                    position_score=float(config.neutral_score),
                    heading_score=float(config.neutral_score),
                    speed_score=float(config.neutral_score),
                    identity_score=float(config.neutral_score),
                    missing_reason="no_ais_coverage",
                )
            )
            continue
        candidates = [
            _assignment_for_pair(int(track_id), rows, mmsi, states, config)
            for mmsi, states in ais_by_mmsi.items()
        ]
        candidates.sort(
            key=lambda item: (
                -float(item.score),
                str(item.assigned_mmsi or (item.component_notes or {}).get("raw_mmsi", "")),
            )
        )
        best = candidates[0]
        assignments.append(best)
    return assignments


def score_ais_position(
    center_xy: list[float] | tuple[float, float],
    frame: int,
    mmsi: int | None,
    aligned_clip,
    *,
    max_distance_px: float,
    neutral_score: float = 0.5,
) -> tuple[float, float | None, str]:
    """Legacy pixel-distance score used by older AIS/ReID bridge tests."""
    if aligned_clip is None or mmsi is None:
        return float(neutral_score), None, "ais_missing"
    vessel = aligned_clip.vessel_at(int(frame), int(mmsi))
    if vessel is None:
        return float(neutral_score), None, "ais_missing"
    distance = math.hypot(float(center_xy[0]) - float(vessel.pixel_x), float(center_xy[1]) - float(vessel.pixel_y))
    return _score_distance(distance, max_distance_px), distance, "ais_position"


def score_ais_identity(
    assigned_mmsi: int | str | None,
    candidate_mmsi: int | str | None,
    *,
    hard_gate: bool = False,
    neutral_score: float = 0.5,
) -> tuple[float, bool, str]:
    """Legacy identity score: missing AIS is neutral, hard mismatches can gate."""
    if assigned_mmsi is None or candidate_mmsi is None:
        return float(neutral_score), True, "ais_missing"
    if str(assigned_mmsi) == str(candidate_mmsi):
        return 1.0, True, "mmsi_match"
    if hard_gate:
        return 0.0, False, "mmsi_hard_mismatch"
    return 0.0, True, "mmsi_mismatch"


def assign_mmsi_per_tracklet(tracklets: list[Any], aligned_clip, *, max_distance_px: float) -> dict[str, int | None]:
    """Assign each tracklet to the nearest AIS MMSI observed during its frame span."""
    assignments: dict[str, int | None] = {}
    for tracklet in tracklets:
        best_mmsi: int | None = None
        best_distance = float("inf")
        start = int(getattr(tracklet, "frame_start", 1))
        end = int(getattr(tracklet, "frame_end", start))
        center = getattr(tracklet, "mean_bbox", None) or getattr(tracklet, "start_center", None)
        if center is None:
            assignments[str(getattr(tracklet, "tracklet_id"))] = None
            continue
        if len(center) >= 4:
            cx = float(center[0]) + (float(center[2]) / 2.0)
            cy = float(center[1]) + (float(center[3]) / 2.0)
        else:
            cx, cy = float(center[0]), float(center[1])

        for frame in range(start, end + 1):
            for mmsi, vessel in aligned_clip.vessels_at(frame).items():
                distance = math.hypot(cx - float(vessel.pixel_x), cy - float(vessel.pixel_y))
                if distance < best_distance:
                    best_distance = distance
                    best_mmsi = int(mmsi)

        assignments[str(getattr(tracklet, "tracklet_id"))] = best_mmsi if best_distance <= float(max_distance_px) else None
    return assignments
