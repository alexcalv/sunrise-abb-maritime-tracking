from __future__ import annotations

from pathlib import Path
from typing import Any

from ais.alignment import build_aligned_clip
from ais.parser import group_fixes_by_mmsi, load_ais_fixes
from ais.schemas import AisAlignedClip
from stitching.schemas import AisConfig, StitchConfig


def load_aligned_clip_for_stitch(
    config: StitchConfig,
    *,
    max_frame: int,
    ais_file_override: str | None = None,
) -> AisAlignedClip | None:
    ais = config.ais
    path_str = ais_file_override or ais.file_path
    if not ais.enabled or not path_str:
        return None

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"AIS file not found: {path}")

    fps = ais.video_fps
    if fps is None or fps <= 0:
        raise ValueError("AisConfig.video_fps must be set when AIS stitching is enabled")

    fixes = load_ais_fixes(path, video_fps=float(fps))
    grouped = group_fixes_by_mmsi(fixes)
    return build_aligned_clip(
        grouped,
        max_frame=max_frame,
        video_fps=float(fps),
        time_offset_ms=int(ais.time_offset_ms),
        geo_affine=ais.geo_affine,
    )


def stamp_tracklet_mmsi(
    tracklets: list[Any],
    aligned: AisAlignedClip | None,
    *,
    max_distance_px: float,
) -> dict[str, int | None]:
    if aligned is None:
        return {t.tracklet_id: None for t in tracklets}
    from ais.fusion import assign_mmsi_per_tracklet

    assignments = assign_mmsi_per_tracklet(tracklets, aligned, max_distance_px=max_distance_px)
    for tracklet in tracklets:
        tracklet.assigned_mmsi = assignments.get(tracklet.tracklet_id)
    return assignments
