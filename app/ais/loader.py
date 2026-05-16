from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ais.parser import fixes_from_json_dict, group_fixes_by_mmsi, load_ais_fixes
from ais.schemas import AisFix, AisPosition, AisTrackLayer


def _timestamp_to_frame(
    timestamp_ms: int,
    *,
    video_fps: float,
    time_offset_ms: int,
) -> int:
    adjusted = timestamp_ms - int(time_offset_ms)
    if adjusted < 0:
        return 1
    zero_based = int(math.floor((adjusted / 1000.0) * float(video_fps)))
    return max(1, zero_based + 1)


def fixes_to_frame_layer(
    fixes: list[AisFix],
    *,
    video_fps: float,
    time_offset_ms: int = 0,
    sync: dict | None = None,
) -> AisTrackLayer:
    """Export normalised fixes to legacy frame-indexed sidecar shape."""
    by_frame: dict[int, list[AisPosition]] = {}
    for fix in fixes:
        frame = _timestamp_to_frame(
            fix.timestamp_ms,
            video_fps=video_fps,
            time_offset_ms=time_offset_ms,
        )
        if fix.latitude_deg is None or fix.longitude_deg is None:
            continue
        by_frame.setdefault(frame, []).append(
            AisPosition(
                mmsi=fix.mmsi,
                latitude_deg=fix.latitude_deg,
                longitude_deg=fix.longitude_deg,
                sog_knots=fix.sog_knots,
                cog_deg=fix.cog_deg,
            )
        )
    frozen = {f: tuple(lst) for f, lst in sorted(by_frame.items())}
    return AisTrackLayer(
        version=1,
        sync={**(sync or {}), "mode": "timestamp_ms"},
        positions_by_frame=frozen,
    )


def _position_from_row(row: dict[str, Any]) -> AisPosition:
    return AisPosition(
        mmsi=int(row["mmsi"]),
        latitude_deg=float(row["latitude_deg"]),
        longitude_deg=float(row["longitude_deg"]),
        sog_knots=float(row["sog_knots"]) if row.get("sog_knots") is not None else None,
        cog_deg=float(row["cog_deg"]) if row.get("cog_deg") is not None else None,
    )


def build_layer_from_json_dict(data: dict[str, Any], *, video_fps: float | None) -> AisTrackLayer:
    """Backward-compatible JSON loader -> frame-indexed AisTrackLayer."""
    sync = dict(data.get("sync") or {})
    time_offset_ms = int(sync.get("time_offset_ms", 0))
    mode = str(sync.get("mode", "frame_index"))
    version = int(data.get("version", 1))
    positions_raw = data.get("positions")
    if not isinstance(positions_raw, list):
        raise ValueError('AIS JSON must contain a list "positions"')

    if mode == "frame_index":
        by_frame: dict[int, list[AisPosition]] = {}
        for idx, row in enumerate(positions_raw):
            if not isinstance(row, dict):
                raise ValueError(f"AIS positions[{idx}] must be an object")
            frame = int(row["frame"])
            if frame < 1:
                raise ValueError(f"AIS frame must be >= 1, got {frame}")
            by_frame.setdefault(frame, []).append(_position_from_row(row))
        frozen = {f: tuple(lst) for f, lst in sorted(by_frame.items())}
        return AisTrackLayer(version=version, sync={**sync, "mode": mode}, positions_by_frame=frozen)

    fps = float(video_fps) if video_fps is not None else sync.get("video_fps")
    if fps is None or float(fps) <= 0:
        raise ValueError("timestamp_ms sync requires positive video_fps (argument or sync.video_fps)")
    fixes = fixes_from_json_dict(data, video_fps=float(fps))
    return fixes_to_frame_layer(fixes, video_fps=float(fps), time_offset_ms=time_offset_ms, sync=sync)


def load_ais_track_layer_from_path(
    path: Path,
    *,
    video_fps: float | None = None,
) -> AisTrackLayer:
    fixes = load_ais_fixes(path, video_fps=video_fps)
    sync: dict[str, Any] = {"mode": "timestamp_ms", "time_offset_ms": 0}
    if path.suffix.lower() == ".json":
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            sync = dict(raw.get("sync") or sync)
    fps = float(video_fps) if video_fps is not None else float(sync.get("video_fps", 30.0))
    return fixes_to_frame_layer(
        fixes,
        video_fps=fps,
        time_offset_ms=int(sync.get("time_offset_ms", 0)),
        sync=sync,
    )


__all__ = [
    "build_layer_from_json_dict",
    "fixes_to_frame_layer",
    "group_fixes_by_mmsi",
    "load_ais_fixes",
    "load_ais_track_layer_from_path",
]
