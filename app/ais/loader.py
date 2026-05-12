from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ais.schemas import AisPosition, AisTrackLayer


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"AIS field {field!r} must be numeric, got {value!r}") from e


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"AIS field {field!r} must be integral, got {value!r}") from e


def _parse_position_row(row: dict[str, Any], *, require_frame: bool) -> tuple[int | None, AisPosition]:
    mmsi = _as_int(row.get("mmsi"), "mmsi")
    lat = _as_float(row.get("latitude_deg"), "latitude_deg")
    lon = _as_float(row.get("longitude_deg"), "longitude_deg")
    sog = row.get("sog_knots")
    cog = row.get("cog_deg")
    pos = AisPosition(
        mmsi=mmsi,
        latitude_deg=lat,
        longitude_deg=lon,
        sog_knots=float(sog) if sog is not None else None,
        cog_deg=float(cog) if cog is not None else None,
    )
    if require_frame:
        frame = _as_int(row.get("frame"), "frame")
        if frame < 1:
            raise ValueError(f"AIS frame must be >= 1, got {frame}")
        return frame, pos
    ts = row.get("timestamp_ms")
    if ts is None:
        raise ValueError('AIS row in timestamp_ms mode requires "timestamp_ms"')
    t_ms = _as_int(ts, "timestamp_ms")
    return t_ms, pos


def _timestamp_to_frame(
    timestamp_ms: int,
    *,
    video_fps: float,
    time_offset_ms: int,
) -> int:
    adjusted = timestamp_ms - int(time_offset_ms)
    if adjusted < 0:
        return 1
    # 1-based MOT frames: first frame at t=0 is frame 1
    zero_based = int(math.floor((adjusted / 1000.0) * float(video_fps)))
    return max(1, zero_based + 1)


def build_layer_from_json_dict(data: dict[str, Any], *, video_fps: float | None) -> AisTrackLayer:
    """
    Parse AIS JSON (version 1) into a frame-indexed layer.

    sync.mode:
      - frame_index: each row has integer "frame" (>=1).
      - timestamp_ms: each row has "timestamp_ms" (ms from clip start); needs video_fps.
    """
    version = int(data.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported AIS JSON version: {version}")

    sync = dict(data.get("sync") or {})
    mode = str(sync.get("mode", "frame_index"))
    positions_raw = data.get("positions")
    if not isinstance(positions_raw, list):
        raise ValueError('AIS JSON must contain a list "positions"')

    by_frame: dict[int, list[AisPosition]] = {}

    if mode == "frame_index":
        for idx, row in enumerate(positions_raw):
            if not isinstance(row, dict):
                raise ValueError(f"AIS positions[{idx}] must be an object")
            frame, pos = _parse_position_row(row, require_frame=True)
            assert frame is not None
            by_frame.setdefault(frame, []).append(pos)
    elif mode == "timestamp_ms":
        if video_fps is None or video_fps <= 0:
            raise ValueError("timestamp_ms sync requires a positive video_fps (file or video probe)")
        time_offset_ms = int(sync.get("time_offset_ms", 0))
        for idx, row in enumerate(positions_raw):
            if not isinstance(row, dict):
                raise ValueError(f"AIS positions[{idx}] must be an object")
            t_ms, pos = _parse_position_row(row, require_frame=False)
            assert t_ms is not None
            frame = _timestamp_to_frame(t_ms, video_fps=float(video_fps), time_offset_ms=time_offset_ms)
            by_frame.setdefault(frame, []).append(pos)
    else:
        raise ValueError(f"Unsupported AIS sync.mode: {mode!r} (expected frame_index|timestamp_ms)")

    frozen: dict[int, tuple[AisPosition, ...]] = {f: tuple(lst) for f, lst in sorted(by_frame.items())}
    return AisTrackLayer(version=version, sync={**sync, "mode": mode}, positions_by_frame=frozen)


def load_ais_track_layer_from_path(
    path: Path,
    *,
    video_fps: float | None = None,
) -> AisTrackLayer:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid AIS JSON: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"AIS file must contain a JSON object: {path}")
    return build_layer_from_json_dict(data, video_fps=video_fps)
