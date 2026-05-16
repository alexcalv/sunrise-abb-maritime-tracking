from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ais.schemas import AisFix


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


def _parse_fix_row(row: dict[str, Any], *, default_timestamp_ms: int | None = None) -> AisFix:
    mmsi = _as_int(row.get("mmsi"), "mmsi")
    ts_raw = row.get("timestamp_ms")
    if ts_raw is not None:
        timestamp_ms = _as_int(ts_raw, "timestamp_ms")
    elif row.get("frame") is not None and default_timestamp_ms is not None:
        # frame-only rows resolved by caller
        raise ValueError("internal: use frame_index parser for frame-only rows")
    elif default_timestamp_ms is not None:
        timestamp_ms = default_timestamp_ms
    else:
        raise ValueError('AIS row requires "timestamp_ms" or "frame" with sync context')

    lat = row.get("latitude_deg")
    lon = row.get("longitude_deg")
    px = row.get("pixel_x", row.get("px"))
    py = row.get("pixel_y", row.get("py"))
    sog = row.get("sog_knots")
    cog = row.get("cog_deg")

    return AisFix(
        mmsi=mmsi,
        timestamp_ms=timestamp_ms,
        latitude_deg=float(lat) if lat is not None else None,
        longitude_deg=float(lon) if lon is not None else None,
        pixel_x=float(px) if px is not None else None,
        pixel_y=float(py) if py is not None else None,
        sog_knots=float(sog) if sog is not None else None,
        cog_deg=float(cog) if cog is not None else None,
    )


def _frame_to_timestamp_ms(frame: int, *, video_fps: float, time_offset_ms: int) -> int:
    zero_based = max(0, int(frame) - 1)
    return int(time_offset_ms + (zero_based / float(video_fps)) * 1000.0)


def fixes_from_json_dict(data: dict[str, Any], *, video_fps: float | None) -> list[AisFix]:
    version = int(data.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported AIS JSON version: {version}")

    sync = dict(data.get("sync") or {})
    mode = str(sync.get("mode", "timestamp_ms"))
    time_offset_ms = int(sync.get("time_offset_ms", 0))
    positions_raw = data.get("positions")
    if not isinstance(positions_raw, list):
        raise ValueError('AIS JSON must contain a list "positions"')

    fixes: list[AisFix] = []

    if mode == "frame_index":
        if video_fps is None or video_fps <= 0:
            raise ValueError("frame_index sync requires positive video_fps")
        for idx, row in enumerate(positions_raw):
            if not isinstance(row, dict):
                raise ValueError(f"AIS positions[{idx}] must be an object")
            frame = _as_int(row.get("frame"), "frame")
            if frame < 1:
                raise ValueError(f"AIS frame must be >= 1, got {frame}")
            ts = _frame_to_timestamp_ms(frame, video_fps=float(video_fps), time_offset_ms=time_offset_ms)
            row_with_ts = {**row, "timestamp_ms": ts}
            fixes.append(_parse_fix_row(row_with_ts))
    elif mode == "timestamp_ms":
        for idx, row in enumerate(positions_raw):
            if not isinstance(row, dict):
                raise ValueError(f"AIS positions[{idx}] must be an object")
            fixes.append(_parse_fix_row(row))
    else:
        raise ValueError(f"Unsupported AIS sync.mode: {mode!r}")

    return fixes


def fixes_from_csv_path(path: Path) -> list[AisFix]:
    fixes: list[AisFix] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"AIS CSV has no header: {path}")
        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                fixes.append(_parse_fix_row(row))
            except ValueError as e:
                raise ValueError(f"AIS CSV {path} line {line_num}: {e}") from e
    return fixes


def load_ais_fixes(
    path: Path,
    *,
    video_fps: float | None = None,
) -> list[AisFix]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return fixes_from_csv_path(path)
    if suffix in {".json", ".geojson"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"AIS file must contain a JSON object: {path}")
        return fixes_from_json_dict(data, video_fps=video_fps)
    raise ValueError(f"Unsupported AIS file type: {suffix} (use .csv or .json)")


def group_fixes_by_mmsi(fixes: list[AisFix]) -> dict[int, list[AisFix]]:
    grouped: dict[int, list[AisFix]] = {}
    for fix in fixes:
        grouped.setdefault(fix.mmsi, []).append(fix)
    for mmsi in grouped:
        grouped[mmsi].sort(key=lambda f: f.timestamp_ms)
    return grouped
