from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import AisConfig, AisRecord, AisTrack


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _column_map(row: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in row.items()}


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        pass
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def parse_affine_matrix(value: str | list[list[float]] | None) -> list[list[float]] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [[float(item) for item in row] for row in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [[float(item) for item in row] for row in parsed]
    except json.JSONDecodeError:
        pass
    parts = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 6:
        return [parts[:3], parts[3:]]
    if len(parts) == 9:
        return [parts[:3], parts[3:6], parts[6:]]
    raise ValueError("Affine matrix must contain 6 or 9 numeric values.")


def _record_from_row(row: dict[str, Any], config: AisConfig) -> AisRecord | None:
    normalized = _column_map(row)
    def get(column_name: str) -> Any:
        return normalized.get(_normalize_key(column_name))

    mmsi = get(config.mmsi_column)
    if mmsi is None or str(mmsi).strip() == "":
        return None
    return AisRecord(
        mmsi=str(mmsi).strip(),
        timestamp=parse_timestamp(get(config.timestamp_column)),
        lat=_parse_float(get(config.lat_column)),
        lon=_parse_float(get(config.lon_column)),
        x=_parse_float(get(config.x_column)),
        y=_parse_float(get(config.y_column)),
        sog=_parse_float(get(config.sog_column)),
        cog=_parse_float(get(config.cog_column)),
        heading=_parse_float(get(config.heading_column)),
        raw=dict(row),
    )


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records: list[dict[str, Any]] = []
        for key, value in payload.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        row = dict(item)
                        row.setdefault("mmsi", key)
                        records.append(row)
            elif isinstance(value, dict):
                row = dict(value)
                row.setdefault("mmsi", key)
                records.append(row)
        return records
    return []


def load_ais_file_with_warnings(path: str | Path, config: AisConfig) -> tuple[dict[str, AisTrack], list[str]]:
    ais_path = Path(path)
    if not ais_path.exists():
        raise FileNotFoundError(f"AIS file not found: {ais_path}")

    suffix = ais_path.suffix.lower()
    input_format = str(config.input_format or "auto").lower()
    rows: list[dict[str, Any]]
    if input_format == "json" or (input_format == "auto" and suffix == ".json"):
        rows = _json_records(json.loads(ais_path.read_text(encoding="utf-8")))
    else:
        with ais_path.open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]

    warnings: list[str] = []
    grouped: dict[str, list[AisRecord]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        record = _record_from_row(row, config)
        if record is None:
            warnings.append(f"row {index}: missing MMSI")
            continue
        if record.timestamp is None:
            warnings.append(f"row {index}: missing or invalid timestamp")
            continue
        grouped[record.mmsi].append(record)

    tracks = {
        mmsi: AisTrack(mmsi=mmsi, records=sorted(records, key=lambda record: float(record.timestamp or 0.0)))
        for mmsi, records in sorted(grouped.items())
        if records
    }
    return tracks, warnings


def load_ais_file(path: str | Path, config: AisConfig) -> dict[str, AisTrack]:
    tracks, _ = load_ais_file_with_warnings(path, config)
    return tracks
