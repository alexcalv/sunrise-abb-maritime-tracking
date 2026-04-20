from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_mot_row(raw_row: list[str], path: Path, line_num: int) -> list[str]:
    row = [part.strip() for part in raw_row if part is not None]
    if not row:
        return []
    if len(row) == 1:
        cell = row[0].strip()
        if not cell or cell.startswith("#"):
            return []
        if "," in cell:
            row = [part.strip() for part in cell.split(",")]
        else:
            row = cell.split()

    if not row or row[0].startswith("#"):
        return []
    if len(row) < 6:
        raise ValueError(f"Invalid MOT row in {path} line {line_num}: expected at least 6 columns, got {len(row)}")
    return row


def load_mot_rows(path: Path, frame_offset: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_num, raw_row in enumerate(reader, start=1):
            row = parse_mot_row(raw_row, path, line_num)
            if not row:
                continue

            rows.append(
                {
                    "frame": int(float(row[0])) + frame_offset,
                    "id": int(float(row[1])),
                    "bbox": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                    "confidence": float(row[6]) if len(row) > 6 else 1.0,
                    "class_id": int(float(row[7])) if len(row) > 7 else 0,
                    "visibility": float(row[8]) if len(row) > 8 else 1.0,
                }
            )
    return rows


def load_mot_frames(path: Path, frame_offset: int = 0) -> dict[int, list[dict[str, Any]]]:
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in load_mot_rows(path, frame_offset=frame_offset):
        rows_by_frame.setdefault(row["frame"], []).append(row)
    return rows_by_frame


def write_mot_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            x, y, w, h = row["bbox"]
            writer.writerow(
                [
                    row["frame"],
                    row["id"],
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{w:.4f}",
                    f"{h:.4f}",
                    f"{float(row['confidence']):.6f}",
                    row["class_id"],
                    f"{float(row['visibility']):.6f}",
                ]
            )


def _track_id_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    return dict(sorted(Counter(row["id"] for row in rows).items()))


def list_track_ids(input_path: str) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = load_mot_rows(path)
    return {
        "input": str(path),
        "num_rows": len(rows),
        "num_track_ids": len(_track_id_counts(rows)),
        "track_id_counts": _track_id_counts(rows),
    }


def merge_tracker_ids_to_gt(
    input_path: str,
    output_path: str,
    source_ids: list[int],
    gt_id: int = 1,
    class_id: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    sort_rows: bool = False,
) -> dict[str, Any]:
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")
    if not source_ids:
        raise ValueError("source_ids must not be empty")

    rows = load_mot_rows(in_path)
    selected: list[dict[str, Any]] = []
    source_id_set = set(source_ids)

    for row in rows:
        if row["id"] < 0:
            continue
        if row["id"] not in source_id_set:
            continue
        if frame_start is not None and row["frame"] < frame_start:
            continue
        if frame_end is not None and row["frame"] > frame_end:
            continue
        selected.append(row)

    if not selected:
        raise RuntimeError("No matching rows found. Inspect the tracker IDs first with --list-ids.")

    if sort_rows:
        selected.sort(key=lambda row: (row["frame"], row["id"]))

    gt_rows = [
        {
            "frame": row["frame"],
            "id": gt_id,
            "bbox": row["bbox"],
            "confidence": 1.0,
            "class_id": row["class_id"] if class_id is None else class_id,
            "visibility": 1.0,
        }
        for row in selected
    ]
    write_mot_rows(out_path, gt_rows)

    frames = [row["frame"] for row in gt_rows]
    output_class_ids = sorted({row["class_id"] for row in gt_rows})
    return {
        "input": str(in_path),
        "output": str(out_path),
        "source_ids": sorted(source_id_set),
        "gt_id": gt_id,
        "rows_written": len(gt_rows),
        "frame_range": [min(frames), max(frames)],
        "class_ids_written": output_class_ids,
    }


def _sorted_by_frame_id(rows: list[dict[str, Any]]) -> bool:
    pairs = [(row["frame"], row["id"]) for row in rows]
    return all(left <= right for left, right in zip(pairs, pairs[1:]))


def summarize_gt_file(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "status": "invalid",
        "error": None,
        "num_rows": 0,
        "frame_start": None,
        "frame_end": None,
        "num_unique_ids": 0,
        "unique_class_ids": [],
        "sorted_by_frame_id": True,
        "nonpositive_frame_rows": 0,
        "nonpositive_id_rows": 0,
        "nonpositive_box_rows": 0,
        "duplicate_frame_id_rows": 0,
        "non_gt_conf_rows": 0,
        "visibility_out_of_range_rows": 0,
    }

    try:
        rows = load_mot_rows(path)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary

    summary["num_rows"] = len(rows)
    if not rows:
        summary["error"] = "No MOT rows found"
        return summary

    frames = [row["frame"] for row in rows]
    summary["frame_start"] = min(frames)
    summary["frame_end"] = max(frames)
    summary["num_unique_ids"] = len({row["id"] for row in rows})
    summary["unique_class_ids"] = sorted({row["class_id"] for row in rows})
    summary["sorted_by_frame_id"] = _sorted_by_frame_id(rows)
    summary["nonpositive_frame_rows"] = sum(1 for row in rows if row["frame"] < 1)
    summary["nonpositive_id_rows"] = sum(1 for row in rows if row["id"] < 1)
    summary["nonpositive_box_rows"] = sum(1 for row in rows if row["bbox"][2] <= 0 or row["bbox"][3] <= 0)
    summary["non_gt_conf_rows"] = sum(1 for row in rows if abs(row["confidence"] - 1.0) > 1e-9)
    summary["visibility_out_of_range_rows"] = sum(1 for row in rows if not 0.0 <= row["visibility"] <= 1.0)

    counts = Counter((row["frame"], row["id"]) for row in rows)
    summary["duplicate_frame_id_rows"] = sum(count - 1 for count in counts.values() if count > 1)

    invalid = any(
        [
            summary["nonpositive_frame_rows"],
            summary["nonpositive_id_rows"],
            summary["nonpositive_box_rows"],
            summary["duplicate_frame_id_rows"],
        ]
    )
    warning = any(
        [
            not summary["sorted_by_frame_id"],
            summary["non_gt_conf_rows"],
            summary["visibility_out_of_range_rows"],
        ]
    )

    if invalid:
        summary["status"] = "invalid"
    elif warning:
        summary["status"] = "warning"
    else:
        summary["status"] = "ok"

    return summary


def validate_gt_dir(gt_dir: str, output_json: str | None = None, output_csv: str | None = None) -> dict[str, Any]:
    gt_root = Path(gt_dir)
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_root}")

    gt_files = sorted(gt_root.glob("*.txt"))
    if not gt_files:
        raise RuntimeError(f"No GT txt files found in {gt_root}")

    file_summaries = [summarize_gt_file(path) for path in gt_files]
    result = {
        "gt_dir": str(gt_root),
        "num_files": len(file_summaries),
        "num_ok": sum(1 for item in file_summaries if item["status"] == "ok"),
        "num_warning": sum(1 for item in file_summaries if item["status"] == "warning"),
        "num_invalid": sum(1 for item in file_summaries if item["status"] == "invalid"),
        "files": file_summaries,
        "output_json": None,
        "output_csv": None,
    }

    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output_json"] = str(json_path)

    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "file",
                "status",
                "error",
                "num_rows",
                "frame_start",
                "frame_end",
                "num_unique_ids",
                "unique_class_ids",
                "sorted_by_frame_id",
                "nonpositive_frame_rows",
                "nonpositive_id_rows",
                "nonpositive_box_rows",
                "duplicate_frame_id_rows",
                "non_gt_conf_rows",
                "visibility_out_of_range_rows",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in file_summaries:
                row = dict(item)
                row["unique_class_ids"] = " ".join(str(value) for value in row["unique_class_ids"])
                writer.writerow({field: row.get(field) for field in fieldnames})
        result["output_csv"] = str(csv_path)

    return result
