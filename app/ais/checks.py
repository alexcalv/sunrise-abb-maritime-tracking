from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from ais import AisConfig, align_ais_tracks_to_frames, assign_ais_to_tracks, load_ais_file
from evaluation.mot import load_mot_rows, write_mot_rows


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "mmsi", "x", "y", "sog", "cog", "heading"])
        writer.writeheader()
        for frame in range(5):
            writer.writerow({"timestamp": frame, "mmsi": "111", "x": 100 + frame * 10, "y": 50, "sog": 10, "cog": 0, "heading": 0})
            writer.writerow({"timestamp": frame, "mmsi": "222", "x": 100 - frame * 10, "y": 80, "sog": 10, "cog": 180, "heading": 180})


def _write_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"timestamp": frame, "mmsi": "333", "x": 20 + frame, "y": 30, "sog": 4, "cog": 0}
                for frame in range(3)
            ]
        ),
        encoding="utf-8",
    )


def _write_mot(path: Path) -> None:
    rows = []
    for frame in range(1, 6):
        rows.append({"frame": frame, "id": 1, "bbox": [95 + (frame - 1) * 10, 45, 10, 10], "confidence": 1.0, "class_id": 0, "visibility": 1.0})
        rows.append({"frame": frame, "id": 2, "bbox": [95 - (frame - 1) * 10, 75, 10, 10], "confidence": 1.0, "class_id": 0, "visibility": 1.0})
    write_mot_rows(path, rows)


def run_checks() -> dict[str, object]:
    """Run lightweight AIS parser, alignment, and neutral-missing-data checks."""

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        csv_path = tmp_dir / "ais.csv"
        json_path = tmp_dir / "ais.json"
        mot_path = tmp_dir / "tracks.txt"
        _write_csv(csv_path)
        _write_json(json_path)
        _write_mot(mot_path)

        config = AisConfig(diagnostics_enabled=True, fps=1.0, video_start_time=0.0)
        csv_tracks = load_ais_file(csv_path, config)
        results.append({"case": "parse_csv", "passed": sorted(csv_tracks) == ["111", "222"]})
        json_tracks = load_ais_file(json_path, config)
        results.append({"case": "parse_json", "passed": sorted(json_tracks) == ["333"]})

        aligned = align_ais_tracks_to_frames(csv_tracks, frame_count=5, fps=1.0, video_start_time=0.0, config=config)
        results.append({"case": "align_xy_to_frames", "passed": len(aligned) == 5 and len(aligned[1]) == 2})

        assignments = assign_ais_to_tracks(load_mot_rows(mot_path), aligned, config)
        assigned = {assignment.track_id: assignment.assigned_mmsi for assignment in assignments}
        results.append({"case": "assign_mmsi_to_mot_track", "passed": assigned.get(1) == "111"})

        missing = assign_ais_to_tracks(load_mot_rows(mot_path), {}, config)
        results.append(
            {
                "case": "missing_ais_neutral",
                "passed": all(abs(item.score - config.neutral_score) < 1e-9 for item in missing),
            }
        )

    return {"passed": all(bool(item["passed"]) for item in results), "cases": results}
