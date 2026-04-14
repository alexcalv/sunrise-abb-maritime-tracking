from __future__ import annotations

import csv
import json
from pathlib import Path


def aggregate_clip(worker_results_dir: str, aggregated_dir: str, clip_id: str) -> dict:
    in_dir = Path(worker_results_dir) / clip_id
    out_dir = Path(aggregated_dir) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fp in sorted(in_dir.glob("*.json")):
        rows.append(json.loads(fp.read_text(encoding="utf-8")))

    if not rows:
        raise RuntimeError(f"No worker results found for clip_id={clip_id}")

    summary_csv = out_dir / "summary.csv"
    mot_txt = out_dir / f"{clip_id}.txt"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_ms", "num_detections", "model_name", "worker_id", "image_path"])
        for row in rows:
            writer.writerow([
                row["frame_index"],
                row.get("timestamp_ms"),
                len(row.get("detections", [])),
                row["model_name"],
                row["worker_id"],
                row["image_path"],
            ])

    with mot_txt.open("w", encoding="utf-8") as f:
        pseudo_track_id = -1
        for row in rows:
            for det in row.get("detections", []):
                pseudo_track_id -= 1
                f.write(
                    f'{det["frame_index"]},{pseudo_track_id},{det["x"]:.4f},{det["y"]:.4f},{det["w"]:.4f},{det["h"]:.4f},{det["confidence"]:.6f},{det["class_id"]},1\n'
                )

    return {
        "clip_id": clip_id,
        "summary_csv": str(summary_csv),
        "mot_txt": str(mot_txt),
        "num_frames": len(rows),
    }
