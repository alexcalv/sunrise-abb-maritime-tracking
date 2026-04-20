from __future__ import annotations

import json
import time
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sequence_name(source: str, result_path: str | None) -> str:
    source_path = Path(source)
    default_name = source_path.stem or source_path.name or "stream"
    if result_path is None:
        return default_name

    path = Path(result_path)
    if source_path.is_dir() and path.suffix.lower() in IMAGE_EXTS:
        return default_name
    return path.stem or default_name


def _write_mot_rows(
    mot_path: Path,
    frame_index: int,
    track_ids: list[int],
    xywh_boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> int:
    rows_written = 0
    with mot_path.open("a", encoding="utf-8") as handle:
        for track_id, box, confidence, class_id in zip(track_ids, xywh_boxes, confidences, class_ids):
            xc, yc, w, h = (float(value) for value in box)
            x = xc - (w / 2.0)
            y = yc - (h / 2.0)
            handle.write(
                f"{frame_index},{track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{confidence:.6f},{class_id},1\n"
            )
            rows_written += 1
    return rows_written


def run_tracking(
    model_path: str,
    source: str,
    tracker_yaml: str,
    device: str,
    project: str,
    name: str,
):
    model = YOLO(model_path)
    started_at = time.time()

    results = model.track(
        source=source,
        tracker=tracker_yaml,
        device=device,
        project=project,
        name=name,
        save=True,
        save_txt=False,
        persist=True,
        stream=True,
        verbose=False,
    )

    output_root: Path | None = None
    mot_dir: Path | None = None
    frame_counters: dict[str, int] = {}
    rows_written: dict[str, int] = {}
    detections_seen: dict[str, int] = {}
    detections_without_ids: dict[str, int] = {}
    track_ids_seen: dict[str, set[int]] = {}

    for result in results:
        if output_root is None:
            save_dir = getattr(result, "save_dir", None)
            output_root = Path(save_dir) if save_dir else Path(project) / name
            mot_dir = output_root / "mot"
            mot_dir.mkdir(parents=True, exist_ok=True)

        seq_name = _sequence_name(source, getattr(result, "path", None))
        frame_index = frame_counters.get(seq_name, 0) + 1
        frame_counters[seq_name] = frame_index

        boxes = result.boxes
        if boxes is None or boxes.xywh is None or len(boxes) == 0:
            continue

        xywh_boxes = boxes.xywh.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xywh_boxes)
        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xywh_boxes)
        detections_seen[seq_name] = detections_seen.get(seq_name, 0) + len(xywh_boxes)

        if boxes.id is None:
            detections_without_ids[seq_name] = detections_without_ids.get(seq_name, 0) + len(xywh_boxes)
            continue

        seq_track_ids = boxes.id.int().cpu().tolist()
        track_ids_seen.setdefault(seq_name, set()).update(seq_track_ids)

        mot_path = mot_dir / f"{seq_name}.txt"
        rows_written[seq_name] = rows_written.get(seq_name, 0) + _write_mot_rows(
            mot_path=mot_path,
            frame_index=frame_index,
            track_ids=seq_track_ids,
            xywh_boxes=xywh_boxes,
            confidences=confidences,
            class_ids=class_ids,
        )

    finished_at = time.time()
    wall_time_seconds = max(finished_at - started_at, 0.0)
    total_frames = sum(frame_counters.values())

    if output_root is None:
        output_root = Path(project) / name
        mot_dir = output_root / "mot"
        mot_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "model_path": model_path,
        "source": source,
        "tracker_yaml": tracker_yaml,
        "device": device,
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "total_frames": total_frames,
        "total_rows_written": sum(rows_written.values()),
        "wall_time_seconds": round(wall_time_seconds, 6),
        "effective_fps": round((total_frames / wall_time_seconds), 6) if wall_time_seconds > 0 else None,
        "sequences": {},
    }

    sequence_names = set(frame_counters) | set(rows_written) | set(detections_seen) | set(detections_without_ids)
    for seq_name in sorted(sequence_names):
        run_summary["sequences"][seq_name] = {
            "frames_processed": frame_counters.get(seq_name, 0),
            "detections_seen": detections_seen.get(seq_name, 0),
            "detections_without_ids": detections_without_ids.get(seq_name, 0),
            "rows_written": rows_written.get(seq_name, 0),
            "unique_track_ids": sorted(track_ids_seen.get(seq_name, set())),
            "num_unique_track_ids": len(track_ids_seen.get(seq_name, set())),
            "mot_path": str(mot_dir / f"{seq_name}.txt"),
        }

    run_summary_path = output_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return {
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "run_summary": str(run_summary_path),
        "frames_processed": total_frames,
        "rows_written": sum(rows_written.values()),
    }
