from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ais.loader import build_layer_from_json_dict
from ais.schemas import AisTrackLayer


def read_video_fps(path: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 1e-3:
        return None
    return fps


def resolve_ais_track_layer(
    ais_path: Path,
    *,
    video_for_fps: Path | None,
    fps_override: float | None,
    time_offset_ms_override: int | None = None,
) -> AisTrackLayer:
    """
    Load AIS file; for timestamp_ms sync, supply fps from override or video container.
    """
    raw: dict[str, Any] = json.loads(ais_path.read_text(encoding="utf-8"))
    if time_offset_ms_override is not None:
        sync = dict(raw.get("sync") or {})
        sync["time_offset_ms"] = int(time_offset_ms_override)
        raw["sync"] = sync

    sync = raw.get("sync") or {}
    mode = str(sync.get("mode", "frame_index"))
    video_fps: float | None = None
    if mode == "timestamp_ms":
        video_fps = float(fps_override) if fps_override is not None else None
        if video_fps is None and video_for_fps is not None:
            video_fps = read_video_fps(video_for_fps)
        if video_fps is None or video_fps <= 0:
            raise ValueError(
                "AIS timestamp_ms mode requires --ais-fps or a video file with a valid FPS in the container"
            )
    return build_layer_from_json_dict(raw, video_fps=video_fps)


def ais_layer_to_sidecar_dict(layer: AisTrackLayer, *, source_path: str) -> dict[str, Any]:
    by_frame_json: dict[str, list[dict[str, Any]]] = {}
    for frame, positions in sorted(layer.positions_by_frame.items()):
        by_frame_json[str(frame)] = [p.to_json_dict() for p in positions]
    return {
        "version": 1,
        "kind": "ais_augmentation_layer",
        "enabled": True,
        "source": source_path,
        "ais_schema_version": layer.version,
        "sync": layer.sync,
        "stats": {
            "frames_with_ais": len(layer.positions_by_frame),
            "total_fixes": layer.total_fixes(),
            "max_frame": layer.max_frame(),
        },
        "by_frame": by_frame_json,
    }


def write_ais_sidecar(output_dir: Path, layer: AisTrackLayer, *, source_path: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ais_layer.json"
    payload = ais_layer_to_sidecar_dict(layer, source_path=source_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def merge_ais_into_fusion_summary(summary_path: Path, ais_sidecar_path: Path) -> None:
    """Attach a pointer to the AIS sidecar into fusion_mot_pair_summary.json (non-destructive merge)."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ais_layer"] = {
        "enabled": True,
        "sidecar": str(ais_sidecar_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def merge_ais_into_run_summary(run_summary_path: Path, ais_sidecar_path: Path) -> None:
    """Attach a pointer to the AIS sidecar into tracker run_summary.json."""
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    summary["ais_layer"] = {
        "enabled": True,
        "sidecar": str(ais_sidecar_path),
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
