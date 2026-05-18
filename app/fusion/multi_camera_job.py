from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MultiCameraJob:
    """Parameters for a two-stream track + fuse + render run (typically loaded from JSON)."""

    main_video: Path
    second_video: Path
    model_path: str | None = None
    project_root: Path | None = None

    
    run_name: str = "multi_cam_run"
    tracker_yaml: str | None = None
    device: str | None = None
    image_width: int = 1280
    image_height: int = 720
    max_center_distance_norm: float = 0.55
    camera_a_id: str = "camera_a"
    camera_b_id: str = "camera_b"
    platform: str = "simulation"
    label_mode: str = "id"
    fusion_match_mode: str = "auto"
    video_layout: str = "side_by_side_hd"
    ais_file: Path | None = None
    ais_fps: float | None = None
    ais_time_offset_ms: int | None = None


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_existing_file(raw: str, base_dir: Path) -> str | None:
    """Return a string path for an optional file; try job dir then /workspace/models (Docker)."""
    p = Path(raw)
    if p.is_absolute():
        return str(p) if p.is_file() else str(p)
    cand = (base_dir / p).resolve()
    if cand.is_file():
        return str(cand)
    docker_models = Path("/workspace/models") / p.name
    if docker_models.is_file():
        return str(docker_models)
    docker_abs = Path("/workspace") / p
    if docker_abs.is_file():
        return str(docker_abs)
    return str(cand)


def load_multi_camera_job(path: Path) -> MultiCameraJob:
    """
    Load a job definition from JSON.

    Relative paths in the file are resolved against the job file's parent directory
    (so you can keep a job next to your repo and use paths like \"../data/videos/a.mp4\").
    """
    path = path.resolve()
    base_dir = path.parent
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def req(key: str) -> Any:
        if key not in data:
            raise KeyError(f"Job file {path} missing required key: {key!r}")
        return data[key]

    model_raw = data.get("model_path")
    tracker_raw = data.get("tracker_yaml")
    ais_raw = data.get("ais_file")

    ais_path: Path | None = None
    if ais_raw is not None:
        if not str(ais_raw).strip():
            raise ValueError(f"Job file {path}: ais_file must be a non-empty path when provided")
        ais_path = _resolve_path(str(ais_raw), base_dir)
        if not ais_path.is_file():
            raise FileNotFoundError(f"AIS file from job not found: {ais_path}")

    ais_fps = data.get("ais_fps")
    ais_time_raw = data.get("ais_time_offset_ms")

    return MultiCameraJob(
        main_video=_resolve_path(req("main_video"), base_dir),
        second_video=_resolve_path(req("second_video"), base_dir),
        model_path=_resolve_existing_file(str(model_raw), base_dir) if model_raw else None,
        project_root=_resolve_path(data["project_root"], base_dir) if data.get("project_root") else None,
        run_name=str(data.get("run_name", "multi_cam_run")),
        tracker_yaml=_resolve_existing_file(str(tracker_raw), base_dir) if tracker_raw else None,
        device=data.get("device"),
        image_width=int(data.get("image_width", 1280)),
        image_height=int(data.get("image_height", 720)),
        max_center_distance_norm=float(data.get("max_center_distance_norm", 0.55)),
        camera_a_id=str(data.get("camera_a_id", "camera_a")),
        camera_b_id=str(data.get("camera_b_id", "camera_b")),
        platform=str(data.get("platform", "simulation")),
        label_mode=str(data.get("label_mode", "id")),
        fusion_match_mode=str(data.get("fusion_match_mode", "auto")),
        video_layout=str(data.get("video_layout", "side_by_side_hd")),
        ais_file=ais_path,
        ais_fps=float(ais_fps) if ais_fps is not None else None,
        ais_time_offset_ms=int(ais_time_raw) if ais_time_raw is not None else None,
    )
