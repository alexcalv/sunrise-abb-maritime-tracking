from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2

from common.config import settings
from evaluation.mot import load_mot_frames, load_mot_rows, write_mot_rows
from fusion.pairwise_center_fusion import PairwiseNormalizedCenterFusion
from fusion.schemas import StreamTrackObservation
from output.render_video import render_combined_video_from_mot, render_video_from_mot
from tracking.tracker_runner import run_tracking


def default_model_path() -> str:
    """Prefer mounted weights under /workspace/models when present (Docker layout)."""
    docker_models = Path("/workspace/models") / settings.yolo_model
    if docker_models.is_file():
        return str(docker_models)
    return settings.yolo_model


def read_video_frame_size(path: Path) -> tuple[int, int]:
    """Return (width, height) from the video container; raises if unavailable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for resolution probe: {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video resolution for {path}: {w}x{h}")
    return w, h


def _safe_run_segment(stem: str) -> str:
    # Ultralytics run names are path segments; avoid separators and odd characters.
    cleaned = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE).strip("._-")
    return cleaned or "stream"


def _missing_mot_help(mot_dir: Path, video: Path, run_summary_path: str | None) -> str:
    lines = [
        f"MOT export missing for {video.name}.",
        f"Expected file (typical): {mot_dir / f'{video.stem}.txt'}",
        f"MOT directory: {mot_dir}",
    ]
    if mot_dir.is_dir():
        found = sorted(mot_dir.glob("*.txt"))
        lines.append(f"Found MOT files: {[p.name for p in found] if found else '(none)'}")
    if run_summary_path and Path(run_summary_path).is_file():
        lines.append(f"Diagnostics: open {run_summary_path}")
        try:
            data = json.loads(Path(run_summary_path).read_text(encoding="utf-8"))
            for seq, info in sorted((data.get("sequences") or {}).items()):
                lines.append(
                    f"  [{seq}] detections_seen={info.get('detections_seen')} "
                    f"detections_without_ids={info.get('detections_without_ids')} "
                    f"rows_written={info.get('rows_written')}"
                )
        except (OSError, json.JSONDecodeError):
            lines.append("  (could not parse run_summary.json)")
        lines.append(
            "If rows_written=0 but detections_seen>0, the tracker never assigned IDs "
            "(try lowering BoT-SORT thresholds or check class filters / model quality)."
        )
    return "\n".join(lines)


def resolve_mot_export_path(mot_dir: Path, video: Path, run_summary_path: str | None) -> Path:
    """
    Locate the MOT text file produced by run_tracking.

    Prefer <video_stem>.txt, then run_summary.json hints, then a single *.txt in mot_dir.
    """
    mot_dir = mot_dir.resolve()
    expected = mot_dir / f"{video.stem}.txt"
    if expected.is_file():
        return expected

    if run_summary_path:
        summary_p = Path(run_summary_path)
        if summary_p.is_file():
            try:
                data = json.loads(summary_p.read_text(encoding="utf-8"))
                seqs: dict[str, Any] = data.get("sequences") or {}
                for seq_key, info in seqs.items():
                    mp = info.get("mot_path")
                    if not mp:
                        continue
                    p = Path(mp)
                    if p.is_file() and (seq_key == video.stem or len(seqs) == 1):
                        return p
            except (OSError, json.JSONDecodeError):
                pass

    candidates = sorted(mot_dir.glob("*.txt"))
    if len(candidates) == 1:
        return candidates[0]
    stem_lower = video.stem.lower()
    for c in candidates:
        if c.stem.lower() == stem_lower:
            return c

    raise FileNotFoundError(_missing_mot_help(mot_dir, video, run_summary_path))


def _rows_to_observations(
    rows: list[dict[str, Any]],
    camera_id: str,
    platform: str,
) -> list[StreamTrackObservation]:
    out: list[StreamTrackObservation] = []
    for row in rows:
        x, y, w, h = row["bbox"]
        out.append(
            StreamTrackObservation(
                camera_id=camera_id,
                platform=platform,
                frame_index=int(row["frame"]),
                local_track_id=int(row["id"]),
                bbox_xywh=(float(x), float(y), float(w), float(h)),
                confidence=float(row["confidence"]),
                class_id=int(row["class_id"]),
            )
        )
    return out


def fuse_two_mot_files_to_disk(
    mot_path_a: Path,
    mot_path_b: Path,
    camera_a_id: str,
    camera_b_id: str,
    platform: str,
    image_width: int,
    image_height: int,
    max_center_distance_norm: float,
    fused_out_a: Path,
    fused_out_b: Path,
    *,
    image_size_by_camera: dict[str, tuple[int, int]] | None = None,
    fusion_match_mode: str = "auto",
) -> dict[str, Any]:
    rows_a = load_mot_rows(mot_path_a)
    rows_b = load_mot_rows(mot_path_b)
    by_frame_a = load_mot_frames(mot_path_a)
    by_frame_b = load_mot_frames(mot_path_b)
    all_frames = sorted(set(by_frame_a) | set(by_frame_b))

    fusion = PairwiseNormalizedCenterFusion(
        reference_camera_id=camera_a_id,
        other_camera_id=camera_b_id,
        max_center_distance_norm=max_center_distance_norm,
        match_mode=fusion_match_mode,
    )
    fusion.reset()

    fused_a: list[dict[str, Any]] = []
    fused_b: list[dict[str, Any]] = []

    for frame_index in all_frames:
        obs_a = _rows_to_observations(by_frame_a.get(frame_index, []), camera_a_id, platform)
        obs_b = _rows_to_observations(by_frame_b.get(frame_index, []), camera_b_id, platform)
        mapping = fusion.fuse_frame(
            frame_index,
            {camera_a_id: obs_a, camera_b_id: obs_b},
            image_width,
            image_height,
            image_size_by_camera=image_size_by_camera,
        )
        for row in by_frame_a.get(frame_index, []):
            gid = mapping[(camera_a_id, int(row["id"]))]
            fused_a.append({**row, "id": gid})
        for row in by_frame_b.get(frame_index, []):
            gid = mapping[(camera_b_id, int(row["id"]))]
            fused_b.append({**row, "id": gid})

    write_mot_rows(fused_out_a, fused_a)
    write_mot_rows(fused_out_b, fused_b)

    key_to_global = fusion.export_key_to_global()

    return {
        "mot_a": str(mot_path_a),
        "mot_b": str(mot_path_b),
        "camera_a": camera_a_id,
        "camera_b": camera_b_id,
        "platform": platform,
        "image_wh": [image_width, image_height],
        "max_center_distance": max_center_distance_norm,
        "fused_mot_a": str(fused_out_a),
        "fused_mot_b": str(fused_out_b),
        "num_frames": len(all_frames),
        "key_to_global_final": key_to_global,
        "num_rows_a": len(rows_a),
        "num_rows_b": len(rows_b),
        "fusion_match_mode": fusion_match_mode,
        "image_size_by_camera": {k: [v[0], v[1]] for k, v in (image_size_by_camera or {}).items()},
    }


def run_dual_camera_track_and_fuse(
    *,
    model_path: str,
    main_video: Path,
    second_video: Path,
    tracker_yaml: str,
    device: str,
    project_root: Path,
    run_name: str,
    image_width: int,
    image_height: int,
    max_center_distance_norm: float,
    camera_a_id: str = "camera_a",
    camera_b_id: str = "camera_b",
    platform: str = "simulation",
    label_mode: str = "id",
    fusion_match_mode: str = "auto",
    video_layout: str = "side_by_side_hd",
    ais_file: Path | str | None = None,
    ais_fps: float | None = None,
    ais_time_offset_ms: int | None = None,
) -> dict[str, Any]:
    """
    Run the tracker on two videos, fuse global identities, write fused MOT + overlay videos.

    Raw Ultralytics outputs live under project_root/<run_name>/raw/<video_stem>/.
    Fused artifacts: project_root/<run_name>/fusion/ and .../videos/.
    """
    project_root = project_root.resolve()
    main_video = main_video.resolve()
    second_video = second_video.resolve()

    stem_a = _safe_run_segment(main_video.stem)
    stem_b = _safe_run_segment(second_video.stem)
    raw_main_name = f"{run_name}/raw/{stem_a}"
    raw_second_name = f"{run_name}/raw/{stem_b}"

    track_main = run_tracking(
        model_path=model_path,
        source=str(main_video),
        tracker_yaml=tracker_yaml,
        device=device,
        project=str(project_root),
        name=raw_main_name,
    )
    track_second = run_tracking(
        model_path=model_path,
        source=str(second_video),
        tracker_yaml=tracker_yaml,
        device=device,
        project=str(project_root),
        name=raw_second_name,
    )

    mot_main = resolve_mot_export_path(
        Path(track_main["mot_dir"]),
        main_video,
        track_main.get("run_summary"),
    )
    mot_second = resolve_mot_export_path(
        Path(track_second["mot_dir"]),
        second_video,
        track_second.get("run_summary"),
    )

    try:
        wh_main = read_video_frame_size(main_video)
    except RuntimeError:
        wh_main = (max(1, image_width), max(1, image_height))
    try:
        wh_second = read_video_frame_size(second_video)
    except RuntimeError:
        wh_second = (max(1, image_width), max(1, image_height))

    image_size_by_camera: dict[str, tuple[int, int]] = {
        camera_a_id: wh_main,
        camera_b_id: wh_second,
    }

    fusion_dir = project_root / run_name / "fusion"
    videos_dir = project_root / run_name / "videos"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    fused_main = fusion_dir / f"{main_video.stem}_fused.txt"
    fused_second = fusion_dir / f"{second_video.stem}_fused.txt"
    fusion_summary = fuse_two_mot_files_to_disk(
        mot_path_a=mot_main,
        mot_path_b=mot_second,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        platform=platform,
        image_width=image_width,
        image_height=image_height,
        max_center_distance_norm=max_center_distance_norm,
        fused_out_a=fused_main,
        fused_out_b=fused_second,
        image_size_by_camera=image_size_by_camera,
        fusion_match_mode=fusion_match_mode,
    )
    fusion_summary["raw_run_summary_main"] = track_main.get("run_summary")
    fusion_summary["raw_run_summary_second"] = track_second.get("run_summary")

    summary_path = fusion_dir / "fusion_mot_pair_summary.json"
    summary_path.write_text(json.dumps(fusion_summary, indent=2), encoding="utf-8")

    ais_sidecar: Path | None = None
    if ais_file is not None:
        from ais.pipeline import merge_ais_into_fusion_summary, resolve_ais_track_layer, write_ais_sidecar

        ap = Path(str(ais_file))
        if not ap.is_file():
            raise FileNotFoundError(f"AIS file not found: {ap}")
        layer = resolve_ais_track_layer(
            ap,
            video_for_fps=main_video,
            fps_override=ais_fps,
            time_offset_ms_override=ais_time_offset_ms,
        )
        ais_sidecar = write_ais_sidecar(fusion_dir, layer, source_path=str(ap.resolve()))
        merge_ais_into_fusion_summary(summary_path, ais_sidecar)
        fusion_summary["ais_layer"] = {"enabled": True, "sidecar": str(ais_sidecar)}

    out_main_mp4 = videos_dir / f"{main_video.stem}_fused_overlay.mp4"
    out_second_mp4 = videos_dir / f"{second_video.stem}_fused_overlay.mp4"
    render_main = render_video_from_mot(
        source_video_path=str(main_video),
        mot_file_path=str(fused_main),
        output_video_path=str(out_main_mp4),
        fps=None,
        label_mode=label_mode,
    )
    render_second = render_video_from_mot(
        source_video_path=str(second_video),
        mot_file_path=str(fused_second),
        output_video_path=str(out_second_mp4),
        fps=None,
        label_mode=label_mode,
    )

    out_combined_mp4 = videos_dir / f"{main_video.stem}_{second_video.stem}_combined_overlay.mp4"
    render_combined = render_combined_video_from_mot(
        source_video_a=str(main_video),
        source_video_b=str(second_video),
        mot_file_a=str(fused_main),
        mot_file_b=str(fused_second),
        output_video_path=str(out_combined_mp4),
        fps=None,
        label_mode=label_mode,
        layout=video_layout,
        camera_a_label=camera_a_id,
        camera_b_label=camera_b_id,
    )

    out: dict[str, Any] = {
        "project_root": str(project_root),
        "run_name": run_name,
        "output_layout": {
            "raw_tracking": str(project_root / run_name / "raw"),
            "fusion_dir": str(fusion_dir),
            "videos_dir": str(videos_dir),
        },
        "mot_main": str(mot_main),
        "mot_second": str(mot_second),
        "fused_mot_main": str(fused_main),
        "fused_mot_second": str(fused_second),
        "fusion_summary_json": str(summary_path),
        "video_main": str(out_main_mp4),
        "video_second": str(out_second_mp4),
        "video_combined": str(out_combined_mp4),
        "render_main": render_main,
        "render_second": render_second,
        "render_combined": render_combined,
        "fusion_match_mode": fusion_match_mode,
        "image_size_by_camera": {k: [v[0], v[1]] for k, v in image_size_by_camera.items()},
    }
    if ais_sidecar is not None:
        out["ais_layer_sidecar"] = str(ais_sidecar)
    return out
