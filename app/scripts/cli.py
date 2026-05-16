from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from broker.celery_app import celery_app
from common.config import settings
from evaluation.localization import evaluate_sparse_localization_dir
from evaluation.mot import list_track_ids, merge_tracker_ids_to_gt, validate_gt_dir
from evaluation.metrics import evaluate_mot_dir
from ingestion.ingestor import build_tasks
from output.aggregator import aggregate_clip


# Default stitch config path (duplicates stitching.runner.DEFAULT_CONFIG_PATH to avoid importing stitch stack at CLI import time).
_DEFAULT_STITCH_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stitching" / "default.yaml"

_VIDEO_EXTS_AIS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".gif"}


def _get_celery_app():
    """Import Celery only when pipeline commands need it (track/render work without celery installed)."""
    from broker.celery_app import celery_app

    return celery_app


def wait_for_tasks(async_results, poll_seconds: float = 2.0) -> None:
    pending = set(range(len(async_results)))
    while pending:
        finished_now = []
        for idx in pending:
            if async_results[idx].ready():
                async_results[idx].get(propagate=True)
                finished_now.append(idx)

        for idx in finished_now:
            pending.remove(idx)

        if pending:
            print(f"Waiting for {len(pending)} batch tasks...")
            time.sleep(poll_seconds)


def cmd_pipeline_ingest(args):
    from ingestion.ingestor import build_tasks

    tasks = build_tasks(
        source=args.source,
        frames_dir=settings.frames_dir,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
    )
    celery_app = _get_celery_app()
    for task in tasks:
        celery_app.send_task("maritime.detect_batch", args=[task.model_dump()])
    print(f"Queued {len(tasks)} batches for clip {Path(args.source).stem}")


def cmd_pipeline_aggregate(args):
    print(aggregate_clip(settings.worker_results_dir, settings.aggregated_dir, args.clip_id))


def cmd_track(args):
    from tracking.tracker_runner import run_tracking

    result = run_tracking(
        model_path=args.model,
        source=args.source,
        tracker_yaml=args.tracker,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    if getattr(args, "ais_file", None):
        from ais.pipeline import merge_ais_into_run_summary, resolve_ais_track_layer, write_ais_sidecar

        ais_p = Path(args.ais_file)
        if not ais_p.is_file():
            raise FileNotFoundError(f"AIS file not found: {ais_p}")
        src = Path(args.source)
        video_probe = src if src.is_file() and src.suffix.lower() in _VIDEO_EXTS_AIS else None
        layer = resolve_ais_track_layer(
            ais_p,
            video_for_fps=video_probe,
            fps_override=getattr(args, "ais_fps", None),
            time_offset_ms_override=getattr(args, "ais_time_offset_ms", None),
        )
        out_root = Path(result["output_root"])
        sidecar = write_ais_sidecar(out_root, layer, source_path=str(ais_p.resolve()))
        merge_ais_into_run_summary(Path(result["run_summary"]), sidecar)
        result["ais_layer_sidecar"] = str(sidecar)
    print(result)


def _default_track_project_dir() -> Path:
    # app/scripts/cli.py -> parents[2] is the workspace root (Docker: /workspace, local: repo root).
    return Path(__file__).resolve().parents[2] / "outputs" / "track"


def _default_tracker_yaml_path() -> str:
    docker_path = Path("/workspace/config/trackers/botsort_maritime.yaml")
    if docker_path.is_file():
        return str(docker_path)
    repo = Path(__file__).resolve().parents[2]
    return str(repo / "config" / "trackers" / "botsort_maritime.yaml")


def _register_track_multi_camera_parsers(sub: Any) -> None:
    """Register both command names so Docker images and --help list them explicitly (no alias quirks)."""
    help_text = (
        "Two videos (any names/paths): run tracker on each, fuse global IDs, write MOT + overlay MP4s. "
        "Use --job <config/multi_camera_job.example.json> or pass --main-video and --second-video. "
        "Outputs: <project>/<run_name>/raw/, fusion/, videos/."
    )
    for cmd_name in ("track-multi-camera", "track-multi-unity-demo"):
        sp = sub.add_parser(
            cmd_name,
            help=help_text if cmd_name == "track-multi-camera" else "Same as track-multi-camera (backward-compatible name).",
        )
        sp.add_argument(
            "--job",
            default=None,
            help="JSON job file (see config/multi_camera_job.example.json). Relative paths resolve from the job file.",
        )
        sp.add_argument("--model", default=None, help="YOLO weights; overrides job; default: YOLO_MODEL or /workspace/models.")
        sp.add_argument("--main-video", default=None, help="Reference camera video path (required if no --job).")
        sp.add_argument("--second-video", default=None, help="Second camera video path (required if no --job).")
        sp.add_argument("--tracker", default=None, help="Tracker YAML; default from job or repo config.")
        sp.add_argument("--device", default=None, help="torch device; default from job or YOLO_DEVICE (use cpu in Docker on Mac).")
        sp.add_argument(
            "--project",
            default=None,
            help="Root folder for track outputs (default: <workspace>/outputs/track or job.project_root).",
        )
        sp.add_argument("--name", default=None, help="Run subdirectory under outputs/track (default: job.run_name or multi_cam_run).")
        sp.add_argument("--image-width", type=int, default=None)
        sp.add_argument("--image-height", type=int, default=None)
        sp.add_argument(
            "--max-center-distance-norm",
            type=float,
            default=None,
            help="Max L2 distance between normalized centers to pair tracks across cameras.",
        )
        sp.add_argument("--camera-a-id", default=None, help="Fusion id for the reference stream (default camera_a or job).")
        sp.add_argument("--camera-b-id", default=None)
        sp.add_argument("--platform", default=None)
        sp.add_argument("--label-mode", default=None, choices=["id", "full", "none"])
        sp.add_argument(
            "--fusion-mode",
            default=None,
            choices=["auto", "normalized_center", "rank_x"],
            help=(
                "Cross-camera pairing: auto (rank_x when equal counts, else normalized centers), "
                "normalized_center (distance in normalized space per camera resolution), "
                "rank_x (sort by center-x; only when counts match)."
            ),
        )
        sp.add_argument(
            "--ais-file",
            default=None,
            help="Optional AIS JSON (see config/ais_layer.example.json). Overrides job ais_file when set.",
        )
        sp.add_argument(
            "--ais-fps",
            type=float,
            default=None,
            help="Override video FPS for AIS sync mode timestamp_ms (else probe main video / job ais_fps).",
        )
        sp.add_argument(
            "--ais-time-offset-ms",
            type=int,
            default=None,
            help="Override AIS sync time_offset_ms for timestamp_ms mode (else JSON / job).",
        )
        sp.set_defaults(func=cmd_track_multi_camera)


def cmd_track_multi_camera(args):
    from fusion.multi_camera_job import load_multi_camera_job
    from fusion.multi_camera_runner import default_model_path, run_dual_camera_track_and_fuse

    job = load_multi_camera_job(Path(args.job)) if getattr(args, "job", None) else None
    if job is None and (not args.main_video or not args.second_video):
        raise SystemExit("Provide --job <job.json> or both --main-video and --second-video.")

    main_video = Path(args.main_video) if args.main_video else job.main_video
    second_video = Path(args.second_video) if args.second_video else job.second_video

    model_path = args.model or (job.model_path if job else None) or default_model_path()
    project_root = (
        Path(args.project)
        if args.project
        else (job.project_root if job and job.project_root else _default_track_project_dir())
    )
    run_name = args.name or (job.run_name if job else None) or "multi_cam_run"
    tracker_yaml = args.tracker or (job.tracker_yaml if job else None) or _default_tracker_yaml_path()
    device = args.device or (job.device if job else None) or settings.yolo_device

    image_width = args.image_width if args.image_width is not None else (job.image_width if job else 1280)
    image_height = args.image_height if args.image_height is not None else (job.image_height if job else 720)
    max_center_distance_norm = (
        args.max_center_distance_norm
        if args.max_center_distance_norm is not None
        else (job.max_center_distance_norm if job else 0.55)
    )
    fusion_match_mode = args.fusion_mode or (job.fusion_match_mode if job else None) or "auto"
    camera_a_id = args.camera_a_id or (job.camera_a_id if job else None) or "camera_a"
    camera_b_id = args.camera_b_id or (job.camera_b_id if job else None) or "camera_b"
    platform = args.platform or (job.platform if job else None) or "simulation"
    label_mode = args.label_mode or (job.label_mode if job else None) or "id"

    ais_file: Path | None = Path(args.ais_file) if getattr(args, "ais_file", None) else None
    if ais_file is None and job and job.ais_file:
        ais_file = job.ais_file
    ais_fps = args.ais_fps if args.ais_fps is not None else (job.ais_fps if job else None)
    ais_time_offset_ms = (
        args.ais_time_offset_ms if args.ais_time_offset_ms is not None else (job.ais_time_offset_ms if job else None)
    )

    if not main_video.is_file():
        raise FileNotFoundError(f"Main video not found: {main_video}")
    if not second_video.is_file():
        raise FileNotFoundError(f"Second camera video not found: {second_video}")

    result = run_dual_camera_track_and_fuse(
        model_path=model_path,
        main_video=main_video,
        second_video=second_video,
        tracker_yaml=tracker_yaml,
        device=device,
        project_root=project_root,
        run_name=run_name,
        image_width=image_width,
        image_height=image_height,
        max_center_distance_norm=max_center_distance_norm,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        platform=platform,
        label_mode=label_mode,
        fusion_match_mode=fusion_match_mode,
        ais_file=ais_file,
        ais_fps=ais_fps,
        ais_time_offset_ms=ais_time_offset_ms,
    )
    print(json.dumps(result, indent=2))


def cmd_evaluate(args):
    result = evaluate_mot_dir(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_csv=args.output,
        output_json=args.summary_json,
        iou_threshold=args.iou_threshold,
        frame_offset=args.frame_offset,
    )
    print(result)


def cmd_evaluate_localization(args):
    result = evaluate_sparse_localization_dir(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_csv=args.output,
        output_json=args.summary_json,
        frame_offset=args.frame_offset,
    )
    print(result)


def cmd_gt_validate(args):
    result = validate_gt_dir(
        gt_dir=args.gt_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(result)


def cmd_gt_merge_ids(args):
    if args.list_ids:
        print(list_track_ids(args.input))
        return

    if not args.output:
        raise ValueError("--output is required unless --list-ids is used")
    if not args.source_ids:
        raise ValueError("--source-ids is required unless --list-ids is used")

    result = merge_tracker_ids_to_gt(
        input_path=args.input,
        output_path=args.output,
        source_ids=args.source_ids,
        gt_id=args.gt_id,
        class_id=args.class_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        sort_rows=args.sort,
    )
    print(result)


def cmd_stitch_tracks(args):
    from stitching.runner import stitch_tracks

    result = stitch_tracks(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        run_summary_path=args.run_summary,
        config_path=args.config,
        stitch_name=args.name,
        ais_file=getattr(args, "ais_file", None),
        ais_fps=getattr(args, "ais_fps", None),
    )
    print(result)


def cmd_stitch_batch(args):
    from stitching.batch import stitch_batch

    result = stitch_batch(
        runs_root=args.runs_root,
        batch_dir=args.batch_dir,
        config_path=args.config,
        stitch_name=args.name,
    )
    print(result)


def cmd_pipeline_full(args):
    from ingestion.ingestor import build_tasks

    clip_id = Path(args.source).stem

    tasks = build_tasks(
        source=args.source,
        frames_dir=settings.frames_dir,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
    )

    celery_app = _get_celery_app()
    async_results = []
    for task in tasks:
        r = celery_app.send_task("maritime.detect_batch", args=[task.model_dump()])
        async_results.append(r)

    print(f"Queued {len(tasks)} batches for clip {clip_id}")
    wait_for_tasks(async_results, poll_seconds=args.poll_seconds)
    print("All batch tasks finished.")

    aggregate_result = aggregate_clip(
        settings.worker_results_dir,
        settings.aggregated_dir,
        clip_id,
    )
    print(aggregate_result)

    from output.render_video import render_annotated_video

    render_result = render_annotated_video(
        frames_dir=settings.frames_dir,
        worker_results_dir=settings.worker_results_dir,
        output_video_path=f"/workspace/outputs/rendered/{clip_id}.mp4",
        clip_id=clip_id,
        fps=args.render_fps,
        label_mode="none" if args.no_labels else "full",
    )
    print(render_result)


def _resolve_render_label_mode(args, default_mode: str) -> str:
    if getattr(args, "no_labels", False):
        return "none"
    if getattr(args, "label_mode", None):
        return args.label_mode
    return default_mode


def cmd_render_video(args):
    from output.render_video import render_annotated_video, render_video_from_mot

    if args.clip_id and (args.source or args.mot_file):
        raise ValueError("render-video accepts either --clip-id or --source with --mot-file, not both")

    if args.clip_id:
        result = render_annotated_video(
            frames_dir=settings.frames_dir,
            worker_results_dir=settings.worker_results_dir,
            output_video_path=args.output,
            clip_id=args.clip_id,
            fps=args.fps,
            label_mode=_resolve_render_label_mode(args, default_mode="full"),
        )
    else:
        if not args.source or not args.mot_file:
            raise ValueError("render-video requires either --clip-id or both --source and --mot-file")
        result = render_video_from_mot(
            source_video_path=args.source,
            mot_file_path=args.mot_file,
            output_video_path=args.output,
            fps=args.fps,
            label_mode=_resolve_render_label_mode(args, default_mode="id"),
        )
    print(result)


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pipeline-ingest")
    sp.add_argument("--source", required=True)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--frame-step", type=int, default=1)
    sp.set_defaults(func=cmd_pipeline_ingest)

    sp = sub.add_parser("pipeline-aggregate")
    sp.add_argument("--clip-id", required=True)
    sp.set_defaults(func=cmd_pipeline_aggregate)

    sp = sub.add_parser("pipeline-full")
    sp.add_argument("--source", required=True)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--frame-step", type=int, default=1)
    sp.add_argument("--poll-seconds", type=float, default=2.0)
    sp.add_argument("--model", default="yolo26l.pt")
    sp.add_argument("--tracker", default="/workspace/config/trackers/botsort_maritime.yaml")
    sp.add_argument("--device", default=settings.yolo_device)
    sp.add_argument("--project", default="/workspace/outputs/track")
    sp.add_argument("--name", default="botsort_run")
    sp.add_argument("--render-fps", type=float, default=30.0)
    sp.add_argument("--no-labels", action="store_true")
    sp.set_defaults(func=cmd_pipeline_full)

    sp = sub.add_parser("track")
    sp.add_argument("--model", required=True)
    sp.add_argument("--source", required=True)
    sp.add_argument("--tracker", default="/workspace/config/trackers/botsort_maritime.yaml")
    sp.add_argument("--device", default=settings.yolo_device)
    sp.add_argument("--project", default="/workspace/outputs/track")
    sp.add_argument("--name", default="botsort_run")
    sp.add_argument(
        "--ais-file",
        default=None,
        help="Optional AIS JSON sidecar source (see config/ais_layer.example.json). Does not change MOT without a consumer.",
    )
    sp.add_argument(
        "--ais-fps",
        type=float,
        default=None,
        help="FPS for AIS timestamp_ms sync when source is not a video with FPS metadata.",
    )
    sp.add_argument(
        "--ais-time-offset-ms",
        type=int,
        default=None,
        help="Override AIS JSON time_offset_ms for timestamp_ms sync.",
    )
    sp.set_defaults(func=cmd_track)

    _register_track_multi_camera_parsers(sub)

    sp = sub.add_parser("evaluate")
    sp.add_argument("--pred-dir", required=True)
    sp.add_argument("--gt-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--summary-json")
    sp.add_argument("--iou-threshold", type=float, default=0.5)
    sp.add_argument("--frame-offset", type=int, default=0)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("evaluate-localization")
    sp.add_argument("--pred-dir", required=True)
    sp.add_argument("--gt-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--summary-json")
    sp.add_argument("--frame-offset", type=int, default=0)
    sp.set_defaults(func=cmd_evaluate_localization)

    sp = sub.add_parser("gt-validate")
    sp.add_argument("--gt-dir", required=True)
    sp.add_argument("--output-json")
    sp.add_argument("--output-csv")
    sp.set_defaults(func=cmd_gt_validate)

    sp = sub.add_parser("gt-merge-ids")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output")
    sp.add_argument("--source-ids", nargs="+", type=int)
    sp.add_argument("--gt-id", type=int, default=1)
    sp.add_argument("--class-id", type=int)
    sp.add_argument("--frame-start", type=int)
    sp.add_argument("--frame-end", type=int)
    sp.add_argument("--sort", action="store_true")
    sp.add_argument("--list-ids", action="store_true")
    sp.set_defaults(func=cmd_gt_merge_ids)

    sp = sub.add_parser("stitch-tracks")
    sp.add_argument("--pred-dir", required=True)
    sp.add_argument("--output-dir")
    sp.add_argument("--run-summary")
    sp.add_argument("--config", default=str(_DEFAULT_STITCH_CONFIG_PATH))
    sp.add_argument("--name")
    sp.add_argument(
        "--ais-file",
        default=None,
        help="Optional AIS CSV/JSON; enables stitching AIS scoring (see config stitching ais: weights).",
    )
    sp.add_argument(
        "--ais-fps",
        type=float,
        default=None,
        help="Video FPS for AIS alignment (else run_summary effective_fps when available).",
    )
    sp.set_defaults(func=cmd_stitch_tracks)

    sp = sub.add_parser("stitch-batch")
    sp.add_argument("--runs-root", default="/workspace/outputs/track")
    sp.add_argument("--batch-dir")
    sp.add_argument("--config", default=str(_DEFAULT_STITCH_CONFIG_PATH))
    sp.add_argument("--name")
    sp.set_defaults(func=cmd_stitch_batch)
    
    sp = sub.add_parser("render-video")
    sp.add_argument("--clip-id")
    sp.add_argument("--source")
    sp.add_argument("--mot-file")
    sp.add_argument("--output", required=True)
    sp.add_argument("--fps", type=float)
    sp.add_argument("--label-mode", choices=["id", "full", "none"])
    sp.add_argument("--no-labels", action="store_true")
    sp.set_defaults(func=cmd_render_video)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
