from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from common.config import settings


# Default stitch config path (kept local to avoid importing the stitch stack at CLI import time).
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
    from output.aggregator import aggregate_clip

    print(aggregate_clip(settings.worker_results_dir, settings.aggregated_dir, args.clip_id))


def cmd_track(args):
    if args.live_reid and args.live_reid_in_loop:
        raise ValueError("--live-reid and --live-reid-in-loop cannot be used together")
    from tracking.tracker_runner import run_tracking

    result = run_tracking(
        model_path=args.model,
        source=args.source,
        tracker_yaml=args.tracker,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        imgsz=args.imgsz,
        live_reid=args.live_reid,
        live_reid_in_loop=args.live_reid_in_loop,
        live_reid_config=args.live_reid_config,
        confirmation_observations=args.confirmation_observations,
        colreg_diagnostics=args.colreg_diagnostics,
        colreg_scoring_experiment=args.colreg_scoring_experiment,
        ais_diagnostics=args.ais_diagnostics,
        ais_file=args.ais_file,
        ais_video_start_time=args.ais_video_start_time,
        ais_affine_matrix=args.ais_affine_matrix,
        visual_continuation=args.visual_continuation,
        continuation_search_radius=args.continuation_search_radius,
        continuation_threshold=args.continuation_threshold,
        continuation_max_gap=args.continuation_max_gap,
        paired_occlusion_prediction=args.paired_occlusion_prediction,
        secondary_model=args.secondary_model,
        detector_fusion=args.detector_fusion,
        fusion_iou_threshold=args.fusion_iou_threshold,
        fusion_confidence_threshold=args.fusion_confidence_threshold,
        fusion_mode=args.fusion_mode,
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
            "--video-layout",
            default=None,
            choices=["side_by_side", "top_bottom", "side_by_side_hd"],
            help=(
                "Combined video layout: side_by_side (full-res horizontal), "
                "top_bottom (vertical stack), side_by_side_hd (horizontal, scaled to 1920px width; default)."
            ),
        )
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
    video_layout = args.video_layout or (job.video_layout if job else None) or "side_by_side_hd"

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
        video_layout=video_layout,
        ais_file=ais_file,
        ais_fps=ais_fps,
        ais_time_offset_ms=ais_time_offset_ms,
    )
    print(json.dumps(result, indent=2))


def cmd_evaluate(args):
    from evaluation.metrics import evaluate_mot_dir

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
    from evaluation.localization import evaluate_sparse_localization_dir

    result = evaluate_sparse_localization_dir(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_csv=args.output,
        output_json=args.summary_json,
        frame_offset=args.frame_offset,
    )
    print(result)


def cmd_gt_validate(args):
    from evaluation.mot import validate_gt_dir

    result = validate_gt_dir(
        gt_dir=args.gt_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(result)


def cmd_gt_merge_ids(args):
    from evaluation.mot import list_track_ids, merge_tracker_ids_to_gt

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


def cmd_replay_stitch(args):
    from stitching.replay import replay_stitch_tracks

    result = replay_stitch_tracks(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        run_summary_path=args.run_summary,
        config_path=args.config,
        stitch_name=args.name,
    )
    print(result)


def cmd_reid_regression(args):
    from reid.regression import check_regression

    summary = check_regression(
        cut28_mot=args.cut28_mot,
        cut28_video=args.cut28_video,
        cut29_mot=args.cut29_mot,
        cut29_video=args.cut29_video,
        config=args.config,
        output_dir=args.output_dir,
        confirmation_observations=args.confirmation_observations,
        colreg_diagnostics=args.colreg_diagnostics,
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def cmd_self_check(args):
    from ais.checks import run_checks as run_ais_checks
    from colreg.checks import run_checks as run_colreg_checks

    run_colreg = bool(args.colreg) or not bool(args.ais)
    run_ais = bool(args.ais) or not bool(args.colreg)
    checks = {}
    if run_colreg:
        checks["colreg"] = run_colreg_checks()
    if run_ais:
        checks["ais"] = run_ais_checks()
    summary = {
        "passed": all(bool(payload.get("passed")) for payload in checks.values()),
        "checks": checks,
    }
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def cmd_fvessel_ais_eval(args):
    from ais.fvessel_identity_eval import run_fvessel_ais_identity_eval

    summary = run_fvessel_ais_identity_eval(
        alignment_root=args.alignment_root,
        run_root=args.run_root,
        output_dir=args.output_dir,
        model=args.model,
        tracker=args.tracker,
        live_reid_config=args.live_reid_config,
        confirmation_observations=args.confirmation_observations,
        min_alignment_confidence=args.min_alignment_confidence,
        max_clips=args.max_clips,
        include_low_confidence=args.include_low_confidence,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(summary, indent=2))
    if summary.get("clips_failed"):
        raise SystemExit(1)


def cmd_sim_eval(args):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.run_sim_eval import run_sim_eval

    summary = run_sim_eval(
        config=args.config,
        sim_out_dir=args.sim_out,
        eval_dir=args.eval_dir,
        unity=args.unity,
        unity_project=args.unity_project,
        model=args.model,
        tracker=args.tracker,
        device=args.device,
        tracking_project=args.tracking_project,
        run_name=args.run_name,
        timeout=args.timeout,
        batchmode=args.batchmode,
        verbose=args.verbose,
        skip_generation=args.skip_generation,
        dry_run=args.dry_run,
        iou_threshold=args.iou_threshold,
        frame_offset=args.frame_offset,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    print(json.dumps(summary, indent=2))


def cmd_pipeline_full(args):
    from ingestion.ingestor import build_tasks
    from output.aggregator import aggregate_clip

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
            video_codec=args.video_codec,
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
            occlusion_predictions_path=args.occlusion_predictions,
            side_by_side=args.side_by_side_demo,
            video_codec=args.video_codec,
            motion_corridor_overlay=args.motion_corridor_overlay,
            prediction_overlay_style=args.prediction_overlay_style,
            hide_uncertainty_circle=args.hide_uncertainty_circle,
            zoom_occlusion_roi=args.zoom_occlusion_roi,
            zoom_padding=args.zoom_padding,
            zoom_min_size=args.zoom_min_size,
            zoom_follow_prediction=args.zoom_follow_prediction,
            prediction_overlay_detail=args.prediction_overlay_detail,
            zoom_reid_recovery=args.zoom_reid_recovery,
            recovery_zoom_pre_frames=args.recovery_zoom_pre_frames,
            recovery_zoom_post_frames=args.recovery_zoom_post_frames,
            recovery_zoom_padding=args.recovery_zoom_padding,
            recovery_zoom_label_duration=args.recovery_zoom_label_duration,
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
    sp.add_argument("--conf", type=float, help="Optional detector confidence threshold passed to Ultralytics tracking.")
    sp.add_argument("--imgsz", type=int, help="Optional image size passed to Ultralytics tracking.")
    sp.add_argument("--secondary-model", help="Optional secondary detector for experimental fusion diagnostics.")
    sp.add_argument(
        "--detector-fusion",
        action="store_true",
        help=(
            "Experimental detector fusion diagnostics. The current tracker loop keeps primary-model "
            "tracking/MOT output unchanged unless a custom pre-tracker fusion loop is added."
        ),
    )
    sp.add_argument("--fusion-iou-threshold", type=float, default=0.55)
    sp.add_argument("--fusion-confidence-threshold", type=float, default=0.25)
    sp.add_argument("--fusion-mode", choices=["union_nms"], default="union_nms")
    sp.add_argument(
        "--live-reid",
        action="store_true",
        help=(
            "Experimental: after raw tracker MOT is written, also run bounded-latency "
            "ReID into <run>/live_reid. Raw MOT is preserved; this is not in-loop tracker integration."
        ),
    )
    sp.add_argument(
        "--live-reid-in-loop",
        action="store_true",
        help=(
            "Experimental: run bounded-latency live ReID inside the tracking frame loop and write "
            "<run>/live_reid_in_loop while preserving raw MOT. This is not zero-latency production "
            "integration. Do not combine with --live-reid."
        ),
    )
    sp.add_argument(
        "--live-reid-config",
        default=str(_DEFAULT_STITCH_CONFIG_PATH),
        help="Config used by experimental --live-reid or --live-reid-in-loop.",
    )
    sp.add_argument(
        "--confirmation-observations",
        type=int,
        default=10,
        help="Bounded-latency observation buffer used by live ReID modes. Default: 10.",
    )
    sp.add_argument(
        "--visual-continuation",
        action="store_true",
        help=(
            "Reporting-only local template continuation for live in-loop occlusion reports. "
            "Does not affect tracker/ReID decisions or MOT output."
        ),
    )
    sp.add_argument("--continuation-search-radius", type=int, default=64)
    sp.add_argument("--continuation-threshold", type=float, default=0.45)
    sp.add_argument("--continuation-max-gap", type=int, default=60)
    sp.add_argument(
        "--paired-occlusion-prediction",
        action="store_true",
        help=(
            "Reporting-only paired-vessel motion corridor prediction for occlusion reports. "
            "Does not affect tracker/ReID decisions or MOT output."
        ),
    )
    sp.add_argument(
        "--colreg-diagnostics",
        action="store_true",
        help=(
            "Reporting-only COLREG diagnostics for live ReID candidate reports. "
            "Does not affect matching scores, gates, remaps, or MOT output."
        ),
    )
    sp.add_argument(
        "--colreg-scoring-experiment",
        action="store_true",
        help=(
            "Experimental and disabled by default: allow ultra-narrow COLREG tie-break scoring "
            "inside live ReID candidate ranking. Implies COLREG diagnostics and must be combined "
            "with --live-reid or --live-reid-in-loop."
        ),
    )
    sp.add_argument(
        "--ais-diagnostics",
        action="store_true",
        help=(
            "Reporting-only AIS fields for live ReID candidate reports. Does not affect scores, "
            "gates, remaps, canonical IDs, tracker decisions, or MOT output; missing AIS is neutral."
        ),
    )
    sp.add_argument("--ais-file", help="Optional AIS CSV/JSON file used only with --ais-diagnostics.")
    sp.add_argument("--ais-video-start-time", help="Optional video start timestamp for AIS frame alignment.")
    sp.add_argument("--ais-affine-matrix", help="Optional 2x3 or 3x3 lon/lat-to-image affine matrix for AIS diagnostics.")
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

    sp = sub.add_parser("stitch-tracks", description="Legacy/reference offline stitcher.")
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

    sp = sub.add_parser("stitch-batch", description="Legacy/reference offline batch stitcher.")
    sp.add_argument("--runs-root", default="/workspace/outputs/track")
    sp.add_argument("--batch-dir")
    sp.add_argument("--config", default=str(_DEFAULT_STITCH_CONFIG_PATH))
    sp.add_argument("--name")
    sp.set_defaults(func=cmd_stitch_batch)

    sp = sub.add_parser("replay-stitch", description="Legacy/reference replay-style stitcher.")
    sp.add_argument("--pred-dir", required=True)
    sp.add_argument("--output-dir")
    sp.add_argument("--run-summary")
    sp.add_argument("--config", default=str(_DEFAULT_STITCH_CONFIG_PATH))
    sp.add_argument("--name")
    sp.set_defaults(func=cmd_replay_stitch)

    sp = sub.add_parser(
        "reid-regression",
        description="Run the saved-MOT ReID v1 regression used for review and CI.",
    )
    sp.add_argument("--cut28-mot")
    sp.add_argument("--cut28-video")
    sp.add_argument("--cut29-mot")
    sp.add_argument("--cut29-video")
    sp.add_argument("--config", help="Optional ReID config applied to both regression cases.")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--confirmation-observations", type=int, default=10)
    sp.add_argument(
        "--colreg-diagnostics",
        action="store_true",
        help="Reporting-only COLREG fields during the regression; decisions and MOT rows must remain unchanged.",
    )
    sp.set_defaults(func=cmd_reid_regression)

    sp = sub.add_parser(
        "self-check",
        description="Run lightweight synthetic checks for diagnostic COLREG and AIS utilities.",
    )
    sp.add_argument("--colreg", action="store_true", help="Run only the COLREG synthetic checks.")
    sp.add_argument("--ais", action="store_true", help="Run only the AIS synthetic checks.")
    sp.set_defaults(func=cmd_self_check)

    sp = sub.add_parser(
        "fvessel-ais-eval",
        description=(
            "Evaluate FVessel aligned-clip MMSI identity evidence against raw/canonical MOT. "
            "AIS remains reporting-only and never changes tracking or ReID decisions."
        ),
    )
    sp.add_argument("--alignment-root", required=True)
    sp.add_argument("--run-root", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--tracker", required=True)
    sp.add_argument("--live-reid-config", required=True)
    sp.add_argument("--confirmation-observations", type=int, default=10)
    sp.add_argument("--min-alignment-confidence", type=float, default=0.10)
    sp.add_argument("--max-clips", type=int, default=3)
    sp.add_argument("--include-low-confidence", action="store_true")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--iou-threshold", type=float, default=0.3)
    sp.set_defaults(func=cmd_fvessel_ais_eval)

    sp = sub.add_parser(
        "sim-eval",
        description="Generate MaritimeSim videos, run tracking per camera, and evaluate against Unity MOT ground truth.",
    )
    sp.add_argument("config", help="Simulation JSON config.")
    sp.add_argument("--sim-out", help="Simulation output directory (default: outputs/sim/<config>).")
    sp.add_argument("--eval-dir", help="Evaluation output directory (default: outputs/sim_eval/<config>).")
    sp.add_argument("--unity", help="Path to Unity executable or set UNITY_PATH.")
    sp.add_argument("--unity-project", help="Unity project root (default: MaritimeSim2).")
    sp.add_argument("--model", help="YOLO model path/name (default: /workspace/models/yolo26l.pt if present).")
    sp.add_argument("--tracker", help="Tracker YAML (default: config/trackers/botsort_maritime.yaml).")
    sp.add_argument("--device", default="cpu", help="Tracking device; use cpu in Docker on Mac.")
    sp.add_argument("--tracking-project", help="Root for raw tracking runs (default: <eval-dir>/track).")
    sp.add_argument("--run-name", help="Tracking run name (default: config stem).")
    sp.add_argument("--timeout", type=float, default=1800, help="Unity timeout in seconds.")
    sp.add_argument("--batchmode", action="store_true", help="Pass -batchmode to Unity (not recommended for Recorder).")
    sp.add_argument("--verbose", action="store_true", help="Enable verbose Unity simulation logs.")
    sp.add_argument("--skip-generation", action="store_true", help="Reuse existing files in --sim-out.")
    sp.add_argument("--dry-run", action="store_true", help="Validate config and print planned steps only.")
    sp.add_argument("--iou-threshold", type=float, default=0.5)
    sp.add_argument("--frame-offset", type=int, default=0)
    sp.add_argument("--conf", type=float, help="Optional detector confidence threshold.")
    sp.add_argument("--imgsz", type=int, help="Optional detector image size.")
    sp.set_defaults(func=cmd_sim_eval)

    sp = sub.add_parser("render-video")
    sp.add_argument("--clip-id")
    sp.add_argument("--source")
    sp.add_argument("--mot-file")
    sp.add_argument("--output", required=True)
    sp.add_argument("--fps", type=float)
    sp.add_argument("--label-mode", choices=["id", "full", "none"])
    sp.add_argument("--no-labels", action="store_true")
    sp.add_argument("--occlusion-predictions", help="Optional reporting-only occlusion prediction JSON overlay.")
    sp.add_argument("--side-by-side-demo", action="store_true", help="Render original and annotated video side by side.")
    sp.add_argument("--motion-corridor-overlay", action="store_true", help="Draw directional corridor ellipses/arrows when present.")
    sp.add_argument(
        "--prediction-overlay-style",
        choices=["circle", "arrow", "corridor", "compact"],
        default="circle",
        help="Occlusion prediction overlay style. circle preserves the older large uncertainty overlay.",
    )
    sp.add_argument(
        "--hide-uncertainty-circle",
        action="store_true",
        help="Suppress the large uncertainty circle and keep compact outlines/corridors only.",
    )
    sp.add_argument("--zoom-occlusion-roi", action="store_true", help="Zoom side-by-side right panel around active occlusion predictions.")
    sp.add_argument("--zoom-padding", type=float, default=1.8)
    sp.add_argument("--zoom-min-size", type=int, default=360)
    sp.add_argument("--zoom-follow-prediction", action="store_true")
    sp.add_argument("--zoom-reid-recovery", action="store_true", help="Prioritize side-by-side zoom around ReID recovery events.")
    sp.add_argument("--recovery-zoom-pre-frames", type=int, default=30)
    sp.add_argument("--recovery-zoom-post-frames", type=int, default=90)
    sp.add_argument("--recovery-zoom-padding", type=float, default=2.0)
    sp.add_argument("--recovery-zoom-label-duration", type=int, default=90)
    sp.add_argument("--prediction-overlay-detail", choices=["clean", "debug"], default="debug")
    sp.add_argument(
        "--video-codec",
        choices=["auto", "mp4v", "h264"],
        default="auto",
        help="Video codec for rendered MP4s. auto prefers Windows-friendly H.264 and falls back clearly.",
    )
    sp.set_defaults(func=cmd_render_video)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
