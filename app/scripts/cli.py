from __future__ import annotations

import argparse
import time
from pathlib import Path

from broker.celery_app import celery_app
from common.config import settings
from evaluation.localization import evaluate_sparse_localization_dir
from evaluation.mot import list_track_ids, merge_tracker_ids_to_gt, validate_gt_dir
from evaluation.metrics import evaluate_mot_dir
from ingestion.ingestor import build_tasks
from output.aggregator import aggregate_clip
from output.render_video import render_annotated_video, render_video_from_mot
from stitching.batch import stitch_batch
from stitching.runner import DEFAULT_CONFIG_PATH, stitch_tracks
from tracking.tracker_runner import run_tracking


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
    tasks = build_tasks(
        source=args.source,
        frames_dir=settings.frames_dir,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
    )
    for task in tasks:
        celery_app.send_task("maritime.detect_batch", args=[task.model_dump()])
    print(f"Queued {len(tasks)} batches for clip {Path(args.source).stem}")


def cmd_pipeline_aggregate(args):
    print(aggregate_clip(settings.worker_results_dir, settings.aggregated_dir, args.clip_id))


def cmd_track(args):
    result = run_tracking(
        model_path=args.model,
        source=args.source,
        tracker_yaml=args.tracker,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(result)


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
    result = stitch_tracks(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        run_summary_path=args.run_summary,
        config_path=args.config,
        stitch_name=args.name,
    )
    print(result)


def cmd_stitch_batch(args):
    result = stitch_batch(
        runs_root=args.runs_root,
        batch_dir=args.batch_dir,
        config_path=args.config,
        stitch_name=args.name,
    )
    print(result)


def cmd_pipeline_full(args):
    clip_id = Path(args.source).stem

    tasks = build_tasks(
        source=args.source,
        frames_dir=settings.frames_dir,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
    )

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
    sp.set_defaults(func=cmd_track)

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
    sp.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sp.add_argument("--name")
    sp.set_defaults(func=cmd_stitch_tracks)

    sp = sub.add_parser("stitch-batch")
    sp.add_argument("--runs-root", default="/workspace/outputs/track")
    sp.add_argument("--batch-dir")
    sp.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
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
