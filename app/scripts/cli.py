from __future__ import annotations

import argparse
import time
from pathlib import Path

from broker.celery_app import celery_app
from common.config import settings
from ingestion.ingestor import build_tasks
from output.aggregator import aggregate_clip
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

    track_result = run_tracking(
        model_path=args.model,
        source=args.source,
        tracker_yaml=args.tracker,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(track_result)


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
    sp.set_defaults(func=cmd_pipeline_full)

    sp = sub.add_parser("track")
    sp.add_argument("--model", required=True)
    sp.add_argument("--source", required=True)
    sp.add_argument("--tracker", default="/workspace/config/trackers/botsort_maritime.yaml")
    sp.add_argument("--device", default=settings.yolo_device)
    sp.add_argument("--project", default="/workspace/outputs/track")
    sp.add_argument("--name", default="botsort_run")
    sp.set_defaults(func=cmd_track)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()