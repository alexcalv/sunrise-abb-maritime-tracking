#!/usr/bin/env python3
"""
run_pipeline.py
Main script for running the vessel detection pipeline.
Usage examples:
    python run_pipeline.py --mode full # Run the entire pipeline
    python run_pipeline.py --mode ingest # Only ingest and queue tasks (workers process them later)
    python run_pipeline.py --mode aggregate # Aggregate already-completed results into a summary
    python run_pipeline.py --mode test --input path/to/file.jpg # Quick test with a single image or video file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def run_full(args):
    """Run the full pipeline: ingest frames, dispatch tasks, wait for completion, and aggregate results."""
    from ingestion.ingestor import collect_all_frames, dispatch_to_queue
    from output.aggregator import aggregate_results, print_summary

    log.info("Starting pipeline...")

    # Stage 1: Collect frames
    frames = collect_all_frames()
    if not frames:
        log.error("No frames collected — check your dataset paths in config/pipeline.yaml")
        sys.exit(1)

    # Stage 2: Dispatch tasks
    log.info("Dispatching tasks to queue...")
    async_results = dispatch_to_queue(frames)

    # Stage 3: Wait for tasks to complete
    log.info(f"Waiting for {len(async_results)} task(s)...")
    completed, failed = 0, 0
    for ar in async_results:
        try:
            ar.get(timeout=600)  # Wait up to 5 minutes per batch
            completed += 1
        except Exception as e:
            log.error(f"Task {ar.id} failed: {e}")
            failed += 1
    log.info(f"Task summary: {completed} succeeded, {failed} failed")

    # Stage 4: Aggregate results
    log.info("Aggregating results...")
    summary_path = aggregate_results()
    print_summary()
    if summary_path:
        log.info(f"Summary saved to {summary_path}")


def run_ingest(args):
    """Collect frames and dispatch tasks to the queue without waiting for results."""
    from ingestion.ingestor import collect_all_frames, dispatch_to_queue

    frames = collect_all_frames()
    if not frames:
        log.error("No frames collected.")
        sys.exit(1)
    async_results = dispatch_to_queue(frames)
    log.info(f"Queued {len(async_results)} tasks. Workers will process them independently.")


def run_aggregate(args):
    """Merge existing result JSONs into a summary file."""
    from output.aggregator import aggregate_results, print_summary

    summary_path = aggregate_results()
    print_summary()
    if summary_path:
        log.info(f"Summary file created: {summary_path}")


def run_test(args):
    """Quick test: run inference on a single image or video file."""
    from ultralytics import YOLO
    import yaml

    cfg_path = Path(__file__).parent / "config" / "pipeline.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    det = cfg["detector"]

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    log.info(f"Loading model {det['model']} …")
    model = YOLO(det["model"])

    log.info(f"Running inference on {input_path} …")
    t0 = time.perf_counter()
    results = model.predict(
        source=str(input_path),
        conf=det["confidence_threshold"],
        iou=det["iou_threshold"],
        imgsz=det["imgsz"],
        device=det["device"],
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    result = results[0]
    print(f"\nTest Results")
    print(f"Input File: {input_path}")
    print(f"Detections: {len(result.boxes)}")
    print(f"Inference Time: {elapsed*1000:.1f} ms")
    for box in result.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        cls = result.names[int(box.cls[0])]
        conf = round(float(box.conf[0]), 3)
        print(f"[{cls}] conf={conf}  bbox=({x1},{y1},{x2},{y2})")
    print("\n")

    import cv2

    out_dir = Path("outputs/test_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        annotated = r.plot()
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), annotated)

    log.info(f"Saved {len(results)} annotated frames to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Vessel detection pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "ingest", "aggregate", "test"],
        default="full",
        help="Select the mode to run",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a single image or video file",
    )
    args = parser.parse_args()

    dispatch = {
        "full": run_full,
        "ingest": run_ingest,
        "aggregate": run_aggregate,
        "test": run_test,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()