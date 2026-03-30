"""
output/aggregator.py
Collects individual frame JSON results and merges them into a single summary file (JSON or CSV) for later review.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def aggregate_results(results_dir: Path | None = None, output_fmt: str | None = None) -> Path:
    """
    Read all per-frame JSON files in results_dir and write a merged summary.
    Returns the path to the aggregated output file.
    """
    cfg = _load_cfg()
    out_cfg = cfg["output"]

    if results_dir is None:
        results_dir = Path(out_cfg["results_dir"])
    if output_fmt is None:
        output_fmt = out_cfg["format"]

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        log.warning(f"No result files found in {results_dir}")
        return None

    log.info(f"Aggregating {len(json_files)} result file(s) from {results_dir}")

    all_records = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        # Flatten detections — one row per detection (or one row if no detections)
        if data["detections"]:
            for det in data["detections"]:
                all_records.append({
                    "frame_path": data["frame_path"],
                    "source_dataset": data["source_dataset"],
                    "sequence_id": data["sequence_id"],
                    "frame_index": data["frame_index"],
                    "inference_ms": data["inference_ms"],
                    "num_detections": data["num_detections"],
                    **det,
                })
        else:
            all_records.append({
                "frame_path": data["frame_path"],
                "source_dataset": data["source_dataset"],
                "sequence_id": data["sequence_id"],
                "frame_index": data["frame_index"],
                "inference_ms": data["inference_ms"],
                "num_detections": 0,
                "bbox_xyxy": None,
                "confidence": None,
                "class_id": None,
                "class_name": None,
            })

    summary_path = results_dir.parent / f"summary.{output_fmt}"

    if output_fmt == "json":
        with open(summary_path, "w") as f:
            json.dump(all_records, f, indent=2)
    elif output_fmt == "csv":
        if all_records:
            fieldnames = list(all_records[0].keys())
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)
    else:
        raise ValueError(f"Unsupported output format: {output_fmt}")

    log.info(f"Summary written to {summary_path} ({len(all_records)} rows)")
    return summary_path


def print_summary(results_dir: Path | None = None) -> None:
    """Print a quick stats overview to stdout."""
    cfg = _load_cfg()
    out_cfg = cfg["output"]

    if results_dir is None:
        results_dir = Path(out_cfg["results_dir"])

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print("No results found.")
        return

    total_frames = len(json_files)
    total_detections = 0
    frames_with_vessels = 0
    dataset_counts: dict[str, int] = {}
    inference_times: list[float] = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        n = data["num_detections"]
        total_detections += n
        if n > 0:
            frames_with_vessels += 1
        ds = data.get("source_dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        inference_times.append(data["inference_ms"])

    avg_ms = sum(inference_times) / len(inference_times) if inference_times else 0

    print("\nVessel Detection Run Summary")
    print(f"Frames processed: {total_frames}")
    print(f"Frames with vessels: {frames_with_vessels} ({100*frames_with_vessels/total_frames:.1f}%)")
    print(f"Total detections: {total_detections}")
    print(f"Avg inference time: {avg_ms:.1f} ms/frame")
    print(f"Dataset breakdown: {dataset_counts}")
    print("\n")
