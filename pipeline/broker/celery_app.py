"""
broker/celery_app.py
Celery application instance and inference task.
Workers are started with: celery -A broker.celery_app worker --loglevel=info --concurrency=4
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml
from celery import Celery
from celery.utils.log import get_task_logger

# Config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"

def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = _load_cfg()
broker_cfg = cfg["broker"]

# Celery application

app = Celery(
    "vessel_pipeline",
    broker=broker_cfg["url"],
    backend=broker_cfg["result_backend"],
)

app.conf.update(
    task_serializer=broker_cfg["task_serializer"],
    result_serializer=broker_cfg["result_serializer"],
    accept_content=broker_cfg["accept_content"],
    worker_prefetch_multiplier=cfg["workers"]["prefetch_multiplier"],
    task_track_started=True,
    task_acks_late=True,          # re-queue if worker crashes mid-task
    worker_max_tasks_per_child=100,  # recycle worker after N tasks (memory safety)
)

logger = get_task_logger(__name__)

_model_cache: dict = {}

def _get_model():
    """Load YOLOv8 model once per worker process and cache it."""
    if "model" not in _model_cache:
        from ultralytics import YOLO  # deferred import so Celery starts fast
        det = cfg["detector"]
        logger.info(f"Loading model {det['model']} on device {det['device']}")
        _model_cache["model"] = YOLO(det["model"])
        _model_cache["cfg"] = det
    return _model_cache["model"], _model_cache["cfg"]


@app.task(bind=True, name="vessel_pipeline.detect_batch", max_retries=3)
def detect_batch(self, batch: list[dict]) -> list[dict]:
    """
    Process a batch of frames through YOLOv8.
    Each item in `batch` is a dict:
        {
            "frame_path": str,       # absolute path to the image file
            "source_dataset": str,   # "SMD" or "MVTD"
            "sequence_id": str,      # e.g. "MVI_0788" or "seq_001"
            "frame_index": int       # 0-based frame number
        }
    Returns a list of result dicts (one per input frame).
    """
    model, det_cfg = _get_model()
    out_cfg = cfg["output"]

    results_dir = Path(out_cfg["results_dir"])
    annotated_dir = Path(out_cfg["annotated_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    batch_results = []

    for item in batch:
        frame_path = Path(item["frame_path"])
        if not frame_path.exists():
            logger.warning(f"Frame not found, skipping: {frame_path}")
            continue

        t0 = time.perf_counter()

        try:
            preds = model.predict(
                source=str(frame_path),
                conf=det_cfg["confidence_threshold"],
                iou=det_cfg["iou_threshold"],
                imgsz=det_cfg["imgsz"],
                device=det_cfg["device"],
                classes=det_cfg["classes"],
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"Inference failed for {frame_path}: {exc}")
            raise self.retry(exc=exc, countdown=5)

        elapsed = time.perf_counter() - t0
        result = preds[0]

        # Parse detections
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(box.conf[0]), 4),
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
            })

        frame_result = {
            "frame_path": str(frame_path),
            "source_dataset": item["source_dataset"],
            "sequence_id": item["sequence_id"],
            "frame_index": item["frame_index"],
            "num_detections": len(detections),
            "detections": detections,
            "inference_ms": round(elapsed * 1000, 1),
        }

        # Save JSON result
        result_file = results_dir / f"{item['sequence_id']}_{item['frame_index']:06d}.json"
        with open(result_file, "w") as f:
            json.dump(frame_result, f, indent=2)

        # Save annotated frame
        if out_cfg["save_annotated"]:
            ann_path = annotated_dir / f"{item['sequence_id']}_{item['frame_index']:06d}.jpg"
            import cv2
            annotated = result.plot()
            cv2.imwrite(str(ann_path), annotated)

        logger.info(
            f"[{item['sequence_id']}] frame {item['frame_index']:>6d} | "
            f"{len(detections)} vessel(s) | {elapsed*1000:.0f}ms"
        )
        batch_results.append(frame_result)

    return batch_results
