from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
    yolo_model: str = os.getenv("YOLO_MODEL", "yolo26l.pt")
    yolo_device: str = os.getenv("YOLO_DEVICE", "auto")
    yolo_conf: float = float(os.getenv("YOLO_CONF", "0.25"))
    yolo_iou: float = float(os.getenv("YOLO_IOU", "0.45"))
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "640"))
    yolo_classes: str = os.getenv("YOLO_CLASSES", "8")
    frames_dir: str = os.getenv("FRAMES_DIR", "/workspace/outputs/frames")
    worker_results_dir: str = os.getenv("WORKER_RESULTS_DIR", "/workspace/outputs/worker_results")
    aggregated_dir: str = os.getenv("AGGREGATED_DIR", "/workspace/outputs/aggregated")


settings = Settings()
