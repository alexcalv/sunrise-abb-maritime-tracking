from __future__ import annotations

import socket
from typing import Any

from broker.celery_app import celery_app
from common.config import settings
from common.logging import setup_logging
from common.schemas import FrameResult, FrameTask
from worker.detector import YoloDetector, get_detector
from worker.persistence import persist_frame_result

setup_logging()


@celery_app.task(name="maritime.detect_batch", bind=True, max_retries=3)
def detect_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
    task = FrameTask(**payload)
    detector = get_detector()
    worker_id = socket.gethostname()

    persisted_files: list[str] = []

    for idx, image_path in enumerate(task.frame_paths):
        frame_index = task.frame_indices[idx]
        timestamp_ms = task.timestamps_ms[idx] if idx < len(task.timestamps_ms) else None
        detections = detector.infer_frame(image_path, frame_index)

        result = FrameResult(
            batch_id=task.batch_id,
            clip_id=task.clip_id,
            source_path=task.source_path,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image_path=image_path,
            model_name=detector.model_name,
            worker_id=worker_id,
            detections=detections,
        )
        persisted_files.append(persist_frame_result(settings.worker_results_dir, result))

    return {
        "batch_id": task.batch_id,
        "clip_id": task.clip_id,
        "worker_id": worker_id,
        "persisted_files": persisted_files,
    }
