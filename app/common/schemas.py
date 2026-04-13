from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FrameTask(BaseModel):
    batch_id: str
    clip_id: str
    source_path: str
    frame_paths: list[str]
    frame_indices: list[int]
    timestamps_ms: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionBox(BaseModel):
    frame_index: int
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    class_name: str | None = None


class FrameResult(BaseModel):
    batch_id: str
    clip_id: str
    source_path: str
    frame_index: int
    timestamp_ms: float | None = None
    image_path: str
    model_name: str
    worker_id: str
    detections: list[DetectionBox] = Field(default_factory=list)
