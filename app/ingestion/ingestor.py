from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2

from common.schemas import FrameTask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _clip_id(source: str) -> str:
    return Path(source).stem


def _extract_video(source: Path, frames_dir: Path, frame_step: int) -> tuple[list[str], list[int], list[float]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    clip_dir = frames_dir / source.stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    frame_indices: list[int] = []
    timestamps_ms: list[float] = []

    raw_idx = 0
    saved_idx = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if raw_idx % frame_step == 0:
            out = clip_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            frame_paths.append(str(out))
            frame_indices.append(saved_idx)
            timestamps_ms.append((raw_idx / fps) * 1000.0 if fps > 0 else float(saved_idx))
            saved_idx += 1

        raw_idx += 1

    cap.release()
    return frame_paths, frame_indices, timestamps_ms


def _collect_images(source: Path, frames_dir: Path) -> tuple[list[str], list[int], list[float]]:
    clip_dir = frames_dir / source.stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    frame_paths: list[str] = []
    frame_indices: list[int] = []
    timestamps_ms: list[float] = []

    for idx, img in enumerate(images, start=1):
        dst = clip_dir / f"{idx:06d}{img.suffix.lower()}"
        shutil.copy2(img, dst)
        frame_paths.append(str(dst))
        frame_indices.append(idx)
        timestamps_ms.append(float(idx - 1))

    return frame_paths, frame_indices, timestamps_ms


def build_tasks(source: str, frames_dir: str, batch_size: int = 16, frame_step: int = 1) -> list[FrameTask]:
    source_path = Path(source)
    frames_root = Path(frames_dir)
    frames_root.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        frame_paths, frame_indices, timestamps_ms = _collect_images(source_path, frames_root)
    elif source_path.suffix.lower() in VIDEO_EXTS:
        frame_paths, frame_indices, timestamps_ms = _extract_video(source_path, frames_root, frame_step)
    else:
        raise ValueError(f"Unsupported source: {source}")

    clip_id = _clip_id(source)
    tasks: list[FrameTask] = []

    for batch_num in range(math.ceil(len(frame_paths) / batch_size)):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, len(frame_paths))
        tasks.append(
            FrameTask(
                batch_id=f"{clip_id}-batch-{batch_num:04d}",
                clip_id=clip_id,
                source_path=str(source_path),
                frame_paths=frame_paths[start:end],
                frame_indices=frame_indices[start:end],
                timestamps_ms=timestamps_ms[start:end],
                metadata={"batch_size": batch_size, "frame_step": frame_step},
            )
        )

    return tasks
