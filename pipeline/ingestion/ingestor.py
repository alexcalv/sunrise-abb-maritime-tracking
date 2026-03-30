"""
ingestion/ingestor.py
Handles reading of video and image datasets, extracting frames, and dispatching batched tasks to the broker queue for processing.
"""

from __future__ import annotations
import logging
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def _load_cfg() -> dict:
    """Load pipeline configuration from YAML file."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _extract_video_frames(video_path: Path, frame_step: int, frame_staging_dir: Path,) -> list[dict]:
    """
    Extract every `frame_step`-th frame from a video and save as JPEG.
    Returns a list of dictionaries containing metadata for each extracted frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Unable to open video: {video_path}")
        return []

    sequence_id = video_path.stem
    staging_dir = frame_staging_dir / sequence_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    frame_idx = 0
    saved_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames // frame_step, desc=f"Processing {video_path.name}", leave=False) as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                out_path = staging_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                frames.append({
                    "frame_path": str(out_path),
                    "source_dataset": "vid",
                    "sequence_id": sequence_id,
                    "frame_index": saved_idx,
                })
                saved_idx += 1
                bar.update(1)
            frame_idx += 1

    cap.release()
    log.info(f"Extracted {saved_idx} frames from {video_path.name}")
    return frames


def _collect_img_frames(img_root: Path, extensions: list[str]) -> list[dict]:
    """
    Walk through the img directory and collect all valid image paths.
    Returns a list of dictionaries containing metadata for each image frame.
    """
    frames = []
    image_files = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in extensions])

    for img_path in image_files:
        # Sequence ID is taken from the parent folder name
        sequence_id = img_path.parent.name or "img_root"
        frames.append({
            "frame_path": str(img_path),
            "source_dataset": "img",
            "sequence_id": sequence_id,
            "frame_index": int(img_path.stem.split("_")[-1]) if img_path.stem[-1].isdigit() else 0,
        })

    log.info(f"Collected {len(frames)} frames from img at {img_root}")
    return frames


def collect_all_frames(frame_staging_dir: Path | None = None) -> list[dict]:
    """
    Collect frames from both vid and img datasets.
    Returns a list of frame metadata dictionaries, ready for task dispatch.
    """
    cfg = _load_cfg()
    ingestion_cfg = cfg["ingestion"]

    if frame_staging_dir is None:
        frame_staging_dir = Path("outputs/staged_frames")
    frame_staging_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[dict] = []

    # Collect frames from videos
    vid_root = Path(ingestion_cfg["vid"]["root_dir"])
    if vid_root.exists():
        video_files = [p for p in vid_root.rglob("*") if p.suffix.lower() in ingestion_cfg["vid"]["supported_extensions"]]
        log.info(f"Found {len(video_files)} vid video(s)")
        for video_file in video_files:
            frames = _extract_video_frames(video_file, ingestion_cfg["vid"]["frame_step"], frame_staging_dir)
            all_frames.extend(frames)
    else:
        log.warning(f"Skipping videos")

    # Collect frames from images
    img_root = Path(ingestion_cfg["img"]["root_dir"])
    if img_root.exists():
        frames = _collect_img_frames(img_root, ingestion_cfg["img"]["supported_extensions"])
        all_frames.extend(frames)
    else:
        log.warning(f"Skipping images")

    log.info(f"Total frames collected: {len(all_frames)}")
    return all_frames


def dispatch_to_queue(all_frames: list[dict]) -> list:
    """
    Dispatch frames as batches to the Celery task queue for processing.
    Returns a list of AsyncResult objects to monitor task progress.
    """
    from broker.celery_app import detect_batch, cfg as pipeline_cfg

    batch_size = pipeline_cfg["ingestion"]["batch_size"]
    batches = [all_frames[i:i + batch_size] for i in range(0, len(all_frames), batch_size)]

    log.info(f"Dispatching {len(batches)} batches ({batch_size} frames per batch)")
    async_results = []
    for batch in tqdm(batches, desc="Queuing batches"):
        ar = detect_batch.delay(batch)
        async_results.append(ar)

    return async_results