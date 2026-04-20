from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from stitching.schemas import AppearanceConfig, Tracklet
from stitching.video import VideoFrameReader


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _import_timm():
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Appearance embedding requires the 'timm' package. Rebuild the project environment after updating requirements.txt."
        ) from exc
    return timm, create_transform, resolve_model_data_config


def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_source_video_path(
    source_video_path: str | None,
    sequence_name: str | None,
) -> Path | None:
    if not source_video_path:
        return None

    source_path = Path(source_video_path)
    if source_path.exists() and source_path.is_file():
        return source_path

    if not source_path.exists() or not source_path.is_dir() or not sequence_name:
        return source_path

    supported_suffixes = [".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"]
    exact_candidates = [source_path / f"{sequence_name}{suffix}" for suffix in supported_suffixes]
    for candidate in exact_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    normalized_sequence = sequence_name.lower()
    for candidate in sorted(source_path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in supported_suffixes:
            if candidate.stem.lower() == normalized_sequence:
                return candidate

    return source_path


@dataclass
class TrackletAppearanceEmbeddings:
    head_embedding: torch.Tensor | None
    tail_embedding: torch.Tensor | None
    metadata: dict[str, Any]


class TimmAppearanceBackend:
    def __init__(self, config: AppearanceConfig) -> None:
        timm, create_transform, resolve_model_data_config = _import_timm()

        checkpoint_path = _resolve_checkpoint_path(config.checkpoint_path)
        self.model_name = config.model_name
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = max(1, int(config.batch_size))
        self.weights_source = "unknown"
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.cache_dir = config.cache_dir

        if checkpoint_path is not None and checkpoint_path.exists():
            self.model = timm.create_model(
                config.model_name,
                pretrained=False,
                num_classes=0,
                checkpoint_path=str(checkpoint_path),
            )
            self.weights_source = "checkpoint"
        elif config.allow_downloads:
            if config.cache_dir:
                cache_path = Path(config.cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(cache_path))
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))
                os.environ.setdefault("TORCH_HOME", str(cache_path))

            self.model = timm.create_model(
                config.model_name,
                pretrained=True,
                num_classes=0,
            )
            self.weights_source = "pretrained_cache_or_download"
        else:
            raise RuntimeError(
                "No local appearance checkpoint found and allow_downloads is false. "
                "Set appearance.checkpoint_path to an existing file or explicitly opt into downloads."
            )

        self.model.eval().to(self.device)
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        self.embedding_dim = int(getattr(self.model, "num_features", 0) or 0)

    def encode_crops(self, crops: list[Any]) -> torch.Tensor | None:
        if not crops:
            return None

        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))

        batches: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                features = self.model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim > 2:
                    features = features.flatten(start_dim=1)
                features = F.normalize(features, dim=1)
                batches.append(features.cpu())
        if not batches:
            return None
        return torch.cat(batches, dim=0)

    def prototype(self, crops: list[Any]) -> torch.Tensor | None:
        embeddings = self.encode_crops(crops)
        if embeddings is None or embeddings.numel() == 0:
            return None
        prototype = embeddings.mean(dim=0, keepdim=True)
        return F.normalize(prototype, dim=1).squeeze(0).cpu()

    def backend_info(self) -> dict[str, Any]:
        return {
            "name": "timm_dinov2",
            "status": "ready",
            "model_name": self.model_name,
            "device": str(self.device),
            "weights_source": self.weights_source,
            "checkpoint_path": self.checkpoint_path,
            "cache_dir": self.cache_dir,
            "embedding_dim": self.embedding_dim,
        }


def _disabled_tracklet_appearance(tracklets: list[Tracklet], reason: str) -> dict[str, TrackletAppearanceEmbeddings]:
    for tracklet in tracklets:
        tracklet.appearance = {
            "status": reason,
            "head_observations": len(tracklet.head_observations),
            "tail_observations": len(tracklet.tail_observations),
            "head_frames": [int(obs.frame) for obs in tracklet.head_observations],
            "tail_frames": [int(obs.frame) for obs in tracklet.tail_observations],
            "head_crops_extracted": 0,
            "tail_crops_extracted": 0,
            "head_embedding_ready": False,
            "tail_embedding_ready": False,
            "embedding_dim": None,
        }
    return {}


def _extract_tracklet_crops(
    reader: VideoFrameReader,
    observations,
    crop_padding: float,
    min_crop_size: int,
) -> list[Any]:
    crops: list[Any] = []
    for observation in observations:
        crop = reader.extract_crop(
            frame_index=int(observation.frame),
            bbox=list(observation.bbox),
            crop_padding=crop_padding,
            min_crop_size=min_crop_size,
        )
        if crop is not None:
            crops.append(crop)
    return crops


def build_tracklet_appearance(
    tracklets: list[Tracklet],
    source_video_path: str | None,
    config: AppearanceConfig,
) -> tuple[dict[str, TrackletAppearanceEmbeddings], dict[str, Any], list[str]]:
    warnings: list[str] = []

    if not config.enabled or config.backend == "none":
        info = {
            "name": config.backend,
            "enabled": config.enabled,
            "status": "disabled",
        }
        return _disabled_tracklet_appearance(tracklets, "disabled"), info, warnings

    if not source_video_path:
        warnings.append("Appearance extraction skipped because source video path is missing.")
        info = {
            "name": config.backend,
            "enabled": config.enabled,
            "status": "no_source",
        }
        return _disabled_tracklet_appearance(tracklets, "no_source"), info, warnings

    sequence_name = tracklets[0].sequence_name if tracklets else None
    source_path = _resolve_source_video_path(source_video_path, sequence_name)
    if source_path is None:
        warnings.append("Appearance extraction skipped because source video path is missing.")
        info = {
            "name": config.backend,
            "enabled": config.enabled,
            "status": "no_source",
        }
        return _disabled_tracklet_appearance(tracklets, "no_source"), info, warnings
    if not source_path.exists() or not source_path.is_file():
        warnings.append(f"Appearance extraction skipped because source video is unavailable: {source_path}")
        info = {
            "name": config.backend,
            "enabled": config.enabled,
            "status": "source_unavailable",
            "source_video": str(source_path),
        }
        return _disabled_tracklet_appearance(tracklets, "source_unavailable"), info, warnings

    if config.backend != "timm_dinov2":
        raise RuntimeError(f"Unsupported appearance backend: {config.backend}")

    backend = TimmAppearanceBackend(config)
    embeddings: dict[str, TrackletAppearanceEmbeddings] = {}
    total_head_crops = 0
    total_tail_crops = 0
    head_ready = 0
    tail_ready = 0

    with VideoFrameReader(str(source_path)) as reader:
        for tracklet in tracklets:
            head_crops = _extract_tracklet_crops(
                reader=reader,
                observations=tracklet.head_observations,
                crop_padding=config.crop_padding,
                min_crop_size=config.min_crop_size,
            )
            tail_crops = _extract_tracklet_crops(
                reader=reader,
                observations=tracklet.tail_observations,
                crop_padding=config.crop_padding,
                min_crop_size=config.min_crop_size,
            )

            head_embedding = backend.prototype(head_crops)
            tail_embedding = backend.prototype(tail_crops)

            total_head_crops += len(head_crops)
            total_tail_crops += len(tail_crops)
            head_ready += 1 if head_embedding is not None else 0
            tail_ready += 1 if tail_embedding is not None else 0

            metadata = {
                "status": "ready" if (head_embedding is not None or tail_embedding is not None) else "no_crops",
                "head_observations": len(tracklet.head_observations),
                "tail_observations": len(tracklet.tail_observations),
                "head_frames": [int(obs.frame) for obs in tracklet.head_observations],
                "tail_frames": [int(obs.frame) for obs in tracklet.tail_observations],
                "head_crops_extracted": len(head_crops),
                "tail_crops_extracted": len(tail_crops),
                "head_embedding_ready": head_embedding is not None,
                "tail_embedding_ready": tail_embedding is not None,
                "embedding_dim": backend.embedding_dim or None,
            }
            tracklet.appearance = metadata
            embeddings[tracklet.tracklet_id] = TrackletAppearanceEmbeddings(
                head_embedding=head_embedding,
                tail_embedding=tail_embedding,
                metadata=metadata,
            )

    backend_info = backend.backend_info()
    backend_info.update(
        {
            "enabled": config.enabled,
            "source_video": str(source_path),
            "tracklets_total": len(tracklets),
            "tracklets_with_head_embedding": head_ready,
            "tracklets_with_tail_embedding": tail_ready,
            "total_head_crops": total_head_crops,
            "total_tail_crops": total_tail_crops,
            "sample_frames": config.sample_frames,
            "crop_padding": config.crop_padding,
            "min_crop_size": config.min_crop_size,
            "use_for_matching": config.use_for_matching,
        }
    )
    return embeddings, backend_info, warnings


def cosine_similarity(
    first: torch.Tensor | None,
    second: torch.Tensor | None,
) -> float | None:
    if first is None or second is None:
        return None
    return _round(F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())
