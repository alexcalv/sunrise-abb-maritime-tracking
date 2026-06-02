"""Video frame extraction via OpenCV."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=4)
def get_video_meta(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, frame_count, width, height) for a video.

    FPS is the *native* video frame rate (used for duration maths), not the
    pipeline's processing speed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 0:
        fps = 25.0
    return (fps, frame_count, width, height)


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Read a single 0-based frame, returned as an RGB numpy array."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
