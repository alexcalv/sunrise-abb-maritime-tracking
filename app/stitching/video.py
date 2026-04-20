from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


class VideoFrameReader:
    def __init__(self, source_path: str, cache_size: int = 32) -> None:
        self.source_path = Path(source_path)
        self.cache_size = max(1, int(cache_size))
        self._capture = cv2.VideoCapture(str(self.source_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open source video: {self.source_path}")
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._cache.clear()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_frame(self, frame_index: int):
        if frame_index < 1:
            return None
        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached.copy()

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None

        self._cache[frame_index] = frame.copy()
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def extract_crop(
        self,
        frame_index: int,
        bbox: list[float],
        crop_padding: float = 0.15,
        min_crop_size: int = 48,
    ):
        frame = self.get_frame(frame_index)
        if frame is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = (float(value) for value in bbox)
        if w <= 1 or h <= 1:
            return None

        pad_w = w * float(crop_padding)
        pad_h = h * float(crop_padding)

        x1 = int(round(x - pad_w))
        y1 = int(round(y - pad_h))
        x2 = int(round(x + w + pad_w))
        y2 = int(round(y + h + pad_h))

        min_size = max(int(min_crop_size), 1)
        if (x2 - x1) < min_size:
            center_x = int(round(x + (w / 2.0)))
            half = max(min_size // 2, 1)
            x1 = center_x - half
            x2 = center_x + half
        if (y2 - y1) < min_size:
            center_y = int(round(y + (h / 2.0)))
            half = max(min_size // 2, 1)
            y1 = center_y - half
            y2 = center_y + half

        x1 = _clamp(x1, 0, max(frame_width - 1, 0))
        y1 = _clamp(y1, 0, max(frame_height - 1, 0))
        x2 = _clamp(x2, 1, frame_width)
        y2 = _clamp(y2, 1, frame_height)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop.copy()
