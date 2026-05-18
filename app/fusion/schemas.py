from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamTrackObservation:
    """Single local-track observation from one camera or node (publisher output shape)."""

    camera_id: str
    # Logical source, e.g. shore | ship | simulation (string keeps adapters decoupled).
    platform: str
    frame_index: int
    local_track_id: int
    # Axis-aligned box in pixel coordinates: top-left x, y, width, height (MOT-style).
    bbox_xywh: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0

    def center_xy(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_xywh
        return x + w * 0.5, y + h * 0.5

    def normalized_center(self, image_width: int, image_height: int) -> tuple[float, float]:
        cx, cy = self.center_xy()
        return cx / float(image_width), cy / float(image_height)
