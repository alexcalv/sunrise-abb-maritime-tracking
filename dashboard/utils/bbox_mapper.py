"""Map a click point to the MOT bounding box that contains it.

MOT bboxes are stored as top-left ``(x, y)`` plus ``(w, h)``, matching the
convention used by ``app/output/render_video.py``.
"""

from __future__ import annotations

from typing import Any


def point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    """True if point (px, py) lies inside bbox [x, y, w, h]."""
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def bbox_area(bbox: list[float]) -> float:
    return float(bbox[2]) * float(bbox[3])


def find_ship_at_point(px: float, py: float, frame_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row whose bbox contains the point.

    When several boxes overlap the point, the smallest one wins so that nested
    or stacked vessels resolve to the most specific target.
    """
    candidates = [row for row in frame_rows if point_in_bbox(px, py, row["bbox"])]
    if not candidates:
        return None
    return min(candidates, key=lambda row: bbox_area(row["bbox"]))
