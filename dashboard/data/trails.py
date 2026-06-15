"""Build per-track movement trails from MOT rows.

A trail is the sequence of bbox-centre points a vessel occupied over the last
``length`` frames up to the current frame. The dashboard draws it as a coloured
polyline so the viewer can see where each ship has been (single-camera history).
"""

from __future__ import annotations

from typing import Any


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def build_trails(
    rows: list[dict[str, Any]],
    current_frame: int,
    length: int = 30,
) -> dict[int, list[tuple[float, float]]]:
    """Return ``{track_id: [(cx, cy), ...]}`` for the frames in
    ``(current_frame - length, current_frame]``, ordered oldest -> newest.

    Negative/unknown ids are skipped. Tracks with a single point are kept (the
    caller decides whether a one-point trail is worth drawing).
    """
    start = current_frame - max(0, length)
    by_id: dict[int, list[tuple[int, float, float]]] = {}
    for row in rows:
        tid = row["id"]
        if tid is None or tid < 0:
            continue
        frame = row["frame"]
        if start < frame <= current_frame:
            cx, cy = _center(row["bbox"])
            by_id.setdefault(tid, []).append((frame, cx, cy))

    trails: dict[int, list[tuple[float, float]]] = {}
    for tid, points in by_id.items():
        points.sort(key=lambda item: item[0])
        trails[tid] = [(cx, cy) for _frame, cx, cy in points]
    return trails
