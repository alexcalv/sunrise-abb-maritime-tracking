"""Occlusion detection from MOT frame gaps.

A track is considered occluded when its id disappears for one or more frames
and then reappears. Each such gap is reported as an occlusion event. Events are
returned newest-first (reverse chronological) for the dashboard log.
"""

from __future__ import annotations

from typing import Any


def detect_occlusions(rows: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    """Find per-track frame gaps and return them as occlusion events.

    Each event: ``track_id``, ``gap_start`` (last frame before gap),
    ``gap_end`` (frame it reappears), ``gap_frames``, ``gap_seconds``.
    """
    frames_by_id: dict[int, list[int]] = {}
    for row in rows:
        if row["id"] < 0:
            continue
        frames_by_id.setdefault(row["id"], []).append(row["frame"])

    events: list[dict[str, Any]] = []
    safe_fps = fps if fps and fps > 0 else 25.0

    for track_id, frames in frames_by_id.items():
        ordered = sorted(set(frames))
        for prev, nxt in zip(ordered, ordered[1:]):
            gap = nxt - prev
            if gap > 1:
                gap_frames = gap - 1
                events.append(
                    {
                        "track_id": track_id,
                        "gap_start": prev,
                        "gap_end": nxt,
                        "gap_frames": gap_frames,
                        "gap_seconds": gap_frames / safe_fps,
                    }
                )

    # Reverse chronological: most recent reappearance first.
    events.sort(key=lambda e: e["gap_end"], reverse=True)
    return events


def format_timecode(frame: int, fps: float) -> str:
    """Convert a frame index to an HH:MM:SS timecode string."""
    safe_fps = fps if fps and fps > 0 else 25.0
    total_seconds = int(frame / safe_fps)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
