"""Aggregate statistics over a MOT sequence for the ship panel header.

Pure helpers (no Streamlit) so they can be unit-tested directly.
"""

from __future__ import annotations

from typing import Any


def sequence_summary(rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    """Summarise a whole sequence: number of distinct vessels and the one
    tracked the longest (by span between its first and last frame).

    Returns ``total_ships``, ``longest_id`` (or ``None``) and
    ``longest_seconds``. Negative/unknown ids are ignored.
    """
    safe_fps = fps if fps and fps > 0 else 25.0
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for row in rows:
        tid = row["id"]
        if tid is None or tid < 0:
            continue
        frame = row["frame"]
        if tid not in first or frame < first[tid]:
            first[tid] = frame
        if tid not in last or frame > last[tid]:
            last[tid] = frame

    longest_id: int | None = None
    longest_frames = -1
    for tid in first:
        span = last[tid] - first[tid]
        if span > longest_frames:
            longest_frames = span
            longest_id = tid

    return {
        "total_ships": len(first),
        "longest_id": longest_id,
        "longest_seconds": max(0, longest_frames) / safe_fps if longest_id is not None else 0.0,
    }


def occlusion_counts(events: list[dict[str, Any]]) -> dict[int, int]:
    """Count occlusion events per track id."""
    counts: dict[int, int] = {}
    for event in events:
        tid = event["track_id"]
        counts[tid] = counts.get(tid, 0) + 1
    return counts
