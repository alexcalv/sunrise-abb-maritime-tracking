"""Per-track presence timeline across the whole sequence.

For each track id, computes the contiguous frame segments during which the
track is present. A gap larger than ``max_gap`` frames splits a track into
separate segments, so occlusions appear as breaks in the timeline.

Unlike the trail overlay, this is motion-independent: it shows *when* a vessel
is tracked rather than *where* it moves, so it stays informative even on
near-static footage.
"""

from __future__ import annotations

from typing import Any


def build_presence_segments(
    rows: list[dict[str, Any]], max_gap: int = 1
) -> dict[int, list[tuple[int, int]]]:
    """Return ``{track_id: [(start_frame, end_frame), ...]}``.

    Each tuple is an inclusive run of frames the track was present for. Frames
    missing for more than ``max_gap`` start a new segment, so a gap shows up as
    a break between two segments. Negative ids (tracker sentinels) are skipped.
    """
    frames_by_id: dict[int, list[int]] = {}
    for row in rows:
        tid = row["id"]
        if tid is None or tid < 0:
            continue
        frames_by_id.setdefault(tid, []).append(row["frame"])

    timelines: dict[int, list[tuple[int, int]]] = {}
    for tid, frames in frames_by_id.items():
        ordered = sorted(set(frames))
        segments: list[tuple[int, int]] = []
        seg_start = prev = ordered[0]
        for frame in ordered[1:]:
            if frame - prev > max_gap:
                segments.append((seg_start, prev))
                seg_start = frame
            prev = frame
        segments.append((seg_start, prev))
        timelines[tid] = segments
    return timelines
