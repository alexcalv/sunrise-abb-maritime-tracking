"""Serialise occlusion events to CSV for download.

Kept separate from the detector so the export format can evolve without
touching detection logic.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from data.occlusion_detector import format_timecode


def occlusions_to_csv(events: list[dict[str, Any]], fps: float) -> str:
    """Return the occlusion events as CSV text (one row per event)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["track_id", "gap_start", "gap_end", "gap_frames", "gap_seconds", "timecode"]
    )
    for event in events:
        writer.writerow(
            [
                event["track_id"],
                event["gap_start"],
                event["gap_end"],
                event["gap_frames"],
                f"{event['gap_seconds']:.3f}",
                format_timecode(event["gap_end"], fps),
            ]
        )
    return buffer.getvalue()
