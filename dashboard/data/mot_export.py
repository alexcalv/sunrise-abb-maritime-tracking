"""Serialise (corrected) MOT rows back to MOT text.

Lets the user download the tracking file with their ID corrections baked in,
without ever touching the original on disk. The format matches the team writer
in ``app/evaluation/mot.py`` (bbox as ``.4f``, confidence/visibility as ``.6f``)
so the exported file round-trips through the same parser.
"""

from __future__ import annotations

from typing import Any


def rows_to_mot_text(rows: list[dict[str, Any]]) -> str:
    """Return MOT-formatted text for ``rows``, sorted by (frame, id)."""
    lines: list[str] = []
    for row in sorted(rows, key=lambda r: (r["frame"], r["id"])):
        x, y, w, h = row["bbox"]
        confidence = float(row.get("confidence", 1.0))
        class_id = row.get("class_id", 0)
        visibility = float(row.get("visibility", 1.0))
        lines.append(
            f"{row['frame']},{row['id']},"
            f"{x:.4f},{y:.4f},{w:.4f},{h:.4f},"
            f"{confidence:.6f},{class_id},{visibility:.6f}"
        )
    return "\n".join(lines) + "\n" if lines else ""
