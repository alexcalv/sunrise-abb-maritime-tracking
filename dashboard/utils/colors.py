"""Track colour palette, kept consistent with ``app/output/render_video.py``.

Mirrored here (rather than imported) so the dashboard does not pull in the
heavy rendering module and its dependencies.
"""

from __future__ import annotations

TRACK_COLORS: list[tuple[int, int, int]] = [
    (80, 220, 255),
    (0, 200, 120),
    (255, 170, 0),
    (220, 80, 80),
    (180, 90, 255),
    (255, 120, 200),
    (120, 210, 70),
    (255, 255, 80),
]


def track_color(track_id: int | None) -> tuple[int, int, int]:
    """Return the RGB colour for a track id (green for unknown/negative)."""
    if track_id is None or track_id < 0:
        return (0, 255, 0)
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def track_color_hex(track_id: int | None) -> str:
    r, g, b = track_color(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"
