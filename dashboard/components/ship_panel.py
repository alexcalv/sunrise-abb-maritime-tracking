"""Feature 1 - Ship Panel.

Lists every vessel visible at the current frame with its id, current location
(bbox centre) and how long it has been tracked so far.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from utils.colors import track_color_hex


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def render_ship_panel(
    frame_rows: list[dict[str, Any]],
    current_frame: int,
    first_seen: dict[int, int],
    fps: float,
) -> None:
    st.subheader("Ships on screen")

    if not frame_rows:
        st.info("No vessel detected at this frame.")
        return

    safe_fps = fps if fps and fps > 0 else 25.0
    st.caption(f"{len(frame_rows)} vessel(s) at frame {current_frame}")

    for row in sorted(frame_rows, key=lambda r: r["id"]):
        tid = row["id"]
        cx, cy = _center(row["bbox"])
        duration_frames = current_frame - first_seen.get(tid, current_frame)
        duration_s = max(0, duration_frames) / safe_fps
        color = track_color_hex(tid)

        st.markdown(
            f"<div style='border-left:6px solid {color};padding:4px 10px;margin-bottom:6px;'>"
            f"<b>Ship #{tid}</b><br>"
            f"position: ({cx:.0f}, {cy:.0f})<br>"
            f"tracked for: {duration_s:.1f}s ({duration_frames} frames)"
            f"</div>",
            unsafe_allow_html=True,
        )
