"""Feature 1 - Ship Panel.

Lists every vessel visible at the current frame with its id, current location
(bbox centre) and how long it has been tracked so far.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from data.stats import occlusion_counts, sequence_summary
from utils.colors import track_color_hex


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def render_ship_panel(
    frame_rows: list[dict[str, Any]],
    current_frame: int,
    first_seen: dict[int, int],
    fps: float,
    all_rows: list[dict[str, Any]] | None = None,
    occlusions: list[dict[str, Any]] | None = None,
    focus_id: int | None = None,
) -> None:
    st.subheader("Ships on screen")

    # Sequence-level summary across the whole run (not just this frame).
    if all_rows:
        summary = sequence_summary(all_rows, fps)
        active = len(frame_rows)
        longest = (
            f"#{summary['longest_id']} ({summary['longest_seconds']:.1f}s)"
            if summary["longest_id"] is not None
            else "-"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total ships", summary["total_ships"])
        c2.metric("Active now", active)
        c3.metric("Longest tracked", longest)

    if not frame_rows:
        st.info("No vessel detected at this frame.")
        return

    rows_to_show = frame_rows
    if focus_id is not None:
        rows_to_show = [row for row in frame_rows if row["id"] == focus_id]
        if not rows_to_show:
            st.info(f"Ship #{focus_id} is not on this frame.")
            return

    safe_fps = fps if fps and fps > 0 else 25.0
    counts = occlusion_counts(occlusions) if occlusions else {}
    st.caption(f"{len(rows_to_show)} vessel(s) at frame {current_frame}")

    for row in sorted(rows_to_show, key=lambda r: r["id"]):
        tid = row["id"]
        cx, cy = _center(row["bbox"])
        duration_frames = current_frame - first_seen.get(tid, current_frame)
        duration_s = max(0, duration_frames) / safe_fps
        color = track_color_hex(tid)
        occ = counts.get(tid, 0)
        occ_line = f"<br>occlusions: {occ}" if occ else ""

        st.markdown(
            f"<div style='border-left:6px solid {color};padding:4px 10px;margin-bottom:6px;'>"
            f"<b>Ship #{tid}</b><br>"
            f"position: ({cx:.0f}, {cy:.0f})<br>"
            f"tracked for: {duration_s:.1f}s ({duration_frames} frames)"
            f"{occ_line}"
            f"</div>",
            unsafe_allow_html=True,
        )
