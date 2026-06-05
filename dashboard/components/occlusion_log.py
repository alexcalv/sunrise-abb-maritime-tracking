"""Feature 2 - Occlusion Log.

Shows occlusion events (track disappear/reappear gaps) newest-first.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from data.occlusion_detector import format_timecode
from utils.colors import track_color_hex


def _jump_to(frame: int) -> None:
    """Move navigation to a frame and stop playback.

    Runs as an ``on_click`` callback so it executes before the navigation
    widgets are instantiated - assigning ``nav_playing`` (a widget key) is only
    allowed before its widget exists in the run.
    """
    st.session_state.nav_frame = int(frame)
    st.session_state.nav_playing = False


def render_occlusion_log(events: list[dict[str, Any]], fps: float, limit: int = 30) -> None:
    st.subheader("Occlusion log")

    if not events:
        st.info("No occlusion detected in this sequence.")
        return

    st.caption(f"{len(events)} event(s) - newest first")

    for index, event in enumerate(events[:limit]):
        tid = event["track_id"]
        color = track_color_hex(tid)
        timecode = format_timecode(event["gap_end"], fps)
        st.markdown(
            f"<div style='border-left:6px solid {color};padding:4px 10px;margin-bottom:2px;'>"
            f"<b>[{timecode}] Ship #{tid}</b><br>"
            f"occluded for {event['gap_seconds']:.1f}s "
            f"({event['gap_frames']} frames)<br>"
            f"<small>frame {event['gap_start']} &rarr; {event['gap_end']}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.button(
            f"Jump to frame {event['gap_start']}",
            key=f"occ_jump_{index}_{tid}_{event['gap_start']}",
            help="Go to the last frame before this ship disappeared",
            on_click=_jump_to,
            args=(event["gap_start"],),
        )
