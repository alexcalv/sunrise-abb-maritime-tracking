"""Per-vessel presence timeline (Gantt-style) across the whole sequence.

Shows when each tracked vessel is present; gaps (occlusions) appear as breaks
in the bars and a dashed marker indicates the current frame. Motion-independent,
so it stays informative even on near-static footage.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from data.timeline import build_presence_segments
from utils.colors import track_color_hex


def render_timeline(rows: list[dict[str, Any]], current_frame: int, max_gap: int = 1) -> None:
    st.subheader("Vessel timeline")

    segments = build_presence_segments(rows, max_gap)
    if not segments:
        st.info("No tracked vessel to show on the timeline.")
        return

    ids = sorted(segments)
    order = [f"#{tid}" for tid in ids]
    color_range = [track_color_hex(tid) for tid in ids]

    records = []
    for tid in ids:
        for start, end in segments[tid]:
            records.append(
                {
                    "ship": f"#{tid}",
                    # +1 so a single-frame presence still renders as a visible bar
                    "start": start,
                    "end": end + 1,
                }
            )
    data = pd.DataFrame(records)

    bars = (
        alt.Chart(data)
        .mark_bar(height=14, cornerRadius=3)
        .encode(
            x=alt.X("start:Q", title="Frame"),
            x2="end:Q",
            y=alt.Y("ship:N", sort=order, title=None),
            color=alt.Color(
                "ship:N",
                scale=alt.Scale(domain=order, range=color_range),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("ship:N", title="Ship"),
                alt.Tooltip("start:Q", title="First frame"),
                alt.Tooltip("end:Q", title="Last frame (excl.)"),
            ],
        )
    )

    marker = (
        alt.Chart(pd.DataFrame({"frame": [current_frame]}))
        .mark_rule(color="#ff4b4b", strokeDash=[4, 4], size=2)
        .encode(x="frame:Q")
    )

    st.altair_chart(bars + marker, use_container_width=True)
    st.caption(
        "Each bar shows when a vessel is tracked; breaks are occlusions. "
        "The dashed line marks the current frame."
    )
