"""Feature 3 - Click-to-edit ID.

Given the vessel the user clicked on, show an input to assign a new id. The
correction is stored as an override applied from the current frame onward; the
original MOT file is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from data.mot_export import rows_to_mot_text
from data.overrides import (
    add_override,
    clear_overrides,
    find_collisions,
    get_overrides,
    remove_override,
)


def render_id_editor(
    selected: dict[str, Any] | None,
    current_frame: int,
    run_key: str,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    st.subheader("Edit ID")

    if selected is None:
        st.info("Click a vessel on the frame to edit its ID.")
    else:
        current_id = selected["id"]
        st.write(f"Selected **Ship #{current_id}** at frame {current_frame}.")
        new_id = st.number_input(
            "New ID",
            min_value=1,
            value=int(current_id),
            step=1,
            key=f"new_id_{run_key}_{current_frame}_{current_id}",
        )
        scope = st.radio(
            "Scope",
            ["From this frame onward (split)", "Whole track (merge / relabel)"],
            key=f"scope_{run_key}_{current_frame}_{current_id}",
            help="Split: only frames from here on get the new id. "
            "Whole track: every frame of this id is relabelled - use it to merge "
            "two ids into one.",
        )
        from_frame = 0 if scope.startswith("Whole") else current_frame

        # Warn (do not block) if the target id already exists in the affected range.
        collisions = []
        if rows and int(new_id) != int(current_id):
            collisions = find_collisions(rows, int(new_id), from_frame)
        if collisions:
            shown = ", ".join(str(f) for f in collisions[:5])
            more = " ..." if len(collisions) > 5 else ""
            st.warning(
                f"Ship #{int(new_id)} already exists at frame(s) {shown}{more}. "
                "Applying this puts two vessels under the same id there."
            )

        if st.button("Apply correction", type="primary"):
            if int(new_id) == int(current_id):
                st.warning("New ID is identical to the current one.")
            else:
                add_override(run_key, old_id=current_id, new_id=int(new_id), from_frame=from_frame)
                st.success(f"Ship #{current_id} -> #{int(new_id)} from frame {from_frame}.")
                st.rerun()

    entries = get_overrides(run_key)
    if entries:
        st.markdown("**Active corrections**")
        for index, entry in enumerate(entries):
            label_col, del_col = st.columns([5, 1])
            label_col.caption(
                f"#{entry['old_id']} -> #{entry['new_id']} (from frame {entry['from_frame']})"
            )
            if del_col.button("✕", key=f"del_{run_key}_{index}", help="Delete this correction"):
                remove_override(run_key, index)
                st.rerun()

        if st.button("Undo last correction"):
            remove_override(run_key, len(entries) - 1)
            st.rerun()

        confirm_key = f"confirm_clear_{run_key}"
        if st.session_state.get(confirm_key):
            st.warning("Clear all corrections? This cannot be undone.")
            yes_col, no_col = st.columns(2)
            if yes_col.button("Yes, clear all", type="primary"):
                clear_overrides(run_key)
                st.session_state[confirm_key] = False
                st.rerun()
            if no_col.button("Cancel"):
                st.session_state[confirm_key] = False
                st.rerun()
        elif st.button("Clear all corrections"):
            st.session_state[confirm_key] = True
            st.rerun()

    if rows:
        safe_name = Path(run_key).stem or "tracking"
        st.download_button(
            "Download corrected MOT",
            data=rows_to_mot_text(rows),
            file_name=f"{safe_name}_corrected.txt",
            mime="text/plain",
            help="Export the tracking file with your ID corrections applied. "
            "The original file is never modified.",
        )
