"""Single-camera tracking dashboard (SUNRISE additional tasks).

Reads MOT tracking outputs already produced by the pipeline and overlays them on
the source video. Lets the user inspect vessels per frame, review occlusion
events, and correct track IDs by clicking on a vessel. Originals are never
modified - corrections live in a sidecar JSON.

Run from the repo root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

# Make sibling packages (components/, data/, utils/) importable regardless of CWD.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from components.id_editor import render_id_editor
from components.occlusion_log import render_occlusion_log
from components.ship_panel import render_ship_panel
from data.mot_loader import first_seen_frames, frame_bounds, load_mot_rows
from data.occlusion_detector import detect_occlusions
from data.overrides import apply_overrides
from data.trails import build_trails
from utils.bbox_mapper import find_ship_at_point
from utils.frame_extractor import get_video_meta, read_frame
from utils.overlay import draw_boxes, draw_trails

REPO_ROOT = Path(_DASHBOARD_DIR).parent
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "track"
VIDEOS_ROOT = REPO_ROOT / "data" / "videos"

st.set_page_config(page_title="Single-Camera Tracking Dashboard", layout="wide")


def discover_mot_files() -> dict[str, Path]:
    """Map a human-readable run key -> MOT .txt path."""
    found: dict[str, Path] = {}
    if not OUTPUTS_ROOT.exists():
        return found
    for mot_path in sorted(OUTPUTS_ROOT.glob("**/mot/*.txt")):
        key = str(mot_path.relative_to(OUTPUTS_ROOT))
        found[key] = mot_path
    return found


def mot_matches_video(mot_path: Path, video_path: Path) -> bool:
    """True when a MOT sequence name corresponds to a video file name."""
    m, v = mot_path.stem, video_path.stem
    return m == v or m in v or v in m


def persist_upload(uploaded_file, suffix: str) -> Path:
    """Write an uploaded file to a stable temp path (cv2 needs a real path)."""
    data = uploaded_file.getvalue()
    digest = hashlib.md5(data).hexdigest()[:12]
    cache_dir = Path(tempfile.gettempdir()) / "sunrise_dashboard_uploads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{digest}{suffix}"
    if not target.exists():
        target.write_bytes(data)
    return target


def select_existing_source() -> tuple[str, Path, Path] | None:
    """Sidebar UI for picking an already-produced run + video. Returns
    (run_key, mot_path, video_path) or None if nothing usable."""
    mot_files = discover_mot_files()
    if not mot_files:
        st.warning(f"No MOT files found under {OUTPUTS_ROOT}.")
        return None
    videos = sorted(VIDEOS_ROOT.glob("*.mp4")) if VIDEOS_ROOT.exists() else []
    if not videos:
        st.error(f"No videos found under {VIDEOS_ROOT}.")
        return None

    # Pick the video first, then only offer MOT files computed on that video so
    # boxes can't be drawn with coordinates from a different resolution.
    video_path = st.selectbox("Video", videos, format_func=lambda p: p.name)
    matching = {k: p for k, p in mot_files.items() if mot_matches_video(p, video_path)}
    if matching:
        options = matching
    else:
        st.warning(f"No MOT file matches {video_path.name}; showing all (check alignment).")
        options = mot_files

    run_key = st.selectbox("Tracking run (MOT file)", list(options.keys()))
    mot_path = options[run_key]
    return run_key, mot_path, video_path


def select_uploaded_source() -> tuple[str, Path, Path] | None:
    """Sidebar UI for uploading a video + MOT file. Returns
    (run_key, mot_path, video_path) or None until both are provided."""
    video_file = st.file_uploader("Raw video (.mp4)", type=["mp4"])
    mot_file = st.file_uploader("Tracking file (.txt MOT)", type=["txt"])
    if video_file is None or mot_file is None:
        st.info("Upload both a video and its MOT file to start.")
        return None
    video_path = persist_upload(video_file, ".mp4")
    mot_path = persist_upload(mot_file, ".txt")
    run_key = f"upload:{mot_file.name}"
    return run_key, mot_path, video_path


def render_navigation(mot_min: int, max_frame: int, fps: float) -> int:
    """Sidebar frame navigation: typed frame box, prev/next step, slider, and a
    play/pause toggle that auto-advances. Returns the selected frame index."""
    max_frame = int(max_frame)

    # Canonical frame value lives in session state so buttons/slider/box agree.
    if "nav_frame" not in st.session_state:
        st.session_state.nav_frame = int(mot_min)
    st.session_state.nav_frame = max(0, min(max_frame, int(st.session_state.nav_frame)))

    def _step(delta: int) -> None:
        st.session_state.nav_frame = max(0, min(max_frame, st.session_state.nav_frame + delta))
        st.session_state.nav_playing = False

    def _from_box() -> None:
        st.session_state.nav_frame = int(st.session_state.nav_box)

    def _from_slider() -> None:
        st.session_state.nav_frame = int(st.session_state.nav_slider)

    # Seed widget-bound keys from the canonical value before instantiation so the
    # box and slider always reflect the current frame (incl. after Play/step).
    st.session_state.nav_box = st.session_state.nav_frame
    st.session_state.nav_slider = st.session_state.nav_frame

    st.header("Navigation")

    prev_col, play_col, next_col = st.columns(3)
    prev_col.button("◀ Prev", on_click=_step, args=(-1,), use_container_width=True)
    next_col.button("Next ▶", on_click=_step, args=(1,), use_container_width=True)
    playing = play_col.toggle("▶ Play", key="nav_playing")

    st.number_input(
        "Go to frame", min_value=0, max_value=max_frame, step=1,
        key="nav_box", on_change=_from_box,
    )
    st.slider("Frame", min_value=0, max_value=max_frame, key="nav_slider", on_change=_from_slider)
    st.caption(f"Video FPS: {fps:.2f} | playing: {'yes' if playing else 'no'}")

    return int(st.session_state.nav_frame)


def main() -> None:
    st.markdown(
        "<h2 style='margin:0;font-weight:700;'>Single-Camera Tracking Dashboard</h2>"
        "<p style='margin:2px 0 0;color:#8b98a5;font-size:0.9rem;'>"
        "SUNRISE &middot; ABB Maritime Tracking &mdash; additional tasks</p>"
        "<hr style='margin:0.6rem 0 1rem;border:none;border-top:1px solid #2b313e;'>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Source")
        mode = st.radio("Input mode", ["Existing runs", "Upload files"], horizontal=False)
        selection = select_existing_source() if mode == "Existing runs" else select_uploaded_source()

    if selection is None:
        return
    run_key, mot_path, video_path = selection

    # Load + correct tracking data.
    rows = apply_overrides(load_mot_rows(mot_path), run_key)
    if not rows:
        st.warning("This MOT file contains no rows.")
        return

    fps, video_frames, vid_w, vid_h = get_video_meta(str(video_path))
    mot_min, mot_max = frame_bounds(rows)
    max_frame = max(mot_max, video_frames - 1)
    first_seen = first_seen_frames(rows)
    occlusions = detect_occlusions(rows, fps)

    # Guard against a MOT file from a differently-sized video: its coordinates
    # would land off-frame or squashed once drawn on this video.
    if vid_w and vid_h:
        max_x = max((row["bbox"][0] + row["bbox"][2]) for row in rows)
        max_y = max((row["bbox"][1] + row["bbox"][3]) for row in rows)
        if max_x > vid_w * 1.02 or max_y > vid_h * 1.02:
            st.warning(
                f"MOT coordinates reach {int(max_x)}x{int(max_y)} but the video is "
                f"{vid_w}x{vid_h}. This MOT file likely doesn't match this video."
            )

    with st.sidebar:
        current_frame = render_navigation(mot_min, max_frame, fps)
        st.caption(f"MOT frames {mot_min}-{mot_max}")
        st.header("Display")
        show_trails = st.toggle("Show vessel trails", value=False)
        trail_length = st.slider(
            "Trail length (frames)", min_value=5, max_value=120, value=30,
            disabled=not show_trails,
        )

    frame_rows = [row for row in rows if row["frame"] == current_frame]

    col_video, col_panels = st.columns([3, 2])

    with col_video:
        head_col, file_col = st.columns([1, 2])
        head_col.subheader(f"Frame {current_frame}")
        file_col.caption(
            f"**MOT:** `{Path(mot_path).name}`  \n**Video:** `{Path(video_path).name}`"
        )
        try:
            frame_rgb = read_frame(str(video_path), current_frame)
        except Exception as exc:  # noqa: BLE001 - surface any decode issue to the UI
            st.error(f"Could not read frame: {exc}")
            return

        image = draw_boxes(frame_rgb, frame_rows)
        if show_trails:
            image = draw_trails(image, build_trails(rows, current_frame, trail_length))

        # Scale the image responsively to the column width (aspect ratio kept,
        # no side-cropping). The component reports the displayed size back, so we
        # map clicks using that instead of a fixed display width.
        click = streamlit_image_coordinates(image, use_column_width="always", key="frame_click")
        st.caption("Click a vessel to select it for ID editing.")

    selected = None
    if click is not None:
        displayed_w = click.get("width") or image.width
        scale = image.width / displayed_w if displayed_w else 1.0
        px, py = click["x"] * scale, click["y"] * scale
        selected = find_ship_at_point(px, py, frame_rows)

    with col_panels:
        render_ship_panel(frame_rows, current_frame, first_seen, fps)
        st.divider()
        render_id_editor(selected, current_frame, run_key)
        st.divider()
        render_occlusion_log(occlusions, fps)

    # Auto-advance when Play is on. Step one frame, pace to ~video FPS, rerun.
    # At the end we simply stop advancing; nav_playing is a widget key and can't
    # be reassigned from the script body after the toggle is instantiated.
    if st.session_state.get("nav_playing") and current_frame < int(max_frame):
        st.session_state.nav_frame = current_frame + 1
        time.sleep(1.0 / max(fps, 1.0))
        st.rerun()


if __name__ == "__main__":
    main()
