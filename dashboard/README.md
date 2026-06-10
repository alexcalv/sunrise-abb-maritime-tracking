# Single-Camera Tracking Dashboard

A Streamlit dashboard for the SUNRISE / ABB maritime tracking project. It
**reads tracking outputs already produced by the pipeline** (it does not run
detection/tracking itself) and overlays them on the source video so you can:

- **Ship panel** — list the vessels visible at the current frame, with their
  position and how long each has been tracked.
- **Occlusion log** — list disappear/reappear gaps (occlusions), newest first.
- **Click-to-edit ID** — click a vessel on the frame and reassign its track ID.
  Corrections apply from the clicked frame onward and are stored in a sidecar
  JSON; the original MOT file and the video are never modified.

## How it works

The video is just pixels — it contains no IDs. The track IDs and box positions
live in a **MOT `.txt` file** (`frame, id, x, y, w, h, ...`). The dashboard draws
the boxes on top of each frame from that file, so they stay clickable/editable.

```
data/videos/Scene0_MainCamera.mp4           ← raw frames (background)
outputs/track/.../mot/Scene0_MainCamera.txt ← tracking results (IDs + boxes)
            │
            ▼
      dashboard (Streamlit) ── draws boxes ── click to edit ── overrides.json
```

## Run

```bash
pip install -r dashboard/requirements-dashboard.txt
streamlit run dashboard/app.py
```

Then open http://localhost:8501.

### Choosing inputs

- **Existing runs** (default): pick a tracking run and a video from dropdowns;
  the dashboard scans `outputs/track/` and `data/videos/`.
- **Upload files**: upload a raw `.mp4` and its MOT `.txt` manually.

## Module layout

| File | Role |
|------|------|
| `app.py` | Streamlit entrypoint, layout, click handling |
| `components/ship_panel.py` | Feature 1 — ships on screen |
| `components/occlusion_log.py` | Feature 2 — occlusion log |
| `components/id_editor.py` | Feature 3 — click-to-edit ID |
| `data/mot_loader.py` | Reuses `app/evaluation/mot.py` to parse MOT files |
| `data/occlusion_detector.py` | Finds frame gaps → occlusion events |
| `data/overrides.py` | Sidecar storage + apply of ID corrections |
| `utils/frame_extractor.py` | Reads a frame + native FPS via OpenCV |
| `utils/bbox_mapper.py` | Click `(x, y)` → bounding box lookup |
| `utils/overlay.py` | Draws boxes + ID labels on a frame |
| `utils/colors.py` | Track colour palette (matches `render_video.py`) |
