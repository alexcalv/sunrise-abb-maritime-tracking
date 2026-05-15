# Maritime Tracking Pipeline

This repo tracks maritime vessels in video, keeps the raw tracker output, and can write a separate ReID-stabilized MOT file when identity continuity matters. The current prototype focuses on the original occlusion problem: a small vessel disappears behind another object, the system predicts where it should be during the gap, and ReID links the reappearing track back to the earlier identity.

The usual workflow is:

`track -> MOT export -> optional ReID -> optional context -> render video -> evaluate`

Raw MOT is always preserved under `mot/`. ReID output, when enabled, is written separately under `live_reid/` or `live_reid_in_loop/`.

## What The Demo Shows

The easiest way to review the prototype is the side-by-side demo video:

| Visual | Meaning |
|---|---|
| Left panel | Original source video. |
| Right panel | Processed video with tracking, ReID, and prediction overlays. |
| Box label `raw` | The tracker ID produced by BoT-SORT or ByteTrack. |
| Box label `canon` | The canonical ReID identity after bounded-latency remapping. |
| Magenta ghost/crosshair | Predicted vessel position while the target is visually missing. |
| Magenta circle | Prediction uncertainty. The radius grows as the gap gets longer. |
| Green recovery label | ReID linked a reappearing raw track back to an earlier canonical identity. |
| COLREG/AIS text | Optional context only. It does not change tracker or ReID decisions. |

Prediction is deterministic constant velocity for now. It is for reporting and visualization, not forced matching.

## Workflow Overview

| Stage | What it does |
|---|---|
| Detection | Runs a YOLO model on video frames. |
| Tracking | Runs BoT-SORT or ByteTrack and writes raw MOT. |
| ReID | Optionally repairs broken IDs after occlusion. |
| Occlusion prediction | Reports expected image-plane position and uncertainty during gaps. |
| Context | Optionally adds COLREG/AIS fields to reports. |
| Rendering | Produces normal or side-by-side demo videos. |
| Evaluation | Runs MOT-style metrics and saved-MOT ReID regression checks. |

## Repository Structure

```text
app/
  ais/           AIS parsing, alignment, soft assignment, and checks
  colreg/        COLREG-inspired image-plane context
  evaluation/    MOT parsing, GT helpers, and evaluators
  output/        aggregation and video rendering
  reid/          saved-MOT regression and parity helpers
  scripts/       main CLI implementation package
  stitching/     active bounded-latency ReID plus legacy stitching references
  tracking/      tracker runner, MOT export, ReID hooks, and occlusion reports

scripts/
  cli.py         thin wrapper around the app CLI
  run_detection.py
                 one-command demo runner for tracking, ReID, context, and rendering

config/
  ais/           AIS/video manifest example
  trackers/      BoT-SORT and ByteTrack configs
  stitching/     ReID configs
```

Runtime logic lives in `app/`. The two files in `scripts/` are convenience entrypoints only.

## Command-Line Workflows

### Demo Command

This is the shortest command for the bounded-latency live demo:

```bash
docker compose run --rm --entrypoint python runner /workspace/scripts/run_detection.py \
  --demo-mode \
  --demo-title "Maritime Occlusion Recovery Demo" \
  --demo-note "ReID recovery with visualization-only occlusion prediction" \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/demo/final_demo \
  --name cut28_final_demo \
  --live-reid-config /workspace/config/stitching/reid_appearance_v2.yaml \
  --confirmation-observations 10 \
  --video-codec auto
```

`--demo-mode` enables tracker-loop ReID, rendering, and side-by-side demo output. It does not enable AIS or COLREG automatically.

Expected outputs:

- raw MOT: `<run>/mot/*.txt`
- canonical MOT: `<run>/live_reid_in_loop/mot/*.txt`
- occlusion report: `<run>/live_reid_in_loop/occlusion_predictions.json`
- side-by-side video: `<run>/<name>_side_by_side_demo.mp4`
- summary: `<run>/demo_summary.json`

This is a bounded-latency live demo, not zero-latency production live ReID.

Demo videos default to `--video-codec auto`, which tries to produce a Windows-friendly H.264 MP4. If Windows Media Player Legacy still refuses a file, VLC should open it; you can also rerun with `--video-codec h264` to require the H.264 path when OpenCV or ffmpeg supports it.

### Raw BoT-SORT

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name botsort_one
```

### Raw ByteTrack

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/bytetrack_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name bytetrack_one
```

### Post-Stage ReID

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/demo/reid_demo \
  --name cut28_live_reid \
  --live-reid \
  --live-reid-config /workspace/config/stitching/reid_appearance_v2.yaml \
  --confirmation-observations 10
```

### Tracker-Loop ReID

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/demo/reid_demo \
  --name cut28_live_reid_in_loop \
  --live-reid-in-loop \
  --live-reid-config /workspace/config/stitching/reid_appearance_v2.yaml \
  --confirmation-observations 10
```

### ReID With Context Fields

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/demo/reid_context_demo \
  --name cut28_context \
  --live-reid-in-loop \
  --colreg-diagnostics \
  --ais-diagnostics \
  --ais-file /workspace/path/to/ais.csv \
  --ais-video-start-time 2026-05-14T12:00:00Z
```

Only use `--ais-diagnostics` with a real AIS file and timing/projection metadata. Missing AIS is neutral.

## ReID Modes

| Mode | Flag | Output | Notes |
|---|---|---|---|
| Raw tracking | no ReID flag | `<run>/mot/` | Default behavior. |
| Post-stage ReID | `--live-reid` | `<run>/live_reid/mot/` | Runs after raw MOT is complete. |
| Tracker-loop ReID | `--live-reid-in-loop` | `<run>/live_reid_in_loop/mot/` | Runs bounded-latency mapping from inside the tracking loop. |

Fresh tracking may produce different raw IDs across local, Docker, CPU, GPU, or model-runtime settings. Use the saved-MOT regression for historical exact repairs.

Reference regression:

```bash
docker compose run --rm runner reid-regression \
  --output-dir /workspace/outputs/demo/reid_v1_regression_ci \
  --confirmation-observations 10
```

Expected anchors:

- `cut28#1`: `6->2`
- `cut29#3`: `275->244` and `277->267`

Replay/offline stitchers remain as legacy/reference code. They are not the active ReID path.

## Benchmark And Evaluation

Render MOT onto video:

```bash
docker compose run --rm runner render-video \
  --source /workspace/data/videos/cut28#1.mp4 \
  --mot-file /workspace/outputs/demo/reid_demo/cut28_live_reid/live_reid/mot/cut28#1.txt \
  --output /workspace/outputs/demo/reid_demo/cut28_live_reid.mp4 \
  --label-mode id
```

Evaluate MOT output:

```bash
docker compose run --rm runner evaluate \
  --pred-dir /workspace/outputs/track/botsort_one/mot \
  --gt-dir /workspace/data/gt/benchmark_v2_continuity/cut28_target \
  --output /workspace/outputs/eval/benchmark_v2_continuity/cut28_target/botsort_one_original.csv \
  --summary-json /workspace/outputs/eval/benchmark_v2_continuity/cut28_target/botsort_one_original.json
```

Review checks:

```bash
python -m compileall app scripts
python scripts/cli.py --help
python scripts/run_detection.py --help
docker compose config
docker compose run --rm --entrypoint python runner /workspace/scripts/run_detection.py --help
docker compose run --rm runner self-check
docker compose run --rm runner reid-regression --output-dir /workspace/outputs/demo/reid_v1_regression_ci --confirmation-observations 10
```

## COLREG And AIS Status

COLREG is image-plane context, not legal navigation compliance. It can add encounter and motion-prior fields to reports. A narrow scoring experiment exists, but it is disabled by default and is not recommended as active scoring yet.

AIS support can parse CSV/JSON, align AIS to frames, and softly assign MMSIs to MOT tracks. It supports pre-projected `x`/`y`, or `lon`/`lat` with an affine matrix. No real AIS files are included in the repo.

For real AIS validation, provide:

- AIS CSV/JSON with `timestamp` and `mmsi`
- matching video and MOT output
- FPS and video start time
- timezone or known time offset if available
- image `x`/`y`, or `lon`/`lat` plus an affine matrix

COLREG and AIS are context for now. They do not change tracker decisions, ReID gates, remaps, canonical IDs, or MOT output by default.

## Dataset Strategy

| Need | Candidate data |
|---|---|
| Detector training and domain adaptation | SeaShips and SMD-style maritime imagery/video. |
| Tracking and occlusion evaluation | MVTD-style multi-vessel tracking sequences and curated occlusion windows. |
| Appearance ReID | VR-VCA / VesselReID-style crops or tracklets. |
| Sensor fusion validation | Real AIS paired with matching video, FPS, start time, and projection metadata. |
| Fixtures only | Synthetic AIS for parser/alignment checks, not real-world validation claims. |

Detector and ReID training are outside this pass. The repo currently focuses on inference, ID recovery, reports, and demo output.

## Practical Notes

- Use small fresh-tracking batches, at most three videos at a time.
- Inspect existing outputs before rerunning, and skip completed MOT files.
- Use `--device cpu` unless GPU availability is confirmed.
- The demo writes new run folders if the requested name already exists.

## Current Limitations

- ReID is ready for review, but not fully production-hardened.
- The occlusion predictor is constant velocity only.
- COLREG is context-only by default.
- AIS needs real files and timing/projection metadata before real validation.
- `HOTA` is not implemented.
- Sparse localization annotation files are still pending.
