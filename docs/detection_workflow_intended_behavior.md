# Detection workflow — what it does and what it does not

US14 — Sprint 3
Karim Zaafrani

## Why this document

The detection part of the pipeline is going to be plugged together with the tracker, the evaluation, and the offline ReID during Sprint 3. Before that happens it helps to have a short note that says what the detection workflow is supposed to do, what it gives back, and, just as important, what it is not supposed to do. That way nobody has to read the code to guess.

This note only covers the detection side. The tracker, the evaluator, and the stitcher are separate and have their own code.

## What "detection workflow" means here

In this repo, "detection" is the distributed path that takes a video, splits it into frames, runs YOLO on each frame through Celery workers, and writes the results to disk. It does not track anything across frames and it does not produce identities.

The CLI exposes it through three commands:

- `pipeline-ingest` — extract frames and send detection batches to the workers
- `pipeline-aggregate` — merge the per-frame JSON files into one summary per clip
- `pipeline-full` — do the three things above in a row and also render a detection overlay video at the end

The actual inference happens inside `app/worker/detector.py` (a small wrapper around Ultralytics YOLO) and is called by the Celery task `maritime.detect_batch` in `app/worker/tasks.py`.

## What goes in

The source can be either a video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or a folder of images (`.jpg`, `.png`, etc.). Anything else raises an error.

Everything else is configured by environment variables. The ones that actually affect the detection output are:

| Variable | Default | What it does |
|---|---|---|
| `YOLO_MODEL` | `yolo26l.pt` | Which weights to use |
| `YOLO_DEVICE` | `auto` | `cpu`, `0`, `cuda:0` |
| `YOLO_CONF` | `0.25` | Minimum confidence kept |
| `YOLO_IOU` | `0.45` | NMS IoU threshold |
| `YOLO_IMGSZ` | `640` | Inference image size |
| `YOLO_CLASSES` | `8` | Comma-separated class ids (empty = keep all) |

On top of that, the CLI takes `--source`, `--batch-size` (default 16 frames per Celery task) and `--frame-step` (default 1, meaning every frame).

## What comes out

Three kinds of files are written:

- The extracted frames, as JPGs under `FRAMES_DIR/<clip_id>/`. They are numbered from 1 and always contiguous, even when `--frame-step` skips raw frames.
- One JSON per frame under `WORKER_RESULTS_DIR`. It follows the `FrameResult` schema in `app/common/schemas.py` and contains the detections (x, y, w, h in pixels, confidence, class id) plus some metadata (clip id, batch id, worker id, timestamp).
- One aggregated summary per clip under `AGGREGATED_DIR` once `pipeline-aggregate` or `pipeline-full` has run.

If you use `pipeline-full`, you also get a rendered overlay video at `/workspace/outputs/rendered/<clip_id>.mp4`. This is meant for quick visual checks of the detector, not as a tracking result.

## What the workflow is supposed to do

Run YOLO detection on every frame (or every Nth frame) of a clip in a distributed way, and write the detections to disk in a format the rest of the code can read later. The key design choices are:

- Batches of frames are dispatched to Celery workers so several workers can run in parallel without extra coordination.
- Per-frame results are persisted immediately, so aggregation can be re-run any time without redoing YOLO.
- Video sources and image-folder sources are normalized into the same layout so downstream code does not have to care which one it is.

## What the workflow is not supposed to do

This part is the one most people get wrong, so it is listed on purpose.

- **No track identity.** The detections have no id across frames. If a boat is detected in frame 10 and again in frame 11, nothing links the two. That is the tracker's job.
- **`pipeline-full` does not call the tracker.** It runs ingest, detection, aggregation and detection rendering. Nothing more. If you want MOT output, you run `track` separately. This is confirmed in the README "Current Limitations".
- **No re-identification.** ReID lives in `app/stitching/` and runs offline on top of tracker output, not on raw detections.
- **No evaluation.** Metrics (IDF1, ID switches, position error) are produced by the `evaluate` and `evaluate-localization` commands, which work on tracker output and on manually annotated GT. The detection workflow produces no metrics.
- **No occlusion handling.** When a boat is fully occluded, the detector just returns no box. Any reasoning about occlusion is done later (tracker, stitcher, localization evaluator).
- **No COLREG or motion priors.** The detector is purely per-frame visual. US10 is a separate concern.
- **No live or streaming input.** A complete file on disk is expected before the workflow starts. No RTSP, no webcam, no partial file.
- **No multi-camera fusion.** Each clip is processed on its own. Multi-node and multi-camera are the subject of US12.

## A few edge cases worth knowing

- If the video cannot be opened by OpenCV, the ingest stage throws `RuntimeError`.
- If YOLO finds nothing on a frame, the frame still gets a JSON file with an empty detections list. Missing boxes are not the same as missing frames.
- If `--frame-step` is larger than 1, only some frames are extracted and detected, and the saved indices are contiguous (1, 2, 3…), not the raw indices. Downstream code should use the saved index.
- If the worker crashes mid-batch, Celery retries the task up to three times before giving up.
- If Redis is unreachable, the CLI fails either at enqueue or while waiting for tasks.
- Running the pipeline a second time on the same clip overwrites the previous frames and JSON files.

## Where the detection fits in

There are really two entry points in this project. One is `track`, which takes a video and runs the tracker directly on it. The other is the detection workflow described here, which runs distributed YOLO and writes detection data.

For the current benchmark (BoT-SORT, ByteTrack, IDF1, IDSW, stitching), everything goes through `track`, not through this workflow. The detection workflow is mostly useful for detector-side experiments, for separating ingestion and detection from the rest, and as the base for the distributed deployment story that US12 is building on.

That is the scope. Anything the team wants to add (tracker inside `pipeline-full`, online ReID, multi-camera, COLREG priors) is a separate change, not a modification of the current detection workflow.
