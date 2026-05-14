# Maritime Tracking Pipeline

This project tracks maritime vessels in video and evaluates identity continuity through occlusion. The workflow combines YOLO-based detection, explicit multi-object tracking, MOT export, video rendering, quantitative evaluation, and offline identity repair.

In practice, the main workflow is:

`track -> MOT export -> render-video -> evaluate -> offline stitch`

That path has been exercised on both BoT-SORT and ByteTrack, and it is the basis for the current benchmark and demo set.

## Workflow Overview

### 1. Ingestion and distributed detection

The project can extract frames, distribute detection batches through Celery and Redis, persist per-frame JSON results, and aggregate those detections back into summary files. This is useful for detector-side experiments and for separating ingestion from later stages.

### 2. Explicit tracking

Tracking runs directly on raw video or image folders with BoT-SORT or ByteTrack. Each run writes tracker MOT output together with a run summary, which makes the tracking stage reproducible and easy to evaluate later.


### 3. Rendering

Rendered videos can be produced directly from a source video and a MOT file. This is the main visualization path for tracker output and stitched output. A legacy detection-overlay mode is still available for the distributed detection path.

### 4. Evaluation

The evaluation utilities work with MOT-style ground truth. They support GT validation, GT construction from curated tracker fragments, and MOT metrics such as IDF1 and ID switch count.

### 5. Offline stitching and ReID

After tracking, the stitcher can reconnect trajectory fragments across occlusion gaps. It uses motion and bounding-box consistency, and can add appearance cues from vessel crops extracted from the source video. The stitched result is written as a separate run so the original tracker output remains intact.

### 6. Sparse localization evaluation

A separate sparse-localization path is available for manually annotated occlusion windows. It computes center-position error on selected frames and stays separate from the identity benchmark.

## Repository Structure

```text
app/
  broker/        Celery wiring
  common/        settings, logging, shared schemas
  evaluation/    MOT parsing, GT helpers, evaluators
  ingestion/     frame extraction and batch creation
  output/        aggregation and video rendering
  scripts/       CLI entrypoint
  stitching/     offline track stitching and appearance matching
  tracking/      tracker runner with MOT export
  worker/        YOLO detector, Celery task, JSON persistence

config/
  trackers/      BoT-SORT and ByteTrack configs
  stitching/     motion-only and appearance-aware stitch configs

data/
  videos/        source clips
  gt/            benchmark GT and manifests

outputs/
  demo/          rendered demo videos
  eval/          evaluation artifacts
  stitching/     batch stitch summaries
  track/         tracker runs and stitched runs
```

## Command-Line Workflows

The Docker image entrypoint is:

```bash
python -m scripts.cli
```

Main commands:

| Command | Purpose |
|---|---|
| `pipeline-ingest` | extract frames and queue distributed detection batches |
| `pipeline-aggregate` | aggregate worker JSON outputs into summaries |
| `pipeline-full` | run ingest, wait for detection batches, aggregate, and render detections |
| `track` | run BoT-SORT or ByteTrack and write MOT plus `run_summary.json` |
| `render-video` | render a detection overlay or MOT overlay back onto video |
| `evaluate` | compute MOT-style identity metrics against MOT ground truth |
| `gt-merge-ids` | build curated GT from tracker fragments |
| `gt-validate` | validate MOT-style GT files |
| `stitch-tracks` | stitch one tracker run into a separate corrected run |
| `stitch-batch` | stitch multiple runs and write a compact batch summary |
| `evaluate-localization` | evaluate sparse manual localization annotations |

### Unit tests in Docker

The image `ENTRYPOINT` is `python -m scripts.cli`, so extra words after `runner` are parsed as CLI subcommands. **`docker compose run --rm runner python -m unittest …` fails** because `python` is not a valid subcommand.

Use either the dedicated **`test`** service (recommended):

```bash
docker compose run --rm test
```

Or override the entrypoint on **`runner`**:

```bash
docker compose run --rm --entrypoint python runner \
  -m unittest discover -s /workspace/tests -p 'test*.py' -v
```

The `runner` service mounts `./tests` to `/workspace/tests` so discovery can see your host test files.

## Representative Commands

### BoT-SORT on a single clip

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name botsort_one
```

### ByteTrack on the same clip

```bash
docker compose run --rm runner track \
  --model /workspace/models/YOLOV8M_CUSTOM.pt \
  --source /workspace/data/videos/cut28#1.mp4 \
  --tracker /workspace/config/trackers/bytetrack_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name bytetrack_one
```

### Render stitched MOT back onto the source video

```bash
docker compose run --rm runner render-video \
  --source /workspace/data/videos/cut28#1.mp4 \
  --mot-file /workspace/outputs/track/botsort_one/stitched/reid_appearance_v2/mot/cut28#1.txt \
  --output /workspace/outputs/demo/stitching_reid_demo.mp4 \
  --label-mode id
```

### Evaluate against curated MOT ground truth

```bash
docker compose run --rm runner evaluate \
  --pred-dir /workspace/outputs/track/botsort_one/mot \
  --gt-dir /workspace/data/gt/benchmark_v2_continuity/cut28_target \
  --output /workspace/outputs/eval/benchmark_v2_continuity/cut28_target/botsort_one_original.csv \
  --summary-json /workspace/outputs/eval/benchmark_v2_continuity/cut28_target/botsort_one_original.json
```

### Build continuity GT from tracker fragments

```bash
docker compose run --rm runner gt-merge-ids \
  --input /workspace/outputs/track/botsort_one/mot/cut28#1.txt \
  --output /workspace/data/gt/benchmark_v2_continuity/cut28_target/cut28#1.txt \
  --source-ids 2 6 \
  --gt-id 1 \
  --class-id 0 \
  --sort
```

### Stitch a tracker run

```bash
docker compose run --rm runner stitch-tracks \
  --pred-dir /workspace/outputs/track/botsort_one/mot \
  --config /workspace/config/stitching/reid_appearance_v2.yaml
```

### Stitch a larger BoT-SORT run and summarize it

```bash
docker compose run --rm runner stitch-batch \
  --runs-root /workspace/outputs/track/botsort_all_videos \
  --batch-dir /workspace/outputs/stitching/batches/reid_appearance_botsort_all_v1_refresh \
  --config /workspace/config/stitching/reid_appearance_v1.yaml
```

### Evaluate sparse localization windows

```bash
docker compose run --rm runner evaluate-localization \
  --pred-dir /workspace/outputs/eval/benchmark_v2_continuity/cut28_target/pred_subsets/botsort_one_original/mot \
  --gt-dir /workspace/data/gt/benchmark_v2_continuity/cut28_target/localization_sparse \
  --output /workspace/outputs/eval/benchmark_v2_continuity_localization/cut28_target/botsort_one_original.csv \
  --summary-json /workspace/outputs/eval/benchmark_v2_continuity_localization/cut28_target/botsort_one_original.json
```

## Benchmark and Evaluation

The current benchmark is `data/gt/benchmark_v2_continuity/benchmark_manifest.json`. It focuses on curated occlusion cases and identity continuity rather than full-scene annotation.

Current cases:

- `cut28_target`
  - BoT-SORT original vs stitched
  - ByteTrack original vs stitched
- `cut29_target`
  - BoT-SORT original vs refreshed stitched output
- `cut34_target`
  - BoT-SORT original vs refreshed stitched output

Ground truth in this benchmark is semi-manual at the identity level, while bounding boxes are inherited from tracker outputs.

The MOT evaluator reports:

- `mota`
- `motp`
- `idf1`
- `idp`
- `idr`
- `num_switches`
- `num_fragmentations`
- `precision`
- `recall`

Current benchmark results are simple:

- each curated case shows one ID switch in the original evaluated tracker output
- the stitched output removes that switch in all three cases
- stitched outputs reach `IDF1 = 1.0` on those cases

BoT-SORT and ByteTrack are directly comparable on `cut28_target`. Broader ByteTrack multi-case parity is still incomplete in the saved workspace.

Sparse localization evaluation is defined separately in `data/gt/benchmark_v2_continuity/localization_sparse_manifest.json`. It uses manually annotated occlusion-window frames and reports:

- `mean_center_error_px`
- `median_center_error_px`
- `max_center_error_px`
- `evaluated_frame_count`

## Practical Notes

On a single machine, the distributed detection path works best with modest settings:

- keep worker concurrency low
- prefer one worker over many aggressive parallel workers
- run explicit tracking as a separate stage

This keeps the pipeline stable on modest hardware and reflects the way the project is currently used.

For the distributed worker path, note that the base `worker` service in `docker-compose.yml` still defaults to `YOLO_DEVICE: "0"`. For CPU-only execution, override the device explicitly.

## Current Limitations

- The distributed detection pipeline is implemented, but most runtime coverage currently comes from tracking and evaluation.
- `pipeline-full` does not call the tracker; it runs ingest, detection, aggregation, and detection rendering.
- ReID remains offline; it does not run inside BoT-SORT runtime.
- `HOTA` is not implemented.
- Mean position error during or immediately after occlusion is not yet established by the current benchmark because manual sparse localization GT has not been committed.
