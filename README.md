# Maritime Tracking Pipeline

Docker-first maritime vessel detection and tracking pipeline with distributed batch detection, aggregation, annotated video rendering, explicit tracker execution, and evaluation support.

## Overview

This project is designed for maritime video analysis with a modular pipeline:

1. **Ingestion**
   - Reads a video file or image folder
   - Extracts frames
   - Splits work into typed batch tasks

2. **Broker**
   - Uses Celery + Redis to dispatch batch jobs

3. **Worker**
   - Runs YOLO detection on frames
   - Persists per-frame JSON outputs

4. **Aggregator**
   - Merges worker outputs
   - Produces:
     - `summary.csv`
     - MOT-style text output

5. **Renderer**
   - Draws detections back onto extracted frames
   - Produces an annotated output video

6. **Tracking**
   - Runs explicit tracking with BoT-SORT or ByteTrack
   - Kept separate from distributed detection for stability and reproducibility

7. **Evaluation**
   - Compares prediction outputs against MOT-style ground truth

---

## Project Structure

```
.
├── app/
│   ├── broker/
│   ├── common/
│   ├── ingestion/
│   ├── output/
│   ├── tracking/
│   ├── worker/
│   └── scripts/
├── config/
│   └── trackers/
├── data/
│   ├── img/
│   ├── vid/
│   ├── gt/
│   └── gt_eval/
├── outputs/
│   ├── frames/
│   ├── worker_results/
│   ├── aggregated/
│   ├── rendered/
│   ├── track/
│   └── eval/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
└── requirements.txt
````

---

## Execution Modes

This project supports two main execution modes:

### 1. Distributed detection pipeline

Use this when you want:

* frame extraction
* distributed YOLO detection
* aggregated outputs
* annotated detection video

### 2. Explicit tracking

Use this when you want:

* full tracker execution
* BoT-SORT or ByteTrack
* tracker outputs separated from distributed batch detection

This separation is intentional. Detection distributes well across batches. Stateful tracking usually does not.

---

## Requirements

* Docker
* Docker Compose
* NVIDIA GPU + NVIDIA Container Toolkit for GPU mode
* or CPU-only mode if GPU is not available

---

## Low-Resource Machines

This project is distributed at the architecture level, but that does not mean it will automatically scale well on a single low-resource PC. If Redis, the worker, detection, and tracking all run on the same machine, they still compete for the same CPU, RAM, and GPU. In that situation, the system can fail because of memory pressure, especially VRAM, if concurrency is too high.

The distributed pipeline helps because it lets you control the workload and, when needed, split components across multiple nodes. On a limited machine, the correct approach is to reduce concurrency and use lighter settings. Instead of launching many workers in parallel, use a single worker with sequential or near-sequential processing. This reduces the risk of out-of-memory failures and makes behavior more predictable.

### Recommended low-resource mode

For modest PCs or a single GPU with limited memory:

* use only 1 worker
* use `--concurrency=1`
* use `--pool=solo`
* reduce `batch-size` to `4` or `8`
* use a lighter model such as `yolo26m.pt` or `yolo26s.pt`
* keep `imgsz` at `640` or reduce it to `512` if needed
* run tracking as a separate stage after the distributed detection pipeline
* avoid multiple CUDA processes at the same time

### Recommended worker configuration

In `docker-compose.yml`, the `worker` service should look like this:

```yaml
worker:
  build: .
  image: maritime-tracker:consolidated
  entrypoint:
    [
      "celery",
      "-A", "worker.tasks",
      "worker",
      "--loglevel=INFO",
      "--pool=solo",
      "--concurrency=1"
    ]
  environment:
    CELERY_BROKER_URL: redis://redis:6379/0
    CELERY_RESULT_BACKEND: redis://redis:6379/1
    YOLO_MODEL: yolo26m.pt
    YOLO_DEVICE: "0"
    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
    WORKER_RESULTS_DIR: /workspace/outputs/worker_results
    FRAMES_DIR: /workspace/outputs/frames
    AGGREGATED_DIR: /workspace/outputs/aggregated
  depends_on:
    - redis
  volumes:
    - ./data:/workspace/data
    - ./outputs:/workspace/outputs
    - ./config:/workspace/config
    - ./models:/workspace/models
```

If you want to force CPU execution, replace:

```yaml
YOLO_DEVICE: "0"
```

with:

```yaml
YOLO_DEVICE: "cpu"
```

### Rule of thumb

On a single weak machine, distributed does not mean aggressive parallelism. It means separating responsibilities and limiting concurrency so the system remains stable.

### Recommended profiles

Weak machine:

* 1 worker
* `concurrency=1`
* `batch-size=4`
* `yolo26s.pt` or `yolo26m.pt`

Mid-range machine:

* 1 worker
* `concurrency=1`
* `batch-size=8` or `16`
* `yolo26m.pt`

Stronger machine with comfortable GPU memory:

* 1 or 2 workers, depending on real VRAM headroom
* `concurrency=1` per worker
* `batch-size=16`
* `yolo26l.pt`

---

## Build

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

### CPU

```bash
docker compose build
```

---

## Start Services

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d redis worker
```

### CPU

```bash
docker compose up -d redis worker
```

Check service status:

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f worker
```

### CPU

```bash
docker compose ps
docker compose logs -f worker
```

---

## Distributed Detection Pipeline

### Detection only

This runs:

* frame extraction
* distributed YOLO detection
* worker result persistence

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-ingest \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 4 \
  --frame-step 1
```

### CPU

```bash
docker compose run --rm --no-deps runner \
  pipeline-ingest \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 4 \
  --frame-step 1
```

Monitor progress:

```bash
find outputs/worker_results/cut29#2 -type f | wc -l
```

Check worker logs:

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=200 worker
```

### CPU

```bash
docker compose logs --tail=200 worker
```

---

## Aggregate Detection Results

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-aggregate \
  --clip-id cut29#2
```

### CPU

```bash
docker compose run --rm --no-deps runner \
  pipeline-aggregate \
  --clip-id cut29#2
```

Outputs:

* `outputs/aggregated/cut29#2/summary.csv`
* `outputs/aggregated/cut29#2/cut29#2.txt`

Inspect:

```bash
head outputs/aggregated/cut29#2/summary.csv
tail outputs/aggregated/cut29#2/summary.csv
head outputs/aggregated/cut29#2/cut29#2.txt
wc -l outputs/aggregated/cut29#2/cut29#2.txt
```

---

## Render Annotated Detection Video

This step draws the distributed detection results back onto the extracted frames and saves a video.

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  render-video \
  --clip-id cut29#2 \
  --output /workspace/outputs/rendered/cut29#2.mp4 \
  --fps 30
```

### CPU

```bash
docker compose run --rm --no-deps runner \
  render-video \
  --clip-id cut29#2 \
  --output /workspace/outputs/rendered/cut29#2.mp4 \
  --fps 30
```

Output:

* `outputs/rendered/cut29#2.mp4`

Check:

```bash
find outputs/rendered -maxdepth 2 -type f | sort
```

---

## Full Detection Pipeline with Annotated Video

This runs:

* ingestion
* task dispatch
* wait for workers
* aggregation
* annotated video rendering

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-full \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 16 \
  --frame-step 1 \
  --poll-seconds 2 \
  --render-fps 30
```

### CPU

```bash
docker compose run --rm --no-deps runner \
  pipeline-full \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 8 \
  --frame-step 1 \
  --poll-seconds 2 \
  --render-fps 30
```

Expected outputs:

* `outputs/frames/cut29#2/...`
* `outputs/worker_results/cut29#2/...`
* `outputs/aggregated/cut29#2/summary.csv`
* `outputs/aggregated/cut29#2/cut29#2.txt`
* `outputs/rendered/cut29#2.mp4`

---

## Tracking

Tracking is intentionally separated from the distributed batch detection pipeline.

### BoT-SORT on GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  track \
  --model yolo26m.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device 0 \
  --project /workspace/outputs/track \
  --name botsort_run
```

### BoT-SORT on CPU

```bash
docker compose run --rm --no-deps runner \
  track \
  --model yolo26m.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name botsort_run
```

### ByteTrack on GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  track \
  --model yolo26m.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/bytetrack_maritime.yaml \
  --device 0 \
  --project /workspace/outputs/track \
  --name bytetrack_run
```

Inspect outputs:

```bash
find outputs/track/botsort_run -maxdepth 3 -type f | sort
find outputs/track/bytetrack_run -maxdepth 3 -type f | sort
```

---

## Evaluation

Evaluation compares prediction outputs against MOT-style ground truth.

Example:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  evaluate \
  --pred-dir /workspace/outputs/track_eval/botsort \
  --gt-dir /workspace/data/gt_eval/video29 \
  --output /workspace/outputs/eval/botsort_metrics.csv
```

Inspect:

```bash
cat outputs/eval/botsort_metrics.csv
```

---

## Where Outputs Go

### Distributed detection

* `outputs/frames/<clip_id>/`
* `outputs/worker_results/<clip_id>/`
* `outputs/aggregated/<clip_id>/summary.csv`
* `outputs/aggregated/<clip_id>/<clip_id>.txt`

### Rendered detection video

* `outputs/rendered/<clip_id>.mp4`

### Tracker runs

* `outputs/track/<run_name>/`

### Evaluation

* `outputs/eval/`

---

## Troubleshooting

### Worker stays idle and pipeline waits forever

Check:

```bash
docker compose logs -f worker
```

Common causes:

* worker import error
* wrong Celery app path
* queue mismatch
* Redis not running

### Redis hostname resolution failure

Make sure `redis` is actually running:

```bash
docker compose ps
```

Avoid hardcoded conflicting container names across different projects.

### GPU out of memory

Reduce load:

* set `--concurrency=1`
* use `--pool=solo`
* lower `batch-size`
* switch from `yolo26l.pt` to `yolo26m.pt` or `yolo26s.pt`

### Tracking does not produce output

Make sure:

* tracker config exists in `config/trackers/`
* required tracker dependencies are installed in the image
* the tracking generator is actually consumed if `stream=True` is used

### Annotated video not generated

The detection pipeline does not produce annotated video unless `render-video` or `pipeline-full` is run.

---

## Recommended Workflow

For a stable single-machine run:

1. start Redis and one worker
2. run distributed detection with a safe batch size
3. wait for `outputs/worker_results` to stop increasing
4. aggregate detections
5. render the annotated detection video
6. run tracking separately
7. evaluate at the end

This workflow is more robust than trying to do everything in parallel on weak hardware.

---

## Stop Everything

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans
```

### CPU

```bash
docker compose down --remove-orphans
```
