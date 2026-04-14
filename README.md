## Running on Low-Resource Machines

This project is distributed at the architecture level, but that does not mean it will automatically scale well on a single low-resource PC. If Redis, the worker, detection, and tracking all run on the same machine, they still compete for the same CPU, RAM, and GPU. In that situation, the system can fail because of memory pressure, especially VRAM, if concurrency is too high.

The distributed pipeline helps because it lets you control the workload and, when needed, split components across multiple nodes. On a limited machine, the correct approach is to reduce concurrency and use lighter settings. Instead of launching many workers in parallel, use a single worker with sequential or near-sequential processing. This reduces the risk of out-of-memory failures and makes behavior more predictable.

### Recommended Low-Resource Mode

For modest PCs or a single GPU with limited memory:

* use only 1 worker
* use `--concurrency=1`
* use `--pool=solo`
* reduce `batch-size` to `4` or `8`
* use a lighter model such as `yolo26m.pt` or `yolo26s.pt`
* keep `imgsz` at `640` or reduce it to `512` if needed
* run tracking as a separate stage after the distributed detection pipeline
* avoid multiple CUDA processes at the same time

### Recommended Worker Configuration

In `docker-compose.yml`, the `worker` service should look like this:

```
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

```
YOLO_DEVICE: "0"
```

with:

```
YOLO_DEVICE: "cpu"
```

### What Happens on a Low-Resource Machine

If the system is badly configured, the pipeline can fail because of:

* `CUDA out of memory`
* lack of RAM
* severe slowdown caused by too many processes
* worker stalls
* queued tasks with no real progress

If it is configured properly, the system does not become more powerful, but it degrades in a controlled way. Instead of crashing, it processes more slowly. That is the practical advantage of the distributed architecture: you can adapt the number of workers and the model size to the available resources.

### Rule of Thumb

On a single weak machine, distributed does not mean aggressive parallelism. It means separating responsibilities and limiting concurrency so the system remains stable.

### Recommended Profiles

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

## Commands to Run

### 1. Build the image

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

CPU:

```
docker compose build
```

### 2. Start Redis and the worker

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d redis worker
```

CPU:

```
docker compose up -d redis worker
```

### 3. Confirm that the worker started correctly

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f worker
```

CPU:

```
docker compose ps
docker compose logs -f worker
```

### 4. Run distributed ingestion

Safe mode for limited hardware:

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-ingest \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 4 \
  --frame-step 1
```

CPU:

```
docker compose run --rm --no-deps runner \
  pipeline-ingest \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 4 \
  --frame-step 1
```

### 5. Monitor progress

```
find outputs/worker_results/cut29#2 -type f | wc -l
```

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=200 worker
```

### 6. Aggregate results

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-aggregate \
  --clip-id cut29#2
```

CPU:

```
docker compose run --rm --no-deps runner \
  pipeline-aggregate \
  --clip-id cut29#2
```

### 7. Run tracking separately

BoT-SORT on GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  track \
  --model yolo26m.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device 0 \
  --project /workspace/outputs/track \
  --name botsort_run
```

BoT-SORT on CPU:

```
docker compose run --rm --no-deps runner \
  track \
  --model yolo26m.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device cpu \
  --project /workspace/outputs/track \
  --name botsort_run
```

### 8. Inspect results

Aggregated summary:

```
head outputs/aggregated/cut29#2/summary.csv
tail outputs/aggregated/cut29#2/summary.csv
```

Aggregated MOT file:

```
head outputs/aggregated/cut29#2/cut29#2.txt
wc -l outputs/aggregated/cut29#2/cut29#2.txt
```

Tracker outputs:

```
find outputs/track/botsort_run -maxdepth 3 -type f | sort
```

### 9. Stop everything

GPU:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans
```

CPU:

```
docker compose down --remove-orphans
```

## Recommended Workflow for Limited Hardware

The most stable workflow for low-resource machines is:

1. start Redis and a single worker
2. run `pipeline-ingest` with `batch-size=4` or `8`
3. wait until results appear in `outputs/worker_results`
4. run `pipeline-aggregate`
5. run `track` separately
6. evaluate at the end

