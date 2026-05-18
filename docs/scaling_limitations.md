# Scaling limitations of the current architecture

US12 — Sprint 3
Karim Zaafrani

## Why this document

US12 is about making the system extendable to multi-camera and distributed deployments. Before deciding how to change the architecture, it helps to have an honest picture of what the current setup does well, where it starts to break, and what would need to be touched to go further. This note is that picture.

It does not propose a new architecture. Anhelina is working on that side. This document only writes down the limits that are already there today.

## What the current setup looks like

The whole stack is described in `docker-compose.yml`. In short:

- One Redis container acts as both the Celery broker and the result backend.
- One worker container runs YOLO, configured with `--pool=solo --concurrency=1`, which means it processes one batch at a time, in a single process.
- One runner container is started on demand for CLI commands (tracking, evaluation, stitching, rendering).
- All containers share the host filesystem through the `./data`, `./outputs`, `./config` and `./models` bind mounts.

The GPU variant (`docker-compose.gpu.yml`) only adds NVIDIA runtime options and pins `CUDA_VISIBLE_DEVICES=0`. It does not change the number of workers.

So the default deployment is really one Redis + one worker + one runner, all on the same machine.

## What scales reasonably today

A few things are already in the right place for growth:

- The detection side is split into small Celery tasks (`maritime.detect_batch`), one per batch of frames. Nothing in the code assumes a single worker; `--concurrency=1` is a configuration choice, not a hard limit.
- `task_acks_late=True` and `worker_prefetch_multiplier=1` in `app/broker/celery_app.py` mean tasks are only acked once they finish, and a worker never hoards more than one task at a time. If a worker crashes mid-batch, the task gets requeued.
- Per-frame results are written to disk immediately, so adding more workers does not create memory pressure: each worker flushes and moves on.
- Aggregation reads from disk, which means it is decoupled from detection and can be rerun safely.

Practically, if you only change `--concurrency` or spin up more `worker` replicas with `docker compose up --scale worker=N`, the detection throughput scales almost linearly up to the CPU or GPU limit of the host.

## Where it starts to break

The rest of the document is about the real limitations. They matter as soon as we move past "one laptop, one clip".

### One Redis, no replica

`docker-compose.yml` defines a single Redis instance. It is the broker, the result backend, and has no persistence tuning. This is a single point of failure and a bandwidth bottleneck. As long as everything runs on one machine it is fine, but it will not hold up for real multi-node use, and there is no fallback if it dies during a long run.

### Solo worker, single GPU

The worker is pinned to `--pool=solo --concurrency=1` and `CUDA_VISIBLE_DEVICES=0`. That makes runs predictable but hides the fact that a single worker cannot use a multi-GPU machine, and cannot process several batches in parallel without spinning up extra replicas by hand. There is no auto-scaling, no queue-length-aware spawn, nothing.

### Local filesystem everywhere

All data paths are local bind mounts: `./data`, `./outputs`, `./config`, `./models`. This is very convenient during development, but it is also the hardest thing to change later. As soon as the worker runs on a different host than the CLI (or the runner), the paths stop making sense. Multi-node simply does not work with the current volume setup.

### Frame extraction is serial and on disk

`ingestion.ingestor._extract_video` reads a video end-to-end with OpenCV and writes every kept frame as a JPG. That means before detection even starts, the pipeline has already gone through the whole video sequentially, and the disk now holds one JPG per frame. For a 30-minute clip at 30 FPS with `frame_step=1`, that is 54,000 files on disk just for one clip. The amount of I/O grows linearly with video length and the filesystem becomes the dominant cost on long clips.

### One JSON per frame

The worker writes one JSON per processed frame via `persist_frame_result`. Same problem as extraction: the file count explodes, and the aggregator later has to open and merge all of them. This is fine for small clips (hundreds of frames), but it does not behave well at the 100k-files scale. There is no batching at the persistence layer.

### No multi-camera coordination

Each clip is ingested and processed on its own. Nothing in the code understands "these two clips come from two cameras looking at the same scene". There is no timestamp alignment, no shared identity space, no calibration layer, no multi-view fusion. This is exactly what US12 is trying to enable and it is correctly listed as out of scope for the current architecture.

### No streaming input

The workflow expects a complete file on disk. There is no RTSP ingestion, no webcam capture, no partial-file tailing. A live deployment on a real boat or port would need a whole new ingestion path.

### CLI waits synchronously

`pipeline-full` polls Celery until all batches are finished (`wait_for_tasks` in `app/scripts/cli.py`). Fine for one clip, but it ties one CLI process to one job for the full duration. There is no job queue at the CLI level, no dashboard, no way to check long-running state without tailing the CLI output.

### Limited observability

There is no Flower, no Prometheus exporter, no health endpoint. The only signal a user has is the CLI print log and the contents of `outputs/`. Debugging a stuck worker means running `docker compose logs` by hand.

## Rough capacity today

These numbers are not a benchmark, they are a quick sanity check to make the limits less abstract.

- On CPU, YOLOv8 medium at 640px runs roughly at 10 to 30 FPS per worker depending on the host. One CPU worker can therefore cover roughly one camera at a reduced frame rate or a recorded clip played back at a lower than real-time speed.
- On a single consumer GPU, the same model runs much faster (often above 100 FPS) and the bottleneck quickly moves to disk I/O and to the extraction stage, not to YOLO itself.
- With `--scale worker=N` on one machine, throughput scales until CPU cores or GPU memory saturates. After that, adding workers only increases contention.
- For multi-camera, the practical ceiling today is: as many cameras as you can fit into one host's CPU or GPU budget, plus enough disk to hold the extracted frames. Beyond that, the single Redis and the local filesystem are the hard blockers, not the model.

## What would need to change to actually scale

This is not a full redesign proposal, just a short list of what US12 (and the Beta proto discussions) will probably have to address.

- **Shared storage** instead of bind mounts. Some network-accessible layer (NFS, object storage such as S3 or MinIO, or a distributed filesystem) so that workers on different hosts can see the same `data/`, `outputs/` and `models/`.
- **Broker that can span nodes.** Either a managed Redis with persistence and replicas, a Redis cluster, or a different broker altogether (RabbitMQ, Kafka for the streaming use case). Today there is nothing to configure on that front.
- **Multi-replica worker configuration.** Change `--pool=solo --concurrency=1` depending on the target hardware, and expose the number of replicas as a deploy-time variable rather than a fixed value in `docker-compose.yml`.
- **Streaming ingestion path.** A new ingestor that can read from RTSP or from a message bus, in parallel to the current file-based one. The rest of the pipeline should not need to care which one is the source.
- **Per-camera identity + multi-view fusion.** The tracker and the schemas need to learn about cameras, not just clips. `FrameTask` and `FrameResult` in `app/common/schemas.py` do not carry a camera id today.
- **Reduced per-frame file pressure.** Either batch JSONs per N frames, or move to a more compact store (SQLite, Parquet, a small database). The current one-file-per-frame pattern is a soft wall somewhere around the low-100k files per clip.
- **Observability.** Celery Flower or a simple Prometheus exporter, plus a health check per service, would remove the "why is nothing happening" blind spot.

## Scope, one more time

This document lists limits of the current architecture. It does not decide which of the changes above must be done first, and it does not propose a final multi-node design. Those decisions are part of US12's architecture work and of the broader discussion about how far the Beta proto should push the distributed story.
