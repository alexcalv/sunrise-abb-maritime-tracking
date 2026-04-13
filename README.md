# Maritime Pipeline Refactor

Docker-based distributed maritime detection pipeline with explicit tracking stage.

## Architecture

camera / video / image folder
-> ingestion
-> broker (Celery + Redis)
-> parallel workers (YOLO detection)
-> aggregator (summary.csv + MOT txt)
-> explicit CLI tracking stage

## Build and run

### CPU
```bash
docker compose build
docker compose up -d redis worker
```

### GPU
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d redis worker
```

## Ingest pipeline
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-ingest \
  --source /workspace/data/vid/cut29#2.mp4 \
  --batch-size 16 \
  --frame-step 1
```

## Aggregate
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  pipeline-aggregate \
  --clip-id cut29#2
```

## Track
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps runner \
  track \
  --model /workspace/models/vessel.pt \
  --source /workspace/data/vid/cut29#2.mp4 \
  --tracker /workspace/config/trackers/botsort_maritime.yaml \
  --device 0 \
  --project /workspace/outputs/track \
  --name botsort_run
```
