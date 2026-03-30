# Vessel Detection Pipeline — Sprint 1 Baseline

Distributed YOLOv8 inference pipeline for maritime vessel detection.
Processes both **SMD** (video sequences) and **MVTD** (image frames) datasets.

---

## Architecture

```
[SMD videos] ──┐
               ├──► [Ingest coordinator] ──► [Redis broker] ──► [YOLOv8 worker × N]
[MVTD frames] ─┘                                                        │
                                                              ┌──────────┴─────────┐
                                                         [JSON results]  [Annotated frames]
```

Workers are independent processes (or containers) that pull tasks from Redis.
Scale horizontally by launching more worker processes.

---

## Quick start (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Redis

```bash
# With Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally (macOS)
brew install redis && redis-server
```

### 3. Place datasets

```
data/
  SMD/          ← unzip Singapore Maritime Dataset here (videos)
  MVTD/         ← unzip MVTD here (image folders)
```

Edit `config/pipeline.yaml` to update `root_dir` paths if needed.

### 4. Start workers

```bash
# In one terminal — start 4 parallel workers
celery -A broker.celery_app worker --loglevel=info --concurrency=4
```

### 5. Run the pipeline

```bash
# In another terminal
python run_pipeline.py --mode full
```

### Smoke test (no broker needed)

```bash
python run_pipeline.py --mode test --input path/to/vessel_image.jpg
```

---

## Distributed / Docker mode

```bash
# Build and start everything (Redis + 4 workers + Flower dashboard)
docker compose up --scale worker=4

# Then from the host, dispatch tasks
python run_pipeline.py --mode ingest

# Monitor at http://localhost:5555
```

---

## Configuration

All settings live in `config/pipeline.yaml`:

| Key | Default | Description |
|---|---|---|
| `detector.model` | `yolov8n.pt` | Model variant (n/s/m/l/x) |
| `detector.confidence_threshold` | `0.25` | Min detection confidence |
| `detector.device` | `cpu` | `cpu`, `0` (GPU 0), `cuda` |
| `detector.classes` | `null` | COCO class filter (e.g. `[8]` = boat only) |
| `ingestion.smd.frame_step` | `5` | Extract every Nth video frame |
| `ingestion.batch_size` | `8` | Frames per queue message |
| `output.format` | `json` | `json` or `csv` |
| `output.save_annotated` | `true` | Save bounding-box overlaid images |

---

## Outputs

```
outputs/
  detections/          ← per-frame JSON files (one per frame)
  annotated/           ← JPEG frames with bounding boxes drawn
  summary.json         ← merged flat table of all detections
```

Each per-frame JSON:
```json
{
  "frame_path": "...",
  "source_dataset": "SMD",
  "sequence_id": "MVI_0788",
  "frame_index": 42,
  "num_detections": 2,
  "inference_ms": 18.4,
  "detections": [
    {
      "bbox_xyxy": [120.0, 80.0, 340.0, 210.0],
      "confidence": 0.8731,
      "class_id": 8,
      "class_name": "boat"
    }
  ]
}
```

---

## Next steps (Sprint 2+)

- Fine-tune YOLOv8 on SMD/MVTD with maritime-specific classes
- Add ReID head for cross-frame vessel tracking (VesselReID / layumi collection)
- Integrate AIS data (FVessel dataset) for sensor fusion
- Evaluate on COLREG-aware scenarios
