# Maritime vessel tracking baseline in Docker

This project gives you a reproducible Docker workflow for Sprint 1:
- train or fine-tune a YOLO detector for vessels
- run multi-object tracking with BoT-SORT or ByteTrack
- export MOT-format tracking files
- evaluate against MOT-format ground truth with `motmetrics`

## Project layout

- `configs/vessel_data.yaml` — Ultralytics dataset config
- `trackers/botsort_maritime.yaml` — BoT-SORT configuration
- `trackers/bytetrack_maritime.yaml` — ByteTrack configuration
- `scripts/cli.py` — one entrypoint for training, tracking, detection, and evaluation
- `data/videos` — input videos
- `data/labels` — YOLO training labels
- `data/gt` — MOT ground-truth text files for evaluation
- `models/vessel.pt` — trained vessel detector checkpoint
- `results/` — generated outputs

## Ground-truth format expected by evaluator

Create one file per video in `data/gt`, e.g. `harbor_01.txt`, in MOTChallenge-like CSV format:

```text
frame,id,x,y,w,h,conf,class,visibility
1,1,100,200,50,40,1,0,1
1,2,300,220,60,45,1,0,1
```

Predictions are converted from Ultralytics label files and matched against those ground-truth files.

## First run

```bash
docker compose build
```

Smoke test with any sample videos you place in `data/videos/`:

```bash
docker compose run --rm tracker detect \
  --model yolo11n.pt \
  --source /workspace/data/videos \
  --project /workspace/results/detect \
  --name smoke
```

Train a vessel detector once `configs/vessel_data.yaml` points at your dataset:

```bash
docker compose run --rm tracker train \
  --data /workspace/configs/vessel_data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --project /workspace/results/train \
  --name vessel_yolo11n
```

Run BoT-SORT:

```bash
docker compose run --rm tracker track \
  --model /workspace/models/vessel.pt \
  --source /workspace/data/videos \
  --tracker /workspace/trackers/botsort_maritime.yaml \
  --project /workspace/results/track \
  --name botsort
```

Run ByteTrack:

```bash
docker compose run --rm tracker track \
  --model /workspace/models/vessel.pt \
  --source /workspace/data/videos \
  --tracker /workspace/trackers/bytetrack_maritime.yaml \
  --project /workspace/results/track \
  --name bytetrack
```

Evaluate either tracker:

```bash
docker compose run --rm tracker evaluate \
  --pred-dir /workspace/results/track/botsort/mot \
  --gt-dir /workspace/data/gt \
  --output /workspace/results/eval/botsort_metrics.csv
```

## Notes

- Ultralytics track mode supports BoT-SORT and ByteTrack through YAML tracker configs.
- Keep the detector output contract consistent: `[x1, y1, x2, y2, confidence, class_id]`.
- For a fair comparison, BoT-SORT and ByteTrack should use the same detector checkpoint and the same video subset.




------ ### ---- 

# Multi-Object Tracking Evaluation (Sprint 1)

## Objective

Evaluate and compare multiple tracking-by-detection algorithms for **occlusion-resilient vessel tracking** using a single-camera setup.

The goal is to determine which tracker maintains **identity consistency under occlusion**, rather than only detection performance.

---

## Evaluated Trackers

* **BoT-SORT**
* **ByteTrack**
* **OC-SORT**

All trackers use the same detector:

* YOLO (`yolo26n.pt`)

All outputs are exported in **MOT format**:

```
[frame, id, x, y, w, h, conf, class, visibility]
```

---

## Evaluation Methodology

### Ground Truth

A **single-target ground truth** was manually constructed by merging fragmented tracker IDs corresponding to the same physical object.

Example:

```
GT ID 1 = {8, 190, 195}
```

This represents one physical object across occlusion events.

---

### Evaluation Approach

Evaluation is performed using IoU-based matching per frame:

* Match threshold: `IoU ≥ 0.5`
* Frame-by-frame association between GT and tracker output

---

### Metrics

The following identity-focused metrics are used:

* **Match Ratio** → detection coverage
* **Number of Tracker IDs Used** → identity fragmentation
* **Number of Segments** → trajectory continuity
* **ID Switches (contiguous)** → identity instability
* **Dominant ID Ratio** → temporal consistency

---

## Results (Single Target – Occlusion Scenario)

| Metric            | Tracker A | Tracker B |
| ----------------- | --------- | --------- |
| Match Ratio       | 1.000     | 0.999     |
| IDs Used          | 4         | 3         |
| Segments          | 34        | 4         |
| ID Switches       | 33        | 3         |
| Dominant ID Ratio | 0.61      | 0.65      |

---

## Analysis

* Both trackers achieve near-perfect detection coverage.
* However, identity continuity differs significantly.

### Tracker A

* High fragmentation (34 segments)
* High number of ID switches (33)
* Unstable identity tracking

### Tracker B

* Low fragmentation (4 segments)
* Minimal ID switches (3)
* Stable identity tracking across occlusion

---

## Key Insight

Detection performance alone is not sufficient for tracker evaluation.

> The critical factor for maritime tracking is **identity preservation during occlusion and reappearance**.

---

## Conclusion

For the evaluated scenario:

* The best-performing tracker is the one with:

  * **lowest number of segments**
  * **fewest ID switches**
  * **highest dominant ID ratio**

This tracker is currently the most suitable candidate for maritime vessel tracking.

---

## Notes on OC-SORT

OC-SORT was integrated as a custom tracker using YOLO detections.

Pipeline:

```
YOLO → detections → OC-SORT → MOT output
```

Unlike BoT-SORT and ByteTrack (native Ultralytics integration), OC-SORT runs as an external module.

---

## Limitations

* Evaluation is based on **single-target ground truth**
* GT derived from tracker outputs (not fully independent)
* Only one occlusion scenario evaluated

---

## Next Steps

1. Extend evaluation to:

   * multiple targets
   * multiple videos
   * varied occlusion conditions

2. Compute standard MOT metrics:

   * IDF1
   * HOTA
   * MOTA

3. Investigate improvements:

   * Re-identification (ReID)
   * Motion models tuned for maritime dynamics
   * Temporal smoothing

4. Automate benchmark pipeline in Docker

---

## Reproducibility

Run trackers:

```bash
# BoT-SORT / ByteTrack (existing pipeline)
docker compose run --rm tracker track ...

# OC-SORT
docker compose run --rm --entrypoint python tracker \
  scripts/run_ocsort.py \
  --model yolo26n.pt \
  --source data/videos \
  --output-dir results/track/ocsort \
  --device cpu
```

Run evaluation:

```bash
python scripts/evaluate_single_target.py \
  --gt data/gt/video_01_gt.txt \
  --pred \
    results/track/botsort/mot/dataset.txt \
    results/track/bytetrack/mot/dataset.txt \
    results/track/ocsort/mot/dataset.txt \
  --iou-thresh 0.5
```

---

## Final Takeaway

For occlusion-heavy maritime scenarios:

* **Identity continuity > detection accuracy**
* Trackers must be evaluated on **temporal consistency**, not just bounding box overlap

