import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import motmetrics as mm
import pandas as pd
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker entrypoint for maritime vessel tracking experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train or fine-tune a YOLO detector")
    train.add_argument("--model", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=1280)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=None)
    train.add_argument("--project", default="/workspace/results/train")
    train.add_argument("--name", default="vessel_detector")
    train.add_argument("--workers", type=int, default=8)

    detect = sub.add_parser("detect", help="Run detector inference on videos/images")
    detect.add_argument("--model", required=True)
    detect.add_argument("--source", required=True)
    detect.add_argument("--conf", type=float, default=0.25)
    detect.add_argument("--imgsz", type=int, default=1280)
    detect.add_argument("--device", default=None)
    detect.add_argument("--project", default="/workspace/results/detect")
    detect.add_argument("--name", default="predict")
    detect.add_argument("--save-txt", action="store_true", default=True)
    detect.add_argument("--save-conf", action="store_true", default=True)

    track = sub.add_parser("track", help="Run BoT-SORT or ByteTrack")
    track.add_argument("--model", required=True)
    track.add_argument("--source", required=True)
    track.add_argument("--tracker", required=True)
    track.add_argument("--conf", type=float, default=0.25)
    track.add_argument("--imgsz", type=int, default=1280)
    track.add_argument("--device", default=None)
    track.add_argument("--project", default="/workspace/results/track")
    track.add_argument("--name", default="run")
    track.add_argument("--save", action="store_true", default=True)
    track.add_argument("--save-video", action="store_true", default=True)
    track.add_argument("--save-frames", action="store_true", default=False)
    track.add_argument("--show", action="store_true", default=False)
    track.add_argument("--stream", action="store_true", default=False)
    track.add_argument("--classes", nargs="*", type=int, default=None)

    evaluate = sub.add_parser("evaluate", help="Evaluate tracker outputs against MOT-format ground truth")
    evaluate.add_argument("--pred-dir", required=True, help="Directory containing Ultralytics tracking label txt files")
    evaluate.add_argument("--gt-dir", required=True, help="Directory containing MOT-format ground truth txt files")
    evaluate.add_argument("--output", required=True, help="Output CSV path")
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.add_argument("--frame-offset", type=int, default=0)

    return parser


def run_train(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        exist_ok=True,
    )


def run_detect(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        exist_ok=True,
    )


def run_track(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    output_root = Path(args.project) / args.name
    mot_dir = output_root / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    frame_counters: Dict[str, int] = {}
    line_counts: Dict[str, int] = {}

    results = model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save_video,
        save_frames=args.save_frames,
        show=args.show,
        stream=True,
        persist=True,
        classes=args.classes,
        exist_ok=True,
    )

    for result in results:
        source_path = Path(getattr(result, "path", "stream"))
        seq_name = source_path.stem or "stream"
        frame_counters[seq_name] = frame_counters.get(seq_name, 0) + 1
        frame_idx = frame_counters[seq_name]
        mot_path = mot_dir / f"{seq_name}.txt"

        boxes = result.boxes
        if boxes is None or boxes.xywh is None or len(boxes) == 0:
            continue

        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
        xywh = boxes.xywh.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(boxes)
        classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(boxes)

        with mot_path.open("a") as f:
            for track_id, box, conf, cls_id in zip(ids, xywh, confs, classes):
                xc, yc, w, h = box
                x = xc - w / 2
                y = yc - h / 2
                # MOT format: frame,id,x,y,w,h,conf,class,visibility
                f.write(f"{frame_idx},{track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{conf:.6f},{cls_id},1\n")
                line_counts[seq_name] = line_counts.get(seq_name, 0) + 1

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps({"frames": frame_counters, "rows_written": line_counts}, indent=2))
    print(json.dumps({"frames": frame_counters, "rows_written": line_counts}, indent=2))


def _load_gt_file(path: Path) -> Dict[int, List[dict]]:
    gt: Dict[int, List[dict]] = {}
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            frame = int(float(row[0]))
            obj_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            cls = int(float(row[7])) if len(row) > 7 else 0
            gt.setdefault(frame, []).append(
                {"id": obj_id, "bbox": [x, y, w, h], "class_id": cls}
            )
    return gt


def _load_pred_file(path: Path, frame_offset: int = 0) -> Dict[int, List[dict]]:
    preds: Dict[int, List[dict]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vals = [float(p) for p in parts]
            if len(vals) < 7:
                raise ValueError(
                    f"Expected at least 7 columns in prediction file {path}, got {len(vals)}"
                )
            frame = int(vals[0]) + frame_offset
            track_id = int(vals[1])
            x, y, w, h = vals[2:6]
            conf = vals[6]
            class_id = int(vals[7]) if len(vals) > 7 else 0
            preds.setdefault(frame, []).append(
                {"id": track_id, "bbox": [x, y, w, h], "conf": conf, "class_id": class_id}
            )
    return preds


def _iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def evaluate_sequence(gt: Dict[int, List[dict]], pred: Dict[int, List[dict]], iou_threshold: float) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.keys()) | set(pred.keys()))
    for frame in frames:
        gt_objs = gt.get(frame, [])
        pred_objs = pred.get(frame, [])
        gt_ids = [obj["id"] for obj in gt_objs]
        pred_ids = [obj["id"] for obj in pred_objs]
        distances = []
        for g in gt_objs:
            row = []
            for p in pred_objs:
                if g["class_id"] != p["class_id"]:
                    row.append(float("nan"))
                    continue
                iou = _iou_xywh(g["bbox"], p["bbox"])
                row.append(1 - iou if iou >= iou_threshold else float("nan"))
            distances.append(row)
        acc.update(gt_ids, pred_ids, distances)
    return acc


def run_evaluate(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.txt"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction txt files found in {pred_dir}")

    accumulators = []
    names = []
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            print(f"[WARN] missing GT for {pred_file.name}, skipping")
            continue
        gt = _load_gt_file(gt_file)
        pred = _load_pred_file(pred_file, frame_offset=args.frame_offset)
        accumulators.append(evaluate_sequence(gt, pred, args.iou_threshold))
        names.append(pred_file.stem)

    if not accumulators:
        raise RuntimeError("No comparable prediction/ground-truth pairs found")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=names,
        metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations", "precision", "recall"],
        generate_overall=True,
    )
    summary.to_csv(output)
    print(summary)
    print(f"Saved metrics to {output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "detect":
        run_detect(args)
    elif args.command == "track":
        run_track(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
