#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "trackers" / "OC_SORT"))

from trackers.ocsort_tracker.ocsort import OCSort

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--class-id", type=int, default=None)
    p.add_argument("--min-box-area", type=float, default=10.0)
    p.add_argument("--track-thresh", type=float, default=0.5)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_videos(source: Path):
    if source.is_file():
        return [source]
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted([p for p in source.iterdir() if p.suffix.lower() in exts])


def to_mot_row(frame_idx, track_id, x1, y1, x2, y2, conf, class_id=0, vis=1):
    w = x2 - x1
    h = y2 - y1
    return [frame_idx, int(track_id), x1, y1, w, h, conf, class_id, vis]


def main():
    args = parse_args()
    model = YOLO(args.model)
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    mot_dir = output_dir / "mot"
    render_dir = output_dir / "rendered"
    ensure_dir(mot_dir)
    ensure_dir(render_dir)

    videos = list_videos(source)

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = OCSort(
            det_thresh=args.track_thresh,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )

        out_video = render_dir / f"{video_path.stem}.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        mot_rows = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls in zip(xyxy, confs, clss):
                    if args.class_id is not None and cls != args.class_id:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area < args.min_box_area:
                        continue
                    detections.append([x1, y1, x2, y2, float(score)])

            det_array = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)

            tracks = tracker.update(det_array, frame.shape, frame.shape)

            for trk in tracks:
                x1, y1, x2, y2, track_id = trk[:5]
                track_id = int(track_id)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"id:{track_id}",
                    (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                mot_rows.append(to_mot_row(
                    frame_idx, track_id, float(x1), float(y1), float(x2), float(y2), 1.0, 0, 1
                ))

            writer.write(frame)

        cap.release()
        writer.release()

        mot_path = mot_dir / f"{video_path.stem}.txt"
        with mot_path.open("w", newline="") as f:
            csv.writer(f).writerows(mot_rows)

        print(f"[OK] {video_path.name} -> {mot_path}")


if __name__ == "__main__":
    main()