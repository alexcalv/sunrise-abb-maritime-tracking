#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a single-target MOT ground truth against one or more tracker MOT files. "
            "The GT should contain one physical target with a fixed GT ID."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth MOT file")
    parser.add_argument(
        "--pred",
        required=True,
        nargs="+",
        help="One or more tracker MOT files to evaluate"
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold to count a tracker box as matched to GT (default: 0.5)"
    )
    parser.add_argument(
        "--gt-id",
        type=int,
        default=None,
        help="Optional GT ID filter. If omitted, the script expects only one GT ID in the file."
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class filter. If set, only rows with this class are used."
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path for summary table"
    )
    parser.add_argument(
        "--dump-frame-matches",
        default=None,
        help="Optional directory where per-frame match CSVs will be written"
    )
    return parser.parse_args()


def safe_int(v):
    return int(float(v.strip()))


def safe_float(v):
    return float(v.strip())


def load_mot(path: Path, class_id=None):
    rows_by_frame = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 9:
                continue
            try:
                frame = safe_int(row[0])
                track_id = safe_int(row[1])
                x = safe_float(row[2])
                y = safe_float(row[3])
                w = safe_float(row[4])
                h = safe_float(row[5])
                conf = safe_float(row[6])
                cls = safe_int(row[7])
                vis = safe_float(row[8])
            except Exception:
                continue

            if class_id is not None and cls != class_id:
                continue

            rows_by_frame[frame].append({
                "frame": frame,
                "track_id": track_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
                "class_id": cls,
                "visibility": vis,
            })
    return rows_by_frame


def xywh_to_xyxy(box):
    x1 = box["x"]
    y1 = box["y"]
    x2 = x1 + box["w"]
    y2 = y1 + box["h"]
    return x1, y1, x2, y2


def iou(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def choose_gt_id(gt_rows_by_frame, gt_id_arg=None):
    ids = sorted({row["track_id"] for rows in gt_rows_by_frame.values() for row in rows})
    if gt_id_arg is not None:
        if gt_id_arg not in ids:
            raise ValueError(f"Requested GT ID {gt_id_arg} not found in GT file. Found: {ids}")
        return gt_id_arg
    if len(ids) != 1:
        raise ValueError(
            f"GT file contains multiple IDs {ids}. Provide --gt-id to select one."
        )
    return ids[0]


def filter_single_gt(rows_by_frame, gt_id):
    out = {}
    for frame, rows in rows_by_frame.items():
        filtered = [r for r in rows if r["track_id"] == gt_id]
        if filtered:
            # for single-target GT there should be exactly one row per visible frame
            out[frame] = filtered[0]
    return out


def evaluate_one(gt_single, pred_rows_by_frame, iou_thresh, tracker_name):
    frames = sorted(gt_single.keys())
    matched_frames = 0
    missed_frames = 0
    matched_ids = []
    per_frame = []
    id_segments = []
    prev_matched_id = None
    prev_had_match = False
    id_switches = 0
    reacquisition_switches = 0

    for frame in frames:
        gt_box = gt_single[frame]
        preds = [r for r in pred_rows_by_frame.get(frame, []) if r["track_id"] >= 0]

        best = None
        best_iou = 0.0
        for p in preds:
            score = iou(gt_box, p)
            if score > best_iou:
                best_iou = score
                best = p

        if best is not None and best_iou >= iou_thresh:
            matched_frames += 1
            current_id = best["track_id"]
            matched_ids.append(current_id)

            if prev_had_match:
                if prev_matched_id is not None and current_id != prev_matched_id:
                    id_switches += 1
            else:
                if prev_matched_id is not None and current_id != prev_matched_id:
                    reacquisition_switches += 1

            if not id_segments or id_segments[-1] != current_id:
                id_segments.append(current_id)

            prev_matched_id = current_id
            prev_had_match = True

            per_frame.append({
                "frame": frame,
                "gt_id": gt_box["track_id"],
                "matched": 1,
                "matched_tracker_id": current_id,
                "iou": round(best_iou, 6),
                "gt_x": gt_box["x"],
                "gt_y": gt_box["y"],
                "gt_w": gt_box["w"],
                "gt_h": gt_box["h"],
                "pred_x": best["x"],
                "pred_y": best["y"],
                "pred_w": best["w"],
                "pred_h": best["h"],
            })
        else:
            missed_frames += 1
            prev_had_match = False
            per_frame.append({
                "frame": frame,
                "gt_id": gt_box["track_id"],
                "matched": 0,
                "matched_tracker_id": "",
                "iou": 0.0,
                "gt_x": gt_box["x"],
                "gt_y": gt_box["y"],
                "gt_w": gt_box["w"],
                "gt_h": gt_box["h"],
                "pred_x": "",
                "pred_y": "",
                "pred_w": "",
                "pred_h": "",
            })

    total_frames = len(frames)
    match_ratio = matched_frames / total_frames if total_frames else 0.0
    unique_ids = sorted(set(matched_ids))
    dominant_id = Counter(matched_ids).most_common(1)[0][0] if matched_ids else None
    dominant_id_ratio = (
        Counter(matched_ids).most_common(1)[0][1] / matched_frames
        if matched_ids else 0.0
    )

    return {
        "tracker": tracker_name,
        "frames_gt": total_frames,
        "frames_matched": matched_frames,
        "frames_missed": missed_frames,
        "match_ratio": round(match_ratio, 6),
        "num_tracker_ids_used": len(unique_ids),
        "tracker_ids_used": " ".join(map(str, unique_ids)),
        "id_segments": " ".join(map(str, id_segments)),
        "num_segments": len(id_segments),
        "id_switches_contiguous": id_switches,
        "id_switches_after_gap": reacquisition_switches,
        "dominant_tracker_id": dominant_id if dominant_id is not None else "",
        "dominant_id_ratio": round(dominant_id_ratio, 6),
        "per_frame": per_frame,
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    gt_path = Path(args.gt)
    pred_paths = [Path(p) for p in args.pred]

    gt_rows = load_mot(gt_path, class_id=args.class_id)
    gt_id = choose_gt_id(gt_rows, args.gt_id)
    gt_single = filter_single_gt(gt_rows, gt_id)

    if not gt_single:
        raise ValueError("No GT rows found after filtering.")

    summaries = []
    per_frame_outputs = []

    for pred_path in pred_paths:
        pred_rows = load_mot(pred_path, class_id=args.class_id)
        tracker_name = pred_path.stem
        result = evaluate_one(
            gt_single=gt_single,
            pred_rows_by_frame=pred_rows,
            iou_thresh=args.iou_thresh,
            tracker_name=tracker_name
        )
        summaries.append({k: v for k, v in result.items() if k != "per_frame"})
        per_frame_outputs.append((tracker_name, result["per_frame"]))

    fieldnames = [
        "tracker",
        "frames_gt",
        "frames_matched",
        "frames_missed",
        "match_ratio",
        "num_tracker_ids_used",
        "tracker_ids_used",
        "id_segments",
        "num_segments",
        "id_switches_contiguous",
        "id_switches_after_gap",
        "dominant_tracker_id",
        "dominant_id_ratio",
    ]

    print()
    print("Single-target evaluation summary")
    print("=" * 80)
    for row in summaries:
        print(f"Tracker:                 {row['tracker']}")
        print(f"  frames_gt:             {row['frames_gt']}")
        print(f"  frames_matched:        {row['frames_matched']}")
        print(f"  frames_missed:         {row['frames_missed']}")
        print(f"  match_ratio:           {row['match_ratio']}")
        print(f"  num_tracker_ids_used:  {row['num_tracker_ids_used']}")
        print(f"  tracker_ids_used:      {row['tracker_ids_used']}")
        print(f"  id_segments:           {row['id_segments']}")
        print(f"  num_segments:          {row['num_segments']}")
        print(f"  id_switches_contig.:   {row['id_switches_contiguous']}")
        print(f"  id_switches_after_gap: {row['id_switches_after_gap']}")
        print(f"  dominant_tracker_id:   {row['dominant_tracker_id']}")
        print(f"  dominant_id_ratio:     {row['dominant_id_ratio']}")
        print("-" * 80)

    if args.output_csv:
        write_csv(Path(args.output_csv), summaries, fieldnames)
        print(f"[OK] Summary CSV written to {args.output_csv}")

    if args.dump_frame_matches:
        out_dir = Path(args.dump_frame_matches)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_frame_fields = [
            "frame", "gt_id", "matched", "matched_tracker_id", "iou",
            "gt_x", "gt_y", "gt_w", "gt_h",
            "pred_x", "pred_y", "pred_w", "pred_h"
        ]
        for tracker_name, rows in per_frame_outputs:
            write_csv(out_dir / f"{tracker_name}_frame_matches.csv", rows, per_frame_fields)
        print(f"[OK] Per-frame match CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
