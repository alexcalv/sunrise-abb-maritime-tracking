#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple tracker IDs into one MOT ground-truth target."
    )
    parser.add_argument("--input", required=True, help="Input MOT txt file")
    parser.add_argument("--output", required=True, help="Output GT txt file")
    parser.add_argument(
        "--source-ids",
        required=True,
        nargs="+",
        type=int,
        help="Tracker IDs to merge, e.g. 8 195"
    )
    parser.add_argument("--gt-id", type=int, default=1, help="GT ID to assign")
    parser.add_argument("--class-id", type=int, default=2, help="Class ID to write")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument(
        "--list-ids",
        action="store_true",
        help="List IDs found in the file and exit"
    )
    return parser.parse_args()


def safe_int(v):
    return int(float(v.strip()))


def safe_float(v):
    return float(v.strip())


def load_mot(path: Path):
    rows = []
    bad_lines = 0

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) != 9:
                bad_lines += 1
                print(f"[WARN] line {line_num}: expected 9 columns, got {len(row)} -> {row}")
                continue

            try:
                frame = safe_int(row[0])
                track_id = safe_int(row[1])
                x = safe_float(row[2])
                y = safe_float(row[3])
                w = safe_float(row[4])
                h = safe_float(row[5])
                conf = safe_float(row[6])
                class_id = safe_int(row[7])
                visibility = safe_float(row[8])
            except Exception as e:
                bad_lines += 1
                print(f"[WARN] line {line_num}: parse error: {e}")
                continue

            rows.append({
                "frame": frame,
                "track_id": track_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
                "class_id": class_id,
                "visibility": visibility,
            })

    print(f"[INFO] Loaded {len(rows)} valid rows")
    if bad_lines:
        print(f"[INFO] Skipped {bad_lines} invalid rows")
    return rows


def list_ids(rows):
    counts = Counter(r["track_id"] for r in rows)
    print("\nIDs found in file:")
    for track_id, n in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  ID {track_id}: {n} rows")


def filter_rows(rows, source_ids, frame_start=None, frame_end=None):
    selected = []
    for r in rows:
        if r["track_id"] < 0:
            continue
        if r["track_id"] not in source_ids:
            continue
        if frame_start is not None and r["frame"] < frame_start:
            continue
        if frame_end is not None and r["frame"] > frame_end:
            continue
        selected.append(r)
    return selected


def convert_to_gt(rows, gt_id, class_id):
    out = []
    for r in rows:
        out.append([
            r["frame"],
            gt_id,
            f"{r['x']:.4f}",
            f"{r['y']:.4f}",
            f"{r['w']:.4f}",
            f"{r['h']:.4f}",
            1,
            class_id,
            1
        ])
    return out


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    rows = load_mot(input_path)

    if args.list_ids:
        list_ids(rows)
        return

    selected = filter_rows(
        rows,
        source_ids=set(args.source_ids),
        frame_start=args.frame_start,
        frame_end=args.frame_end
    )

    if not selected:
        print("[WARN] No matching rows found.")
        print("[INFO] Run with --list-ids first to inspect available IDs.")
        return

    if args.sort:
        selected.sort(key=lambda r: (r["frame"], r["track_id"]))

    gt_rows = convert_to_gt(selected, args.gt_id, args.class_id)
    write_rows(output_path, gt_rows)

    frames = [r["frame"] for r in selected]
    print("[OK] Wrote GT file")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")
    print(f"  source ids: {args.source_ids}")
    print(f"  gt id: {args.gt_id}")
    print(f"  rows: {len(gt_rows)}")
    print(f"  frame range: {min(frames)} -> {max(frames)}")


if __name__ == "__main__":
    main()