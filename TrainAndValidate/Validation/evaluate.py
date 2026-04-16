import os
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import time

import yaml

with open("eval.yaml", "r") as f:
    config = yaml.safe_load(f)

gt_dir = config["dataset"]["gt_dir"]
trk_dir = config["dataset"]["trk_dir"]
iou_thr = config["evaluation"]["iou_threshold"]

# IO
def load_mot_file(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split(",")
            frame = int(vals[0])
            tid   = int(vals[1])
            x, y, w, h = map(float, vals[2:6])
            data.append([frame, tid, x, y, w, h])
    return np.array(data, dtype=np.float64)


# IoU  (top-left xywh)
def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter + 1e-9

    return inter / union


def iou_matrix(gt_boxes, trk_boxes):
    M = np.zeros((len(gt_boxes), len(trk_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_boxes):
        for j, t in enumerate(trk_boxes):
            M[i, j] = iou(g, t)
    return M


# MOTA
def evaluate_mota(gt, trk, iou_thr=0.5):
    frames = sorted(set(gt[:, 0]).union(set(trk[:, 0])))

    fp = fn = id_switches = matches = 0
    gt_id_to_trk_id = {}   # last known tracker id for each GT id

    for f in frames:
        gt_f  = gt[gt[:, 0] == f]
        trk_f = trk[trk[:, 0] == f]

        if len(gt_f) == 0:
            fp += len(trk_f)
            continue
        if len(trk_f) == 0:
            fn += len(gt_f)
            continue

        cost = iou_matrix(gt_f[:, 2:6], trk_f[:, 2:6])
        row_ind, col_ind = linear_sum_assignment(-cost)

        matched_gt  = set()
        matched_trk = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < iou_thr:
                continue
            gt_id  = int(gt_f[r, 1])
            trk_id = int(trk_f[c, 1])

            matches += 1
            matched_gt.add(r)
            matched_trk.add(c)

            if gt_id in gt_id_to_trk_id and gt_id_to_trk_id[gt_id] != trk_id:
                id_switches += 1
            gt_id_to_trk_id[gt_id] = trk_id

        fp += len(trk_f) - len(matched_trk)
        fn += len(gt_f)  - len(matched_gt)

    total_gt = len(gt)
    mota = 1.0 - (fn + fp + id_switches) / (total_gt + 1e-9)

    return {
        "MOTA":    round(float(mota), 4),
        "IDSW":    id_switches,
        "FP":      fp,
        "FN":      fn,
        "MATCHES": matches,
        "GT":      total_gt,
    }


# IDF1
#   IDTP = sum of assigned counts
#   IDFP = total pred detections - IDTP
#   IDFN = total GT   detections - IDTP
#   IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
def compute_idf1(gt, trk, iou_thr=0.5):
    gt_frames  = defaultdict(list)
    trk_frames = defaultdict(list)

    for g in gt:
        gt_frames[int(g[0])].append(g)
    for t in trk:
        trk_frames[int(t[0])].append(t)

    # co-occurrence: how many times GT id i was matched to pred id j
    cooccur = defaultdict(lambda: defaultdict(int))

    for f in set(gt_frames.keys()).union(trk_frames.keys()):
        gts = np.array(gt_frames.get(f, []))
        trs = np.array(trk_frames.get(f, []))

        if len(gts) == 0 or len(trs) == 0:
            continue

        cost = iou_matrix(gts[:, 2:6], trs[:, 2:6])
        row_ind, col_ind = linear_sum_assignment(-cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= iou_thr:
                gt_id  = int(gts[r, 1])
                trk_id = int(trs[c, 1])
                cooccur[gt_id][trk_id] += 1

    # Build cost matrix for global ID assignment
    gt_ids  = sorted(cooccur.keys())
    all_trk_ids = sorted(set(tid for v in cooccur.values() for tid in v.keys()))

    if not gt_ids or not all_trk_ids:
        return 0.0, 0.0, 0.0

    gt_idx  = {g: i for i, g in enumerate(gt_ids)}
    trk_idx = {t: i for i, t in enumerate(all_trk_ids)}

    C = np.zeros((len(gt_ids), len(all_trk_ids)), dtype=np.float64)
    for g, trk_map in cooccur.items():
        for t, cnt in trk_map.items():
            C[gt_idx[g], trk_idx[t]] = cnt

    row_ind, col_ind = linear_sum_assignment(-C)

    idtp = int(C[row_ind, col_ind].sum())
    idfp = len(trk) - idtp
    idfn = len(gt)  - idtp

    precision = idtp / (idtp + idfp + 1e-9)
    recall    = idtp / (idtp + idfn + 1e-9)
    idf1      = 2 * precision * recall / (precision + recall + 1e-9)

    return round(float(idf1), 4), round(float(precision), 4), round(float(recall), 4)



# HOTA
# For each alpha in [0.05, 0.10, ..., 0.95]:
#   Per frame, Hungarian-match GT↔pred at IoU ≥ alpha.
#   DetA(alpha) = TP / (TP + FP + FN)          (detection jaccard)
#   AssA(alpha) = mean over matched pairs of
#                 |frames where same GT-pred pair matched| /
#                 |frames where either appears matched|    (association jaccard)
#   HOTA(alpha) = sqrt(DetA * AssA)
# HOTA = mean over alphas
def compute_hota(gt, trk, alphas=None):
    if alphas is None:
        alphas = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05..0.95

    gt_frames  = defaultdict(list)
    trk_frames = defaultdict(list)
    for g in gt:
        gt_frames[int(g[0])].append(g)
    for t in trk:
        trk_frames[int(t[0])].append(t)

    frames = sorted(set(gt_frames.keys()).union(trk_frames.keys()))

    hota_scores = []

    for alpha in alphas:
        tp = fp = fn = 0

        # Track matched pair counts for AssA
        pair_tp = defaultdict(int)
        gt_matched  = defaultdict(int)
        trk_matched = defaultdict(int)

        for f in frames:
            gts = np.array(gt_frames.get(f, []))
            trs = np.array(trk_frames.get(f, []))

            if len(gts) == 0 and len(trs) == 0:
                continue
            if len(gts) == 0:
                fp += len(trs)
                continue
            if len(trs) == 0:
                fn += len(gts)
                continue

            cost = iou_matrix(gts[:, 2:6], trs[:, 2:6])
            row_ind, col_ind = linear_sum_assignment(-cost)

            matched_g = set()
            matched_t = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= alpha:
                    tp += 1
                    matched_g.add(r)
                    matched_t.add(c)
                    g_id = int(gts[r, 1])
                    t_id = int(trs[c, 1])
                    pair_tp[(g_id, t_id)] += 1
                    gt_matched[g_id]  += 1
                    trk_matched[t_id] += 1

            fp += len(trs) - len(matched_t)
            fn += len(gts) - len(matched_g)

        det_a = tp / (tp + fp + fn + 1e-9)

        # AssA: for each matched pair compute association jaccard, then average
        ass_scores = []
        for (g_id, t_id), p_tp in pair_tp.items():
            p_union = gt_matched[g_id] + trk_matched[t_id] - p_tp
            ass_scores.append(p_tp / (p_union + 1e-9))

        ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0

        hota_scores.append(np.sqrt(det_a * ass_a))

    return round(float(np.mean(hota_scores)), 4)



# RUN ALL
def evaluate_dataset(gt_dir, trk_dir):
    seqs = [f.replace(".txt", "") for f in os.listdir(gt_dir) if f.endswith(".txt")]
    results = []

    for seq in sorted(seqs):
        gt_path  = os.path.join(gt_dir,  seq + ".txt")
        trk_path = os.path.join(trk_dir, seq + ".txt")

        if not os.path.exists(trk_path):
            print("WARNING: no tracking output for", seq)
            continue

        gt  = load_mot_file(gt_path)
        trk = load_mot_file(trk_path)

        mot              = evaluate_mota(gt, trk, iou_thr)
        idf1, prec, rec  = compute_idf1(gt, trk, iou_thr)
        hota             = compute_hota(gt, trk)

        results.append({
            "seq":       seq,
            "MOTA":      mot["MOTA"],
            "IDF1":      idf1,
            "HOTA":      hota,
            "IDSW":      mot["IDSW"],
            "FP":        mot["FP"],
            "FN":        mot["FN"],
            "Precision": prec,
            "Recall":    rec,
            "GT":        mot["GT"],
        })

    return results


if __name__ == "__main__":
    start = time.perf_counter()
    results = evaluate_dataset(gt_dir, trk_dir)
    end = time.perf_counter()
    runtime = end - start

    print("\n=== FINAL RESULTS ===\n")
    header = "{:<20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10} {:>8} {:>8}".format(
        "seq", "MOTA", "IDF1", "HOTA", "IDSW", "FP", "FN", "Precision", "Recall", "Runtime")
    print(header)
    print("-" * len(header))
    for r in results:
        print("{:<20} {:>6.3f} {:>6.3f} {:>6.3f} {:>6} {:>6} {:>6} {:>10.3f} {:>8.3f} {:>8.3f}".format(
            r["seq"], r["MOTA"], r["IDF1"], r["HOTA"],
            r["IDSW"], r["FP"], r["FN"], r["Precision"], r["Recall"], runtime))
