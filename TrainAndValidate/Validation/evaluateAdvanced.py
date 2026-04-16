import os
import numpy as np
import time
import yaml
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# CONFIG
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
            if not line.strip() or line.startswith("#"):
                continue
            vals = line.strip().split(",")

            frame = int(vals[0])
            tid   = int(vals[1])
            x, y, w, h = map(float, vals[2:6])

            data.append([frame, tid, x, y, w, h])

    return np.asarray(data, dtype=np.float64)



# IoU
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

    fp = fn = idsw = matches = 0
    gt2trk = {}

    for f in frames:
        gt_f = gt[gt[:, 0] == f]
        trk_f = trk[trk[:, 0] == f]

        if len(gt_f) == 0:
            fp += len(trk_f)
            continue
        if len(trk_f) == 0:
            fn += len(gt_f)
            continue

        cost = iou_matrix(gt_f[:, 2:6], trk_f[:, 2:6])
        r, c = linear_sum_assignment(-cost)

        matched_g = set()

        for i, j in zip(r, c):
            if cost[i, j] < iou_thr:
                continue

            g_id = int(gt_f[i, 1])
            t_id = int(trk_f[j, 1])

            matches += 1
            matched_g.add(i)

            if g_id in gt2trk and gt2trk[g_id] != t_id:
                idsw += 1

            gt2trk[g_id] = t_id

        fp += len(trk_f) - len(r)
        fn += len(gt_f) - len(matched_g)

    mota = 1.0 - (fn + fp + idsw) / (len(gt) + 1e-9)

    return mota, fp, fn, idsw



# IDF1
def compute_idf1(gt, trk, iou_thr=0.5):
    gt_f = defaultdict(list)
    tr_f = defaultdict(list)

    for g in gt:
        gt_f[int(g[0])].append(g)
    for t in trk:
        tr_f[int(t[0])].append(t)

    cooccur = defaultdict(lambda: defaultdict(int))

    for f in set(gt_f.keys()).union(tr_f.keys()):
        gts = np.array(gt_f.get(f, []))
        trs = np.array(tr_f.get(f, []))

        if len(gts) == 0 or len(trs) == 0:
            continue

        cost = iou_matrix(gts[:, 2:6], trs[:, 2:6])
        r, c = linear_sum_assignment(-cost)

        for i, j in zip(r, c):
            if cost[i, j] >= iou_thr:
                cooccur[int(gts[i, 1])][int(trs[j, 1])] += 1

    tp = sum(max(v.values()) for v in cooccur.values())
    fp = len(trk) - tp
    fn = len(gt) - tp

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return f1, prec, rec



# HOTA
def compute_hota(gt, trk, alphas=None):
    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 19)

    gt_f = defaultdict(list)
    tr_f = defaultdict(list)

    for g in gt:
        gt_f[int(g[0])].append(g)
    for t in trk:
        tr_f[int(t[0])].append(t)

    frames = set(gt_f.keys()).union(tr_f.keys())

    scores = []

    for a in alphas:
        tp = fp = fn = 0

        for f in frames:
            gts = np.array(gt_f.get(f, []))
            trs = np.array(tr_f.get(f, []))

            if len(gts) == 0:
                fp += len(trs)
                continue
            if len(trs) == 0:
                fn += len(gts)
                continue

            cost = iou_matrix(gts[:, 2:6], trs[:, 2:6])
            r, c = linear_sum_assignment(-cost)

            matched_g = set()
            matched_t = set()

            for i, j in zip(r, c):
                if cost[i, j] >= a:
                    tp += 1
                    matched_g.add(i)
                    matched_t.add(j)

            fp += len(trs) - len(matched_t)
            fn += len(gts) - len(matched_g)

        det_a = tp / (tp + fp + fn + 1e-9)
        ass_a = tp / (tp + 1e-9)

        scores.append(np.sqrt(det_a * ass_a))

    return float(np.mean(scores))



# MAIN EVAL
def evaluate_dataset():
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".txt")])

    results = []

    total_gt = total_fp = total_fn = total_idsw = 0

    for f in gt_files:
        seq = f.replace(".txt", "")

        gt_path  = os.path.join(gt_dir, f)
        trk_path = os.path.join(trk_dir, f)

        if not os.path.exists(trk_path):
            print(f"[WARN] Missing: {seq}")
            continue

        gt  = load_mot_file(gt_path)
        trk = load_mot_file(trk_path)

        mota, fp, fn, idsw = evaluate_mota(gt, trk, iou_thr)
        idf1, prec, rec    = compute_idf1(gt, trk, iou_thr)
        hota               = compute_hota(gt, trk)

        total_gt += len(gt)
        total_fp += fp
        total_fn += fn
        total_idsw += idsw

        results.append([seq, mota, idf1, hota, idsw, fp, fn, prec, rec])

    # GLOBAL METRICS
    global_mota = 1 - (total_fp + total_fn + total_idsw) / (total_gt + 1e-9)

    return results, global_mota


if __name__ == "__main__":
    start = time.perf_counter()

    results, global_mota = evaluate_dataset()

    end = time.perf_counter()

    print("\n=== PER-SEQUENCE RESULTS ===\n")

    print(f"{'seq':<15} {'MOTA':>6} {'IDF1':>6} {'HOTA':>6} {'IDSW':>6} {'FP':>6} {'FN':>6} {'P':>6} {'R':>6}")

    for r in results:
        print(f"{r[0]:<15} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6} {r[5]:6} {r[6]:6} {r[7]:6.3f} {r[8]:6.3f}")

    print("\n=== GLOBAL SUMMARY ===")
    print("Global MOTA:", round(global_mota, 4))
    print("Runtime:", round(end - start, 3), "sec")