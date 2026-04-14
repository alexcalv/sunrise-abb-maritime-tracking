from __future__ import annotations

from pathlib import Path


def pair_prediction_and_gt(pred_dir: str, gt_dir: str) -> list[tuple[Path, Path]]:
    pred_files = {p.name: p for p in Path(pred_dir).glob("*.txt")}
    gt_files = {p.name: p for p in Path(gt_dir).glob("*.txt")}

    pairs: list[tuple[Path, Path]] = []
    for name, pred in pred_files.items():
        gt = gt_files.get(name)
        if gt is not None:
            pairs.append((pred, gt))
    return pairs
