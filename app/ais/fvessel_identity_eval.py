from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MotRow:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float | None = None


@dataclass(frozen=True)
class GtFusionRow:
    index: int
    frame: int
    second: float
    mmsi: str
    x: float
    y: float
    width: float
    height: float
    confidence: float | None = None


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _safe_clip_name(value: str) -> str:
    stem = Path(value).stem or "clip"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return safe or "clip"


def _path_basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _looks_like_windows_path(value: str) -> bool:
    text = value.strip()
    return (len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}) or text.startswith("\\\\")


def _resolve_path(manifest_path: Path, value: Any, default_name: str | None = None) -> Path:
    raw = value if value not in (None, "") else default_name
    if raw is None:
        raise ValueError(f"Missing path value in {manifest_path}")
    text = str(raw).strip()
    path = Path(text)
    basename = _path_basename(text)

    # Alignment manifests can be generated on Windows and later consumed in Docker.
    # If a Windows absolute path is not valid here, prefer the clip folder copy.
    if _looks_like_windows_path(text):
        local_copy = manifest_path.parent / basename
        if local_copy.exists():
            return local_copy.resolve()
        return local_copy

    if path.is_absolute():
        if path.exists():
            return path
        local_copy = manifest_path.parent / basename
        if basename and local_copy.exists():
            return local_copy.resolve()
        return path

    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate.resolve()
    local_copy = manifest_path.parent / basename
    if basename and local_copy.exists():
        return local_copy.resolve()
    return candidate.resolve()


def _manifest_value(manifest: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    normalized = {_normalize_key(key): value for key, value in manifest.items()}
    for key in keys:
        normalized_key = _normalize_key(key)
        if normalized_key in normalized:
            return normalized[normalized_key]
    return default


def load_alignment_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Alignment manifest must be a JSON object: {manifest_path}")
    payload["_manifest_path"] = str(manifest_path)
    return payload


def load_aligned_ais(path: str | Path) -> list[dict[str, Any]]:
    ais_path = Path(path)
    with ais_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [{_normalize_key(key): value for key, value in row.items()} for row in csv.DictReader(handle)]
    return rows


def _row_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        key = _normalize_key(name)
        if key in row:
            return row[key]
    return None


def load_gt_fusion_clip(path: str | Path, fps: float) -> list[GtFusionRow]:
    gt_path = Path(path)
    rows: list[GtFusionRow] = []
    with gt_path.open("r", newline="", encoding="utf-8") as handle:
        for index, raw_row in enumerate(csv.DictReader(handle)):
            row = {_normalize_key(key): value for key, value in raw_row.items()}
            mmsi = _row_value(row, "mmsi", "MMSI")
            if mmsi is None or str(mmsi).strip() == "":
                continue
            frame = _int_or_none(_row_value(row, "frame", "frame_id"))
            second = _float_or_none(_row_value(row, "second", "time", "timestamp")) or 0.0
            if frame is None:
                frame = int(round(second * fps)) + 1
            x = _float_or_none(_row_value(row, "bb_left", "left", "x", "bbox_left"))
            y = _float_or_none(_row_value(row, "bb_top", "top", "y", "bbox_top"))
            width = _float_or_none(_row_value(row, "bb_width", "width", "w", "bbox_width"))
            height = _float_or_none(_row_value(row, "bb_height", "height", "h", "bbox_height"))
            if x is None or y is None or width is None or height is None:
                continue
            rows.append(
                GtFusionRow(
                    index=index,
                    frame=int(frame),
                    second=float(second),
                    mmsi=str(mmsi).strip(),
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    confidence=_float_or_none(_row_value(row, "conf", "confidence")),
                )
            )
    return rows


def load_mot(path: str | Path) -> list[MotRow]:
    mot_path = Path(path)
    rows: list[MotRow] = []
    with mot_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if len(raw) < 6:
                continue
            try:
                rows.append(
                    MotRow(
                        frame=int(float(raw[0])),
                        track_id=int(float(raw[1])),
                        x=float(raw[2]),
                        y=float(raw[3]),
                        width=float(raw[4]),
                        height=float(raw[5]),
                        confidence=float(raw[6]) if len(raw) > 6 and raw[6] != "" else None,
                    )
                )
            except ValueError:
                continue
    return rows


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    inter_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    inter_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area = lw * lh + rw * rh - inter
    return inter / area if area > 0.0 else 0.0


def match_mot_to_gt_fusion_by_iou(
    mot_rows: list[MotRow],
    gt_rows: list[GtFusionRow],
    iou_threshold: float = 0.3,
) -> tuple[list[dict[str, Any]], int, int]:
    gt_by_frame: dict[int, list[GtFusionRow]] = defaultdict(list)
    for gt in gt_rows:
        gt_by_frame[gt.frame].append(gt)

    matches: list[dict[str, Any]] = []
    matched_gt_indices: set[int] = set()
    for mot in mot_rows:
        best_gt: GtFusionRow | None = None
        best_iou = 0.0
        for frame in (mot.frame, mot.frame - 1, mot.frame + 1):
            for gt in gt_by_frame.get(frame, []):
                iou = _bbox_iou((mot.x, mot.y, mot.width, mot.height), (gt.x, gt.y, gt.width, gt.height))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
        if best_gt is None or best_iou < iou_threshold:
            continue
        matched_gt_indices.add(best_gt.index)
        matches.append(
            {
                "frame": mot.frame,
                "track_id": mot.track_id,
                "mmsi": best_gt.mmsi,
                "iou": round(best_iou, 6),
                "gt_index": best_gt.index,
                "gt_frame": best_gt.frame,
                "gt_second": best_gt.second,
            }
        )
    unmatched_mot_rows = len(mot_rows) - len(matches)
    unmatched_gt_rows = len(gt_rows) - len(matched_gt_indices)
    return matches, unmatched_mot_rows, unmatched_gt_rows


def assign_dominant_mmsi_per_track(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_track[int(match["track_id"])].append(match)

    assignments: list[dict[str, Any]] = []
    for track_id, track_matches in sorted(by_track.items()):
        counts = Counter(str(match["mmsi"]) for match in track_matches)
        dominant_mmsi, dominant_count = counts.most_common(1)[0]
        total = len(track_matches)
        mean_iou = sum(float(match["iou"]) for match in track_matches) / total if total else 0.0
        assignments.append(
            {
                "track_id": track_id,
                "dominant_mmsi": dominant_mmsi,
                "dominant_count": dominant_count,
                "matched_rows": total,
                "purity": round(dominant_count / total, 6) if total else 0.0,
                "mean_iou": round(mean_iou, 6),
                "mmsi_counts": dict(sorted(counts.items())),
            }
        )
    return assignments


def compute_track_purity(assignments: list[dict[str, Any]]) -> float | None:
    if not assignments:
        return None
    return round(sum(float(row["purity"]) for row in assignments) / len(assignments), 6)


def compute_mmsi_fragmentation(assignments: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    by_mmsi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        mmsi = str(assignment.get("dominant_mmsi") or "")
        if not mmsi:
            continue
        by_mmsi[mmsi].append(assignment)

    rows: list[dict[str, Any]] = []
    for mmsi, mmsi_assignments in sorted(by_mmsi.items()):
        track_ids = [int(row["track_id"]) for row in mmsi_assignments]
        rows.append(
            {
                "mode": mode,
                "mmsi": mmsi,
                "track_count": len(track_ids),
                "track_ids": " ".join(str(track_id) for track_id in sorted(track_ids)),
                "id_switch_like_count": max(0, len(track_ids) - 1),
                "mean_purity": round(
                    sum(float(row["purity"]) for row in mmsi_assignments) / len(mmsi_assignments),
                    6,
                ),
            }
        )
    return rows


def _assignment_by_track(assignments: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["track_id"]): row for row in assignments}


def _parse_remap(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str) and "->" in value:
        left, right = value.split("->", 1)
        try:
            return int(left.strip()), int(right.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        source = _int_or_none(value.get("source") or value.get("source_track_id") or value.get("from"))
        target = _int_or_none(value.get("target") or value.get("target_track_id") or value.get("to"))
        if source is not None and target is not None:
            return source, target
    return None


def evaluate_ais_support_for_reid_remaps(
    accepted_remaps: list[Any],
    raw_assignments: list[dict[str, Any]],
    canonical_assignments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_by_track = _assignment_by_track(raw_assignments)
    canonical_by_track = _assignment_by_track(canonical_assignments)
    rows: list[dict[str, Any]] = []
    counts = {
        "accepted_remaps_total": 0,
        "ais_supported_remaps": 0,
        "ais_penalized_remaps": 0,
        "ais_unavailable_remaps": 0,
    }
    for remap in accepted_remaps:
        parsed = _parse_remap(remap)
        if parsed is None:
            continue
        source, target = parsed
        counts["accepted_remaps_total"] += 1
        source_mmsi = (raw_by_track.get(source) or {}).get("dominant_mmsi")
        target_mmsi = (canonical_by_track.get(target) or {}).get("dominant_mmsi")
        if not source_mmsi or not target_mmsi:
            verdict = "unavailable"
            counts["ais_unavailable_remaps"] += 1
        elif str(source_mmsi) == str(target_mmsi):
            verdict = "supported"
            counts["ais_supported_remaps"] += 1
        else:
            verdict = "penalized"
            counts["ais_penalized_remaps"] += 1
        rows.append(
            {
                "remap": f"{source}->{target}",
                "source_raw_track_id": source,
                "target_canonical_track_id": target,
                "source_dominant_mmsi": source_mmsi,
                "target_dominant_mmsi": target_mmsi,
                "ais_verdict": verdict,
            }
        )
    return rows, counts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _single_mot_file(mot_dir: Path) -> Path | None:
    files = sorted(mot_dir.glob("*.txt"))
    return files[0] if len(files) == 1 else None


def _load_demo_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "demo_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _find_run_output(run_root: Path, safe_name: str) -> Path | None:
    candidates = [path for path in run_root.glob(f"{safe_name}*") if path.is_dir() and (path / "demo_summary.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _fps_from_manifest(manifest: dict[str, Any]) -> float:
    value = _manifest_value(manifest, ("fps", "clip_fps", "video_fps", "source_fps"), 30.0)
    parsed = _float_or_none(value)
    return parsed if parsed and parsed > 0 else 30.0


def _alignment_confidence(manifest: dict[str, Any]) -> float | None:
    return _float_or_none(
        _manifest_value(
            manifest,
            ("offset_confidence", "alignment_confidence", "confidence", "best_offset_confidence"),
        )
    )


def _clip_start_time(manifest: dict[str, Any]) -> str | None:
    value = _manifest_value(
        manifest,
        ("clip_effective_start_time", "effective_start_time", "clip_start_time", "video_start_time", "start_time"),
    )
    return str(value) if value not in (None, "") else None


def _collect_manifests(alignment_root: Path) -> list[Path]:
    if not alignment_root.exists():
        return []
    return sorted(alignment_root.rglob("fvessel_clip_alignment_manifest.json"))


def _manifest_paths(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    clip_video = _resolve_path(
        manifest_path,
        _manifest_value(manifest, ("clip_video", "video_file", "video_path", "clip_path")),
    )
    ais_file = _resolve_path(
        manifest_path,
        _manifest_value(manifest, ("ais_clip_aligned", "ais_file", "ais_clip_aligned_csv")),
        "ais_clip_aligned.csv",
    )
    gt_file = _resolve_path(
        manifest_path,
        _manifest_value(manifest, ("gt_fusion_clip_aligned", "gt_fusion_file", "gt_fusion_clip_aligned_csv")),
        "gt_fusion_clip_aligned.csv",
    )
    return {"clip_video": clip_video, "ais_file": ais_file, "gt_file": gt_file}


def _clip_row_counts(manifest: dict[str, Any], ais_file: Path, gt_file: Path) -> tuple[int, int]:
    ais_rows = _int_or_none(_manifest_value(manifest, ("ais_rows_in_clip", "ais_row_count"))) or 0
    gt_rows = _int_or_none(_manifest_value(manifest, ("gt_fusion_rows_in_clip", "gt_rows_in_clip", "gt_row_count"))) or 0
    if ais_rows <= 0 and ais_file.exists():
        with ais_file.open("r", newline="", encoding="utf-8") as handle:
            ais_rows = max(0, sum(1 for _ in csv.DictReader(handle)))
    if gt_rows <= 0 and gt_file.exists():
        with gt_file.open("r", newline="", encoding="utf-8") as handle:
            gt_rows = max(0, sum(1 for _ in csv.DictReader(handle)))
    return ais_rows, gt_rows


def _skip_reason(
    manifest: dict[str, Any],
    paths: dict[str, Path],
    min_alignment_confidence: float,
    include_low_confidence: bool,
) -> str | None:
    if not paths["clip_video"].exists():
        return "missing_clip_video"
    if not paths["ais_file"].exists():
        return "missing_ais_clip_aligned_csv"
    if not paths["gt_file"].exists():
        return "missing_gt_fusion_clip_aligned_csv"
    ais_rows, gt_rows = _clip_row_counts(manifest, paths["ais_file"], paths["gt_file"])
    if ais_rows <= 0:
        return "no_ais_rows_in_clip"
    if gt_rows <= 0:
        return "no_gt_fusion_rows_in_clip"
    confidence = _alignment_confidence(manifest)
    if confidence is None:
        return "missing_offset_confidence" if not include_low_confidence else None
    if confidence < min_alignment_confidence and not include_low_confidence:
        return f"low_offset_confidence:{confidence:.4f}"
    return None


def _run_demo_for_clip(
    *,
    clip_video: Path,
    ais_file: Path,
    clip_start_time: str | None,
    run_root: Path,
    safe_name: str,
    model: str,
    tracker: str,
    live_reid_config: str,
    confirmation_observations: int,
    device: str,
    eval_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    script_path = Path("/workspace/scripts/run_detection.py")
    if not script_path.exists():
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_detection.py"
    command = [
        sys.executable,
        str(script_path),
        "--demo-mode",
        "--demo-title",
        f"FVessel AIS Identity Evaluation - {safe_name}",
        "--demo-note",
        "AIS diagnostics are reporting-only; evaluation compares MOT identities to gt_fusion MMSI labels.",
        "--model",
        model,
        "--source",
        str(clip_video),
        "--tracker",
        tracker,
        "--device",
        device,
        "--project",
        str(run_root),
        "--name",
        safe_name,
        "--live-reid-config",
        live_reid_config,
        "--confirmation-observations",
        str(int(confirmation_observations)),
        "--visual-continuation",
        "--paired-occlusion-prediction",
        "--motion-corridor-overlay",
        "--ais-diagnostics",
        "--ais-file",
        str(ais_file),
    ]
    if clip_start_time:
        command.extend(["--ais-video-start-time", clip_start_time])

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "run_detection_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (eval_dir / "run_detection_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    run_dir = _find_run_output(run_root, safe_name)
    if run_dir is None:
        raise FileNotFoundError(f"Could not locate run output for {safe_name} under {run_root}")
    return run_dir, _load_demo_summary(run_dir)


def _evaluate_run(
    *,
    clip_name: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    gt_file: Path,
    run_dir: Path,
    demo_summary: dict[str, Any],
    output_dir: Path,
    fps: float,
    iou_threshold: float,
) -> dict[str, Any]:
    raw_mot = Path(demo_summary.get("raw_mot_path") or "") if demo_summary.get("raw_mot_path") else _single_mot_file(run_dir / "mot")
    canonical_mot = (
        Path(demo_summary.get("canonical_mot_path"))
        if demo_summary.get("canonical_mot_path")
        else _single_mot_file(run_dir / "live_reid_in_loop" / "mot")
    )
    if raw_mot is None or not raw_mot.exists():
        raise FileNotFoundError(f"Raw MOT missing for {clip_name}: {raw_mot}")
    gt_rows = load_gt_fusion_clip(gt_file, fps=fps)

    raw_rows = load_mot(raw_mot)
    raw_matches, raw_unmatched_mot, raw_unmatched_gt = match_mot_to_gt_fusion_by_iou(
        raw_rows,
        gt_rows,
        iou_threshold=iou_threshold,
    )
    raw_assignments = assign_dominant_mmsi_per_track(raw_matches)

    canonical_rows: list[MotRow] = []
    canonical_matches: list[dict[str, Any]] = []
    canonical_assignments: list[dict[str, Any]] = []
    canonical_unmatched_mot = 0
    canonical_unmatched_gt = len(gt_rows)
    if canonical_mot is not None and canonical_mot.exists():
        canonical_rows = load_mot(canonical_mot)
        canonical_matches, canonical_unmatched_mot, canonical_unmatched_gt = match_mot_to_gt_fusion_by_iou(
            canonical_rows,
            gt_rows,
            iou_threshold=iou_threshold,
        )
        canonical_assignments = assign_dominant_mmsi_per_track(canonical_matches)

    raw_fragmentation = compute_mmsi_fragmentation(raw_assignments, mode="raw")
    canonical_fragmentation = compute_mmsi_fragmentation(canonical_assignments, mode="canonical")
    remap_rows, remap_counts = evaluate_ais_support_for_reid_remaps(
        list(demo_summary.get("accepted_remaps") or []),
        raw_assignments,
        canonical_assignments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "track_mmsi_assignment_raw.csv", raw_assignments)
    _write_csv(output_dir / "track_mmsi_assignment_canonical.csv", canonical_assignments)
    _write_csv(output_dir / "mmsi_fragmentation.csv", raw_fragmentation + canonical_fragmentation)
    _write_csv(output_dir / "ais_remap_support.csv", remap_rows)

    raw_switch_like = sum(int(row["id_switch_like_count"]) for row in raw_fragmentation)
    canonical_switch_like = sum(int(row["id_switch_like_count"]) for row in canonical_fragmentation)
    raw_fragmentation_mean = (
        round(sum(int(row["track_count"]) for row in raw_fragmentation) / len(raw_fragmentation), 6)
        if raw_fragmentation
        else None
    )
    canonical_fragmentation_mean = (
        round(sum(int(row["track_count"]) for row in canonical_fragmentation) / len(canonical_fragmentation), 6)
        if canonical_fragmentation
        else None
    )
    summary = {
        "clip_name": clip_name,
        "manifest": str(manifest_path),
        "run_dir": str(run_dir),
        "raw_mot_path": str(raw_mot),
        "canonical_mot_path": str(canonical_mot) if canonical_mot else None,
        "gt_fusion_path": str(gt_file),
        "fps": fps,
        "iou_threshold": iou_threshold,
        "alignment_confidence": _alignment_confidence(manifest),
        "gt_mmsi_count": len({row.mmsi for row in gt_rows}),
        "raw_track_count": len({row.track_id for row in raw_rows}),
        "canonical_track_count": len({row.track_id for row in canonical_rows}) if canonical_rows else 0,
        "raw_track_mmsi_purity_mean": compute_track_purity(raw_assignments),
        "canonical_track_mmsi_purity_mean": compute_track_purity(canonical_assignments),
        "raw_mmsi_fragmentation_mean": raw_fragmentation_mean,
        "canonical_mmsi_fragmentation_mean": canonical_fragmentation_mean,
        "raw_id_switch_like_count_by_mmsi": raw_switch_like,
        "canonical_id_switch_like_count_by_mmsi": canonical_switch_like,
        "raw_unmatched_mot_rows": raw_unmatched_mot,
        "canonical_unmatched_mot_rows": canonical_unmatched_mot,
        "raw_unmatched_gt_rows": raw_unmatched_gt,
        "canonical_unmatched_gt_rows": canonical_unmatched_gt,
        "accepted_remaps": list(demo_summary.get("accepted_remaps") or []),
        **remap_counts,
        "ais_remap_support_rows": remap_rows,
        "note": "FVessel AIS identity evaluation is reporting-only and does not affect MOT/ReID decisions.",
    }
    (output_dir / "ais_identity_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_fvessel_ais_identity_eval(
    *,
    alignment_root: str | Path,
    run_root: str | Path,
    output_dir: str | Path,
    model: str,
    tracker: str,
    live_reid_config: str,
    confirmation_observations: int = 10,
    min_alignment_confidence: float = 0.1,
    max_clips: int = 3,
    include_low_confidence: bool = False,
    device: str = "cpu",
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    alignment_root = Path(alignment_root)
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    manifests = _collect_manifests(alignment_root)
    skipped: list[dict[str, Any]] = []
    selected: list[tuple[Path, dict[str, Any], dict[str, Path]]] = []
    for manifest_path in manifests:
        try:
            manifest = load_alignment_manifest(manifest_path)
            paths = _manifest_paths(manifest_path, manifest)
            reason = _skip_reason(manifest, paths, min_alignment_confidence, include_low_confidence)
        except Exception as exc:  # noqa: BLE001 - this command should report bad manifests, not crash a batch.
            skipped.append({"manifest": str(manifest_path), "reason": f"manifest_error:{exc}"})
            continue
        if reason:
            skipped.append({"manifest": str(manifest_path), "reason": reason})
            continue
        selected.append((manifest_path, manifest, paths))
        if len(selected) >= max(0, int(max_clips)):
            break

    if not manifests:
        skipped.append({"manifest": str(alignment_root), "reason": "no_fvessel_alignment_manifests_found"})

    evaluated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for manifest_path, manifest, paths in selected:
        clip_name = _safe_clip_name(str(_manifest_value(manifest, ("clip_id", "clip_name"), paths["clip_video"].stem)))
        eval_clip_dir = output_dir / clip_name
        fps = _fps_from_manifest(manifest)
        try:
            run_dir, demo_summary = _run_demo_for_clip(
                clip_video=paths["clip_video"],
                ais_file=paths["ais_file"],
                clip_start_time=_clip_start_time(manifest),
                run_root=run_root,
                safe_name=clip_name,
                model=model,
                tracker=tracker,
                live_reid_config=live_reid_config,
                confirmation_observations=confirmation_observations,
                device=device,
                eval_dir=eval_clip_dir,
            )
            evaluated.append(
                _evaluate_run(
                    clip_name=clip_name,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    gt_file=paths["gt_file"],
                    run_dir=run_dir,
                    demo_summary=demo_summary,
                    output_dir=eval_clip_dir,
                    fps=fps,
                    iou_threshold=iou_threshold,
                )
            )
        except subprocess.CalledProcessError as exc:
            eval_clip_dir.mkdir(parents=True, exist_ok=True)
            (eval_clip_dir / "run_detection_stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
            (eval_clip_dir / "run_detection_stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
            failed.append({"clip_name": clip_name, "manifest": str(manifest_path), "reason": f"run_failed:{exc.returncode}"})
        except Exception as exc:  # noqa: BLE001 - keep batch summaries complete.
            failed.append({"clip_name": clip_name, "manifest": str(manifest_path), "reason": str(exc)})

    improved = [
        row
        for row in evaluated
        if row.get("canonical_mmsi_fragmentation_mean") is not None
        and row.get("raw_mmsi_fragmentation_mean") is not None
        and float(row["canonical_mmsi_fragmentation_mean"]) < float(row["raw_mmsi_fragmentation_mean"])
    ]
    supported = [row for row in evaluated if int(row.get("ais_supported_remaps") or 0) > 0]
    penalized = [row for row in evaluated if int(row.get("ais_penalized_remaps") or 0) > 0]
    poor_alignment = [
        {"manifest": str(path), "reason": reason}
        for path, reason in (
            (item.get("manifest"), item.get("reason")) for item in skipped if "confidence" in str(item.get("reason"))
        )
    ]

    recommendation = "real_fvessel_alignment_pending"
    if evaluated:
        if penalized:
            recommendation = "keep_ais_reporting_only_penalized_remaps_need_review"
        elif supported or improved:
            recommendation = "keep_ais_reporting_only_and_consider_opt_in_ais_assisted_reid_experiment"
        else:
            recommendation = "keep_ais_reporting_only_collect_more_aligned_clips"

    summary = {
        "passed": not failed,
        "alignment_root": str(alignment_root),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "clips_evaluated": len(evaluated),
        "clips_skipped": skipped,
        "clips_failed": failed,
        "evaluated": evaluated,
        "best_ais_supported_clip": max(supported, key=lambda row: int(row.get("ais_supported_remaps") or 0), default=None),
        "best_canonical_improvement_clip": min(
            improved,
            key=lambda row: float(row.get("canonical_mmsi_fragmentation_mean") or math.inf),
            default=None,
        ),
        "clips_where_ais_would_penalize_current_reid": penalized,
        "clips_with_poor_alignment": poor_alignment,
        "recommendation": recommendation,
        "note": "AIS is evaluated as reporting-only identity evidence; it is not used for gates, remaps, canonical IDs, tracker decisions, or MOT output.",
    }
    (output_dir / "fvessel_ais_identity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
