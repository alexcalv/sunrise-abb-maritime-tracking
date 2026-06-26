#!/usr/bin/env python3
"""Run the automated MaritimeSim collection, detection, and MOT evaluation loop."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from tools.generate_videos import camera_names, collect_outputs, generate_videos, validate_outputs
except ModuleNotFoundError:
    from generate_videos import camera_names, collect_outputs, generate_videos, validate_outputs


REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_app_importable() -> None:
    app_root = REPO_ROOT / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))


def _safe_name(raw: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", raw, flags=re.UNICODE).strip("._-")
    return cleaned or "camera"


def _default_model_path() -> str:
    docker_model = Path("/workspace/models/yolo26l.pt")
    if docker_model.is_file():
        return str(docker_model)
    return "yolo26l.pt"


def _default_tracker_yaml() -> str:
    docker_tracker = Path("/workspace/config/trackers/botsort_maritime.yaml")
    if docker_tracker.is_file():
        return str(docker_tracker)
    return str(REPO_ROOT / "config" / "trackers" / "botsort_maritime.yaml")


def _resolve_mot_path(mot_dir: Path, video_path: Path, run_summary: str | None) -> Path:
    _ensure_app_importable()
    from fusion.multi_camera_runner import resolve_mot_export_path

    return resolve_mot_export_path(mot_dir, video_path, run_summary)


def _track_one_camera(
    *,
    camera: str,
    video_path: Path,
    model: str,
    tracker: str,
    device: str,
    project: Path,
    run_name: str,
    conf: float | None,
    imgsz: int | None,
) -> dict[str, Any]:
    _ensure_app_importable()
    from tracking.tracker_runner import run_tracking

    safe_camera = _safe_name(camera)
    result = run_tracking(
        model_path=model,
        source=str(video_path),
        tracker_yaml=tracker,
        device=device,
        project=str(project),
        name=f"{run_name}/raw/{safe_camera}",
        conf=conf,
        imgsz=imgsz,
    )
    mot_path = _resolve_mot_path(Path(result["mot_dir"]), video_path, result.get("run_summary"))
    return {
        "camera": camera,
        "video": str(video_path),
        "mot": str(mot_path),
        "run_summary": result.get("run_summary"),
        "raw_result": result,
    }


def _evaluate(pred_dir: Path, gt_dir: Path, eval_dir: Path, iou_threshold: float, frame_offset: int) -> dict[str, Any]:
    _ensure_app_importable()
    from evaluation.metrics import evaluate_mot_dir

    return evaluate_mot_dir(
        pred_dir=str(pred_dir),
        gt_dir=str(gt_dir),
        output_csv=str(eval_dir / "metrics.csv"),
        output_json=str(eval_dir / "metrics.json"),
        iou_threshold=iou_threshold,
        frame_offset=frame_offset,
    )


def run_sim_eval(
    *,
    config: str | Path,
    sim_out_dir: str | Path | None = None,
    eval_dir: str | Path | None = None,
    unity: str | None = None,
    unity_project: str | Path | None = None,
    model: str | None = None,
    tracker: str | None = None,
    device: str = "cpu",
    tracking_project: str | Path | None = None,
    run_name: str | None = None,
    timeout: float | None = 1800,
    batchmode: bool = False,
    verbose: bool = False,
    skip_generation: bool = False,
    dry_run: bool = False,
    iou_threshold: float = 0.5,
    frame_offset: int = 0,
    conf: float | None = None,
    imgsz: int | None = None,
) -> dict[str, Any]:
    """
    Execute the automated evaluation loop.

    Full mode renders videos with Unity, runs detection/tracking per camera, then
    evaluates predicted MOT against Unity's per-camera ground truth. ``dry_run``
    validates the config and returns the planned steps without requiring Unity,
    model weights, or generated videos.
    """
    config_path = Path(config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    names = camera_names(config_path)
    stem = config_path.stem
    sim_output = Path(sim_out_dir).resolve() if sim_out_dir else REPO_ROOT / "outputs" / "sim" / stem
    eval_output = Path(eval_dir).resolve() if eval_dir else REPO_ROOT / "outputs" / "sim_eval" / stem
    track_project = Path(tracking_project).resolve() if tracking_project else eval_output / "track"
    run = run_name or stem
    model_path = model or _default_model_path()
    tracker_yaml = tracker or _default_tracker_yaml()

    summary: dict[str, Any] = {
        "config": str(config_path),
        "sim_out_dir": str(sim_output),
        "eval_dir": str(eval_output),
        "camera_names": names,
        "model": model_path,
        "tracker": tracker_yaml,
        "device": device,
        "run_name": run,
        "dry_run": dry_run,
        "skip_generation": skip_generation,
    }

    if dry_run:
        summary["planned_steps"] = [
            "generate videos with Unity" if not skip_generation else "reuse existing simulation outputs",
            "run detector/tracker once per camera video",
            "copy predicted MOT files into one evaluation directory",
            "evaluate predictions against Unity MOT ground truth",
        ]
        return summary

    eval_output.mkdir(parents=True, exist_ok=True)

    if skip_generation:
        outputs = collect_outputs(config_path, sim_output)
        missing = validate_outputs(outputs)
        if missing:
            raise RuntimeError("Existing simulation outputs are not usable:\n  " + "\n  ".join(missing))
    else:
        outputs = generate_videos(
            config=config_path,
            out_dir=sim_output,
            unity=unity,
            project=unity_project,
            timeout=timeout,
            batchmode=batchmode,
            verbose=verbose,
        )

    pred_dir = eval_output / "pred_mot"
    pred_dir.mkdir(parents=True, exist_ok=True)

    tracking_results: dict[str, Any] = {}
    for camera, files in outputs.items():
        video_path = Path(files["video"])
        gt_path = Path(files["mot"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Generated video not found for {camera}: {video_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Generated MOT ground truth not found for {camera}: {gt_path}")

        track_result = _track_one_camera(
            camera=camera,
            video_path=video_path,
            model=model_path,
            tracker=tracker_yaml,
            device=device,
            project=track_project,
            run_name=run,
            conf=conf,
            imgsz=imgsz,
        )
        tracking_results[camera] = track_result
        shutil.copyfile(track_result["mot"], pred_dir / f"{camera}.txt")

    metrics = _evaluate(
        pred_dir=pred_dir,
        gt_dir=sim_output,
        eval_dir=eval_output,
        iou_threshold=iou_threshold,
        frame_offset=frame_offset,
    )

    summary.update(
        {
            "simulation_outputs": outputs,
            "pred_dir": str(pred_dir),
            "tracking_results": tracking_results,
            "metrics": metrics,
        }
    )
    summary_path = eval_output / "sim_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MaritimeSim video generation, tracking, and evaluation.")
    parser.add_argument("config", help="Simulation JSON config.")
    parser.add_argument("--sim-out", help="Simulation output directory (default: outputs/sim/<config>).")
    parser.add_argument("--eval-dir", help="Evaluation output directory (default: outputs/sim_eval/<config>).")
    parser.add_argument("--unity", help="Path to Unity executable or set UNITY_PATH.")
    parser.add_argument("--unity-project", help="Unity project root (default: MaritimeSim2).")
    parser.add_argument("--model", help="YOLO model path/name (default: /workspace/models/yolo26l.pt if present).")
    parser.add_argument("--tracker", help="Tracker YAML (default: config/trackers/botsort_maritime.yaml).")
    parser.add_argument("--device", default="cpu", help="Tracking device; use cpu in Docker on Mac.")
    parser.add_argument("--tracking-project", help="Root for raw tracking runs (default: <eval-dir>/track).")
    parser.add_argument("--run-name", help="Tracking run name (default: config stem).")
    parser.add_argument("--timeout", type=float, default=1800, help="Unity timeout in seconds.")
    parser.add_argument("--batchmode", action="store_true", help="Pass -batchmode to Unity (not recommended for Recorder).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Unity simulation logs.")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse existing files in --sim-out.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned steps only.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--conf", type=float, help="Optional detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, help="Optional detector image size.")
    args = parser.parse_args(argv)

    try:
        result = run_sim_eval(
            config=args.config,
            sim_out_dir=args.sim_out,
            eval_dir=args.eval_dir,
            unity=args.unity,
            unity_project=args.unity_project,
            model=args.model,
            tracker=args.tracker,
            device=args.device,
            tracking_project=args.tracking_project,
            run_name=args.run_name,
            timeout=args.timeout,
            batchmode=args.batchmode,
            verbose=args.verbose,
            skip_generation=args.skip_generation,
            dry_run=args.dry_run,
            iou_threshold=args.iou_threshold,
            frame_offset=args.frame_offset,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print a concise failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
