from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_app_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    app_root = repo_root / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    return repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command maritime tracking demo runner. It preserves raw MOT and can add "
            "bounded-latency ReID, side-by-side rendering, and reporting-only context."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--project", default="/workspace/outputs/demo/prototype")
    parser.add_argument("--name", default="prototype_run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, help="Optional detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, help="Optional detector image size.")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help=(
            "Convenience preset for the bounded-latency live demo: enables --live-reid-in-loop, "
            "--render, and --side-by-side-demo. Does not enable AIS or COLREG automatically."
        ),
    )
    parser.add_argument("--demo-title", help="Optional title shown in demo outputs and demo_summary.json.")
    parser.add_argument("--demo-note", help="Optional short note shown in demo outputs and demo_summary.json.")
    parser.add_argument(
        "--video-codec",
        choices=["auto", "mp4v", "h264"],
        default="auto",
        help="Video codec for rendered demos. auto prefers Windows-friendly H.264 and falls back clearly.",
    )
    parser.add_argument("--live-reid", action="store_true", help="Run post-stage bounded-latency ReID.")
    parser.add_argument("--live-reid-in-loop", action="store_true", help="Run tracker-loop bounded-latency ReID v1.")
    parser.add_argument("--live-reid-config", help="Optional ReID config path.")
    parser.add_argument("--confirmation-observations", type=int, default=10)
    parser.add_argument(
        "--colreg-diagnostics",
        action="store_true",
        help="Add reporting-only COLREG candidate diagnostics. Does not affect decisions by default.",
    )
    parser.add_argument(
        "--colreg-scoring-experiment",
        action="store_true",
        help="Experimental opt-in COLREG tie-break scoring. Disabled by default; not production scoring.",
    )
    parser.add_argument(
        "--ais-diagnostics",
        action="store_true",
        help="Add reporting-only AIS fields. Missing AIS remains neutral and does not affect decisions.",
    )
    parser.add_argument("--ais-file", help="AIS CSV/JSON file for reporting-only diagnostics.")
    parser.add_argument("--ais-video-start-time", help="Video start timestamp for AIS alignment.")
    parser.add_argument("--ais-affine-matrix", help="Optional 2x3 or 3x3 lon/lat-to-image affine matrix.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the selected raw/canonical MOT for single-video runs after tracking completes.",
    )
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="Write a presentable bounded-latency demo video. This is not a zero-latency production UI.",
    )
    parser.add_argument(
        "--side-by-side-demo",
        action="store_true",
        help="Render original video beside the annotated bounded-latency demo video.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the selected MOT output after tracking.")
    parser.add_argument("--gt-dir", help="Ground-truth MOT directory used with --evaluate.")
    parser.add_argument("--eval-output", help="Optional evaluation CSV output path.")
    parser.add_argument("--eval-summary-json", help="Optional evaluation JSON summary path.")
    return parser


def _select_output_mot_dir(result: dict) -> Path:
    if result.get("live_reid_in_loop"):
        return Path(result["live_reid_in_loop"]["mot_dir"])
    elif result.get("live_reid"):
        return Path(result["live_reid"]["mot_dir"])
    return Path(result["mot_dir"])


def _select_output_mot_kind(result: dict) -> str:
    if result.get("live_reid_in_loop"):
        return "live_reid_in_loop"
    if result.get("live_reid"):
        return "live_reid"
    return "raw"


def _select_render_mot(result: dict) -> tuple[Path, str]:
    mot_dir = _select_output_mot_dir(result)
    mot_files = sorted(mot_dir.glob("*.txt"))
    if len(mot_files) != 1:
        raise ValueError(f"--render expects exactly one MOT file in {mot_dir}, found {len(mot_files)}")
    return mot_files[0], _select_output_mot_kind(result)


def _single_mot_file(mot_dir: str | None) -> str | None:
    if not mot_dir:
        return None
    mot_files = sorted(Path(mot_dir).glob("*.txt"))
    if len(mot_files) != 1:
        return None
    return str(mot_files[0])


def _load_occlusion_summary(path: str | None) -> dict:
    if not path:
        return {}
    report_path = Path(path)
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _count_recovery_events(path: str | None) -> int:
    if not path:
        return 0
    report_path = Path(path)
    if not report_path.exists():
        return 0
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    events = set()
    for sequence_payload in (payload.get("sequences") or {}).values():
        for prediction in sequence_payload.get("predictions") or []:
            if not prediction.get("recovered_by_remap"):
                continue
            decision_frame = prediction.get("remap_decision_frame")
            if decision_frame is None:
                continue
            events.add(
                (
                    int(decision_frame),
                    prediction.get("remap_target_raw_track_id"),
                    prediction.get("canonical_id"),
                    prediction.get("raw_track_id", prediction.get("track_id")),
                )
            )
    return len(events)


def _write_demo_summary(args: argparse.Namespace, result: dict, summary: dict) -> Path:
    output_root = Path(result["output_root"])
    occlusion_summary = _load_occlusion_summary(summary.get("occlusion_predictions"))
    demo_summary = {
        "demo_mode_enabled": bool(args.demo_mode),
        "side_by_side_demo_enabled": bool(summary.get("side_by_side_demo")),
        "demo_title": args.demo_title,
        "demo_note": args.demo_note,
        "source": args.source,
        "source_video": args.source,
        "tracker": args.tracker,
        "tracker_config": args.tracker,
        "model": args.model,
        "mode": summary.get("rendered_mot_kind") or ("live_reid_in_loop" if summary.get("live_reid_mot_dir") else "raw"),
        "raw_mot_path": _single_mot_file(result.get("mot_dir")),
        "raw_mot_dir": result.get("mot_dir"),
        "canonical_mot_path": _single_mot_file(summary.get("live_reid_mot_dir")),
        "canonical_mot_dir": summary.get("live_reid_mot_dir"),
        "occlusion_predictions_path": summary.get("occlusion_predictions"),
        "occlusion_prediction_report": summary.get("occlusion_predictions"),
        "rendered_video_path": summary.get("render_output"),
        "side_by_side_video_path": summary.get("render_output") if summary.get("side_by_side_demo") else None,
        "final_demo_video_path": summary.get("render_output"),
        "rendered_mot_kind": summary.get("rendered_mot_kind"),
        "occlusion_overlay_enabled": bool(summary.get("occlusion_overlay_enabled")),
        "video_codec_requested": summary.get("video_codec_requested"),
        "video_codec_used": summary.get("video_codec_used"),
        "ffmpeg_transcode_used": bool(summary.get("ffmpeg_transcode_used")),
        "windows_friendly_video": bool(summary.get("windows_friendly_video")),
        "accepted_remaps": summary.get("accepted_remaps", []),
        "prediction_count": int(occlusion_summary.get("prediction_count") or 0),
        "recovered_prediction_count": int(occlusion_summary.get("recovered_prediction_count") or 0),
        "recovery_event_count": int(summary.get("recovery_event_count") or 0),
        "max_gap_frames": int(occlusion_summary.get("max_gap_frames") or 0),
        "mean_uncertainty_radius": occlusion_summary.get("mean_uncertainty_radius"),
        "colreg_diagnostics_enabled": bool(args.colreg_diagnostics),
        "ais_diagnostics_enabled": bool(args.ais_diagnostics),
        "reporting_context_note": "COLREG/AIS diagnostics are reporting-only and do not affect tracking or ReID decisions.",
        "bounded_latency_statement": "bounded-latency demo, not zero-latency production live ReID",
    }
    summary_path = output_root / "demo_summary.json"
    summary_path.write_text(json.dumps(demo_summary, indent=2), encoding="utf-8")
    return summary_path


def _print_human_summary(summary: dict) -> None:
    accepted_remaps = summary.get("accepted_remaps") or []
    print("\nDemo summary")
    print(f"  raw MOT: {summary.get('raw_mot_dir')}")
    print(f"  canonical MOT: {summary.get('live_reid_mot_dir') or 'none'}")
    print(f"  occlusion predictions: {summary.get('occlusion_predictions') or 'none'}")
    print(f"  rendered video: {summary.get('render_output') or 'none'}")
    print(f"  side-by-side demo: {summary.get('render_output') if summary.get('side_by_side_demo') else 'none'}")
    print(f"  video codec: {summary.get('video_codec_used') or 'none'}")
    print(f"  ffmpeg transcode: {'yes' if summary.get('ffmpeg_transcode_used') else 'no'}")
    print(f"  Windows-friendly video: {'yes' if summary.get('windows_friendly_video') else 'no'}")
    print(f"  accepted remaps: {accepted_remaps if accepted_remaps else 'none'}")
    print(f"  prediction count: {summary.get('prediction_count', 0)}")
    print(f"  recovered predictions: {summary.get('recovered_prediction_count', 0)}")
    print(f"  max gap frames: {summary.get('max_gap_frames', 0)}")
    print(f"  recovery events: {summary.get('recovery_event_count', 0)}")
    print(f"  COLREG diagnostics: {'enabled' if summary.get('colreg_diagnostics') else 'disabled'}")
    print(f"  AIS diagnostics: {'enabled' if summary.get('ais_diagnostics') else 'disabled'}")
    if summary.get("colreg_diagnostics") or summary.get("ais_diagnostics"):
        print("  note: AIS/COLREG are reporting-only and do not affect decisions.")


def main() -> None:
    _add_app_to_path()
    args = build_parser().parse_args()
    if args.demo_mode:
        # Demo mode is only a convenience preset. It does not change ReID scoring.
        if not args.live_reid:
            args.live_reid_in_loop = True
        args.render = True
        args.side_by_side_demo = True

    from evaluation.metrics import evaluate_mot_dir
    from output.render_video import render_video_from_mot
    from tracking.tracker_runner import run_tracking

    result = run_tracking(
        model_path=args.model,
        source=args.source,
        tracker_yaml=args.tracker,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        imgsz=args.imgsz,
        live_reid=args.live_reid,
        live_reid_in_loop=args.live_reid_in_loop,
        live_reid_config=args.live_reid_config,
        confirmation_observations=args.confirmation_observations,
        colreg_diagnostics=args.colreg_diagnostics,
        colreg_scoring_experiment=args.colreg_scoring_experiment,
        ais_diagnostics=args.ais_diagnostics,
        ais_file=args.ais_file,
        ais_video_start_time=args.ais_video_start_time,
        ais_affine_matrix=args.ais_affine_matrix,
    )

    if args.render or args.live_preview or args.side_by_side_demo:
        mot_file, mot_kind = _select_render_mot(result)
        output_root = Path(result["output_root"])
        if args.side_by_side_demo:
            render_output = output_root / f"{args.name}_side_by_side_demo.mp4"
        elif args.live_preview:
            render_output = output_root / f"{args.name}_live_demo.mp4"
        else:
            render_output = output_root / f"{args.name}_rendered.mp4"
        occlusion_predictions = None
        if result.get("live_reid_in_loop"):
            occlusion_predictions = result["live_reid_in_loop"].get("occlusion_predictions")
        result["render"] = render_video_from_mot(
            source_video_path=args.source,
            mot_file_path=str(mot_file),
            output_video_path=str(render_output),
            label_mode="id",
            occlusion_predictions_path=occlusion_predictions,
            side_by_side=args.side_by_side_demo,
            raw_mot_file_path=_single_mot_file(result.get("mot_dir")),
            rendered_mot_kind=mot_kind,
            status_context={
                "mode": mot_kind,
                "reid_enabled": bool(result.get("live_reid") or result.get("live_reid_in_loop")),
                "occlusion_predictions_enabled": bool(occlusion_predictions),
                "colreg_diagnostics": bool(args.colreg_diagnostics),
                "ais_diagnostics": bool(args.ais_diagnostics),
                "demo_title": args.demo_title,
                "demo_note": args.demo_note,
            },
            video_codec=args.video_codec,
        )
        result["render"]["rendered_mot_kind"] = mot_kind
        result["render"]["rendered_mot_path"] = str(mot_file)
        if args.live_preview:
            result["live_preview"] = {
                "mode": "bounded_latency_demo_video",
                "output": str(render_output),
                "note": "Rendered sidecar demo video; not zero-latency production live UI.",
            }

    if args.evaluate:
        if not args.gt_dir:
            raise ValueError("--evaluate requires --gt-dir")
        output_root = Path(result["output_root"])
        eval_output = Path(args.eval_output) if args.eval_output else output_root / "evaluation.csv"
        eval_summary = Path(args.eval_summary_json) if args.eval_summary_json else output_root / "evaluation_summary.json"
        result["evaluation"] = evaluate_mot_dir(
            pred_dir=str(_select_output_mot_dir(result)),
            gt_dir=args.gt_dir,
            output_csv=str(eval_output),
            output_json=str(eval_summary),
        )

    summary = {
        "output_root": result.get("output_root"),
        "raw_mot_dir": result.get("mot_dir"),
        "live_reid_mot_dir": None,
        "occlusion_predictions": None,
        "render_output": None,
        "rendered_mot_path": None,
        "rendered_mot_kind": None,
        "occlusion_overlay_enabled": False,
        "occlusion_predictions_rendered": 0,
        "video_codec_requested": args.video_codec,
        "video_codec_used": None,
        "ffmpeg_transcode_used": False,
        "windows_friendly_video": False,
        "side_by_side_demo": bool(args.side_by_side_demo),
        "evaluation": result.get("evaluation"),
        "accepted_remaps": [],
        "colreg_diagnostics": bool(args.colreg_diagnostics),
        "ais_diagnostics": bool(args.ais_diagnostics),
    }
    if result.get("live_reid_in_loop"):
        summary["live_reid_mot_dir"] = result["live_reid_in_loop"].get("mot_dir")
        summary["occlusion_predictions"] = result["live_reid_in_loop"].get("occlusion_predictions")
        summary["accepted_remaps"] = result["live_reid_in_loop"].get("accepted_remaps", [])
    elif result.get("live_reid"):
        summary["live_reid_mot_dir"] = result["live_reid"].get("mot_dir")
        summary["accepted_remaps"] = result["live_reid"].get("accepted_remaps", [])
    if result.get("render"):
        summary["render_output"] = result["render"].get("output_video")
        summary["rendered_mot_path"] = result["render"].get("rendered_mot_path")
        summary["rendered_mot_kind"] = result["render"].get("rendered_mot_kind")
        summary["occlusion_overlay_enabled"] = bool(result["render"].get("occlusion_overlay_enabled"))
        summary["occlusion_predictions_rendered"] = int(result["render"].get("occlusion_predictions_rendered") or 0)
        summary["recovery_event_count"] = int(result["render"].get("recovery_event_count") or 0)
        summary["video_codec_requested"] = result["render"].get("video_codec_requested")
        summary["video_codec_used"] = result["render"].get("video_codec_used")
        summary["ffmpeg_transcode_used"] = bool(result["render"].get("ffmpeg_transcode_used"))
        summary["windows_friendly_video"] = bool(result["render"].get("windows_friendly_video"))
        summary["video_codec_warning"] = result["render"].get("video_codec_warning")
    occlusion_summary = _load_occlusion_summary(summary.get("occlusion_predictions"))
    summary["prediction_count"] = int(occlusion_summary.get("prediction_count") or 0)
    summary["recovered_prediction_count"] = int(occlusion_summary.get("recovered_prediction_count") or 0)
    if "recovery_event_count" not in summary:
        summary["recovery_event_count"] = _count_recovery_events(summary.get("occlusion_predictions"))
    summary["max_gap_frames"] = int(occlusion_summary.get("max_gap_frames") or 0)
    summary["mean_uncertainty_radius"] = occlusion_summary.get("mean_uncertainty_radius")
    summary["demo_summary"] = str(_write_demo_summary(args, result, summary))
    summary["raw_result"] = result
    _print_human_summary(summary)
    print("\nMachine-readable summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
