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
    parser.add_argument("--secondary-model", help="Optional secondary detector for experimental fusion diagnostics.")
    parser.add_argument(
        "--detector-fusion",
        action="store_true",
        help=(
            "Experimental detector fusion diagnostics. The current tracker loop preserves primary-model "
            "tracking/MOT output unless true pre-tracker fusion is added later."
        ),
    )
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.55)
    parser.add_argument("--fusion-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--fusion-mode", choices=["union_nms"], default="union_nms")
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
    parser.add_argument(
        "--prediction-overlay-style",
        choices=["circle", "arrow", "corridor", "compact"],
        help=(
            "Occlusion overlay style. In --demo-mode, defaults to compact; otherwise preserves "
            "the older circle style unless set explicitly."
        ),
    )
    parser.add_argument(
        "--hide-uncertainty-circle",
        action="store_true",
        help="Suppress the large magenta uncertainty circle and use compact outlines/corridors instead.",
    )
    parser.add_argument("--zoom-occlusion-roi", action="store_true", help="Zoom the right demo panel around the active occlusion/recovery area.")
    parser.add_argument("--zoom-padding", type=float, default=1.8, help="Padding multiplier around the occlusion zoom ROI. Default: 1.8.")
    parser.add_argument("--zoom-min-size", type=int, default=360, help="Minimum width/height of the occlusion zoom ROI. Default: 360.")
    parser.add_argument("--zoom-follow-prediction", action="store_true", help="Let the zoom ROI follow the active prediction more closely.")
    parser.add_argument("--zoom-reid-recovery", action="store_true", help="Prioritize the zoom panel around ReID recovery events after occlusion.")
    parser.add_argument("--recovery-zoom-pre-frames", type=int, default=30, help="Frames before a ReID recovery event to keep the recovery zoom active.")
    parser.add_argument("--recovery-zoom-post-frames", type=int, default=90, help="Frames after a ReID recovery event to keep the recovery zoom active.")
    parser.add_argument("--recovery-zoom-padding", type=float, default=2.0, help="Padding multiplier around the ReID recovery zoom ROI.")
    parser.add_argument("--recovery-zoom-label-duration", type=int, default=90, help="Frames to keep ReID recovery labels visible.")
    parser.add_argument(
        "--prediction-overlay-detail",
        choices=["clean", "debug"],
        help="clean hides large uncertainty overlays; debug keeps more corridor/search details.",
    )
    parser.add_argument("--live-reid", action="store_true", help="Run post-stage bounded-latency ReID.")
    parser.add_argument("--live-reid-in-loop", action="store_true", help="Run tracker-loop bounded-latency ReID v1.")
    parser.add_argument("--live-reid-config", help="Optional ReID config path.")
    parser.add_argument("--confirmation-observations", type=int, default=10)
    parser.add_argument(
        "--visual-continuation",
        action="store_true",
        help=(
            "Add reporting-only local template continuation during occlusion gaps. "
            "Does not alter tracking, ReID, or MOT output."
        ),
    )
    parser.add_argument(
        "--paired-occlusion-prediction",
        action="store_true",
        help=(
            "Add reporting-only paired-vessel motion corridor fields when an occluder can be estimated. "
            "Does not alter tracking, ReID, or MOT output."
        ),
    )
    parser.add_argument(
        "--motion-corridor-overlay",
        action="store_true",
        help="Draw directional corridor ellipses/arrows for paired occlusion predictions in rendered demos.",
    )
    parser.add_argument(
        "--continuation-search-radius",
        type=int,
        default=64,
        help="Pixel radius for the visual-continuation local search window. Default: 64.",
    )
    parser.add_argument(
        "--continuation-threshold",
        type=float,
        default=0.45,
        help="Minimum decayed template-match confidence for continuation to be marked active. Default: 0.45.",
    )
    parser.add_argument(
        "--continuation-max-gap",
        type=int,
        default=60,
        help="Maximum gap, in frames, for visual-continuation matching. Default: 60.",
    )
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
        "primary_model": args.model,
        "secondary_model": args.secondary_model,
        "detector_fusion_enabled": bool(args.detector_fusion),
        "detector_fusion_report": (result.get("detector_fusion") or {}).get("report_path"),
        "fused_detection_count": (result.get("detector_fusion") or {}).get("fused_detection_count"),
        "secondary_only_detection_count": (result.get("detector_fusion") or {}).get("secondary_only_detection_count"),
        "duplicate_removed_count": (result.get("detector_fusion") or {}).get("duplicate_removed_count"),
        "detector_fusion_applied_to_tracker": (result.get("detector_fusion") or {}).get("applied_to_tracker"),
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
        "motion_corridor_overlay_enabled": bool(args.motion_corridor_overlay),
        "prediction_overlay_style": summary.get("prediction_overlay_style"),
        "uncertainty_circle_hidden": bool(summary.get("uncertainty_circle_hidden")),
        "motion_arrow_enabled": bool(summary.get("motion_arrow_enabled")),
        "predicted_bbox_overlay_enabled": bool(summary.get("predicted_bbox_overlay_enabled")),
        "zoom_occlusion_roi_enabled": bool(summary.get("zoom_occlusion_roi_enabled")),
        "frames_with_zoom_roi": int(summary.get("frames_with_zoom_roi") or 0),
        "zoom_padding": summary.get("zoom_padding"),
        "zoom_min_size": summary.get("zoom_min_size"),
        "zoom_follow_prediction": bool(summary.get("zoom_follow_prediction")),
        "zoom_reid_recovery_enabled": bool(summary.get("zoom_reid_recovery_enabled")),
        "recovery_zoom_event_count": int(summary.get("recovery_zoom_event_count") or 0),
        "recovery_zoom_pre_frames": int(summary.get("recovery_zoom_pre_frames") or 0),
        "recovery_zoom_post_frames": int(summary.get("recovery_zoom_post_frames") or 0),
        "recovery_zoom_active_frames": int(summary.get("recovery_zoom_active_frames") or 0),
        "recovery_zoom_label_duration": int(summary.get("recovery_zoom_label_duration") or 0),
        "recovery_events": summary.get("recovery_events") or [],
        "prediction_overlay_detail": summary.get("prediction_overlay_detail"),
        "clean_demo_overlay_enabled": bool(summary.get("clean_demo_overlay_enabled")),
        "large_uncertainty_overlay_enabled": bool(summary.get("large_uncertainty_overlay_enabled")),
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
        "visual_continuation_enabled": bool(occlusion_summary.get("visual_continuation_enabled")),
        "visual_continuation_prediction_count": int(occlusion_summary.get("visual_continuation_prediction_count") or 0),
        "visual_continuation_active_count": int(occlusion_summary.get("visual_continuation_active_count") or 0),
        "mean_visual_continuation_confidence": occlusion_summary.get("mean_visual_continuation_confidence"),
        "continuation_part_matching_enabled": bool(occlusion_summary.get("continuation_part_matching_enabled")),
        "continuation_prediction_count": int(occlusion_summary.get("continuation_prediction_count") or 0),
        "continuation_active_count": int(occlusion_summary.get("continuation_active_count") or 0),
        "mean_continuation_confidence": occlusion_summary.get("mean_continuation_confidence"),
        "mean_best_part_score": occlusion_summary.get("mean_best_part_score"),
        "matched_part_distribution": occlusion_summary.get("matched_part_distribution") or {},
        "visual_continuation_note": occlusion_summary.get("visual_continuation_note"),
        "paired_occlusion_prediction_enabled": bool(occlusion_summary.get("paired_occlusion_prediction_enabled")),
        "paired_prediction_count": int(occlusion_summary.get("paired_prediction_count") or 0),
        "occluder_pair_count": int(occlusion_summary.get("occluder_pair_count") or 0),
        "mean_occlusion_pair_confidence": occlusion_summary.get("mean_occlusion_pair_confidence"),
        "corridor_prediction_count": int(occlusion_summary.get("corridor_prediction_count") or 0),
        "corridor_direction_valid_count": int(occlusion_summary.get("corridor_direction_valid_count") or 0),
        "corridor_direction_inconsistent_count": int(occlusion_summary.get("corridor_direction_inconsistent_count") or 0),
        "corridor_direction_unknown_count": int(occlusion_summary.get("corridor_direction_unknown_count") or 0),
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
    print(f"  prediction overlay: {summary.get('prediction_overlay_style') or 'none'}")
    print(f"  uncertainty circle hidden: {'yes' if summary.get('uncertainty_circle_hidden') else 'no'}")
    print(f"  zoom occlusion ROI: {'enabled' if summary.get('zoom_occlusion_roi_enabled') else 'disabled'}")
    if summary.get("zoom_occlusion_roi_enabled"):
        print(f"  frames with zoom ROI: {summary.get('frames_with_zoom_roi', 0)}")
        print(f"  overlay detail: {summary.get('prediction_overlay_detail')}")
    print(f"  ReID recovery zoom: {'enabled' if summary.get('zoom_reid_recovery_enabled') else 'disabled'}")
    if summary.get("zoom_reid_recovery_enabled"):
        print(f"  recovery zoom events: {summary.get('recovery_zoom_event_count', 0)}")
        print(f"  recovery zoom active frames: {summary.get('recovery_zoom_active_frames', 0)}")
    print(f"  video codec: {summary.get('video_codec_used') or 'none'}")
    print(f"  ffmpeg transcode: {'yes' if summary.get('ffmpeg_transcode_used') else 'no'}")
    print(f"  Windows-friendly video: {'yes' if summary.get('windows_friendly_video') else 'no'}")
    print(f"  accepted remaps: {accepted_remaps if accepted_remaps else 'none'}")
    print(f"  prediction count: {summary.get('prediction_count', 0)}")
    print(f"  recovered predictions: {summary.get('recovered_prediction_count', 0)}")
    print(f"  max gap frames: {summary.get('max_gap_frames', 0)}")
    print(f"  visual continuation: {'enabled' if summary.get('visual_continuation_enabled') else 'disabled'}")
    if summary.get("visual_continuation_enabled"):
        print(f"  continuation active predictions: {summary.get('visual_continuation_active_count', 0)}")
        print(f"  mean continuation confidence: {summary.get('mean_continuation_confidence')}")
        print(f"  matched parts: {summary.get('matched_part_distribution') or {}}")
    print(f"  paired occlusion prediction: {'enabled' if summary.get('paired_occlusion_prediction_enabled') else 'disabled'}")
    if summary.get("paired_occlusion_prediction_enabled"):
        print(f"  paired predictions: {summary.get('paired_prediction_count', 0)}")
        print(f"  occluder pairs: {summary.get('occluder_pair_count', 0)}")
        print(f"  mean pair confidence: {summary.get('mean_occlusion_pair_confidence')}")
        print(
            "  corridor direction: "
            f"valid={summary.get('corridor_direction_valid_count', 0)}, "
            f"inconsistent={summary.get('corridor_direction_inconsistent_count', 0)}, "
            f"unknown={summary.get('corridor_direction_unknown_count', 0)}"
        )
    print(f"  detector fusion: {'enabled' if summary.get('detector_fusion_enabled') else 'disabled'}")
    if summary.get("detector_fusion_enabled"):
        print(f"  fusion report: {summary.get('detector_fusion_report')}")
        print(f"  fused detections: {summary.get('fused_detection_count')}")
        print(f"  secondary-only detections: {summary.get('secondary_only_detection_count')}")
        print(f"  duplicate detections removed: {summary.get('duplicate_removed_count')}")
        print(f"  fusion applied to tracker: {summary.get('detector_fusion_applied_to_tracker')}")
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
        if args.prediction_overlay_style is None:
            args.prediction_overlay_style = "compact"
        if args.prediction_overlay_detail is None:
            args.prediction_overlay_detail = "clean"
    if args.prediction_overlay_style is None:
        args.prediction_overlay_style = "circle"
    if args.prediction_overlay_detail is None:
        args.prediction_overlay_detail = "debug"

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
        visual_continuation=args.visual_continuation,
        continuation_search_radius=args.continuation_search_radius,
        continuation_threshold=args.continuation_threshold,
        continuation_max_gap=args.continuation_max_gap,
        paired_occlusion_prediction=args.paired_occlusion_prediction,
        secondary_model=args.secondary_model,
        detector_fusion=args.detector_fusion,
        fusion_iou_threshold=args.fusion_iou_threshold,
        fusion_confidence_threshold=args.fusion_confidence_threshold,
        fusion_mode=args.fusion_mode,
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
                "motion_corridor_overlay": bool(args.motion_corridor_overlay),
                "prediction_overlay_style": args.prediction_overlay_style,
                "zoom_occlusion_roi": bool(args.zoom_occlusion_roi),
                "zoom_reid_recovery": bool(args.zoom_reid_recovery),
                "colreg_diagnostics": bool(args.colreg_diagnostics),
                "ais_diagnostics": bool(args.ais_diagnostics),
                "demo_title": args.demo_title,
                "demo_note": args.demo_note,
            },
            video_codec=args.video_codec,
            motion_corridor_overlay=args.motion_corridor_overlay,
            prediction_overlay_style=args.prediction_overlay_style,
            hide_uncertainty_circle=args.hide_uncertainty_circle,
            zoom_occlusion_roi=args.zoom_occlusion_roi,
            zoom_padding=args.zoom_padding,
            zoom_min_size=args.zoom_min_size,
            zoom_follow_prediction=args.zoom_follow_prediction,
            prediction_overlay_detail=args.prediction_overlay_detail,
            zoom_reid_recovery=args.zoom_reid_recovery,
            recovery_zoom_pre_frames=args.recovery_zoom_pre_frames,
            recovery_zoom_post_frames=args.recovery_zoom_post_frames,
            recovery_zoom_padding=args.recovery_zoom_padding,
            recovery_zoom_label_duration=args.recovery_zoom_label_duration,
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
        "motion_corridor_overlay_enabled": bool(args.motion_corridor_overlay),
        "prediction_overlay_style": args.prediction_overlay_style,
        "uncertainty_circle_hidden": bool(args.hide_uncertainty_circle),
        "motion_arrow_enabled": args.prediction_overlay_style in {"arrow", "corridor", "compact"} or bool(args.motion_corridor_overlay),
        "predicted_bbox_overlay_enabled": args.prediction_overlay_style in {"arrow", "corridor", "compact"},
        "zoom_occlusion_roi_enabled": False,
        "frames_with_zoom_roi": 0,
        "zoom_padding": args.zoom_padding,
        "zoom_min_size": args.zoom_min_size,
        "zoom_follow_prediction": bool(args.zoom_follow_prediction),
        "zoom_reid_recovery_enabled": False,
        "recovery_zoom_event_count": 0,
        "recovery_zoom_pre_frames": args.recovery_zoom_pre_frames,
        "recovery_zoom_post_frames": args.recovery_zoom_post_frames,
        "recovery_zoom_active_frames": 0,
        "recovery_zoom_label_duration": args.recovery_zoom_label_duration,
        "recovery_events": [],
        "prediction_overlay_detail": args.prediction_overlay_detail,
        "clean_demo_overlay_enabled": args.prediction_overlay_detail == "clean",
        "large_uncertainty_overlay_enabled": args.prediction_overlay_detail != "clean" and not args.hide_uncertainty_circle and args.prediction_overlay_style == "circle",
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
        "visual_continuation_enabled": bool(args.visual_continuation),
        "paired_occlusion_prediction_enabled": bool(args.paired_occlusion_prediction),
        "detector_fusion_enabled": bool(args.detector_fusion),
        "detector_fusion_report": None,
        "fused_detection_count": None,
        "secondary_only_detection_count": None,
        "duplicate_removed_count": None,
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
        summary["motion_corridor_overlay_enabled"] = bool(result["render"].get("motion_corridor_overlay"))
        summary["prediction_overlay_style"] = result["render"].get("prediction_overlay_style")
        summary["uncertainty_circle_hidden"] = bool(result["render"].get("uncertainty_circle_hidden"))
        summary["motion_arrow_enabled"] = bool(result["render"].get("motion_arrow_enabled"))
        summary["predicted_bbox_overlay_enabled"] = bool(result["render"].get("predicted_bbox_overlay_enabled"))
        summary["zoom_occlusion_roi_enabled"] = bool(result["render"].get("zoom_occlusion_roi_enabled"))
        summary["frames_with_zoom_roi"] = int(result["render"].get("frames_with_zoom_roi") or 0)
        summary["zoom_padding"] = result["render"].get("zoom_padding")
        summary["zoom_min_size"] = result["render"].get("zoom_min_size")
        summary["zoom_follow_prediction"] = bool(result["render"].get("zoom_follow_prediction"))
        summary["zoom_reid_recovery_enabled"] = bool(result["render"].get("zoom_reid_recovery_enabled"))
        summary["recovery_zoom_event_count"] = int(result["render"].get("recovery_zoom_event_count") or 0)
        summary["recovery_zoom_pre_frames"] = int(result["render"].get("recovery_zoom_pre_frames") or 0)
        summary["recovery_zoom_post_frames"] = int(result["render"].get("recovery_zoom_post_frames") or 0)
        summary["recovery_zoom_active_frames"] = int(result["render"].get("recovery_zoom_active_frames") or 0)
        summary["recovery_zoom_label_duration"] = int(result["render"].get("recovery_zoom_label_duration") or 0)
        summary["recovery_events"] = result["render"].get("recovery_events") or []
        summary["prediction_overlay_detail"] = result["render"].get("prediction_overlay_detail")
        summary["clean_demo_overlay_enabled"] = bool(result["render"].get("clean_demo_overlay_enabled"))
        summary["large_uncertainty_overlay_enabled"] = bool(result["render"].get("large_uncertainty_overlay_enabled"))
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
    summary["visual_continuation_enabled"] = bool(occlusion_summary.get("visual_continuation_enabled"))
    summary["visual_continuation_prediction_count"] = int(occlusion_summary.get("visual_continuation_prediction_count") or 0)
    summary["visual_continuation_active_count"] = int(occlusion_summary.get("visual_continuation_active_count") or 0)
    summary["mean_visual_continuation_confidence"] = occlusion_summary.get("mean_visual_continuation_confidence")
    summary["continuation_part_matching_enabled"] = bool(occlusion_summary.get("continuation_part_matching_enabled"))
    summary["continuation_prediction_count"] = int(occlusion_summary.get("continuation_prediction_count") or 0)
    summary["continuation_active_count"] = int(occlusion_summary.get("continuation_active_count") or 0)
    summary["mean_continuation_confidence"] = occlusion_summary.get("mean_continuation_confidence")
    summary["mean_best_part_score"] = occlusion_summary.get("mean_best_part_score")
    summary["matched_part_distribution"] = occlusion_summary.get("matched_part_distribution") or {}
    summary["visual_continuation_note"] = occlusion_summary.get("visual_continuation_note")
    summary["paired_occlusion_prediction_enabled"] = bool(occlusion_summary.get("paired_occlusion_prediction_enabled"))
    summary["paired_prediction_count"] = int(occlusion_summary.get("paired_prediction_count") or 0)
    summary["occluder_pair_count"] = int(occlusion_summary.get("occluder_pair_count") or 0)
    summary["mean_occlusion_pair_confidence"] = occlusion_summary.get("mean_occlusion_pair_confidence")
    summary["corridor_prediction_count"] = int(occlusion_summary.get("corridor_prediction_count") or 0)
    summary["corridor_direction_valid_count"] = int(occlusion_summary.get("corridor_direction_valid_count") or 0)
    summary["corridor_direction_inconsistent_count"] = int(occlusion_summary.get("corridor_direction_inconsistent_count") or 0)
    summary["corridor_direction_unknown_count"] = int(occlusion_summary.get("corridor_direction_unknown_count") or 0)
    fusion_result = result.get("detector_fusion") or {}
    if fusion_result:
        summary["detector_fusion_report"] = fusion_result.get("report_path")
        summary["fused_detection_count"] = fusion_result.get("fused_detection_count")
        summary["secondary_only_detection_count"] = fusion_result.get("secondary_only_detection_count")
        summary["duplicate_removed_count"] = fusion_result.get("duplicate_removed_count")
        summary["detector_fusion_applied_to_tracker"] = fusion_result.get("applied_to_tracker")
    summary["demo_summary"] = str(_write_demo_summary(args, result, summary))
    summary["raw_result"] = result
    _print_human_summary(summary)
    print("\nMachine-readable summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
