from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from evaluation.mot import load_mot_frames

LABEL_MODES = {"id", "full", "none"}
VIDEO_CODECS = {"auto", "mp4v", "h264"}
DEFAULT_FPS = 30.0
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_PADDING = 4
LABEL_LINE_GAP = 4
TRACK_COLORS = [
    (80, 220, 255),
    (0, 200, 120),
    (255, 170, 0),
    (220, 80, 80),
    (180, 90, 255),
    (255, 120, 200),
    (120, 210, 70),
    (255, 255, 80),
]
OCCLUSION_COLOR = (255, 80, 255)
CONTINUATION_COLOR = (255, 220, 80)
CONTINUATION_ACTIVE_COLOR = (80, 255, 255)
OCCLUSION_FILL_ALPHA = 0.18
RECOVERY_COLOR = (80, 255, 180)
STATUS_BG_COLOR = (10, 20, 30)
# Keep recovery banners on screen long enough for a human reviewer to notice.
RECOVERY_EVENT_DURATION_FRAMES = 30


def _load_frame_result(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _normalize_label_mode(label_mode: str | None, default: str) -> str:
    mode = (label_mode or default).lower()
    if mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}. Expected one of {sorted(LABEL_MODES)}")
    return mode


def _normalize_video_codec(video_codec: str | None) -> str:
    codec = (video_codec or "auto").lower()
    if codec not in VIDEO_CODECS:
        raise ValueError(f"Unsupported video codec: {video_codec}. Expected one of {sorted(VIDEO_CODECS)}")
    return codec


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _center_bbox_to_corners(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def _mot_bbox_to_corners(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None or track_id < 0:
        return 0, 255, 0
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def _load_occlusion_predictions(occlusion_predictions_path: str | None) -> dict[int, list[dict]]:
    if not occlusion_predictions_path:
        return {}
    path = Path(occlusion_predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Occlusion prediction report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict]] = {}
    for sequence_payload in (payload.get("sequences") or {}).values():
        for prediction in sequence_payload.get("predictions") or []:
            frame = int(prediction.get("current_frame", prediction.get("frame")))
            by_frame.setdefault(frame, []).append(prediction)
    return by_frame


def _load_recovery_events(occlusion_predictions_path: str | None) -> dict[int, list[dict]]:
    """Build display-only ReID recovery banners from the occlusion report."""

    if not occlusion_predictions_path:
        return {}
    path = Path(occlusion_predictions_path)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    events_by_frame: dict[int, list[dict]] = {}
    seen: set[tuple[int, int | None, int | None, int | None]] = set()
    for sequence_payload in (payload.get("sequences") or {}).values():
        for prediction in sequence_payload.get("predictions") or []:
            if not prediction.get("recovered_by_remap"):
                continue
            decision_frame = prediction.get("remap_decision_frame")
            if decision_frame is None:
                continue
            frame = int(decision_frame)
            canonical_id = prediction.get("canonical_id")
            recovered_raw_id = prediction.get("remap_target_raw_track_id")
            lost_raw_id = prediction.get("raw_track_id", prediction.get("track_id"))
            key = (frame, int(recovered_raw_id) if recovered_raw_id is not None else None, int(canonical_id) if canonical_id is not None else None, int(lost_raw_id) if lost_raw_id is not None else None)
            if key in seen:
                continue
            seen.add(key)
            event = {
                "frame": frame,
                "recovered_raw_id": recovered_raw_id,
                "lost_raw_id": lost_raw_id,
                "canonical_id": canonical_id,
            }
            for display_frame in range(frame, frame + RECOVERY_EVENT_DURATION_FRAMES):
                events_by_frame.setdefault(display_frame, []).append(event)
    return events_by_frame


def _count_recovery_events(events_by_frame: dict[int, list[dict]]) -> int:
    unique_events = {
        (
            int(event.get("frame", -1)),
            event.get("recovered_raw_id"),
            event.get("canonical_id"),
            event.get("lost_raw_id"),
        )
        for events in events_by_frame.values()
        for event in events
    }
    unique_events.discard((-1, None, None, None))
    return len(unique_events)


def _ship_label(track_id: int | str | None) -> str:
    return f"ship {track_id}" if track_id is not None else "ship ?"


def _display_prediction_source(source: object) -> str:
    text = str(source or "predicted").replace("_", " ")
    return text if text else "predicted"


def _occlusion_label_lines(prediction: dict) -> list[str]:
    raw_track_id = prediction.get("raw_track_id", prediction.get("track_id"))
    canonical_id = prediction.get("canonical_id")
    display_id = canonical_id if canonical_id is not None else raw_track_id
    source = _display_prediction_source(prediction.get("source", "prediction"))
    lines = [
        _ship_label(display_id),
        f"gap {int(prediction.get('gap_frames', 0))} | predicted",
        f"source {source}",
    ]
    occluder_id = prediction.get("occluder_canonical_id") or prediction.get("occluder_track_id")
    if occluder_id is not None:
        lines.append(f"occluded by {_ship_label(occluder_id)}")
    expected_side = prediction.get("expected_reappearance_side")
    if expected_side and expected_side != "unknown":
        lines.append(f"expected exit: {expected_side}")
    continuation = prediction.get("visual_continuation")
    if isinstance(continuation, dict):
        status = str(continuation.get("status") or "lost")
        confidence = float(continuation.get("confidence") or 0.0)
        part_name = continuation.get("matched_part_name")
        if part_name:
            lines.append(f"{status} {confidence:.2f} | part {part_name}")
        else:
            lines.append(f"continuation {status} {confidence:.2f}")
    if bool(prediction.get("recovered_by_remap")):
        lines.append("recovered by ReID")
    return lines


def _draw_recovery_events(frame, events: list[dict]) -> None:
    if not events:
        return
    lines = []
    for event in events[:2]:
        recovered_raw_id = event.get("recovered_raw_id")
        canonical_id = event.get("canonical_id")
        if recovered_raw_id is not None and canonical_id is not None:
            lines.append(f"ReID recovered: {_ship_label(recovered_raw_id)} -> {_ship_label(canonical_id)}")
        else:
            lines.append("ReID recovered identity after gap")
    _draw_label_block(frame, 16, 52, lines, RECOVERY_COLOR)


def _draw_motion_corridor(frame, prediction: dict, center_x: int, center_y: int) -> bool:
    major = prediction.get("uncertainty_major_axis")
    minor = prediction.get("uncertainty_minor_axis")
    if major is None or minor is None:
        return False
    angle = float(prediction.get("uncertainty_angle_deg") or 0.0)
    axes = (
        max(2, int(round(float(major)))),
        max(2, int(round(float(minor)))),
    )
    overlay = frame.copy()
    cv2.ellipse(overlay, (center_x, center_y), axes, angle, 0, 360, OCCLUSION_COLOR, -1)
    cv2.addWeighted(overlay, OCCLUSION_FILL_ALPHA, frame, 1.0 - OCCLUSION_FILL_ALPHA, 0, frame)
    cv2.ellipse(frame, (center_x, center_y), axes, angle, 0, 360, OCCLUSION_COLOR, 2)

    start = prediction.get("motion_corridor_start") or {}
    end = prediction.get("motion_corridor_end") or {}
    if start and end:
        sx = _clamp(int(round(float(start.get("x", center_x)))), 0, frame.shape[1] - 1)
        sy = _clamp(int(round(float(start.get("y", center_y)))), 0, frame.shape[0] - 1)
        ex = _clamp(int(round(float(end.get("x", center_x)))), 0, frame.shape[1] - 1)
        ey = _clamp(int(round(float(end.get("y", center_y)))), 0, frame.shape[0] - 1)
        cv2.arrowedLine(frame, (sx, sy), (ex, ey), OCCLUSION_COLOR, 2, cv2.LINE_AA, tipLength=0.18)
    return True


def _draw_occlusion_prediction(frame, prediction: dict, *, motion_corridor_overlay: bool = False) -> None:
    """Draw a missing-vessel prediction; optional corridor fields stay visualization-only."""

    frame_height, frame_width = frame.shape[:2]
    center = prediction.get("predicted_center") or {}
    x = float(center.get("x", prediction.get("predicted_x", 0.0)))
    y = float(center.get("y", prediction.get("predicted_y", 0.0)))
    radius = max(2, int(round(float(prediction.get("uncertainty_radius") or 0.0))))
    center_x = _clamp(int(round(x)), 0, max(frame_width - 1, 0))
    center_y = _clamp(int(round(y)), 0, max(frame_height - 1, 0))

    drew_corridor = motion_corridor_overlay and _draw_motion_corridor(frame, prediction, center_x, center_y)
    if not drew_corridor:
        overlay = frame.copy()
        # The translucent fill makes uncertainty visible without hiding the vessel scene.
        cv2.circle(overlay, (center_x, center_y), radius, OCCLUSION_COLOR, -1)
        cv2.addWeighted(overlay, OCCLUSION_FILL_ALPHA, frame, 1.0 - OCCLUSION_FILL_ALPHA, 0, frame)
        cv2.circle(frame, (center_x, center_y), radius, OCCLUSION_COLOR, 2)
    cv2.drawMarker(
        frame,
        (center_x, center_y),
        OCCLUSION_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    continuation = prediction.get("visual_continuation")
    if isinstance(continuation, dict):
        search_window = continuation.get("search_window") or {}
        if search_window:
            x1 = int(search_window.get("x", 0))
            y1 = int(search_window.get("y", 0))
            x2 = x1 + int(search_window.get("w", 0))
            y2 = y1 + int(search_window.get("h", 0))
            color = CONTINUATION_ACTIVE_COLOR if continuation.get("active") else CONTINUATION_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        best_center = continuation.get("best_center") or {}
        if best_center:
            best_x = _clamp(int(round(float(best_center.get("x", center_x)))), 0, max(frame_width - 1, 0))
            best_y = _clamp(int(round(float(best_center.get("y", center_y)))), 0, max(frame_height - 1, 0))
            cv2.circle(frame, (best_x, best_y), 5, CONTINUATION_ACTIVE_COLOR, -1)
    _draw_label_block(frame, center_x + 8, center_y, _occlusion_label_lines(prediction), OCCLUSION_COLOR)


def _bbox_key(row: dict) -> tuple[int, int, int, int]:
    x, y, w, h = row["bbox"]
    return tuple(int(round(float(value) * 10)) for value in (x, y, w, h))


def _build_raw_id_lookup(raw_mot_file_path: str | None) -> dict[int, dict[tuple[int, int, int, int], int]]:
    if not raw_mot_file_path:
        return {}
    raw_path = Path(raw_mot_file_path)
    if not raw_path.exists():
        return {}
    lookup: dict[int, dict[tuple[int, int, int, int], int]] = {}
    for frame, rows in load_mot_frames(raw_path).items():
        frame_lookup = lookup.setdefault(frame, {})
        for row in rows:
            frame_lookup[_bbox_key(row)] = int(row["id"])
    return lookup


def _draw_status_strip(frame, frame_index: int, context: dict | None) -> None:
    context = context or {}
    frame_height, frame_width = frame.shape[:2]
    mode = context.get("mode") or "raw"
    reid_enabled = "yes" if context.get("reid_enabled") else "no"
    occlusion_enabled = "yes" if context.get("occlusion_predictions_enabled") else "no"
    corridor_enabled = "yes" if context.get("motion_corridor_overlay") else "no"
    colreg_enabled = "yes" if context.get("colreg_diagnostics") else "no"
    ais_enabled = "yes" if context.get("ais_diagnostics") else "no"
    lines: list[str] = []
    demo_title = context.get("demo_title")
    demo_note = context.get("demo_note")
    if demo_title:
        lines.append(str(demo_title))
    lines.extend(
        [
            f"Frame {frame_index} | mode: {mode} | ReID: {reid_enabled} | occlusion prediction: {occlusion_enabled}",
            f"motion corridor overlay: {corridor_enabled} | paired prediction is visualization/reporting only",
            f"COLREG diagnostics: {colreg_enabled} | AIS diagnostics: {ais_enabled} | bounded-latency live demo",
            "Left: original | Right: tracking + ReID + occlusion prediction",
            "Legend: ship label = displayed vessel ID | magenta ghost+circle = prediction + uncertainty",
            "Yellow box/dot = visual continuation search and best local template match when enabled",
            "Green banner = ReID recovery event | COLREG/AIS are reporting-only when enabled",
        ]
    )
    if demo_note:
        lines.append(f"Note: {demo_note}")

    line_gap = 22
    strip_height = min(frame_height, 14 + (line_gap * len(lines)))
    y1 = max(0, frame_height - strip_height)
    cv2.rectangle(frame, (0, y1), (frame_width, frame_height), STATUS_BG_COLOR, -1)

    text_y = y1 + 18
    for line in lines:
        cv2.putText(frame, str(line)[:170], (16, text_y), LABEL_FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        text_y += line_gap


def _resolve_fps(requested_fps: float | None, fallback_fps: float | None = None) -> float:
    if requested_fps is not None and requested_fps > 0:
        return requested_fps
    if fallback_fps is not None and fallback_fps > 0:
        return fallback_fps
    return DEFAULT_FPS


def _codec_candidates(output_path: Path, video_codec: str) -> list[str]:
    if video_codec == "mp4v":
        return ["mp4v"]
    if video_codec == "h264":
        return ["avc1", "H264", "X264", "mp4v"]
    if output_path.suffix.lower() == ".avi":
        return ["XVID", "mp4v"]
    return ["avc1", "H264", "X264", "mp4v"]


def _is_h264_codec(codec: str) -> bool:
    return codec.lower() in {"avc1", "h264", "x264"}


def _open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    *,
    video_codec: str = "auto",
) -> tuple[cv2.VideoWriter, str]:
    codecs = _codec_candidates(output_path, video_codec)
    for codec in codecs:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {output_path}")


def _transcode_to_h264(input_path: Path, output_path: Path) -> tuple[bool, str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False, "ffmpeg_not_found"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffmpeg_transcode_failed"
        return False, message
    return True, None


def _finalize_video_codec(output_path: Path, requested_codec: str, writer_codec: str) -> dict[str, object]:
    writer_is_h264 = _is_h264_codec(writer_codec)
    info: dict[str, object] = {
        "video_codec_requested": requested_codec,
        "video_writer_codec": writer_codec,
        "video_codec_used": "h264" if writer_is_h264 else writer_codec.lower(),
        "ffmpeg_transcode_used": False,
        "windows_friendly_video": writer_is_h264 and output_path.suffix.lower() == ".mp4",
        "video_codec_warning": None,
    }
    if writer_is_h264 or requested_codec == "mp4v" or output_path.suffix.lower() != ".mp4":
        return info

    temp_output = output_path.with_name(f"{output_path.stem}.h264_tmp{output_path.suffix}")
    ok, warning = _transcode_to_h264(output_path, temp_output)
    if ok:
        temp_output.replace(output_path)
        info.update(
            {
                "video_codec_used": "h264",
                "ffmpeg_transcode_used": True,
                "windows_friendly_video": True,
            }
        )
    else:
        if temp_output.exists():
            temp_output.unlink()
        info["video_codec_warning"] = warning or "h264_transcode_unavailable"
    return info


def _draw_label_block(
    frame,
    anchor_x: int,
    anchor_y: int,
    lines: list[str],
    border_color: tuple[int, int, int],
) -> None:
    if not lines:
        return

    frame_height, frame_width = frame.shape[:2]
    text_metrics: list[tuple[str, int, int, int]] = []
    block_width = 0
    text_height = 0

    for line in lines:
        (line_width, line_height), baseline = cv2.getTextSize(
            line,
            LABEL_FONT,
            LABEL_SCALE,
            LABEL_THICKNESS,
        )
        text_metrics.append((line, line_width, line_height, baseline))
        block_width = max(block_width, line_width)
        text_height += line_height + baseline

    total_width = block_width + (LABEL_PADDING * 2)
    total_height = text_height + (LABEL_PADDING * 2) + (LABEL_LINE_GAP * max(len(text_metrics) - 1, 0))

    origin_x = _clamp(anchor_x, 0, max(frame_width - total_width, 0))
    preferred_y = anchor_y - total_height - 6
    if preferred_y < 0:
        preferred_y = anchor_y + 6
    origin_y = _clamp(preferred_y, 0, max(frame_height - total_height, 0))

    cv2.rectangle(
        frame,
        (origin_x, origin_y),
        (origin_x + total_width, origin_y + total_height),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        frame,
        (origin_x, origin_y),
        (origin_x + total_width, origin_y + total_height),
        border_color,
        1,
    )

    text_y = origin_y + LABEL_PADDING
    for index, (line, _, line_height, baseline) in enumerate(text_metrics):
        text_y += line_height
        cv2.putText(
            frame,
            line,
            (origin_x + LABEL_PADDING, text_y),
            LABEL_FONT,
            LABEL_SCALE,
            (255, 255, 255),
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )
        text_y += baseline
        if index < len(text_metrics) - 1:
            text_y += LABEL_LINE_GAP


def _draw_annotation(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    label_lines: list[str],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    x1 = _clamp(x1, 0, max(frame_width - 1, 0))
    y1 = _clamp(y1, 0, max(frame_height - 1, 0))
    x2 = _clamp(x2, 0, max(frame_width - 1, 0))
    y2 = _clamp(y2, 0, max(frame_height - 1, 0))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _draw_label_block(frame, x1, y1, label_lines, color)


def _detection_label_lines(det: dict, label_mode: str) -> list[str]:
    if label_mode == "none":
        return []

    if label_mode == "id":
        return []

    confidence = float(det.get("confidence", 0.0))
    return [f"conf {confidence:.2f}"]


def _track_label_lines(
    row: dict,
    label_mode: str,
    raw_track_id: int | None = None,
    rendered_mot_kind: str = "raw",
) -> list[str]:
    if label_mode == "none":
        return []

    track_id = int(row["id"])
    if label_mode == "id":
        return [_ship_label(track_id)]

    confidence = float(row.get("confidence", 0.0))
    return [_ship_label(track_id), f"conf {confidence:.2f}"]


def render_annotated_video(
    frames_dir: str,
    worker_results_dir: str,
    output_video_path: str,
    clip_id: str,
    fps: float | None = None,
    label_mode: str = "full",
    video_codec: str = "auto",
) -> dict:
    label_mode = _normalize_label_mode(label_mode, default="full")
    requested_codec = _normalize_video_codec(video_codec)
    frame_dir = Path(frames_dir) / clip_id
    result_dir = Path(worker_results_dir) / clip_id
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("*.png"))

    if not frame_files:
        raise RuntimeError(f"No frames found for clip_id={clip_id} in {frame_dir}")

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_files[0]}")

    height, width = first.shape[:2]

    resolved_fps = _resolve_fps(fps)
    writer, codec = _open_video_writer(output_path, resolved_fps, width, height, video_codec=requested_codec)

    frames_written = 0
    frames_with_annotations = 0

    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        frame_index = int(frame_path.stem)
        json_path = result_dir / f"{frame_index:06d}.json"

        if json_path.exists():
            result = _load_frame_result(json_path)
            detections = result.get("detections", [])
            if detections:
                frames_with_annotations += 1
            for det in detections:
                x = float(det["x"])
                y = float(det["y"])
                w = float(det["w"])
                h = float(det["h"])
                x1, y1, x2, y2 = _center_bbox_to_corners(x, y, w, h)
                _draw_annotation(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    (0, 255, 0),
                    _detection_label_lines(det, label_mode),
                )

        writer.write(frame)
        frames_written += 1

    writer.release()
    codec_info = _finalize_video_codec(output_path, requested_codec, codec)

    result = {
        "clip_id": clip_id,
        "output_video": str(output_path),
        "frames_written": frames_written,
        "frames_with_annotations": frames_with_annotations,
        "fps": resolved_fps,
        "label_mode": label_mode,
        "codec": codec,
    }
    result.update(codec_info)
    return result


def render_video_from_mot(
    source_video_path: str,
    mot_file_path: str,
    output_video_path: str,
    fps: float | None = None,
    label_mode: str = "id",
    occlusion_predictions_path: str | None = None,
    side_by_side: bool = False,
    raw_mot_file_path: str | None = None,
    rendered_mot_kind: str = "raw",
    status_context: dict | None = None,
    video_codec: str = "auto",
    motion_corridor_overlay: bool = False,
) -> dict:
    label_mode = _normalize_label_mode(label_mode, default="id")
    requested_codec = _normalize_video_codec(video_codec)
    source_path = Path(source_video_path)
    mot_path = Path(mot_file_path)
    output_path = Path(output_video_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    if not mot_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_frame = load_mot_frames(mot_path)
    occlusion_by_frame = _load_occlusion_predictions(occlusion_predictions_path)
    recovery_events_by_frame = _load_recovery_events(occlusion_predictions_path)
    recovery_event_count = _count_recovery_events(recovery_events_by_frame)
    raw_id_lookup = _build_raw_id_lookup(raw_mot_file_path)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source_path}")

    capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine source frame size: {source_path}")

    resolved_fps = _resolve_fps(fps, fallback_fps=capture_fps)
    writer_width = width * 2 if side_by_side else width
    writer, codec = _open_video_writer(output_path, resolved_fps, writer_width, height, video_codec=requested_codec)

    frames_written = 0
    frames_with_annotations = 0
    rows_rendered = 0
    frames_with_occlusion_predictions = 0
    occlusion_predictions_rendered = 0
    frame_index = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        original_frame = frame.copy() if side_by_side else None
        frame_rows = rows_by_frame.get(frame_index, [])
        if frame_rows:
            frames_with_annotations += 1
        for row in frame_rows:
            x, y, w, h = row["bbox"]
            x1, y1, x2, y2 = _mot_bbox_to_corners(x, y, w, h)
            raw_track_id = raw_id_lookup.get(frame_index, {}).get(_bbox_key(row))
            _draw_annotation(
                frame,
                x1,
                y1,
                x2,
                y2,
                _track_color(int(row["id"])),
                _track_label_lines(row, label_mode, raw_track_id=raw_track_id, rendered_mot_kind=rendered_mot_kind),
            )
            rows_rendered += 1

        _draw_recovery_events(frame, recovery_events_by_frame.get(frame_index, []))

        frame_predictions = occlusion_by_frame.get(frame_index, [])
        if frame_predictions:
            frames_with_occlusion_predictions += 1
        for prediction in frame_predictions:
            _draw_occlusion_prediction(frame, prediction, motion_corridor_overlay=motion_corridor_overlay)
            occlusion_predictions_rendered += 1

        if side_by_side and original_frame is not None:
            cv2.putText(original_frame, "Left: original", (16, 28), LABEL_FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Right: tracking + ReID + prediction", (16, 28), LABEL_FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            combined = cv2.hconcat([original_frame, frame])
            _draw_status_strip(combined, frame_index, status_context)
            writer.write(combined)
        else:
            writer.write(frame)
        frames_written += 1
        frame_index += 1

    capture.release()
    writer.release()
    codec_info = _finalize_video_codec(output_path, requested_codec, codec)

    result = {
        "source_video": str(source_path),
        "mot_file": str(mot_path),
        "output_video": str(output_path),
        "frames_written": frames_written,
        "frames_with_annotations": frames_with_annotations,
        "rows_rendered": rows_rendered,
        "occlusion_predictions_path": occlusion_predictions_path,
        "occlusion_overlay_enabled": bool(occlusion_by_frame),
        "frames_with_occlusion_predictions": frames_with_occlusion_predictions,
        "occlusion_predictions_rendered": occlusion_predictions_rendered,
        "recovery_event_count": recovery_event_count,
        "side_by_side": bool(side_by_side),
        "motion_corridor_overlay": bool(motion_corridor_overlay),
        "raw_mot_file": raw_mot_file_path,
        "rendered_mot_kind": rendered_mot_kind,
        "mot_frames": len(rows_by_frame),
        "fps": resolved_fps,
        "source_fps": capture_fps,
        "label_mode": label_mode,
        "codec": codec,
    }
    result.update(codec_info)
    return result


# ---------------------------------------------------------------------------
# Combined dual-camera video
# ---------------------------------------------------------------------------

CAMERA_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_LABEL_SCALE = 0.7
CAMERA_LABEL_THICKNESS = 2
CAMERA_LABEL_MARGIN = 10


def _annotate_frame(frame, frame_rows: list[dict], label_mode: str) -> None:
    """Draw all MOT annotations on *frame* in-place."""
    for row in frame_rows:
        x, y, w, h = row["bbox"]
        x1, y1, x2, y2 = _mot_bbox_to_corners(x, y, w, h)
        _draw_annotation(
            frame,
            x1,
            y1,
            x2,
            y2,
            _track_color(int(row["id"])),
            _track_label_lines(row, label_mode),
        )


def _pad_to_height(frame, target_height: int):
    """Centre-pad a frame vertically with black to reach *target_height*."""
    h = frame.shape[0]
    if h >= target_height:
        return frame
    top = (target_height - h) // 2
    bottom = target_height - h - top
    return cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _pad_to_width(frame, target_width: int):
    """Centre-pad a frame horizontally with black to reach *target_width*."""
    w = frame.shape[1]
    if w >= target_width:
        return frame
    left = (target_width - w) // 2
    right = target_width - w - left
    return cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _draw_camera_label(frame, label: str) -> None:
    """Draw a semi-transparent banner with *label* at the top-left of *frame*."""
    (text_w, text_h), baseline = cv2.getTextSize(
        label, CAMERA_LABEL_FONT, CAMERA_LABEL_SCALE, CAMERA_LABEL_THICKNESS,
    )
    banner_w = text_w + CAMERA_LABEL_MARGIN * 2
    banner_h = text_h + baseline + CAMERA_LABEL_MARGIN * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        label,
        (CAMERA_LABEL_MARGIN, CAMERA_LABEL_MARGIN + text_h),
        CAMERA_LABEL_FONT,
        CAMERA_LABEL_SCALE,
        (255, 255, 255),
        CAMERA_LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def _compute_layout(
    w_a: int,
    h_a: int,
    w_b: int,
    h_b: int,
    layout: str,
    target_width: int,
) -> dict:
    """Return scaling info for the chosen *layout*."""
    if layout == "side_by_side_hd":
        half_w = target_width // 2
        scale_a = half_w / w_a
        scale_b = half_w / w_b
        new_h_a = int(h_a * scale_a)
        new_h_b = int(h_b * scale_b)
        combined_h = max(new_h_a, new_h_b)
        return {
            "mode": "horizontal",
            "size_a": (half_w, new_h_a),
            "size_b": (half_w, new_h_b),
            "combined": (half_w * 2, combined_h),
            "divider_x": half_w,
        }

    if layout == "side_by_side":
        target_h = max(h_a, h_b)
        scale_a = target_h / h_a
        scale_b = target_h / h_b
        new_w_a = int(w_a * scale_a)
        new_w_b = int(w_b * scale_b)
        return {
            "mode": "horizontal",
            "size_a": (new_w_a, target_h),
            "size_b": (new_w_b, target_h),
            "combined": (new_w_a + new_w_b, target_h),
            "divider_x": new_w_a,
        }

    if layout == "top_bottom":
        target_w = max(w_a, w_b)
        scale_a = target_w / w_a
        scale_b = target_w / w_b
        new_h_a = int(h_a * scale_a)
        new_h_b = int(h_b * scale_b)
        return {
            "mode": "vertical",
            "size_a": (target_w, new_h_a),
            "size_b": (target_w, new_h_b),
            "combined": (target_w, new_h_a + new_h_b),
            "divider_y": new_h_a,
        }

    raise ValueError(f"Unknown layout: {layout!r}. Expected side_by_side_hd, side_by_side, or top_bottom.")


def render_combined_video_from_mot(
    source_video_a: str,
    source_video_b: str,
    mot_file_a: str,
    mot_file_b: str,
    output_video_path: str,
    fps: float | None = None,
    label_mode: str = "id",
    layout: str = "side_by_side_hd",
    camera_a_label: str = "Camera A",
    camera_b_label: str = "Camera B",
    target_width: int = 1920,
) -> dict:
    """Render a single video combining two camera feeds side-by-side (or stacked)."""
    label_mode = _normalize_label_mode(label_mode, default="id")

    path_a = Path(source_video_a)
    path_b = Path(source_video_b)
    mot_a = Path(mot_file_a)
    mot_b = Path(mot_file_b)
    output_path = Path(output_video_path)

    for p, desc in [(path_a, "Source video A"), (path_b, "Source video B"), (mot_a, "MOT file A"), (mot_b, "MOT file B")]:
        if not p.exists():
            raise FileNotFoundError(f"{desc} not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_frame_a = load_mot_frames(mot_a)
    rows_by_frame_b = load_mot_frames(mot_b)

    cap_a = cv2.VideoCapture(str(path_a))
    cap_b = cv2.VideoCapture(str(path_b))
    if not cap_a.isOpened():
        raise RuntimeError(f"Could not open source video A: {path_a}")
    if not cap_b.isOpened():
        cap_a.release()
        raise RuntimeError(f"Could not open source video B: {path_b}")

    w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_a = float(cap_a.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_b = float(cap_b.get(cv2.CAP_PROP_FPS) or 0.0)

    if w_a <= 0 or h_a <= 0 or w_b <= 0 or h_b <= 0:
        cap_a.release()
        cap_b.release()
        raise RuntimeError(f"Invalid video dimensions: A={w_a}x{h_a}, B={w_b}x{h_b}")

    info = _compute_layout(w_a, h_a, w_b, h_b, layout, target_width)
    combined_w, combined_h = info["combined"]
    resolved_fps = _resolve_fps(fps, fallback_fps=max(fps_a, fps_b))
    writer, codec = _open_video_writer(output_path, resolved_fps, combined_w, combined_h)

    frames_written = 0
    last_frame_a = None
    last_frame_b = None
    frame_index = 1

    while True:
        ok_a, raw_a = cap_a.read()
        ok_b, raw_b = cap_b.read()

        if ok_a:
            last_frame_a = raw_a
        if ok_b:
            last_frame_b = raw_b

        if not ok_a and not ok_b:
            break

        frame_a = last_frame_a
        frame_b = last_frame_b
        if frame_a is None or frame_b is None:
            break

        # Annotate at original resolution (MOT coords are in original pixel space)
        ann_a = frame_a.copy()
        ann_b = frame_b.copy()
        _annotate_frame(ann_a, rows_by_frame_a.get(frame_index, []), label_mode)
        _annotate_frame(ann_b, rows_by_frame_b.get(frame_index, []), label_mode)

        # Scale
        scaled_a = cv2.resize(ann_a, info["size_a"], interpolation=cv2.INTER_LINEAR)
        scaled_b = cv2.resize(ann_b, info["size_b"], interpolation=cv2.INTER_LINEAR)

        if info["mode"] == "horizontal":
            scaled_a = _pad_to_height(scaled_a, combined_h)
            scaled_b = _pad_to_height(scaled_b, combined_h)
            _draw_camera_label(scaled_a, camera_a_label)
            _draw_camera_label(scaled_b, camera_b_label)
            combined = np.hstack([scaled_a, scaled_b])
            cv2.line(combined, (info["divider_x"], 0), (info["divider_x"], combined_h), (200, 200, 200), 2)
        else:
            scaled_a = _pad_to_width(scaled_a, combined_w)
            scaled_b = _pad_to_width(scaled_b, combined_w)
            _draw_camera_label(scaled_a, camera_a_label)
            _draw_camera_label(scaled_b, camera_b_label)
            combined = np.vstack([scaled_a, scaled_b])
            cv2.line(combined, (0, info["divider_y"]), (combined_w, info["divider_y"]), (200, 200, 200), 2)

        writer.write(combined)
        frames_written += 1
        frame_index += 1

    cap_a.release()
    cap_b.release()
    writer.release()

    return {
        "source_video_a": str(path_a),
        "source_video_b": str(path_b),
        "mot_file_a": str(mot_a),
        "mot_file_b": str(mot_b),
        "output_video": str(output_path),
        "frames_written": frames_written,
        "layout": layout,
        "combined_resolution": f"{combined_w}x{combined_h}",
        "fps": resolved_fps,
        "label_mode": label_mode,
        "codec": codec,
    }
