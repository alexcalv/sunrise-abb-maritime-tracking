from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.mot import load_mot_frames

LABEL_MODES = {"id", "full", "none"}
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


def _load_frame_result(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _normalize_label_mode(label_mode: str | None, default: str) -> str:
    mode = (label_mode or default).lower()
    if mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}. Expected one of {sorted(LABEL_MODES)}")
    return mode


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


def _resolve_fps(requested_fps: float | None, fallback_fps: float | None = None) -> float:
    if requested_fps is not None and requested_fps > 0:
        return requested_fps
    if fallback_fps is not None and fallback_fps > 0:
        return fallback_fps
    return DEFAULT_FPS


def _open_video_writer(output_path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, str]:
    codecs = ["XVID", "mp4v"] if output_path.suffix.lower() == ".avi" else ["mp4v", "XVID"]
    for codec in codecs:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {output_path}")


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


def _track_label_lines(row: dict, label_mode: str) -> list[str]:
    if label_mode == "none":
        return []

    track_id = int(row["id"])
    if label_mode == "id":
        return [f"ID {track_id}"]

    confidence = float(row.get("confidence", 0.0))
    return [f"ID {track_id}", f"conf {confidence:.2f}"]


def render_annotated_video(
    frames_dir: str,
    worker_results_dir: str,
    output_video_path: str,
    clip_id: str,
    fps: float | None = None,
    label_mode: str = "full",
) -> dict:
    label_mode = _normalize_label_mode(label_mode, default="full")
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
    writer, codec = _open_video_writer(output_path, resolved_fps, width, height)

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

    return {
        "clip_id": clip_id,
        "output_video": str(output_path),
        "frames_written": frames_written,
        "frames_with_annotations": frames_with_annotations,
        "fps": resolved_fps,
        "label_mode": label_mode,
        "codec": codec,
    }


def render_video_from_mot(
    source_video_path: str,
    mot_file_path: str,
    output_video_path: str,
    fps: float | None = None,
    label_mode: str = "id",
) -> dict:
    label_mode = _normalize_label_mode(label_mode, default="id")
    source_path = Path(source_video_path)
    mot_path = Path(mot_file_path)
    output_path = Path(output_video_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    if not mot_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_frame = load_mot_frames(mot_path)

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
    writer, codec = _open_video_writer(output_path, resolved_fps, width, height)

    frames_written = 0
    frames_with_annotations = 0
    rows_rendered = 0
    frame_index = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_rows = rows_by_frame.get(frame_index, [])
        if frame_rows:
            frames_with_annotations += 1
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
            rows_rendered += 1

        writer.write(frame)
        frames_written += 1
        frame_index += 1

    capture.release()
    writer.release()

    return {
        "source_video": str(source_path),
        "mot_file": str(mot_path),
        "output_video": str(output_path),
        "frames_written": frames_written,
        "frames_with_annotations": frames_with_annotations,
        "rows_rendered": rows_rendered,
        "mot_frames": len(rows_by_frame),
        "fps": resolved_fps,
        "source_fps": capture_fps,
        "label_mode": label_mode,
        "codec": codec,
    }


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
