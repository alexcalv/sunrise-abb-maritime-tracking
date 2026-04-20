from __future__ import annotations

import json
from pathlib import Path

import cv2

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
