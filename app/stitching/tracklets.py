from __future__ import annotations

from collections import defaultdict
from typing import Any

from stitching.schemas import TrackObservation, Tracklet


def _round_list(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _center_from_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x + (w / 2.0), y + (h / 2.0)]


def _velocity_from_rows(rows: list[dict[str, Any]], leading: bool) -> list[float]:
    if len(rows) < 2:
        return [0.0, 0.0]

    first = rows[0] if leading else rows[-2]
    second = rows[1] if leading else rows[-1]

    frame_delta = second["frame"] - first["frame"]
    if frame_delta <= 0:
        return [0.0, 0.0]

    c1 = _center_from_bbox(first["bbox"])
    c2 = _center_from_bbox(second["bbox"])
    return _round_list(
        [
            (c2[0] - c1[0]) / frame_delta,
            (c2[1] - c1[1]) / frame_delta,
        ]
    )


def _build_observations(rows: list[dict[str, Any]], sample_frames: int, from_start: bool) -> list[TrackObservation]:
    if sample_frames <= 0 or not rows:
        return []

    selected_rows = rows[:sample_frames] if from_start else rows[-sample_frames:]
    observations: list[TrackObservation] = []
    for row in selected_rows:
        bbox = [float(value) for value in row["bbox"]]
        observations.append(
            TrackObservation(
                frame=int(row["frame"]),
                bbox=_round_list(bbox),
                confidence=round(float(row["confidence"]), 6),
                center=_round_list(_center_from_bbox(bbox)),
            )
        )
    return observations


def _build_tracklet(
    sequence_name: str,
    source_track_id: int,
    segment_index: int,
    segment_rows: list[dict[str, Any]],
    frame_gaps: list[int],
    observation_samples: int,
) -> Tracklet:
    frames = [row["frame"] for row in segment_rows]
    boxes = [row["bbox"] for row in segment_rows]
    confidences = [float(row["confidence"]) for row in segment_rows]
    class_ids = sorted({int(row["class_id"]) for row in segment_rows})

    mean_bbox = [
        sum(box[axis] for box in boxes) / len(boxes)
        for axis in range(4)
    ]
    areas = [float(box[2]) * float(box[3]) for box in boxes]
    aspect_ratios = [
        (float(box[2]) / float(box[3])) if float(box[3]) > 0 else 0.0
        for box in boxes
    ]

    return Tracklet(
        sequence_name=sequence_name,
        tracklet_id=f"{sequence_name}:{source_track_id}:{segment_index}",
        source_track_id=source_track_id,
        canonical_track_id=source_track_id,
        segment_index=segment_index,
        frame_start=min(frames),
        frame_end=max(frames),
        num_rows=len(segment_rows),
        num_frames=len(set(frames)),
        class_ids=class_ids,
        mean_confidence=round(sum(confidences) / len(confidences), 6),
        first_bbox=_round_list(list(boxes[0])),
        last_bbox=_round_list(list(boxes[-1])),
        mean_bbox=_round_list(mean_bbox),
        mean_area=round(sum(areas) / len(areas), 6),
        mean_aspect_ratio=round(sum(aspect_ratios) / len(aspect_ratios), 6),
        frame_gaps=frame_gaps,
        start_center=_round_list(_center_from_bbox(list(boxes[0]))),
        end_center=_round_list(_center_from_bbox(list(boxes[-1]))),
        start_velocity=_velocity_from_rows(segment_rows, leading=True),
        end_velocity=_velocity_from_rows(segment_rows, leading=False),
        head_observations=_build_observations(segment_rows, observation_samples, from_start=True),
        tail_observations=_build_observations(segment_rows, observation_samples, from_start=False),
    )


def build_tracklets(
    sequence_name: str,
    rows: list[dict[str, Any]],
    split_gap: int = 1,
    min_length: int = 1,
    observation_samples: int = 0,
) -> list[Tracklet]:
    rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_track[int(row["id"])].append(row)

    tracklets: list[Tracklet] = []
    for source_track_id, track_rows in sorted(rows_by_track.items()):
        sorted_rows = sorted(track_rows, key=lambda row: (row["frame"], row["id"]))
        segment_rows: list[dict[str, Any]] = []
        frame_gaps: list[int] = []
        segment_index = 0
        previous_frame: int | None = None

        for row in sorted_rows:
            current_frame = int(row["frame"])
            if previous_frame is not None:
                missing_frames = current_frame - previous_frame - 1
                if missing_frames > split_gap:
                    if len(segment_rows) >= min_length:
                        tracklets.append(
                            _build_tracklet(
                                sequence_name=sequence_name,
                                source_track_id=source_track_id,
                                segment_index=segment_index,
                                segment_rows=segment_rows,
                                frame_gaps=frame_gaps,
                                observation_samples=observation_samples,
                            )
                        )
                    segment_index += 1
                    segment_rows = []
                    frame_gaps = []
                elif missing_frames > 0:
                    frame_gaps.append(missing_frames)

            segment_rows.append(row)
            previous_frame = current_frame

        if len(segment_rows) >= min_length:
            tracklets.append(
                _build_tracklet(
                    sequence_name=sequence_name,
                    source_track_id=source_track_id,
                    segment_index=segment_index,
                    segment_rows=segment_rows,
                    frame_gaps=frame_gaps,
                    observation_samples=observation_samples,
                )
            )

    return sorted(tracklets, key=lambda tracklet: (tracklet.frame_start, tracklet.source_track_id, tracklet.segment_index))
