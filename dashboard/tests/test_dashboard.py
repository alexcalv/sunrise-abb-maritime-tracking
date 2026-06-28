"""Smoke tests for the dashboard data layer (no Streamlit runtime needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mot_export import rows_to_mot_text
from data.occlusion_detector import detect_occlusions, format_timecode
from data.occlusion_export import occlusions_to_csv
from data.overrides import apply_overrides
from data.stats import occlusion_counts, sequence_summary
from data.timeline import build_presence_segments
from data.trails import build_trails
from utils.bbox_mapper import find_ship_at_point, point_in_bbox


def _row(frame: int, tid: int, bbox: list[float]) -> dict:
    return {"frame": frame, "id": tid, "bbox": bbox, "confidence": 1.0, "class_id": 0, "visibility": 1.0}


def test_point_in_bbox():
    assert point_in_bbox(15, 25, [10, 20, 10, 10]) is True
    assert point_in_bbox(5, 25, [10, 20, 10, 10]) is False


def test_find_smallest_box_wins():
    rows = [_row(1, 1, [0, 0, 100, 100]), _row(1, 2, [40, 40, 20, 20])]
    hit = find_ship_at_point(50, 50, rows)
    assert hit is not None and hit["id"] == 2


def test_detect_occlusions_gap():
    rows = [_row(1, 5, [0, 0, 10, 10]), _row(2, 5, [0, 0, 10, 10]), _row(6, 5, [0, 0, 10, 10])]
    events = detect_occlusions(rows, fps=25.0)
    assert len(events) == 1
    assert events[0]["gap_start"] == 2 and events[0]["gap_end"] == 6
    assert events[0]["gap_frames"] == 3


def test_detect_occlusions_ignores_negative_ids():
    rows = [_row(1, -1, [0, 0, 10, 10]), _row(6, -1, [0, 0, 10, 10])]
    assert detect_occlusions(rows, fps=25.0) == []


def test_apply_overrides_from_frame(tmp_path, monkeypatch):
    import data.overrides as ov

    monkeypatch.setattr(ov, "_STORE_PATH", tmp_path / "overrides.json")
    rows = [_row(1, 5, [0, 0, 10, 10]), _row(10, 5, [0, 0, 10, 10])]
    ov.add_override("run", old_id=5, new_id=3, from_frame=10)
    patched = ov.apply_overrides(rows, "run")
    assert patched[0]["id"] == 5  # before from_frame: untouched
    assert patched[1]["id"] == 3  # at/after from_frame: corrected


def test_format_timecode():
    assert format_timecode(0, 25.0) == "00:00:00"
    assert format_timecode(1500, 25.0) == "00:01:00"


def test_build_trails_orders_and_windows():
    rows = [
        _row(1, 7, [0, 0, 10, 10]),
        _row(3, 7, [10, 10, 10, 10]),
        _row(5, 7, [20, 20, 10, 10]),
    ]
    trails = build_trails(rows, current_frame=5, length=3)
    # window is (5-3, 5] -> frames 3 and 5; centres are bbox centre points.
    assert trails[7] == [(15.0, 15.0), (25.0, 25.0)]


def test_build_trails_skips_negative_ids():
    rows = [_row(1, -1, [0, 0, 10, 10]), _row(2, -1, [0, 0, 10, 10])]
    assert build_trails(rows, current_frame=2, length=10) == {}


def test_sequence_summary_longest_track():
    rows = [
        _row(1, 1, [0, 0, 10, 10]),
        _row(2, 1, [0, 0, 10, 10]),
        _row(1, 2, [0, 0, 10, 10]),
        _row(51, 2, [0, 0, 10, 10]),
    ]
    summary = sequence_summary(rows, fps=25.0)
    assert summary["total_ships"] == 2
    assert summary["longest_id"] == 2
    assert summary["longest_seconds"] == 50 / 25.0


def test_occlusion_counts():
    events = [{"track_id": 1}, {"track_id": 1}, {"track_id": 2}]
    assert occlusion_counts(events) == {1: 2, 2: 1}


def test_rows_to_mot_text_roundtrip():
    text = rows_to_mot_text([_row(2, 5, [1, 2, 3, 4]), _row(1, 5, [1, 2, 3, 4])])
    lines = text.strip().splitlines()
    # sorted by (frame, id): frame 1 first.
    assert lines[0].startswith("1,5,1.0000,2.0000,3.0000,4.0000,")
    assert lines[1].startswith("2,5,")


def test_occlusions_to_csv_header_and_rows():
    events = detect_occlusions(
        [_row(1, 5, [0, 0, 10, 10]), _row(6, 5, [0, 0, 10, 10])], fps=25.0
    )
    csv_text = occlusions_to_csv(events, fps=25.0)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "track_id,gap_start,gap_end,gap_frames,gap_seconds,timecode"
    assert lines[1].startswith("5,1,6,4,")


def test_build_presence_segments_splits_on_gap():
    rows = [
        _row(1, 5, [0, 0, 10, 10]),
        _row(2, 5, [0, 0, 10, 10]),
        _row(6, 5, [0, 0, 10, 10]),
        _row(7, 5, [0, 0, 10, 10]),
    ]
    segments = build_presence_segments(rows)
    assert segments[5] == [(1, 2), (6, 7)]


def test_build_presence_segments_skips_negative_ids():
    rows = [_row(1, -1, [0, 0, 10, 10]), _row(2, -1, [0, 0, 10, 10])]
    assert build_presence_segments(rows) == {}
