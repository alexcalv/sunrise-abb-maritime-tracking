"""Smoke tests for the dashboard data layer (no Streamlit runtime needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.occlusion_detector import detect_occlusions, format_timecode
from data.overrides import apply_overrides
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
