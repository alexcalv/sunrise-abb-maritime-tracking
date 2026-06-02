"""MOT data loading for the single-camera dashboard.

Reuses the team's existing parser in ``app/evaluation/mot.py`` so the dashboard
stays consistent with the rest of the pipeline. The module is loaded by file
path to avoid triggering heavy package ``__init__`` side effects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOT_MODULE_PATH = REPO_ROOT / "app" / "evaluation" / "mot.py"


def _load_team_mot_module():
    spec = importlib.util.spec_from_file_location("sunrise_mot", _MOT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load team MOT parser at {_MOT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_team_mot = _load_team_mot_module()
load_mot_rows = _team_mot.load_mot_rows
load_mot_frames = _team_mot.load_mot_frames


def first_seen_frames(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Map each track id to the first frame it appears in."""
    first: dict[int, int] = {}
    for row in rows:
        tid = row["id"]
        if tid not in first or row["frame"] < first[tid]:
            first[tid] = row["frame"]
    return first


def frame_bounds(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (min_frame, max_frame) across all rows; (0, 0) if empty."""
    if not rows:
        return (0, 0)
    frames = [row["frame"] for row in rows]
    return (min(frames), max(frames))
