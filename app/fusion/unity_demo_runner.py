"""Deprecated module name: use fusion.multi_camera_runner and fusion.multi_camera_job."""

from __future__ import annotations

from fusion.multi_camera_runner import (
    default_model_path,
    fuse_two_mot_files_to_disk,
    run_dual_camera_track_and_fuse,
)

# Backward-compatible name used in older snippets.
run_unity_dual_camera_track_and_fuse = run_dual_camera_track_and_fuse

__all__ = [
    "default_model_path",
    "fuse_two_mot_files_to_disk",
    "run_dual_camera_track_and_fuse",
    "run_unity_dual_camera_track_and_fuse",
]
