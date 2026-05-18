__all__ = ["OnlineLiveReIDMapper", "stitch_tracks", "stitch_batch", "replay_stitch_tracks"]
"""Offline track stitching package ."""


def __getattr__(name):
    if name == "OnlineLiveReIDMapper":
        from stitching.online_live_reid import OnlineLiveReIDMapper

        return OnlineLiveReIDMapper
    if name == "stitch_tracks":
        from stitching.runner import stitch_tracks

        return stitch_tracks
    if name == "stitch_batch":
        from stitching.batch import stitch_batch

        return stitch_batch
    if name == "replay_stitch_tracks":
        from stitching.replay import replay_stitch_tracks

        return replay_stitch_tracks
    raise AttributeError(f"module 'stitching' has no attribute {name!r}")
