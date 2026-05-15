from __future__ import annotations

from bisect import bisect_left

from .parser import parse_timestamp
from .types import AisConfig, AisFrameState, AisTrack


def project_lonlat_to_xy(lon: float, lat: float, affine_matrix: list[list[float]] | None) -> tuple[float, float] | None:
    if affine_matrix is None:
        return None
    if len(affine_matrix) == 2 and all(len(row) == 3 for row in affine_matrix):
        x = (affine_matrix[0][0] * lon) + (affine_matrix[0][1] * lat) + affine_matrix[0][2]
        y = (affine_matrix[1][0] * lon) + (affine_matrix[1][1] * lat) + affine_matrix[1][2]
        return float(x), float(y)
    if len(affine_matrix) == 3 and all(len(row) == 3 for row in affine_matrix):
        denom = (affine_matrix[2][0] * lon) + (affine_matrix[2][1] * lat) + affine_matrix[2][2]
        if abs(denom) < 1e-9:
            return None
        x = ((affine_matrix[0][0] * lon) + (affine_matrix[0][1] * lat) + affine_matrix[0][2]) / denom
        y = ((affine_matrix[1][0] * lon) + (affine_matrix[1][1] * lat) + affine_matrix[1][2]) / denom
        return float(x), float(y)
    return None


def _lerp(left: float | None, right: float | None, ratio: float) -> float | None:
    if left is None or right is None:
        return left if ratio <= 0.5 else right
    return float(left) + ((float(right) - float(left)) * float(ratio))


def _record_xy(record: object, config: AisConfig) -> tuple[float | None, float | None]:
    x = getattr(record, "x", None)
    y = getattr(record, "y", None)
    if x is not None and y is not None:
        return float(x), float(y)
    lon = getattr(record, "lon", None)
    lat = getattr(record, "lat", None)
    if lon is not None and lat is not None and bool(config.affine_enabled):
        projected = project_lonlat_to_xy(float(lon), float(lat), config.affine_matrix)
        if projected is not None:
            return projected
    return None, None


def _interpolate(track: AisTrack, timestamp: float, config: AisConfig) -> AisFrameState | None:
    records = track.records
    if not records:
        return None
    times = [float(record.timestamp or 0.0) for record in records]
    index = bisect_left(times, timestamp)
    if index == 0:
        nearest = records[0]
        if abs(float(nearest.timestamp or 0.0) - timestamp) > float(config.max_time_gap_seconds):
            return None
        x, y = _record_xy(nearest, config)
        return AisFrameState(
            frame=0,
            timestamp=timestamp,
            mmsi=track.mmsi,
            x=x,
            y=y,
            lat=nearest.lat,
            lon=nearest.lon,
            sog=nearest.sog,
            cog=nearest.cog,
            heading=nearest.heading,
            interpolated=False,
            position_available=x is not None and y is not None,
            reason="nearest_first_record",
        )
    if index >= len(records):
        nearest = records[-1]
        if abs(timestamp - float(nearest.timestamp or 0.0)) > float(config.max_time_gap_seconds):
            return None
        x, y = _record_xy(nearest, config)
        return AisFrameState(
            frame=0,
            timestamp=timestamp,
            mmsi=track.mmsi,
            x=x,
            y=y,
            lat=nearest.lat,
            lon=nearest.lon,
            sog=nearest.sog,
            cog=nearest.cog,
            heading=nearest.heading,
            interpolated=False,
            position_available=x is not None and y is not None,
            reason="nearest_last_record",
        )

    left = records[index - 1]
    right = records[index]
    left_time = float(left.timestamp or 0.0)
    right_time = float(right.timestamp or 0.0)
    if min(abs(timestamp - left_time), abs(right_time - timestamp)) > float(config.max_time_gap_seconds):
        return None
    span = max(right_time - left_time, 1e-9)
    ratio = (timestamp - left_time) / span
    left_x, left_y = _record_xy(left, config)
    right_x, right_y = _record_xy(right, config)
    x = _lerp(left_x, right_x, ratio)
    y = _lerp(left_y, right_y, ratio)
    return AisFrameState(
        frame=0,
        timestamp=timestamp,
        mmsi=track.mmsi,
        x=x,
        y=y,
        lat=_lerp(left.lat, right.lat, ratio),
        lon=_lerp(left.lon, right.lon, ratio),
        sog=_lerp(left.sog, right.sog, ratio),
        cog=_lerp(left.cog, right.cog, ratio),
        heading=_lerp(left.heading, right.heading, ratio),
        interpolated=True,
        position_available=x is not None and y is not None,
        reason="interpolated",
    )


def align_ais_tracks_to_frames(
    ais_tracks: dict[str, AisTrack],
    frame_count: int,
    fps: float | None,
    video_start_time: str | float | int | None,
    config: AisConfig,
) -> dict[int, list[AisFrameState]]:
    if not ais_tracks or frame_count <= 0:
        return {}
    resolved_fps = float(fps or config.fps or 30.0)
    if resolved_fps <= 0:
        resolved_fps = 30.0

    start_time = parse_timestamp(video_start_time if video_start_time is not None else config.video_start_time)
    if start_time is None:
        timestamps = [
            float(record.timestamp or 0.0)
            for track in ais_tracks.values()
            for record in track.records
            if record.timestamp is not None
        ]
        start_time = min(timestamps) if timestamps else 0.0

    aligned: dict[int, list[AisFrameState]] = {}
    for frame in range(1, int(frame_count) + 1):
        timestamp = float(start_time) + ((frame - 1) / resolved_fps)
        states: list[AisFrameState] = []
        for track in ais_tracks.values():
            state = _interpolate(track, timestamp, config)
            if state is not None:
                states.append(
                    AisFrameState(
                        **{
                            **state.to_dict(),
                            "frame": int(frame),
                        }
                    )
                )
        if states:
            aligned[frame] = states
    return aligned
