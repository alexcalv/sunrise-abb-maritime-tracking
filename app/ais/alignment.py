from __future__ import annotations

import math
from typing import Sequence

from ais.schemas import AisAlignedClip, AisAlignedVesselFrame, AisFix


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _timestamp_to_frame(timestamp_ms: int, *, video_fps: float, time_offset_ms: int) -> int:
    adjusted = timestamp_ms - int(time_offset_ms)
    if adjusted < 0:
        return 1
    zero_based = int(math.floor((adjusted / 1000.0) * float(video_fps)))
    return max(1, zero_based + 1)


def _frame_to_timestamp_ms(frame: int, *, video_fps: float, time_offset_ms: int) -> int:
    zero_based = max(0, int(frame) - 1)
    return int(time_offset_ms + (zero_based / float(video_fps)) * 1000.0)


def _apply_geo_affine(
    lon: float,
    lat: float,
    affine: Sequence[Sequence[float]],
) -> tuple[float, float]:
    if len(affine) != 2 or len(affine[0]) != 3 or len(affine[1]) != 3:
        raise ValueError("geo_affine must be 2x3 [[a,b,c],[d,e,f]] mapping lon,lat -> x,y")
    x = float(affine[0][0]) * lon + float(affine[0][1]) * lat + float(affine[0][2])
    y = float(affine[1][0]) * lon + float(affine[1][1]) * lat + float(affine[1][2])
    return x, y


def _fix_to_pixel(fix: AisFix, geo_affine: Sequence[Sequence[float]] | None) -> tuple[float, float] | None:
    if fix.has_pixel():
        assert fix.pixel_x is not None and fix.pixel_y is not None
        return fix.pixel_x, fix.pixel_y
    if fix.has_geo() and geo_affine is not None:
        assert fix.longitude_deg is not None and fix.latitude_deg is not None
        return _apply_geo_affine(fix.longitude_deg, fix.latitude_deg, geo_affine)
    return None


def _interpolate_fixes(
    fixes: list[AisFix],
    timestamp_ms: int,
    geo_affine: Sequence[Sequence[float]] | None,
) -> tuple[float, float] | None:
    if not fixes:
        return None
    if len(fixes) == 1:
        return _fix_to_pixel(fixes[0], geo_affine)

    if timestamp_ms <= fixes[0].timestamp_ms:
        return _fix_to_pixel(fixes[0], geo_affine)
    if timestamp_ms >= fixes[-1].timestamp_ms:
        return _fix_to_pixel(fixes[-1], geo_affine)

    for left, right in zip(fixes, fixes[1:]):
        if left.timestamp_ms <= timestamp_ms <= right.timestamp_ms:
            if right.timestamp_ms == left.timestamp_ms:
                return _fix_to_pixel(left, geo_affine)
            t = (timestamp_ms - left.timestamp_ms) / float(right.timestamp_ms - left.timestamp_ms)
            left_px = _fix_to_pixel(left, geo_affine)
            right_px = _fix_to_pixel(right, geo_affine)
            if left_px is None or right_px is None:
                return left_px or right_px
            return _lerp(left_px[0], right_px[0], t), _lerp(left_px[1], right_px[1], t)
    return None


def build_aligned_clip(
    fixes_by_mmsi: dict[int, list[AisFix]],
    *,
    max_frame: int,
    video_fps: float,
    time_offset_ms: int = 0,
    geo_affine: Sequence[Sequence[float]] | None = None,
) -> AisAlignedClip:
    if max_frame < 1:
        max_frame = 1
    if video_fps <= 0:
        raise ValueError("video_fps must be positive for AIS alignment")

    frames: dict[int, dict[int, AisAlignedVesselFrame]] = {}
    for frame in range(1, max_frame + 1):
        ts = _frame_to_timestamp_ms(frame, video_fps=video_fps, time_offset_ms=time_offset_ms)
        frame_vessels: dict[int, AisAlignedVesselFrame] = {}
        for mmsi, track in fixes_by_mmsi.items():
            px = _interpolate_fixes(track, ts, geo_affine)
            if px is None:
                continue
            interpolated = not any(f.timestamp_ms == ts for f in track)
            frame_vessels[mmsi] = AisAlignedVesselFrame(
                mmsi=mmsi,
                pixel_x=px[0],
                pixel_y=px[1],
                interpolated=interpolated,
            )
        if frame_vessels:
            frames[frame] = frame_vessels

    return AisAlignedClip(
        frames=frames,
        max_frame=max_frame,
        video_fps=float(video_fps),
        time_offset_ms=int(time_offset_ms),
    )
