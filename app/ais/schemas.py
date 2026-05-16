from __future__ import annotations

from dataclasses import dataclass


AIS_NEUTRAL_SCORE = 0.5


@dataclass(frozen=True)
class AisFix:
    """Single AIS observation normalised to MMSI + time (+ position)."""

    mmsi: int
    timestamp_ms: int
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    pixel_x: float | None = None
    pixel_y: float | None = None
    sog_knots: float | None = None
    cog_deg: float | None = None

    def has_pixel(self) -> bool:
        return self.pixel_x is not None and self.pixel_y is not None

    def has_geo(self) -> bool:
        return self.latitude_deg is not None and self.longitude_deg is not None


@dataclass(frozen=True)
class AisPosition:
    """Legacy pixel/geo fix used in frame-indexed sidecars (kept for track CLI sidecar)."""

    mmsi: int
    latitude_deg: float
    longitude_deg: float
    sog_knots: float | None = None
    cog_deg: float | None = None

    def to_json_dict(self) -> dict:
        d: dict = {
            "mmsi": self.mmsi,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
        }
        if self.sog_knots is not None:
            d["sog_knots"] = self.sog_knots
        if self.cog_deg is not None:
            d["cog_deg"] = self.cog_deg
        return d


@dataclass(frozen=True)
class AisAlignedVesselFrame:
    """AIS position for one vessel on one video frame (pixel space when available)."""

    mmsi: int
    pixel_x: float
    pixel_y: float
    interpolated: bool = False


@dataclass
class AisAlignedClip:
    """Per-frame AIS targets for scoring (frame index 1-based, MOT convention)."""

    frames: dict[int, dict[int, AisAlignedVesselFrame]]
    max_frame: int
    video_fps: float
    time_offset_ms: int

    def vessels_at(self, frame: int) -> dict[int, AisAlignedVesselFrame]:
        return self.frames.get(frame, {})

    def vessel_at(self, frame: int, mmsi: int) -> AisAlignedVesselFrame | None:
        return self.frames.get(frame, {}).get(mmsi)


@dataclass(frozen=True)
class AisTrackLayer:
    """
    Frame-indexed AIS layer for optional track/fusion sidecars (legacy export shape).
    """

    version: int
    sync: dict
    positions_by_frame: dict[int, tuple[AisPosition, ...]]

    def max_frame(self) -> int | None:
        if not self.positions_by_frame:
            return None
        return max(self.positions_by_frame)

    def total_fixes(self) -> int:
        return sum(len(v) for v in self.positions_by_frame.values())
