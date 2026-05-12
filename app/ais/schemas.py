from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AisPosition:
    """One AIS-derived fix (vessel identity + WGS-84 position)."""

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
class AisTrackLayer:
    """
    AIS observations indexed by video frame (1-based, aligned with MOT frame column).

    The layer is optional: consumers must treat missing frames as no AIS data.
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
