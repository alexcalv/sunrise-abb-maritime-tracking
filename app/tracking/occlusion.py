from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class TrackVisibilityState(str, Enum):
    VISIBLE = "visible"
    TEMPORARILY_LOST = "temporarily_lost"
    OCCLUDED = "occluded"
    REAPPEARED = "reappeared"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class OcclusionPrediction:
    """Display/report prediction for a raw track that temporarily disappeared."""

    track_id: int
    frame: int
    predicted_x: float
    predicted_y: float
    uncertainty_x: float
    uncertainty_y: float
    uncertainty_radius: float
    gap_frames: int
    source: str
    reason: str
    canonical_id: int | None = None
    last_visible_frame: int | None = None
    visibility_state: TrackVisibilityState = TrackVisibilityState.TEMPORARILY_LOST

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["visibility_state"] = self.visibility_state.value
        return payload
