from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from fusion.schemas import StreamTrackObservation


@runtime_checkable
class GlobalTrackFusion(Protocol):
    """Abstract fusion entry point: multiple camera streams in, consistent global IDs out."""

    def reset(self) -> None:
        """Clear all internal association state (new sequence or new run)."""

    def fuse_frame(
        self,
        frame_index: int,
        observations_by_camera: Mapping[str, list[StreamTrackObservation]],
        image_width: int,
        image_height: int,
    ) -> dict[tuple[str, int], int]:
        """
        Consume all observations for one synchronized frame index.

        Returns a map (camera_id, local_track_id) -> global_track_id for every observation
        passed in this call. Implementations may update durable mappings for future frames.
        """
