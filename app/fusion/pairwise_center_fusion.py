from __future__ import annotations

import math
from typing import Mapping

from fusion.schemas import StreamTrackObservation


def _l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PairwiseNormalizedCenterFusion:
    """
    Two-camera fusion: associate local tracks across a reference and a second view each frame.

    Matching strategies:
    - normalized_center: greedy min distance in normalized image space (per-camera resolution).
    - rank_x: when both sides have the same detection count, pair by ascending center-x order
      (works well for synchronized maritime / Unity views with similar left-to-right ordering).
    - auto: rank_x when counts match and both non-empty; otherwise normalized_center.
    """

    def __init__(
        self,
        reference_camera_id: str,
        other_camera_id: str,
        max_center_distance_norm: float = 0.55,
        match_mode: str = "auto",
    ) -> None:
        if match_mode not in {"auto", "normalized_center", "rank_x"}:
            raise ValueError(f"match_mode must be auto|normalized_center|rank_x, got {match_mode!r}")
        self._reference_camera_id = reference_camera_id
        self._other_camera_id = other_camera_id
        self._max_d = max_center_distance_norm
        self._match_mode = match_mode
        self._local_key_to_global: dict[tuple[str, int], int] = {}
        self._next_global = 1

    def reset(self) -> None:
        self._local_key_to_global.clear()
        self._next_global = 1

    def _alloc_global(self) -> int:
        gid = self._next_global
        self._next_global += 1
        return gid

    def _ensure_global(self, camera_id: str, local_id: int) -> int:
        key = (camera_id, local_id)
        if key not in self._local_key_to_global:
            self._local_key_to_global[key] = self._alloc_global()
        return self._local_key_to_global[key]

    def _resolve_pair_globals(self, g_ref: int | None, g_other: int | None) -> int:
        # Reference camera wins on conflict so IDs stay stable for the primary view.
        if g_ref is not None:
            return g_ref
        if g_other is not None:
            return g_other
        return self._alloc_global()

    @staticmethod
    def _pairs_rank_x(
        ref_list: list[StreamTrackObservation],
        other_list: list[StreamTrackObservation],
    ) -> list[tuple[int, int]]:
        ref_ord = sorted(range(len(ref_list)), key=lambda i: ref_list[i].center_xy()[0])
        oth_ord = sorted(range(len(other_list)), key=lambda j: other_list[j].center_xy()[0])
        return list(zip(ref_ord, oth_ord))

    def _greedy_min_cost_matching(
        self,
        ref_obs: list[StreamTrackObservation],
        other_obs: list[StreamTrackObservation],
        wh_ref: tuple[int, int],
        wh_other: tuple[int, int],
    ) -> list[tuple[int, int]]:
        wr, hr = wh_ref
        wo, ho = wh_other
        pairs: list[tuple[float, int, int]] = []
        for i, a in enumerate(ref_obs):
            ca = a.normalized_center(wr, hr)
            for j, b in enumerate(other_obs):
                cb = b.normalized_center(wo, ho)
                d = _l2(ca, cb)
                if d <= self._max_d:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda t: t[0])
        used_i: set[int] = set()
        used_j: set[int] = set()
        chosen: list[tuple[int, int]] = []
        for _, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            chosen.append((i, j))
        return chosen

    def _resolve_wh(
        self,
        image_width: int,
        image_height: int,
        image_size_by_camera: Mapping[str, tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        default = (max(1, int(image_width)), max(1, int(image_height)))
        if not image_size_by_camera:
            return default, default
        wr, hr = image_size_by_camera.get(self._reference_camera_id, default)
        wo, ho = image_size_by_camera.get(self._other_camera_id, default)
        return (max(1, int(wr)), max(1, int(hr))), (max(1, int(wo)), max(1, int(ho)))

    def _match_frame(
        self,
        ref_list: list[StreamTrackObservation],
        other_list: list[StreamTrackObservation],
        wh_ref: tuple[int, int],
        wh_other: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if self._match_mode == "rank_x":
            if len(ref_list) == len(other_list) and len(ref_list) > 0:
                return self._pairs_rank_x(ref_list, other_list)
            return self._greedy_min_cost_matching(ref_list, other_list, wh_ref, wh_other)
        if self._match_mode == "normalized_center":
            return self._greedy_min_cost_matching(ref_list, other_list, wh_ref, wh_other)
        # auto
        if len(ref_list) == len(other_list) and len(ref_list) > 0:
            return self._pairs_rank_x(ref_list, other_list)
        return self._greedy_min_cost_matching(ref_list, other_list, wh_ref, wh_other)

    def fuse_frame(
        self,
        frame_index: int,
        observations_by_camera: Mapping[str, list[StreamTrackObservation]],
        image_width: int,
        image_height: int,
        image_size_by_camera: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[tuple[str, int], int]:
        _ = frame_index
        ref_list = list(observations_by_camera.get(self._reference_camera_id, []))
        other_list = list(observations_by_camera.get(self._other_camera_id, []))
        wh_ref, wh_other = self._resolve_wh(image_width, image_height, image_size_by_camera)

        matches = self._match_frame(ref_list, other_list, wh_ref, wh_other)
        matched_ref_idx = {i for i, _ in matches}
        matched_other_idx = {j for _, j in matches}

        for i, j in matches:
            rloc = ref_list[i].local_track_id
            oloc = other_list[j].local_track_id
            g_ref = self._local_key_to_global.get((self._reference_camera_id, rloc))
            g_other = self._local_key_to_global.get((self._other_camera_id, oloc))
            g = self._resolve_pair_globals(g_ref, g_other)
            self._local_key_to_global[(self._reference_camera_id, rloc)] = g
            self._local_key_to_global[(self._other_camera_id, oloc)] = g

        for idx, obs in enumerate(ref_list):
            if idx in matched_ref_idx:
                continue
            self._ensure_global(self._reference_camera_id, obs.local_track_id)

        for idx, obs in enumerate(other_list):
            if idx in matched_other_idx:
                continue
            self._ensure_global(self._other_camera_id, obs.local_track_id)

        out: dict[tuple[str, int], int] = {}
        for cam, obs_list in observations_by_camera.items():
            for obs in obs_list:
                out[(cam, obs.local_track_id)] = self._ensure_global(cam, obs.local_track_id)
        return out

    def export_key_to_global(self) -> dict[str, int]:
        """Human-readable map 'camera_id|local_track_id' -> global_track_id after processing."""
        return {f"{cam}|{lid}": gid for (cam, lid), gid in sorted(self._local_key_to_global.items())}
