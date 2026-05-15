from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from stitching.live_reid import (
    LIVE_REID_VERSION,
    _allocate_canonical_id,
    _annotate_live_appearance,
    _apply_chain_control,
    _apply_short_gap_gate,
    _build_appearance_runtime,
    _build_observation,
    _canonical_merge_meta,
    _close_appearance_runtime,
    _colreg_scoring_summary,
    _configure_ais_options,
    _configure_colreg_options,
    _decision_latency_summary,
    _evaluate_live_candidate,
    _finalize_lost_embedding,
    _make_confirmed_state,
    _make_pending_state,
    _mark_colreg_acceptance,
    _annotate_ais_diagnostics,
    _pending_summary,
    _rank_eligible_candidates,
    _sequence_summary,
    _update_confirmed_state,
    _update_pending_state,
    _use_preview_pending_summary,
)
from stitching.reid_scoring import _appearance_margin_threshold, _round, _winner_margin_threshold
from stitching.runner import DEFAULT_CONFIG_PATH, _load_stitch_config
from stitching.schemas import StitchConfig

IN_LOOP_LIVE_REID_MODE = "in_loop_bounded_latency"


class OnlineLiveReIDMapper:
    """Bounded-latency mapper used by the tracker-loop ReID v1 path.

    The mapper consumes raw tracker detections frame by frame, buffers only the
    configured confirmation window, and emits canonical MOT rows separately
    from raw tracker MOT. Optional AIS/COLREG inputs are reporting-only unless
    an explicit experimental COLREG scoring flag is enabled.
    """

    def __init__(
        self,
        config: StitchConfig | None = None,
        confirmation_observations: int = 10,
        source_video_path: str | None = None,
        config_path: str | None = None,
        sequence_name: str = "sequence",
        safe_mode: bool = True,
        colreg_diagnostics: bool = False,
        colreg_scoring_experiment: bool = False,
        ais_diagnostics: bool = False,
        ais_file: str | None = None,
        ais_video_start_time: str | None = None,
        ais_affine_matrix: str | list[list[float]] | None = None,
        ais_assignments: dict[int, Any] | None = None,
    ) -> None:
        self.config = deepcopy(config) if config is not None else _load_stitch_config(config_path or str(DEFAULT_CONFIG_PATH))
        if safe_mode:
            self.config.matching.chain_control_enabled = True
            self.config.matching.short_gap_gate_enabled = True
        _configure_colreg_options(
            self.config,
            diagnostics_enabled=colreg_diagnostics,
            scoring_experiment=colreg_scoring_experiment,
        )
        _configure_ais_options(
            self.config,
            diagnostics_enabled=ais_diagnostics,
            ais_file=ais_file,
            video_start_time=ais_video_start_time,
            affine_matrix=ais_affine_matrix,
            fps=None,
        )

        self.sequence_name = sequence_name
        self.confirmation_observations = max(1, int(confirmation_observations))
        self.source_video_path = source_video_path
        self.config_path = str(config_path or DEFAULT_CONFIG_PATH)
        self.safe_mode = bool(safe_mode)
        self.head_window = max(1, self.confirmation_observations)
        self.tail_window = max(2, self.confirmation_observations)

        self.appearance_runtime, self.warnings = _build_appearance_runtime(
            sequence_name=sequence_name,
            source_video_path=source_video_path,
            config=self.config,
        )
        self.appearance_ready = self.appearance_runtime is not None

        self.active_tracks: dict[int, dict[str, Any]] = {}
        self.lost_tracks: dict[int, dict[str, Any]] = {}
        self.pending_tracks: dict[int, dict[str, Any]] = {}
        self.source_to_canonical: dict[int, int] = {}
        self.used_canonical_ids: set[int] = set()
        self.canonical_merge_state: dict[int, dict[str, Any]] = {}
        self.next_canonical_id = 1
        self.current_frame = 0
        self.closed = False

        self.rows_in = 0
        self.rows_out = 0
        self.input_track_ids_seen: set[int] = set()
        self.output_track_ids_seen: set[int] = set()
        self.decisions: list[dict[str, Any]] = []
        self.accepted_remaps: list[str] = []
        self.matches_applied = 0
        self.fresh_canonical_ids_assigned = 0
        self.appearance_supported_remap_count = 0
        self.appearance_rejected_count = 0
        self.mapping_records: dict[int, dict[str, Any]] = {}
        self.ais_assignments = ais_assignments or {}

    def update(self, frame_idx: int, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.closed:
            raise RuntimeError("OnlineLiveReIDMapper.update() called after flush().")

        frame_idx = int(frame_idx)
        self.current_frame = max(self.current_frame, frame_idx)
        rows_now = sorted(
            [self._row_from_detection(frame_idx, detection) for detection in detections],
            key=lambda row: (int(row["id"]), float(row["bbox"][0]), float(row["bbox"][1])),
        )
        self.rows_in += len(rows_now)
        current_source_ids = {int(row["id"]) for row in rows_now}
        self.input_track_ids_seen.update(current_source_ids)
        output_rows: list[dict[str, Any]] = []

        for source_id in [source_id for source_id in list(self.pending_tracks) if source_id not in current_source_ids]:
            output_rows.extend(self._finalize_pending(source_id, decision_frame=frame_idx))

        for source_id in [source_id for source_id in list(self.active_tracks) if source_id not in current_source_ids]:
            self.lost_tracks[source_id] = _finalize_lost_embedding(
                self.active_tracks.pop(source_id),
                runtime=self.appearance_runtime,
                config=self.config,
            )

        for source_id in [
            source_id
            for source_id, state in self.lost_tracks.items()
            if max(0, frame_idx - int(state["last_frame"]) - 1) > int(self.config.memory.max_frame_gap)
        ]:
            self.lost_tracks.pop(source_id, None)

        for row in rows_now:
            source_track_id = int(row["id"])
            self.next_canonical_id = max(self.next_canonical_id, source_track_id + 1)

            if source_track_id in self.active_tracks:
                canonical_track_id = int(self.active_tracks[source_track_id]["canonical_track_id"])
                self.active_tracks[source_track_id] = _update_confirmed_state(
                    self.active_tracks[source_track_id],
                    row=row,
                    head_window=self.head_window,
                    tail_window=self.tail_window,
                )
                output_rows.append(self._remap_row(row, canonical_track_id))
                self._update_mapping_last_frame(source_track_id, frame_idx)
                continue

            if source_track_id in self.source_to_canonical:
                canonical_track_id = int(self.source_to_canonical[source_track_id])
                output_rows.append(self._remap_row(row, canonical_track_id))
                self.lost_tracks.pop(source_track_id, None)
                self.active_tracks[source_track_id] = _make_confirmed_state(
                    source_track_id=source_track_id,
                    canonical_track_id=canonical_track_id,
                    observations=[_build_observation(row)],
                    head_window=self.head_window,
                    tail_window=self.tail_window,
                )
                self._update_mapping_last_frame(source_track_id, frame_idx)
                continue

            if source_track_id not in self.pending_tracks:
                self.pending_tracks[source_track_id] = _make_pending_state(source_track_id, row)
            else:
                self.pending_tracks[source_track_id] = _update_pending_state(self.pending_tracks[source_track_id], row)

            if len(self.pending_tracks[source_track_id]["observations"]) >= self.confirmation_observations:
                output_rows.extend(self._finalize_pending(source_track_id, decision_frame=frame_idx))

        self._record_output_rows(output_rows)
        return output_rows

    def flush(self) -> list[dict[str, Any]]:
        if self.closed:
            return []

        output_rows: list[dict[str, Any]] = []
        final_frame = int(self.current_frame)
        for source_id in list(self.pending_tracks):
            output_rows.extend(self._finalize_pending(source_id, decision_frame=final_frame))
        self._record_output_rows(output_rows)
        _close_appearance_runtime(self.appearance_runtime)
        self.closed = True
        return output_rows

    def report(self) -> dict[str, Any]:
        accepted_remap_count = len(self.accepted_remaps)
        rejected_candidate_count = sum(1 for decision in self.decisions if not bool(decision.get("applied")))
        candidate_evaluation_count = len(self.decisions)
        below_score_threshold_count = sum(
            1 for decision in self.decisions if str(decision.get("reason") or "").startswith("below_score_threshold:")
        )
        margin_rejected_count = sum(
            1 for decision in self.decisions if str(decision.get("reason") or "").startswith("margin_too_small:")
        )
        chain_control_checked_count = sum(1 for decision in self.decisions if bool(decision.get("chain_control_checked")))
        chain_control_rejected_count = sum(
            1 for decision in self.decisions if str(decision.get("reason") or "").startswith("chain_control_")
        )
        short_gap_gate_checked_count = sum(1 for decision in self.decisions if bool(decision.get("short_gap_gate_checked")))
        short_gap_gate_rejected_count = sum(
            1 for decision in self.decisions if str(decision.get("reason") or "").startswith("short_gap_gate_")
        )
        sequence_report = {
            "mode": IN_LOOP_LIVE_REID_MODE,
            "version": LIVE_REID_VERSION,
            "sequence_name": self.sequence_name,
            "num_rows_in": self.rows_in,
            "num_rows_out": self.rows_out,
            "unique_track_ids_in": sorted(self.input_track_ids_seen),
            "unique_track_ids_out": sorted(self.output_track_ids_seen),
            "identity_map": {str(source_id): int(canonical_id) for source_id, canonical_id in sorted(self.source_to_canonical.items())},
            "matches_applied": self.matches_applied,
            "accepted_remaps": list(self.accepted_remaps),
            "accepted_remap_count": accepted_remap_count,
            "rejected_candidate_count": rejected_candidate_count,
            "candidate_evaluation_count": candidate_evaluation_count,
            "fresh_canonical_ids_assigned": self.fresh_canonical_ids_assigned,
            "fresh_id_count": self.fresh_canonical_ids_assigned,
            "below_score_threshold_count": below_score_threshold_count,
            "margin_rejected_count": margin_rejected_count,
            "chain_control_checked_count": chain_control_checked_count,
            "chain_control_rejected_count": chain_control_rejected_count,
            "short_gap_gate_checked_count": short_gap_gate_checked_count,
            "short_gap_gate_rejected_count": short_gap_gate_rejected_count,
            "appearance_supported_remaps": self._appearance_supported_remaps(),
            "appearance_supported_remap_count": self.appearance_supported_remap_count,
            "appearance_rejection_count": self.appearance_rejected_count,
            "appearance_rejected_count": self.appearance_rejected_count,
            "appearance": {} if self.appearance_runtime is None else dict(self.appearance_runtime["info"]),
            "appearance_warnings": list(self.warnings),
            "confirmation_observations": self.confirmation_observations,
            "config_path": self.config_path,
            "source_video_path": self.source_video_path,
            "decisions": self.decisions,
            **_decision_latency_summary(self.decisions),
            **_colreg_scoring_summary(self.decisions),
        }
        sequence_report["decision_latency_mean"] = sequence_report["decision_latency_frames_mean"]
        sequence_report["decision_latency_max"] = sequence_report["decision_latency_frames_max"]
        sequence_report["summary"] = _sequence_summary(sequence_report)
        return sequence_report

    def mapping_summary(self) -> list[dict[str, Any]]:
        return [
            dict(record)
            for _, record in sorted(self.mapping_records.items(), key=lambda item: (int(item[1]["first_frame"]), int(item[0])))
        ]

    def _row_from_detection(self, frame_idx: int, detection: dict[str, Any]) -> dict[str, Any]:
        if "bbox" in detection:
            bbox = [float(value) for value in detection["bbox"]]
        else:
            bbox = [
                float(detection["x"]),
                float(detection["y"]),
                float(detection["w"]),
                float(detection["h"]),
            ]
        raw_track_id = int(detection.get("raw_track_id", detection.get("track_id", detection.get("id"))))
        return {
            "frame": int(frame_idx),
            "id": raw_track_id,
            "bbox": bbox,
            "confidence": float(detection.get("confidence", 1.0)),
            "class_id": int(detection.get("class_id", 0)),
            "visibility": float(detection.get("visibility", 1.0)),
        }

    def _finalize_pending(self, source_track_id: int, decision_frame: int) -> list[dict[str, Any]]:
        state = self.pending_tracks.pop(source_track_id, None)
        if state is None:
            return []

        preview_observation_limit = None
        if self.config.appearance.enabled and self.config.appearance.use_for_matching:
            preview_observation_limit = max(1, int(self.config.appearance.sample_frames))
        pending_full = _pending_summary(state, match_observation_limit=None)
        pending_preview = _pending_summary(state, match_observation_limit=preview_observation_limit)

        candidate_decisions: list[dict[str, Any]] = []
        for lost_state in self.lost_tracks.values():
            pending_for_candidate = (
                pending_preview
                if _use_preview_pending_summary(lost_state=lost_state, pending=pending_full, config=self.config)
                else pending_full
            )
            decision = _evaluate_live_candidate(
                lost_state=lost_state,
                pending=pending_for_candidate,
                config=self.config,
                decision_frame=decision_frame,
            )
            _annotate_live_appearance(
                decision=decision,
                lost_state=lost_state,
                pending=pending_preview,
                runtime=self.appearance_runtime,
                config=self.config,
            )
            _annotate_ais_diagnostics(decision, config=self.config, ais_assignments=self.ais_assignments)
            candidate_decisions.append(decision)
        self.decisions.extend(candidate_decisions)

        accepted = False
        canonical_track_id = None
        accepted_decision: dict[str, Any] | None = None
        if candidate_decisions:
            eligible = [decision for decision in candidate_decisions if bool(decision["gating"]["passes_all"])]
            if eligible:
                eligible = _rank_eligible_candidates(eligible, self.config)
                best = eligible[0]
                runner_up = eligible[1] if len(eligible) > 1 else None
                winner_margin_threshold = None
                if float(best["score"] or 0.0) < float(self.config.matching.min_match_score):
                    best["reason"] = (
                        f"below_score_threshold:{float(best['score'] or 0.0):.3f}"
                        f"<{float(self.config.matching.min_match_score):.3f}"
                    )
                else:
                    if runner_up is not None:
                        winner_margin = round(float(best["score"] or 0.0) - float(runner_up["score"] or 0.0), 6)
                        winner_margin_threshold = _winner_margin_threshold(
                            config=self.config,
                            gap_frames=int(best["gap_frames"]),
                        )
                        best["winner_margin"] = _round(winner_margin)
                        best["winner_margin_threshold"] = _round(winner_margin_threshold)
                        if winner_margin < winner_margin_threshold:
                            if bool(best.get("colreg_score_used")) and bool(best.get("colreg_changed_ranking")):
                                best["reason"] = "eligible"
                            else:
                                best["reason"] = f"margin_too_small:{winner_margin:.3f}<{winner_margin_threshold:.3f}"
                            if best["reason"] != "eligible":
                                self._try_appearance_tiebreak(eligible, best, winner_margin_threshold)
                            if any(bool(decision.get("appearance_assisted")) for decision in eligible):
                                best = next(decision for decision in eligible if bool(decision.get("appearance_assisted")))

                    if best["reason"] == "eligible":
                        _apply_chain_control(
                            decision=best,
                            config=self.config,
                            canonical_merge_state=self.canonical_merge_state,
                            winner_margin_threshold=winner_margin_threshold,
                        )
                    if best["reason"] == "eligible":
                        _apply_short_gap_gate(
                            decision=best,
                            config=self.config,
                            canonical_merge_state=self.canonical_merge_state,
                        )
                    if best["reason"] == "eligible":
                        best["applied"] = True
                        best["reason"] = "accepted"
                        if bool(best.get("appearance_assisted")):
                            best["reason"] = "accepted_with_appearance_fallback"
                            self.appearance_supported_remap_count += 1
                        _mark_colreg_acceptance(best)
                        canonical_track_id = int(best["candidate_canonical_track_id"])
                        merge_meta = _canonical_merge_meta(self.canonical_merge_state, canonical_track_id)
                        merge_meta["absorbed_count"] = int(merge_meta.get("absorbed_count") or 0) + 1
                        merge_meta["last_merge_frame"] = int(decision_frame)
                        self.source_to_canonical[source_track_id] = canonical_track_id
                        self.used_canonical_ids.add(canonical_track_id)
                        self.matches_applied += 1
                        self.accepted_remaps.append(f"{source_track_id}->{int(best['candidate_source_track_id'])}")
                        self.lost_tracks.pop(int(best["candidate_source_track_id"]), None)
                        accepted = True
                        accepted_decision = best
                        for decision in eligible[1:]:
                            if decision["reason"] == "eligible":
                                if bool(best.get("appearance_assisted")):
                                    decision["reason"] = (
                                        f"outscored_by_appearance:{source_track_id}->{int(best['candidate_source_track_id'])}"
                                    )
                                else:
                                    decision["reason"] = f"outscored_by:{source_track_id}->{int(best['candidate_source_track_id'])}"

        if not accepted:
            canonical_track_id, self.next_canonical_id = _allocate_canonical_id(
                source_track_id=source_track_id,
                used_canonical_ids=self.used_canonical_ids,
                next_canonical_id=self.next_canonical_id,
            )
            self.source_to_canonical[source_track_id] = canonical_track_id
            _canonical_merge_meta(self.canonical_merge_state, canonical_track_id)
            self.fresh_canonical_ids_assigned += 1

        remapped_rows = [self._remap_row(row, int(canonical_track_id)) for row in state["rows"]]
        self.active_tracks[source_track_id] = _make_confirmed_state(
            source_track_id=source_track_id,
            canonical_track_id=int(canonical_track_id),
            observations=state["observations"],
            head_window=self.head_window,
            tail_window=self.tail_window,
        )
        self._record_mapping(source_track_id, int(canonical_track_id), state, accepted_decision, decision_frame)
        return remapped_rows

    def _try_appearance_tiebreak(
        self,
        eligible: list[dict[str, Any]],
        best: dict[str, Any],
        winner_margin_threshold: float,
    ) -> None:
        if self.appearance_runtime is None:
            return
        ambiguous_candidates = [
            decision
            for decision in eligible
            if (float(best["score"] or 0.0) - float(decision["score"] or 0.0)) < winner_margin_threshold
        ]
        appearance_candidates = [
            decision
            for decision in ambiguous_candidates
            if bool(decision.get("appearance_supported"))
            and float(decision.get("score") or 0.0) >= float(self.config.matching.min_match_score)
        ]
        if not appearance_candidates:
            self.appearance_rejected_count += 1
            return

        appearance_candidates.sort(
            key=lambda decision: (
                -float(decision.get("appearance_similarity") or float("-inf")),
                -float(decision.get("score") or 0.0),
                float(decision.get("motion_distance") or float("inf")),
            )
        )
        appearance_best = appearance_candidates[0]
        appearance_runner_up = appearance_candidates[1] if len(appearance_candidates) > 1 else None
        appearance_margin_threshold = _appearance_margin_threshold(self.config)
        appearance_margin = None
        if appearance_runner_up is not None:
            appearance_margin = round(
                float(appearance_best.get("appearance_similarity") or 0.0)
                - float(appearance_runner_up.get("appearance_similarity") or 0.0),
                6,
            )
        appearance_best["appearance_margin_threshold"] = _round(appearance_margin_threshold)
        appearance_best["appearance_margin"] = _round(appearance_margin)
        if appearance_margin is None or appearance_margin >= appearance_margin_threshold:
            appearance_best["appearance_assisted"] = True
            appearance_best["reason"] = "eligible"
        else:
            self.appearance_rejected_count += 1
            best["appearance_margin_threshold"] = _round(appearance_margin_threshold)
            best["appearance_margin"] = _round(appearance_margin)
            best["reason"] = f"appearance_margin_too_small:{appearance_margin:.3f}<{appearance_margin_threshold:.3f}"

    def _remap_row(self, row: dict[str, Any], canonical_track_id: int) -> dict[str, Any]:
        remapped_row = dict(row)
        remapped_row["raw_track_id"] = int(row["id"])
        remapped_row["canonical_track_id"] = int(canonical_track_id)
        remapped_row["id"] = int(canonical_track_id)
        return remapped_row

    def _record_output_rows(self, rows: list[dict[str, Any]]) -> None:
        self.rows_out += len(rows)
        self.output_track_ids_seen.update(int(row["id"]) for row in rows)

    def _record_mapping(
        self,
        source_track_id: int,
        canonical_track_id: int,
        state: dict[str, Any],
        accepted_decision: dict[str, Any] | None,
        decision_frame: int,
    ) -> None:
        accepted_remap = accepted_decision is not None
        self.mapping_records[source_track_id] = {
            "raw_track_id": int(source_track_id),
            "canonical_track_id": int(canonical_track_id),
            "first_frame": int(state["first_frame"]),
            "last_frame": int(state["last_frame"]),
            "accepted_remap": accepted_remap,
            "matched_raw_track_id": None if accepted_decision is None else int(accepted_decision["candidate_source_track_id"]),
            "reason": "fresh_canonical_id" if accepted_decision is None else str(accepted_decision["reason"]),
            "decision_frame": int(decision_frame),
            "decision_latency": int(decision_frame - int(state["first_frame"])),
        }

    def _update_mapping_last_frame(self, source_track_id: int, frame_idx: int) -> None:
        record = self.mapping_records.get(source_track_id)
        if record is not None:
            record["last_frame"] = int(frame_idx)

    def _appearance_supported_remaps(self) -> list[str]:
        return [
            f"{int(decision['new_source_track_id'])}->{int(decision['candidate_source_track_id'])}"
            for decision in self.decisions
            if bool(decision.get("applied")) and bool(decision.get("appearance_assisted"))
        ]


def sort_mot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["frame"]), int(row["id"]), float(row["bbox"][0]), float(row["bbox"][1])))


def aggregate_online_reports(
    *,
    sequence_reports: dict[str, dict[str, Any]],
    output_root: Path,
    mot_dir: Path,
    config_path: str | None,
    source_video_path: str | None,
    safe_mode: bool,
    confirmation_observations: int,
) -> dict[str, Any]:
    all_decisions = [decision for report in sequence_reports.values() for decision in report["decisions"]]
    latency_summary = _decision_latency_summary(all_decisions)
    accepted_remaps = [
        remap
        for report in sequence_reports.values()
        for remap in report["accepted_remaps"]
    ]
    appearance_supported_remaps = [
        remap
        for report in sequence_reports.values()
        for remap in report["appearance_supported_remaps"]
    ]
    summary = {
        "num_sequences_processed": len(sequence_reports),
        "accepted_remaps": accepted_remaps,
        "accepted_remap_count": sum(int(report["accepted_remap_count"]) for report in sequence_reports.values()),
        "rejected_candidate_count": sum(int(report["rejected_candidate_count"]) for report in sequence_reports.values()),
        "candidate_evaluation_count": sum(int(report["candidate_evaluation_count"]) for report in sequence_reports.values()),
        "fresh_id_count": sum(int(report["fresh_id_count"]) for report in sequence_reports.values()),
        "fresh_canonical_ids_assigned": sum(int(report["fresh_canonical_ids_assigned"]) for report in sequence_reports.values()),
        "unique_ids_before": {name: len(report["unique_track_ids_in"]) for name, report in sequence_reports.items()},
        "unique_ids_after": {name: len(report["unique_track_ids_out"]) for name, report in sequence_reports.items()},
        "appearance_supported_remaps": appearance_supported_remaps,
        "appearance_supported_remap_count": sum(int(report["appearance_supported_remap_count"]) for report in sequence_reports.values()),
        "appearance_rejection_count": sum(int(report["appearance_rejection_count"]) for report in sequence_reports.values()),
        "chain_control_checked_count": sum(int(report["chain_control_checked_count"]) for report in sequence_reports.values()),
        "chain_control_rejected_count": sum(int(report["chain_control_rejected_count"]) for report in sequence_reports.values()),
        "short_gap_gate_checked_count": sum(int(report["short_gap_gate_checked_count"]) for report in sequence_reports.values()),
        "short_gap_gate_rejected_count": sum(int(report["short_gap_gate_rejected_count"]) for report in sequence_reports.values()),
        "decision_latency_mean": latency_summary["decision_latency_frames_mean"],
        "decision_latency_max": latency_summary["decision_latency_frames_max"],
        "confirmation_observations": int(confirmation_observations),
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "source_video_path": source_video_path,
        **latency_summary,
        **_colreg_scoring_summary(all_decisions),
    }
    return {
        "version": LIVE_REID_VERSION,
        "mode": IN_LOOP_LIVE_REID_MODE,
        "output_root": str(output_root),
        "mot_dir": str(mot_dir),
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "source_video_path": source_video_path,
        "safe_mode": bool(safe_mode),
        "confirmation_observations": int(confirmation_observations),
        "total_sequences": len(sequence_reports),
        "accepted_remaps": summary["accepted_remaps"],
        "accepted_remap_count": summary["accepted_remap_count"],
        "rejected_candidate_count": summary["rejected_candidate_count"],
        "fresh_id_count": summary["fresh_id_count"],
        "unique_ids_before": summary["unique_ids_before"],
        "unique_ids_after": summary["unique_ids_after"],
        "appearance_supported_remaps": summary["appearance_supported_remaps"],
        "appearance_supported_remap_count": summary["appearance_supported_remap_count"],
        "appearance_rejection_count": summary["appearance_rejection_count"],
        "chain_control_checked_count": summary["chain_control_checked_count"],
        "chain_control_rejected_count": summary["chain_control_rejected_count"],
        "short_gap_gate_checked_count": summary["short_gap_gate_checked_count"],
        "short_gap_gate_rejected_count": summary["short_gap_gate_rejected_count"],
        "decision_latency_mean": summary["decision_latency_mean"],
        "decision_latency_max": summary["decision_latency_max"],
        "summary": summary,
        "sequences": sequence_reports,
        "warnings": [
            warning
            for report in sequence_reports.values()
            for warning in report.get("appearance_warnings", [])
        ],
    }
