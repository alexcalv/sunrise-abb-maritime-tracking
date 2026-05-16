from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ais.alignment import build_aligned_clip  # pyright: ignore[reportMissingImports]
from ais.fusion import assign_mmsi_per_tracklet, score_ais_identity, score_ais_position  # pyright: ignore[reportMissingImports]
from ais.parser import fixes_from_csv_path, group_fixes_by_mmsi  # pyright: ignore[reportMissingImports]
from ais.schemas import AisFix  # pyright: ignore[reportMissingImports]
from stitching.schemas import AisConfig, Tracklet  # pyright: ignore[reportMissingImports]


def _minimal_tracklet(tid: str, frame_start: int, frame_end: int, center: list[float]) -> Tracklet:
    return Tracklet(
        sequence_name="seq",
        tracklet_id=tid,
        source_track_id=1,
        canonical_track_id=1,
        segment_index=0,
        frame_start=frame_start,
        frame_end=frame_end,
        num_rows=frame_end - frame_start + 1,
        num_frames=frame_end - frame_start + 1,
        class_ids=[0],
        mean_confidence=0.9,
        first_bbox=[center[0], center[1], 10, 10],
        last_bbox=[center[0], center[1], 10, 10],
        mean_bbox=[center[0], center[1], 10, 10],
        mean_area=100.0,
        mean_aspect_ratio=1.0,
        frame_gaps=[],
        start_center=center,
        end_center=center,
        start_velocity=[0.0, 0.0],
        end_velocity=[0.0, 0.0],
    )


class TestAisParserCsv(unittest.TestCase):
    def test_csv_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ais.csv"
            p.write_text(
                "mmsi,timestamp_ms,pixel_x,pixel_y\n"
                "123,0,100.0,200.0\n"
                "123,1000,110.0,210.0\n",
                encoding="utf-8",
            )
            fixes = fixes_from_csv_path(p)
            self.assertEqual(len(fixes), 2)
            self.assertEqual(fixes[0].mmsi, 123)
            self.assertTrue(fixes[0].has_pixel())


class TestAisAlignment(unittest.TestCase):
    def test_interpolate_to_frames(self) -> None:
        fixes = [
            AisFix(mmsi=1, timestamp_ms=0, pixel_x=0.0, pixel_y=0.0),
            AisFix(mmsi=1, timestamp_ms=1000, pixel_x=100.0, pixel_y=0.0),
        ]
        grouped = group_fixes_by_mmsi(fixes)
        clip = build_aligned_clip(grouped, max_frame=30, video_fps=30.0)
        self.assertIn(1, clip.vessels_at(1))
        self.assertIn(15, clip.frames)


class TestAisFusionScores(unittest.TestCase):
    def test_missing_ais_is_neutral(self) -> None:
        cfg = AisConfig(neutral_score=0.5)
        score, _, _ = score_ais_position([0, 0], 1, None, None, max_distance_px=100, neutral_score=cfg.neutral_score)
        self.assertEqual(score, 0.5)
        id_score, gate, status = score_ais_identity(None, 123, hard_gate=False, neutral_score=0.5)
        self.assertEqual(id_score, 0.5)
        self.assertTrue(gate)
        self.assertEqual(status, "ais_missing")

    def test_hard_identity_mismatch_fails_gate(self) -> None:
        _, gate, status = score_ais_identity(1, 2, hard_gate=True)
        self.assertFalse(gate)
        self.assertEqual(status, "mmsi_hard_mismatch")

    def test_mmsi_assignment(self) -> None:
        fixes = [AisFix(mmsi=99, timestamp_ms=0, pixel_x=50.0, pixel_y=50.0)]
        clip = build_aligned_clip(group_fixes_by_mmsi(fixes), max_frame=10, video_fps=30.0)
        t = _minimal_tracklet("t1", 1, 5, [52.0, 48.0])
        assigned = assign_mmsi_per_tracklet([t], clip, max_distance_px=50.0)
        self.assertEqual(assigned["t1"], 99)
        self.assertEqual(t.assigned_mmsi, None)  # stamp is separate helper


if __name__ == "__main__":
    unittest.main()
