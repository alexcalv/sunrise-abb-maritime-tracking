from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ais.loader import build_layer_from_json_dict  # pyright: ignore[reportMissingImports]
from ais.pipeline import ais_layer_to_sidecar_dict, merge_ais_into_run_summary, resolve_ais_track_layer  # pyright: ignore[reportMissingImports]


class TestAisLoader(unittest.TestCase):
    def test_frame_index_mode(self) -> None:
        data = {
            "version": 1,
            "sync": {"mode": "frame_index"},
            "positions": [
                {"frame": 1, "mmsi": 1, "latitude_deg": 0.0, "longitude_deg": 0.0},
                {"frame": 1, "mmsi": 2, "latitude_deg": 1.0, "longitude_deg": 1.0},
                {"frame": 3, "mmsi": 1, "latitude_deg": 0.1, "longitude_deg": 0.1},
            ],
        }
        layer = build_layer_from_json_dict(data, video_fps=None)
        self.assertEqual(layer.total_fixes(), 3)
        self.assertEqual(len(layer.positions_by_frame[1]), 2)
        self.assertEqual(len(layer.positions_by_frame[3]), 1)
        self.assertEqual(layer.positions_by_frame[1][0].mmsi, 1)

    def test_timestamp_ms_mode(self) -> None:
        data = {
            "version": 1,
            "sync": {"mode": "timestamp_ms", "time_offset_ms": 0},
            "positions": [
                {"timestamp_ms": 0, "mmsi": 10, "latitude_deg": 0.0, "longitude_deg": 0.0},
                {"timestamp_ms": 1000, "mmsi": 10, "latitude_deg": 0.0, "longitude_deg": 0.1},
            ],
        }
        layer = build_layer_from_json_dict(data, video_fps=30.0)
        self.assertIn(1, layer.positions_by_frame)
        self.assertIn(31, layer.positions_by_frame)

    def test_timestamp_ms_offset(self) -> None:
        data = {
            "version": 1,
            "sync": {"mode": "timestamp_ms", "time_offset_ms": 500},
            "positions": [{"timestamp_ms": 500, "mmsi": 1, "latitude_deg": 0.0, "longitude_deg": 0.0}],
        }
        layer = build_layer_from_json_dict(data, video_fps=30.0)
        self.assertIn(1, layer.positions_by_frame)

    def test_timestamp_requires_fps(self) -> None:
        data = {
            "version": 1,
            "sync": {"mode": "timestamp_ms"},
            "positions": [{"timestamp_ms": 0, "mmsi": 1, "latitude_deg": 0.0, "longitude_deg": 0.0}],
        }
        with self.assertRaises(ValueError):
            build_layer_from_json_dict(data, video_fps=None)

    def test_resolve_cli_time_offset_overrides_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ais.json"
            p.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "sync": {"mode": "timestamp_ms", "time_offset_ms": 0},
                        "positions": [{"timestamp_ms": 1000, "mmsi": 1, "latitude_deg": 0.0, "longitude_deg": 0.0}],
                    }
                ),
                encoding="utf-8",
            )
            layer = resolve_ais_track_layer(
                p,
                video_for_fps=None,
                fps_override=30.0,
                time_offset_ms_override=1000,
            )
            # (1000 - 1000 offset) * 30 / 1000 = 0 -> frame 1
            self.assertIn(1, layer.positions_by_frame)


class TestAisSidecar(unittest.TestCase):
    def test_merge_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rs = root / "run_summary.json"
            rs.write_text(json.dumps({"sequences": {}}, indent=2), encoding="utf-8")
            sc = root / "ais_layer.json"
            sc.write_text("{}", encoding="utf-8")
            merge_ais_into_run_summary(rs, sc)
            data = json.loads(rs.read_text(encoding="utf-8"))
            self.assertTrue(data["ais_layer"]["enabled"])
            self.assertEqual(data["ais_layer"]["sidecar"], str(sc))

    def test_sidecar_dict_shape(self) -> None:
        data = {
            "version": 1,
            "sync": {"mode": "frame_index"},
            "positions": [{"frame": 2, "mmsi": 99, "latitude_deg": 1.0, "longitude_deg": 2.0}],
        }
        layer = build_layer_from_json_dict(data, video_fps=None)
        d = ais_layer_to_sidecar_dict(layer, source_path="/tmp/x.json")
        self.assertEqual(d["stats"]["total_fixes"], 1)
        self.assertEqual(d["by_frame"]["2"][0]["mmsi"], 99)


if __name__ == "__main__":
    unittest.main()
