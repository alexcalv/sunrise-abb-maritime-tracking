from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.generate_videos import camera_names, collect_outputs, generate_videos
from tools.run_sim_eval import run_sim_eval


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "video": {"width": 640, "height": 360, "fps": 10, "duration": 1.0},
                "ships": [
                    {
                        "name": "ship_a",
                        "prefab": "Ship1",
                        "position": [0, 0, 0],
                        "heading": 90,
                        "speed": 2.0,
                    }
                ],
                "cameras": [
                    {"name": "cam_north", "position": [0, 20, -50], "rotation": [10, 0, 0]},
                    {"position": [50, 20, 0], "rotation": [10, -90, 0]},
                ],
            }
        ),
        encoding="utf-8",
    )


class SimEvalToolTests(unittest.TestCase):
    def test_camera_names_and_collect_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "scenario.json"
            out_dir = root / "out"
            _write_config(config)

            self.assertEqual(camera_names(config), ["cam_north", "camera_1"])
            outputs = collect_outputs(config, out_dir)
            self.assertEqual(outputs["cam_north"]["video"], str((out_dir / "cam_north.mp4").resolve()))
            self.assertEqual(outputs["camera_1"]["mot"], str((out_dir / "camera_1.txt").resolve()))

    def test_run_sim_eval_dry_run_does_not_require_unity_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "scenario.json"
            _write_config(config)

            summary = run_sim_eval(config=config, dry_run=True)

            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["camera_names"], ["cam_north", "camera_1"])
            self.assertIn("run detector/tracker once per camera video", summary["planned_steps"])

    def test_generate_videos_invokes_unity_and_validates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "scenario.json"
            out_dir = root / "generated"
            project = root / "MaritimeSim2"
            project.mkdir()
            _write_config(config)

            def fake_run(cmd, timeout):
                sim_out = Path(cmd[cmd.index("-simOut") + 1])
                for name in ("cam_north", "camera_1"):
                    (sim_out / f"{name}.mp4").write_bytes(b"video")
                    (sim_out / f"{name}.txt").write_text("1,1,0,0,10,10,1,8,1\n", encoding="utf-8")
                    (sim_out / f"{name}.json").write_text("[]\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("tools.generate_videos.find_unity", return_value="/fake/Unity"), mock.patch(
                "tools.generate_videos.subprocess.run",
                side_effect=fake_run,
            ) as run_mock:
                outputs = generate_videos(config, out_dir=out_dir, project=project, quiet=True)

            self.assertEqual(sorted(outputs), ["cam_north", "camera_1"])
            invoked = run_mock.call_args.args[0]
            self.assertIn("-executeMethod", invoked)
            self.assertIn("GeneratorEntry.Run", invoked)
            self.assertEqual(invoked[invoked.index("-projectPath") + 1], str(project.resolve()))


if __name__ == "__main__":
    unittest.main()
