# Simulation Evaluation Tools

`generate_videos.py` belongs in this `tools/` directory. It is a Python wrapper
around the Unity editor entry point `GeneratorEntry.Run` in `MaritimeSim2`.

The full automated loop is:

```bash
python tools/run_sim_eval.py MaritimeSim2/Assets/configs/example_config.json \
  --sim-out outputs/sim/example_config \
  --eval-dir outputs/sim_eval/example_config \
  --model models/yolo26l.pt \
  --device cpu
```

In Docker, use the integrated CLI:

```bash
docker compose run --rm runner sim-eval /workspace/MaritimeSim2/Assets/configs/example_config.json \
  --sim-out /workspace/outputs/sim/example_config \
  --eval-dir /workspace/outputs/sim_eval/example_config \
  --model /workspace/models/yolo26l.pt \
  --device cpu
```

On machines without Unity, validate the wiring without rendering:

```bash
docker compose run --rm runner sim-eval /workspace/MaritimeSim2/Assets/configs/example_config.json --dry-run
```

If videos and Unity ground-truth files already exist, reuse them with
`--skip-generation --sim-out <existing-dir>`. The existing directory must contain
one `<camera>.mp4`, `<camera>.txt`, and `<camera>.json` for every camera in the
config.
