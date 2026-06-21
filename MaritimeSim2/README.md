# MaritimeSim2 — Synthetic Maritime Video & Ground-Truth Generator

A Unity project that turns a single JSON config into **rendered MP4 videos plus
frame-aligned tracking ground truth** (MOT `.txt` + JSON) for maritime
detection/tracking datasets. Ships move on a Crest ocean; one or more virtual
cameras observe the scene and each produces its own video and annotations.

- **Unity:** 2022.3.62f3 (URP)
- **Key packages:** Unity Recorder (MP4 encoding), Crest (ocean)

## How it works

```
config.json ─► Build Generator Scene ─► SimulationDirector ─► per-camera:
              (one-time, from Sample.unity)   spawns ships+cameras     MP4 + MOT.txt + JSON
```

1. **`GeneratorSceneBuilder`** strips `Sample.unity` down to one template camera
   + ocean and saves it as `Assets/Scenes/Generator.unity`, registering every
   prefab in `Assets/ShipsPrefabs`.
2. **`SimulationDirector`** (runs on Play) loads the config, spawns the ships and
   cameras it describes, and gives each camera its own `RenderTexture` +
   `GroundTruthRecorder`.
3. Capture is **deterministic**: `Time.captureDeltaTime` is pinned to `1/fps`, so
   ship motion, waves and ground truth are reproducible and frame-aligned with
   the video.
4. **`MultiCameraRecorder`** encodes one MP4 per camera; **`GroundTruthRecorder`**
   writes per-frame bounding boxes, stable track IDs, class IDs and occlusion.

## Entry points

### Unity Editor
1. `Tools ▸ MaritimeSim ▸ Build Generator Scene` — run **once** (rebuild after
   changing `Sample.unity` or adding ship prefabs).
2. `Tools ▸ MaritimeSim ▸ Generate Videos from Config…` — pick a JSON config; the
   project enters Play mode, renders, and writes outputs.

> For repeated in-Editor runs you can instead set **Config Path Override** /
> **Out Dir Override** on the `__SimulationDirector` object and just press Play.

### Command line (headless / batch)
```bash
Unity.exe -projectPath <path-to-MaritimeSim2> \
          -executeMethod GeneratorEntry.Run \
          -simConfig <config.json> [-simOut <output-dir>] [-simVerbose]
```
Runs a full generation and quits with an exit code (`0` success, non-zero on
error — e.g. `2` = missing `-simConfig`). Rendering needs a GPU, so do **not**
pass `-nographics`.

## Config format

See [`configs/example_config.json`](configs/example_config.json). Summary:

```jsonc
{
  "video":  { "width": 1280, "height": 720, "fps": 30, "duration": 10.0 },
  "ships": [
    { "name": "tanker_A", "prefab": "Ship1", "position": [-30, 0, 0],
      "heading": 90, "speed": 5.0, "classId": 8 }
  ],
  "cameras": [
    { "name": "cam_south", "position": [0, 28, -85], "rotation": [10, 0, 0], "fov": 50 }
  ]
}
```

- **ship.prefab** — name of a prefab in `Assets/ShipsPrefabs` (`Ship1`…`Ship6`).
- **ship.position** — `[x, z]` or `[x, y, z]` world space; **heading** in degrees
  (0 = +Z, 90 = +X); **speed** in m/s; **classId** = MOT class column.
- **camera.name** — must be unique; it becomes the output file name.
  **rotation** = `[pitch, yaw, roll]`; **fov** = vertical FOV.
- MP4 needs even dimensions — odd `width`/`height` are rounded up with a warning.

## Outputs

Written to `-simOut`, else `Recordings/<config-name>/`. Per camera `<name>`:

| File | Contents |
|------|----------|
| `<name>.mp4` | rendered video |
| `<name>.txt` | MOT: `frame, track_id, x, y, w, h, conf, class_id, visibility` |
| `<name>.json` | full per-frame annotations (bbox, occlusion ratio, visibility, occluded) |
| `config.json` | copy of the exact input config (provenance) |

Bounding boxes are image-space (origin top-left); occlusion is estimated by
camera→vessel raycast sampling and exposed as `occlusion_ratio` / `visibility`.
