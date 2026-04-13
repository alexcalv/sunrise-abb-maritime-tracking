from __future__ import annotations

from ultralytics import YOLO


def run_tracking(
    model_path: str,
    source: str,
    tracker_yaml: str,
    device: str,
    project: str,
    name: str,
):
    model = YOLO(model_path)
    return model.track(
        source=source,
        tracker=tracker_yaml,
        device=device,
        project=project,
        name=name,
        save=True,
        save_txt=True,
        persist=True,
        verbose=False,
    )
