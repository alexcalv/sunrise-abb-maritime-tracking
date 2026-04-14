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

    results = model.track(
        source=source,
        tracker=tracker_yaml,
        device=device,
        project=project,
        name=name,
        save=True,
        save_txt=True,
        persist=True,
        stream=True,
        verbose=False,
    )

    frames_processed = 0
    for _ in results:
        frames_processed += 1

    return {"frames_processed": frames_processed}