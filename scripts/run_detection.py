from ultralytics import YOLO
import yaml

with open("configs/detection_baseline.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = YOLO(cfg["model"])
model.predict(
    source=cfg["input_path"],
    save=True,
    conf=cfg["conf_threshold"],
    device=cfg["device"],
    project="outputs",
    name="baseline_detection",
)