from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset_yolo/data.yaml",
        epochs=35,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        cos_lr=True,
        lr0=0.01
    )

if __name__ == "__main__":
    main()