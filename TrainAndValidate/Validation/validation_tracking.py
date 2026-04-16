import os
from ultralytics import YOLO

VIDEO_DIR = "dataset/videos"
OUT_DIR = "dataset/tracking"
MODEL_PATH = "best.pt"

os.makedirs(OUT_DIR, exist_ok=True)


def process_video(video_path, model, out_path):

    print(f"Processing: {video_path}")

    results = model.track(
        source=video_path,                 # IMPORTANT (not folder)
        tracker="botsort.yaml",    # safe config
        conf=0.35,
        iou=0.5,
        imgsz=640,
        device=0,
        persist=True,
        classes=[0],                      # vessel only
        stream=True,                      # IMPORTANT for speed + memory
        verbose=False
    )

    frame_id = 0

    with open(out_path, "w") as f:

        for r in results:

            if r.boxes is None or len(r.boxes) == 0:
                frame_id += 1
                continue

            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id
            confs = r.boxes.conf.cpu().numpy()

            if ids is None:
                frame_id += 1
                continue

            ids = ids.cpu().numpy()

            for i in range(len(xywh)):

                if confs[i] < 0.25:
                    continue

                tid = int(ids[i])
                cx, cy, w, h = xywh[i]

                x = cx - w / 2
                y = cy - h / 2

                # MOT format:
                # frame,id,x,y,w,h,conf,-1,-1,-1
                f.write(f"{frame_id},{tid},{x},{y},{w},{h},{confs[i]},-1,-1,-1\n")

            frame_id += 1

    print(f"Saved: {out_path}")


def main():
    model = YOLO(MODEL_PATH)

    for v in sorted(os.listdir(VIDEO_DIR)):
        if not v.endswith(".mp4"):
            continue

        process_video(
            os.path.join(VIDEO_DIR, v),
            model,
            os.path.join(OUT_DIR, v.replace(".mp4", ".txt"))
        )


if __name__ == "__main__":
    main()