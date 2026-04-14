from __future__ import annotations

from ultralytics import YOLO

from common.config import settings
from common.schemas import DetectionBox


class YoloDetector:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.yolo_model
        self.device = device or settings.yolo_device
        self.model = YOLO(self.model_name)

    def infer_frame(self, image_path: str, frame_index: int) -> list[DetectionBox]:
        classes = None
        if settings.yolo_classes.strip():
            classes = [int(x) for x in settings.yolo_classes.split(",")]

        preds = self.model.predict(
            source=image_path,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
            imgsz=settings.yolo_imgsz,
            device=self.device,
            classes=classes,
            verbose=False,
        )

        if not preds:
            return []

        result = preds[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names
        xywh = boxes.xywh.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.cpu().tolist()

        out: list[DetectionBox] = []
        for box, conf, cls_id in zip(xywh, confs, clss):
            out.append(
                DetectionBox(
                    frame_index=frame_index,
                    x=float(box[0]),
                    y=float(box[1]),
                    w=float(box[2]),
                    h=float(box[3]),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=names.get(int(cls_id)),
                )
            )
        return out


_DETECTOR: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = YoloDetector()
    return _DETECTOR