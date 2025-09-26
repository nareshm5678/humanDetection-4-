import os
import time
from typing import List, Tuple, Dict, Iterable

import numpy as np
import torch
from ultralytics import YOLO


class YOLOEngine:
    """Thin wrapper around Ultralytics YOLOv8 for person-only detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        prefer_cuda: bool = True,
        classes: Iterable[int] = (0,),  # 0 == person in COCO
        conf: float = 0.50,
        iou: float = 0.45,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.classes = set(int(c) for c in classes)
        self.conf = float(conf)
        self.iou = float(iou)

        self.device = "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(self.model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # Optional warmup for stable FPS
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.predict(dummy)
        except Exception:
            pass

    def set_thresholds(self, conf: float, iou: float) -> None:
        self.conf = float(conf)
        self.iou = float(iou)

    def _dynamic_imgsz(self, h: int, w: int) -> int:
        # Choose imgsz as nearest multiple of 32 based on ROI size, clamped to [320, 640]
        side = max(h, w)
        # round up to multiple of 32
        imgsz = int((side + 31) // 32 * 32)
        imgsz = max(320, min(640, imgsz))
        return imgsz

    def predict(self, bgr_image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Run inference on a BGR image (numpy array).

        Returns:
            detections: list of dicts with keys {xyxy, conf, cls}
            infer_ms: inference time in milliseconds
        """
        h, w = bgr_image.shape[:2]
        imgsz = self._dynamic_imgsz(h, w)
        t0 = time.perf_counter()
        results = self.model(
            bgr_image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            imgsz=imgsz,
            classes=list(self.classes) if self.classes else None,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if self.classes and cls_id not in self.classes:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                detections.append({
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "cls": cls_id,
                })
        return detections, infer_ms
