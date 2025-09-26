from typing import List, Dict, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage


GREEN = (80, 220, 100)
YELLOW = (60, 200, 255)
CYAN = (0, 255, 255)
WHITE = (240, 240, 240)
GRAY = (90, 90, 90)


def draw_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color=(0, 255, 0), thickness: int = 2) -> None:
    x, y, w, h = rect
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness, cv2.LINE_AA)


def draw_detections(img: np.ndarray, dets: List[Dict], color=CYAN) -> None:
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        conf = float(d.get("conf", 0.0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"person {conf:.2f}", (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_metrics(img: np.ndarray, text_lines: List[str]) -> None:
    x, y = 10, 24
    for line in text_lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        y += 22


def to_qimage(bgr_img: np.ndarray) -> QImage:
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimg.copy()  # ensure memory ownership
