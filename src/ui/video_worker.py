import os
import time
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from src.inference.yolo_engine import YOLOEngine
from src.roi.roi_controller import ROIController, Rect
from src.tracking.tracker import SmoothTargetTracker
from src.utils.draw import draw_rect, draw_detections, draw_metrics, to_qimage, GREEN, YELLOW


class VideoWorker(QObject):
    frameReady = Signal(object)  # QImage
    metricsReady = Signal(str)
    status = Signal(str)
    sizeChanged = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.engine: Optional[YOLOEngine] = None
        self.roi = ROIController()
        self.tracker = SmoothTargetTracker(alpha=0.25, lost_max=15)

        # settings
        self.source_type = "webcam"  # or "video"
        self.cam_index = 0
        self.video_path: Optional[str] = None

        self.conf = 0.50
        self.iou = 0.45
        self.mode = 1
        self.autofocus = False
        self.prefer_cuda = True

        self.frame_W = 0
        self.frame_H = 0

        self.base_subframe_percent = (0.8, 0.8)

        self._last_fps_t = None
        self._ema_fps = None
        self._lost = False

    @Slot()
    def stop(self) -> None:
        self.running = False

    def _open_source(self) -> bool:
        if self.source_type == "video" and self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        ok = bool(self.cap.isOpened())
        if not ok:
            self.status.emit("Failed to open source")
        return ok

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            # loop videos; for webcam, just return None transiently
            if self.source_type == "video":
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                except Exception:
                    return None
            else:
                return None
        return frame

    def _ensure_engine(self) -> None:
        if self.engine is None:
            self.engine = YOLOEngine(
                model_path=os.path.join(os.getcwd(), "yolov8n.pt"),
                prefer_cuda=self.prefer_cuda,
                classes=(0,),
                conf=self.conf,
                iou=self.iou,
            )

    def _ensure_sizes(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_W, self.frame_H):
            self.frame_W, self.frame_H = w, h
            self.roi.set_frame_size(w, h)
            self.roi.set_mode(self.mode)
            # ensure base subframe size applied
            self.roi.set_subframe_size_percent(self.base_subframe_percent[0], self.base_subframe_percent[1])
            self.sizeChanged.emit(w, h)

    def _update_fps(self) -> float:
        t = time.perf_counter()
        if self._last_fps_t is None:
            self._last_fps_t = t
            return 0.0
        dt = t - self._last_fps_t
        self._last_fps_t = t
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if self._ema_fps is None:
            self._ema_fps = inst_fps
        else:
            self._ema_fps = 0.9 * self._ema_fps + 0.1 * inst_fps
        return self._ema_fps

    def _move_percent_towards(self, cur: Tuple[float, float], target: Tuple[float, float], step: float) -> Tuple[float, float]:
        nx, ny = cur
        tx, ty = target
        dx, dy = tx - nx, ty - ny
        ax, ay = abs(dx), abs(dy)
        sx = step if ax > step else ax
        sy = step if ay > step else ay
        nx += sx * (1 if dx >= 0 else -1)
        ny += sy * (1 if dy >= 0 else -1)
        return (max(0.2, min(1.0, nx)), max(0.2, min(1.0, ny)))

    @Slot()
    def run(self) -> None:
        if not self._open_source():
            return
        try:
            self._ensure_engine()
        except Exception as e:
            self.status.emit(f"Engine init error: {e}")
            return

        self.running = True
        self.status.emit("Running")

        while self.running:
            frame = self._read_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            self._ensure_sizes(frame)

            # Determine current rects
            subframe_rect, roi_rect = self.roi.get_rects()

            # Crop ROI and infer
            x, y, w, h = roi_rect.as_tuple()
            x2, y2 = x + w, y + h
            x, y = max(0, x), max(0, y)
            x2, y2 = min(self.frame_W, x2), min(self.frame_H, y2)
            crop = frame[y:y2, x:x2]

            detections_global: List[Dict] = []
            infer_ms = 0.0
            if crop.size > 0:
                dets_local, infer_ms = self.engine.predict(crop)
                # Map local -> global
                for d in dets_local:
                    lx1, ly1, lx2, ly2 = d["xyxy"]
                    d_glob = {
                        "xyxy": [lx1 + x, ly1 + y, lx2 + x, ly2 + y],
                        "conf": d.get("conf", 0.0),
                        "cls": d.get("cls", 0),
                    }
                    detections_global.append(d_glob)

            # Update ROI/tracking for next frame
            if self.mode == 4 and self.autofocus:
                # Smooth target center tracking and adaptive subframe sizing
                # Default center: current subframe center or frame center
                s_cx, s_cy = self.roi.subframe_rect.center()
                default_center = (s_cx if s_cx > 0 else self.frame_W * 0.5,
                                   s_cy if s_cy > 0 else self.frame_H * 0.5)
                center, lost_flag = self.tracker.update(detections_global, default_center)
                self._lost = lost_flag

                # Adjust subframe size: enlarge when lost, shrink back to base when found
                curp = (self.roi.subframe_percent[0], self.roi.subframe_percent[1])
                if lost_flag:
                    targetp = (1.0, 1.0)
                    newp = self._move_percent_towards(curp, targetp, step=0.06)
                else:
                    targetp = self.base_subframe_percent
                    newp = self._move_percent_towards(curp, targetp, step=0.04)
                self.roi.set_subframe_size_percent(newp[0], newp[1])

                # Center subframe on smoothed center and keep ROI centered in it
                fw = int(self.roi.subframe_percent[0] * self.frame_W)
                fh = int(self.roi.subframe_percent[1] * self.frame_H)
                cx, cy = int(center[0]), int(center[1])
                self.roi.subframe_rect = Rect(int(cx - fw * 0.5), int(cy - fh * 0.5), fw, fh).clamp(self.frame_W, self.frame_H)
                self.roi.click_set_roi_center(cx, cy)
            else:
                self.roi.step(detections_global, autofocus=self.autofocus)

            # Visualize
            vis = frame.copy()
            # Fetch updated rects post-update
            subframe_rect, roi_rect = self.roi.get_rects()
            if self.mode == 4:
                draw_rect(vis, subframe_rect.as_tuple(), color=YELLOW, thickness=2)
            draw_rect(vis, roi_rect.as_tuple(), color=GREEN, thickness=2)
            draw_detections(vis, detections_global)

            # Metrics
            fps = self._update_fps()
            roi_area = roi_rect.w * roi_rect.h
            full_area = max(1, self.frame_W * self.frame_H)
            roi_pct = 100.0 * roi_area / full_area
            human_count = len(detections_global)
            metrics = [
                f"FPS: {fps:5.1f}",
                f"Infer: {infer_ms:5.1f} ms",
                f"Humans: {human_count}",
                f"ROI: {roi_pct:4.1f}% of frame",
                f"Est speedup: {full_area / max(1, roi_area):.2f}x",
                f"Mode: {self.mode} | Lost: {self._lost}",
            ]
            self.metricsReady.emit("\n".join(metrics))

            qimg = to_qimage(vis)
            self.frameReady.emit(qimg)

        self.status.emit("Stopped")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # Settings updaters (thread-safe enough for this demo)
    @Slot(int)
    def set_mode(self, mode: int) -> None:
        self.mode = int(mode)
        self.roi.set_mode(self.mode)

    @Slot(float)
    def set_conf(self, conf: float) -> None:
        self.conf = float(conf)
        if self.engine:
            self.engine.set_thresholds(self.conf, self.iou)

    @Slot(float)
    def set_iou(self, iou: float) -> None:
        self.iou = float(iou)
        if self.engine:
            self.engine.set_thresholds(self.conf, self.iou)

    @Slot(bool)
    def set_autofocus(self, enabled: bool) -> None:
        self.autofocus = bool(enabled)

    @Slot(int, int)
    def set_roi_center(self, cx: int, cy: int) -> None:
        # Only allow manual ROI movement in modes 3 and 4
        if self.mode in (3, 4):
            self.roi.click_set_roi_center(cx, cy)

    def configure(self, source_type: str, cam_index: int, video_path: Optional[str], conf: float, iou: float, mode: int, autofocus: bool, prefer_cuda: bool) -> None:
        self.source_type = source_type
        self.cam_index = int(cam_index)
        self.video_path = video_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.mode = int(mode)
        self.autofocus = bool(autofocus)
        self.prefer_cuda = bool(prefer_cuda)
