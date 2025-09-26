from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, W: int, H: int) -> "Rect":
        self.x = max(0, min(self.x, max(0, W - 1)))
        self.y = max(0, min(self.y, max(0, H - 1)))
        self.w = max(1, min(self.w, max(1, W - self.x)))
        self.h = max(1, min(self.h, max(1, H - self.y)))
        return self

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def center(self) -> Tuple[int, int]:
        return int(self.x + self.w * 0.5), int(self.y + self.h * 0.5)


class ROIController:
    """Manages ROI/subframe rectangles for 4 modes.

    Modes:
      1: Full fixed frame as ROI
      2: Fixed frame with a fixed sub-ROI (does not move automatically)
      3: Fixed frame with a movable ROI (autofocus optional)
      4: Movable frame with a movable ROI
    """

    def __init__(self, roi_percent: Tuple[float, float] = (0.5, 0.5), subframe_percent: Tuple[float, float] = (0.8, 0.8)) -> None:
        self.W = 0
        self.H = 0
        self.mode = 1
        self.roi_percent = list(roi_percent)
        self.subframe_percent = list(subframe_percent)

        self.roi_rect = Rect(0, 0, 0, 0)
        self.subframe_rect = Rect(0, 0, 0, 0)

        self.manual_roi_center = None  # (cx, cy)
        self.manual_roi_offset = (0, 0)  # relative to subframe center for mode 4

        # Tracking state for mode 4
        self.tracked_center = None  # (cx, cy)

    def set_frame_size(self, W: int, H: int) -> None:
        self.W, self.H = int(W), int(H)
        self._reset_defaults()

    def _reset_defaults(self) -> None:
        if self.W <= 0 or self.H <= 0:
            return
        rw = int(self.roi_percent[0] * self.W)
        rh = int(self.roi_percent[1] * self.H)
        rx = int((self.W - rw) * 0.5)
        ry = int((self.H - rh) * 0.5)
        self.roi_rect = Rect(rx, ry, rw, rh).clamp(self.W, self.H)
        self.subframe_rect = Rect(0, 0, self.W, self.H)
        self.tracked_center = (self.W * 0.5, self.H * 0.5)

    def set_mode(self, mode: int) -> None:
        self.mode = int(max(1, min(4, mode)))
        if self.mode != 4:
            self.subframe_rect = Rect(0, 0, self.W, self.H)
        self.manual_roi_offset = (0, 0)

    def set_roi_size_percent(self, wp: float, hp: float) -> None:
        self.roi_percent = [max(0.05, min(0.98, float(wp))), max(0.05, min(0.98, float(hp)))]
        self._resize_roi()

    def set_subframe_size_percent(self, wp: float, hp: float) -> None:
        self.subframe_percent = [max(0.2, min(1.0, float(wp))), max(0.2, min(1.0, float(hp)))]
        self._resize_subframe()

    def _resize_roi(self) -> None:
        rw = int(self.roi_percent[0] * self.W)
        rh = int(self.roi_percent[1] * self.H)
        cx, cy = self.roi_rect.center()
        self.roi_rect = Rect(int(cx - rw * 0.5), int(cy - rh * 0.5), rw, rh).clamp(self.W, self.H)

    def _resize_subframe(self) -> None:
        if self.mode != 4:
            self.subframe_rect = Rect(0, 0, self.W, self.H)
            return
        fw = int(self.subframe_percent[0] * self.W)
        fh = int(self.subframe_percent[1] * self.H)
        cx, cy = self.subframe_rect.center()
        self.subframe_rect = Rect(int(cx - fw * 0.5), int(cy - fh * 0.5), fw, fh).clamp(self.W, self.H)

    def click_set_roi_center(self, cx: int, cy: int) -> None:
        self.manual_roi_center = (int(cx), int(cy))
        self._move_roi_to_center(self.manual_roi_center)

    def _move_roi_to_center(self, center: Tuple[int, int]) -> None:
        rw, rh = self.roi_rect.w, self.roi_rect.h
        self.roi_rect = Rect(int(center[0] - rw * 0.5), int(center[1] - rh * 0.5), rw, rh).clamp(self.W, self.H)

    def _center_inside_subframe(self, cx: float, cy: float) -> Tuple[int, int]:
        # Clamp desired center into subframe bounds
        sx, sy, sw, sh = self.subframe_rect.as_tuple()
        cx = max(sx, min(cx, sx + sw))
        cy = max(sy, min(cy, sy + sh))
        return int(cx), int(cy)

    def get_rects(self) -> Tuple[Rect, Rect]:
        """Return current (subframe_rect, roi_rect) in global coordinates."""
        if self.mode == 1:
            full = Rect(0, 0, self.W, self.H)
            return full, full
        if self.mode == 4:
            # Ensure ROI stays within subframe
            sx, sy, sw, sh = self.subframe_rect.as_tuple()
            rw, rh = self.roi_rect.w, self.roi_rect.h
            cx, cy = self.roi_rect.center()
            cx = max(sx + rw * 0.5, min(cx, sx + sw - rw * 0.5))
            cy = max(sy + rh * 0.5, min(cy, sy + sh - rh * 0.5))
            self._move_roi_to_center((int(cx), int(cy)))
        return self.subframe_rect, self.roi_rect

    def step(self, detections: List[Dict], autofocus: bool) -> None:
        """Update internal subframe/roi centers based on detections and mode.
        Call this AFTER running detection on the current ROI.
        """
        if self.W <= 0 or self.H <= 0:
            return

        if not autofocus or not detections:
            # nothing to update
            return

        # Compute a simple target center (highest conf)
        best = None
        best_key = (-1.0, -1.0)
        for d in detections:
            x1, y1, x2, y2 = d["xyxy"]
            conf = float(d.get("conf", 0.0))
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            key = (conf, area)
            if key > best_key:
                best_key = key
                best = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        if best is None:
            return

        if self.mode == 3:
            self._move_roi_to_center((int(best[0]), int(best[1])))
        elif self.mode == 4:
            # Move subframe center towards target; keep ROI centered in subframe by default
            fw = int(self.subframe_percent[0] * self.W)
            fh = int(self.subframe_percent[1] * self.H)
            cx, cy = int(best[0]), int(best[1])
            self.subframe_rect = Rect(int(cx - fw * 0.5), int(cy - fh * 0.5), fw, fh).clamp(self.W, self.H)
            # Keep ROI centered on the target as default behavior
            self._move_roi_to_center(self._center_inside_subframe(cx, cy))
