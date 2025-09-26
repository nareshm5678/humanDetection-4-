from typing import List, Dict, Optional, Tuple


class SmoothTargetTracker:
    """Simple center tracker with exponential smoothing."""

    def __init__(self, alpha: float = 0.25, lost_max: int = 15) -> None:
        self.alpha = float(alpha)
        self.lost_max = int(lost_max)
        self.center: Optional[Tuple[float, float]] = None
        self.lost_count = 0

    @staticmethod
    def _pick_target(dets: List[Dict]) -> Optional[Tuple[float, float]]:
        if not dets:
            return None
        # Prefer highest confidence; tie-break by area
        best = None
        best_key = (-1.0, -1.0)
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            conf = float(d.get("conf", 0.0))
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            key = (conf, area)
            if key > best_key:
                best_key = key
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                best = (cx, cy)
        return best

    def update(self, dets: List[Dict], default_center: Tuple[float, float]) -> Tuple[Tuple[float, float], bool]:
        target = self._pick_target(dets)
        if target is None:
            self.lost_count += 1
            if self.center is None:
                self.center = default_center
            return self.center, self.lost_count >= self.lost_max

        self.lost_count = 0
        if self.center is None:
            self.center = target
        else:
            ax = self.alpha
            self.center = (ax * target[0] + (1 - ax) * self.center[0],
                           ax * target[1] + (1 - ax) * self.center[1])
        return self.center, False
