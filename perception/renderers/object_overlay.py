"""ObjectOverlay — bounding boxes + class labels for detected road agents."""
from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from perception.contracts import PerceptionState

# Per-class BGR colors. Anything outside this dict gets a neutral gray.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car":        (255, 128,   0),  # orange
    "truck":      (255,  64,   0),  # red-orange
    "person":     (  0, 200, 255),  # yellow
    "bicycle":    (180, 100, 255),  # magenta
    "motorcycle": (255,   0, 200),  # pink
}
_DEFAULT_COLOR = (200, 200, 200)


class ObjectOverlay:
    name: ClassVar[str] = "objects"

    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray:
        out = state.objects()
        for det in out.detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
            color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label_top = max(0, y1 - th - 6)
            cv2.rectangle(canvas, (x1, y_label_top), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                canvas, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )
        return canvas
