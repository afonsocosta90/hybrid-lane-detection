"""SteeringOverlay — draws the lookahead dot and the steering angle text."""
from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from perception.contracts import PerceptionState


class SteeringOverlay:
    name: ClassVar[str] = "steering"

    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray:
        s = state.steering()
        u, v = s.lookahead_pixel_uv
        h, w = canvas.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return canvas

        # Lookahead dot — yellow, ringed for visibility against any background.
        cv2.circle(canvas, (u, v), 10, (0, 0, 0), 2)
        cv2.circle(canvas, (u, v), 7, (0, 255, 255), -1)

        # Vertical line from the bottom-center of the frame to the lookahead
        # dot — a visual "future trajectory" cue.
        bottom_center = (w // 2, h - 1)
        cv2.line(canvas, bottom_center, (u, v), (0, 255, 255), 1)

        # Angle text + cross-track distance, just below the dot.
        label_y = min(h - 5, v + 26)
        text = f"{s.angle_deg:+.1f} deg | CTE {s.cross_track_error_m:+.2f}m"
        cv2.putText(
            canvas, text, (max(5, u - 100), label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
        )
        cv2.putText(
            canvas, text, (max(5, u - 100), label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        return canvas
