"""HudOverlay — pipeline-wide dashboard: FPS, per-module ms, error badges.

This is the one overlay allowed to look across modules. Drawn last so it
sits on top of everything else.
"""
from __future__ import annotations

import time
from collections import deque
from typing import ClassVar

import cv2
import numpy as np

from perception.contracts import PerceptionState

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class HudOverlay:
    name: ClassVar[str] = "hud"

    def __init__(self, fps_window: int = 30):
        self._frame_times: deque[float] = deque(maxlen=fps_window)

    def _fps(self) -> float:
        now = time.perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray:
        fps = self._fps()
        h, w = canvas.shape[:2]

        x, y = 12, 12
        lh = 22  # line height

        panel_w = 360
        panel_h = lh * (3 + len(state.module_timings_ms) + len(state.errors)) + 14
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

        cy = y + lh
        cv2.putText(canvas, f"FPS  {fps:5.1f}", (x + 10, cy), _FONT, 0.6, (0, 255, 0), 2)
        cy += lh

        # Per-module ms — sorted by descending so the bottleneck reads top.
        ordered = sorted(state.module_timings_ms.items(), key=lambda kv: -kv[1])
        for name, ms in ordered:
            line = f"{name:<10} {ms:6.1f} ms"
            cv2.putText(canvas, line, (x + 10, cy), _FONT, 0.5, (220, 220, 220), 1)
            cy += lh

        cy += 4

        # Lane status (if available)
        try:
            lane = state.lane()
            line = f"lane  conf {lane.classical.confidence * 100:5.1f}%  UFLD {'ON' if lane.dl_active else 'OFF'}"
            cv2.putText(canvas, line, (x + 10, cy), _FONT, 0.5, (180, 230, 255), 1)
            cy += lh
        except Exception:  # noqa: BLE001 — accessor may raise; HUD must not crash
            pass

        # Steering summary (if available)
        try:
            s = state.steering()
            line = f"steer {s.angle_deg:+5.1f} deg  look {s.lookahead_world_xy[1]:4.1f} m"
            cv2.putText(canvas, line, (x + 10, cy), _FONT, 0.5, (200, 255, 200), 1)
            cy += lh
        except Exception:  # noqa: BLE001
            pass

        # Object count (if available)
        try:
            o = state.objects()
            line = f"objs  {len(o.detections):3d}"
            cv2.putText(canvas, line, (x + 10, cy), _FONT, 0.5, (200, 200, 255), 1)
            cy += lh
        except Exception:  # noqa: BLE001
            pass

        # Error badges — red so failures are visible, not silent.
        for name in state.errors:
            cv2.putText(canvas, f"!! {name}: ERROR", (x + 10, cy), _FONT, 0.55, (0, 0, 255), 2)
            cy += lh

        return canvas
