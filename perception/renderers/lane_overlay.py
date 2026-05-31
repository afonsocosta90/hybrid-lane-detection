"""LaneOverlay — wraps the existing LaneRenderer methods.

The legacy renderer's render_prediction_tile composes the classical carpet,
UFLD dots, and a lane-specific HUD. We keep that composition for now;
pipeline-wide HUD (FPS, errors) is its own overlay later.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from hybrid_lane_detection.config import RenderConfig as LegacyRenderConfig
from hybrid_lane_detection.renderers import LaneRenderer
from perception.config import RenderConfig
from perception.contracts import PerceptionState


class LaneOverlay:
    name: ClassVar[str] = "lane"

    def __init__(self, cfg: RenderConfig):
        self._inner = LaneRenderer(LegacyRenderConfig(**cfg.model_dump()))

    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray:
        lane = state.lane()
        out = self._inner.render_classical_overlay(canvas, lane.classical)
        out = self._inner.render_dl_overlay(out, lane.dl)
        out = self._inner.draw_description(out, lane.classical, lane.dl_active)
        return out
