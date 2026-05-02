"""Public wrapper for the hybrid lane detection pipeline.

Designed to be imported from another project:

    from hybrid_lane_detection import HybridLaneDetector

    detector = HybridLaneDetector(weights_path="weights/culane_18.pth")
    result = detector.process(frame)        # headless: structured output only
    overlay = detector.render(frame, result) # 1-up overlay for embedding in a UI
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import CONFIG, Config
from .models.classical import ClassicalBackend
from .models.dl_backend import DLBackend
from .renderers import LaneRenderer
from .utils.data_types import LaneData

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LaneResult:
    """Single-frame output bundling both backends plus reusable intermediates."""
    classical: LaneData
    dl: Any                  # ufld.LaneResult or None — kept loose to avoid hard-coupling
    gray: np.ndarray
    edges: np.ndarray
    dl_active: bool


class HybridLaneDetector:
    def __init__(
        self,
        config: Config = CONFIG,
        weights_path: Optional[str] = None,
    ):
        self._config = config
        self._classical = ClassicalBackend(config.classical)
        self._renderer = LaneRenderer(config.render)
        self._dl = self._try_init_dl(weights_path or config.dl.weights_path)

    @property
    def dl_active(self) -> bool:
        return self._dl is not None

    def process(self, frame: np.ndarray) -> LaneResult:
        classical_data, gray, edges = self._classical.process(frame)
        dl_result = self._dl.process(frame) if self._dl is not None else None
        return LaneResult(
            classical=classical_data,
            dl=dl_result,
            gray=gray,
            edges=edges,
            dl_active=self._dl is not None,
        )

    def render(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """1-up frame with classical carpet, UFLD dots, and HUD."""
        return self._renderer.render_prediction_tile(
            frame.copy(), result.classical, result.dl, result.dl_active
        )

    def render_debug(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """4-tile grid: original, grayscale+blur, canny+ROI, prediction."""
        return self._renderer.create_debug_tile(
            frame, result.gray, result.edges, result.classical, result.dl, result.dl_active
        )

    def _try_init_dl(self, weights_path: str) -> Optional[DLBackend]:
        cfg = self._config.dl
        if not os.path.isfile(weights_path) or os.path.getsize(weights_path) == 0:
            log.warning("UFLD weights not found at '%s' — running classical only.", weights_path)
            return None
        try:
            return DLBackend(
                weights_path=weights_path,
                dataset=cfg.dataset,
                backbone=cfg.backbone,
                device=cfg.device,
            )
        except Exception as e:
            log.warning("Failed to init UFLD backend (%s) — running classical only.", e)
            return None
