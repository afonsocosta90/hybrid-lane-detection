"""Lane module — adapter wrapping the existing HybridLaneDetector.

Converts pydantic config to the legacy frozen-dataclass config once at warmup so
the classical backend never sees pydantic. UFLD is intentionally allowed to
degrade (classical-only) if weights are missing, matching the existing library
behavior — the lane module is the one place where soft fallback is OK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from hybrid_lane_detection.config import (
    ClassicalConfig as LegacyClassicalConfig,
)
from hybrid_lane_detection.config import (
    Config as LegacyConfig,
)
from hybrid_lane_detection.config import (
    DLConfig as LegacyDLConfig,
)
from hybrid_lane_detection.config import (
    RenderConfig as LegacyRenderConfig,
)
from hybrid_lane_detection.config import (
    VideoConfig as LegacyVideoConfig,
)
from hybrid_lane_detection.lane_detector import HybridLaneDetector
from hybrid_lane_detection.utils.data_types import LaneData
from perception.config import AppConfig
from perception.contracts import PerceptionFrame, PerceptionState


@dataclass(frozen=True, slots=True)
class LaneOutput:
    classical: LaneData
    dl: Any  # ufld.LaneResult or None — loose to avoid hard-coupling to UFLD types
    dl_active: bool


def _legacy_config_from(app: AppConfig) -> LegacyConfig:
    return LegacyConfig(
        video=LegacyVideoConfig(**app.video.model_dump()),
        dl=LegacyDLConfig(**app.dl.model_dump()),
        classical=LegacyClassicalConfig(**app.classical.model_dump()),
        render=LegacyRenderConfig(**app.render.model_dump()),
    )


class LaneModule:
    name: ClassVar[str] = "lane"
    stage: ClassVar[int] = 0

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._detector: HybridLaneDetector | None = None

    def warmup(self) -> None:
        legacy = _legacy_config_from(self._cfg)
        self._detector = HybridLaneDetector(config=legacy)

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> LaneOutput:
        assert self._detector is not None, "LaneModule.process called before warmup"
        result = self._detector.process(frame.bgr)
        return LaneOutput(
            classical=result.classical,
            dl=result.dl,
            dl_active=result.dl_active,
        )

    @property
    def dl_active(self) -> bool:
        return self._detector is not None and self._detector.dl_active
