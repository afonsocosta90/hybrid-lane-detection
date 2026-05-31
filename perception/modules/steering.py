"""Steering module — stage-1, reads LaneOutput, emits SteeringOutput.

All math lives in `perception.geometry` so this file is just glue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from perception import geometry
from perception.config import ClassicalConfig, SteeringConfig, VideoConfig
from perception.contracts import PerceptionFrame, PerceptionState


@dataclass(frozen=True, slots=True)
class SteeringOutput:
    angle_deg: float
    lookahead_world_xy: tuple[float, float]
    lookahead_pixel_uv: tuple[int, int]
    cross_track_error_m: float
    confidence: float


class SteeringModule:
    name: ClassVar[str] = "steering"
    stage: ClassVar[int] = 1

    def __init__(
        self,
        cfg: SteeringConfig,
        classical_cfg: ClassicalConfig,
        video_cfg: VideoConfig,
        frame_wh_fallback: tuple[int, int] = (1280, 720),
    ):
        self._cfg = cfg
        self._classical_cfg = classical_cfg
        self._video_cfg = video_cfg
        self._frame_wh_fallback = frame_wh_fallback
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._frame_wh: tuple[int, int] | None = None

    def warmup(self) -> None:
        # We don't know the actual frame size until the first frame arrives,
        # so we build the homography against the fallback (matching the ROI
        # tuning) and rebuild lazily if the first frame's size disagrees.
        self._frame_wh = self._frame_wh_fallback
        self._build_homography(self._frame_wh)

    def _build_homography(self, frame_wh: tuple[int, int]) -> None:
        self._H = geometry.build_road_homography(
            self._classical_cfg.roi_top_left,
            self._classical_cfg.roi_top_right,
            frame_wh,
            road_depth_m=self._cfg.road_depth_m,
            lane_width_m=self._cfg.lane_width_m,
        )
        self._H_inv = np.linalg.inv(self._H)
        self._frame_wh = frame_wh

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> SteeringOutput:
        lane = state.lane()
        h, w = frame.bgr.shape[:2]
        wh = (w, h)
        if wh != self._frame_wh:
            self._build_homography(wh)

        assert self._H is not None and self._H_inv is not None

        left = lane.classical.left_fit
        right = lane.classical.right_fit
        y_eval = h - self._cfg.y_eval_offset_px

        angle = geometry.steering_angle_deg(left, right, wh, y_eval)
        x_world, y_world, u, v = geometry.lookahead_point(
            self._H, self._H_inv, left, right, self._cfg.lookahead_m,
        )

        return SteeringOutput(
            angle_deg=angle,
            lookahead_world_xy=(x_world, y_world),
            lookahead_pixel_uv=(int(u), int(v)),
            cross_track_error_m=x_world,
            confidence=float(lane.classical.confidence),
        )
