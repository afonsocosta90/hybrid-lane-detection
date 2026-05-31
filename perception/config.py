"""Pydantic v2 config tree, loaded from YAML at startup.

`load_config(path)` is the fail-loud entry point. Pydantic catches structural,
range, and cross-field invariants before any module is constructed. A second
fail-loud pass happens in each module's warmup() for disk / device / matrix
checks pydantic can't see.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class _Strict(BaseModel):
    """Base model: forbid extra fields so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VideoConfig(_Strict):
    path: str
    window_name: str = "Perception Pipeline"
    wait_key_ms: Annotated[int, Field(ge=1, le=1000)] = 25


class DLConfig(_Strict):
    weights_path: str
    dataset: str = "culane"
    backbone: str = "18"
    device: str = "auto"

    @field_validator("dataset")
    @classmethod
    def _dataset_known(cls, v: str) -> str:
        if v not in {"culane", "tusimple"}:
            raise ValueError(f"dataset must be 'culane' or 'tusimple', got {v!r}")
        return v

    @field_validator("backbone")
    @classmethod
    def _backbone_known(cls, v: str) -> str:
        if v not in {"18", "34"}:
            raise ValueError(f"backbone must be '18' or '34', got {v!r}")
        return v


class ClassicalConfig(_Strict):
    blur_kernel: tuple[int, int] = (3, 3)
    canny_low: Annotated[int, Field(ge=0, le=500)] = 50
    canny_high: Annotated[int, Field(ge=0, le=500)] = 150

    roi_top_left: tuple[int, int] = (400, 350)
    roi_top_right: tuple[int, int] = (1050, 350)

    hough_rho: Annotated[float, Field(gt=0.0)] = 1.5
    hough_theta_divisor: Annotated[int, Field(ge=1)] = 360
    hough_threshold: Annotated[int, Field(ge=1)] = 40
    hough_min_line_length: Annotated[int, Field(ge=1)] = 100
    hough_max_line_gap: Annotated[int, Field(ge=0)] = 50

    min_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 30.0
    max_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 150.0
    min_points_per_side: Annotated[int, Field(ge=1)] = 5

    ema_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    min_trust: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4

    width_min_px: Annotated[float, Field(gt=0.0)] = 500.0
    width_max_px: Annotated[float, Field(gt=0.0)] = 1000.0
    width_target_px: Annotated[float, Field(gt=0.0)] = 750.0
    width_falloff_px: Annotated[float, Field(gt=0.0)] = 400.0
    convergence_ratio_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    convergence_ratio_max: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    convergence_ratio_target: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    convergence_falloff: Annotated[float, Field(gt=0.0)] = 2.0

    width_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    convergence_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    heading_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25

    y_near_offset_px: Annotated[int, Field(ge=0)] = 50
    y_far_factor: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.55

    @field_validator("heading_weight")
    @classmethod
    def _weights_sum_to_one(cls, v: float, info: ValidationInfo) -> float:
        w = info.data.get("width_weight", 0.0) + info.data.get("convergence_weight", 0.0) + v
        if abs(w - 1.0) > 1e-6:
            raise ValueError(
                f"classical scoring weights must sum to 1.0, got {w} "
                f"(width={info.data.get('width_weight')}, "
                f"convergence={info.data.get('convergence_weight')}, heading={v})"
            )
        return v

    @field_validator("max_angle_deg")
    @classmethod
    def _max_gt_min(cls, v: float, info: ValidationInfo) -> float:
        mn = info.data.get("min_angle_deg", 0.0)
        if v <= mn:
            raise ValueError(f"max_angle_deg ({v}) must be > min_angle_deg ({mn})")
        return v


class ObjectsConfig(_Strict):
    weights_path: str = "weights/yolov8n.pt"
    device: str = "auto"
    conf_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    keep_classes: tuple[str, ...] = ("car", "truck", "person", "bicycle", "motorcycle")
    imgsz: Annotated[int, Field(ge=64, le=2048)] = 640


class SteeringConfig(_Strict):
    lookahead_m: Annotated[float, Field(gt=0.0)] = 20.0
    road_depth_m: Annotated[float, Field(gt=0.0)] = 30.0
    lane_width_m: Annotated[float, Field(gt=0.0)] = 3.5
    y_eval_offset_px: Annotated[int, Field(ge=0)] = 50


class RenderConfig(_Strict):
    render_y_top: Annotated[int, Field(ge=0)] = 500
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    hud_box_width: Annotated[int, Field(ge=1)] = 340
    hud_box_height: Annotated[int, Field(ge=1)] = 130
    hud_margin: Annotated[int, Field(ge=0)] = 10
    debug_tile_size: tuple[int, int] = (1280, 720)


class AppConfig(_Strict):
    video: VideoConfig
    dl: DLConfig
    classical: ClassicalConfig
    objects: ObjectsConfig
    steering: SteeringConfig
    render: RenderConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML config. Raises pydantic ValidationError on any failure."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig.model_validate(data)
