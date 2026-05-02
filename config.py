"""Centralized tunables for the hybrid lane detection pipeline.

Edit values here to retune the pipeline for a different video, camera,
or scene. Each section corresponds to the module that consumes it.

Import the singleton:
    from config import CONFIG
    CONFIG.classical.ema_alpha
"""
from dataclasses import dataclass, field
from typing import Tuple


# ----- Video / I/O -----
@dataclass(frozen=True)
class VideoConfig:
    path: str = "test-videos/test_video_0.mp4"
    window_name: str = "Lane Detection Pipeline"
    # cv2.waitKey delay in ms (~40 fps at 25 ms).
    wait_key_ms: int = 25


# ----- UFLD deep-learning backend -----
@dataclass(frozen=True)
class DLConfig:
    weights_path: str = "weights/culane_18.pth"
    dataset: str = "culane"          # "culane" or "tusimple"
    backbone: str = "18"             # "18" or "34"
    device: str = "auto"             # "auto" -> DirectML -> CUDA -> CPU


# ----- Classical CV backend -----
@dataclass(frozen=True)
class ClassicalConfig:
    # Pre-processing
    blur_kernel: Tuple[int, int] = (3, 3)
    canny_low: int = 50
    canny_high: int = 150

    # ROI polygon top corners (bottom corners always hug frame edges).
    # Tuned for 1280x720 dashcam frames — retune for different resolutions.
    roi_top_left: Tuple[int, int] = (400, 350)
    roi_top_right: Tuple[int, int] = (1050, 350)

    # Hough line detection
    hough_rho: float = 1.5
    hough_theta_divisor: int = 360   # theta = pi / hough_theta_divisor
    hough_threshold: int = 40
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 50

    # Line filtering
    min_angle_deg: float = 30.0
    max_angle_deg: float = 150.0
    min_points_per_side: int = 5

    # EMA smoothing of polynomial coefficients across frames
    ema_alpha: float = 0.3           # 0..1, larger = more responsive
    min_trust: float = 0.4           # raw conf below this skips EMA update

    # Confidence — perspective-aware sanity check.
    # Lane-width score (in pixels at the "near" row).
    width_min_px: float = 500.0
    width_max_px: float = 1000.0
    width_target_px: float = 750.0
    width_falloff_px: float = 400.0
    # Perspective convergence (far row narrower than near row).
    convergence_ratio_min: float = 0.1
    convergence_ratio_max: float = 0.7
    convergence_ratio_target: float = 0.4
    convergence_falloff: float = 2.0
    # Final-score weights (must sum to 1.0).
    width_weight: float = 0.5
    convergence_weight: float = 0.25
    heading_weight: float = 0.25
    # Sample-row geometry for the score.
    y_near_offset_px: int = 50       # rows above frame bottom
    y_far_factor: float = 0.55       # fraction of frame height


# ----- Rendering -----
@dataclass(frozen=True)
class RenderConfig:
    # Green-carpet overlay
    render_y_top: int = 500          # top row of the carpet
    confidence_threshold: float = 0.35   # below this: hide carpet, show SEARCHING

    # HUD box (top-right corner)
    hud_box_width: int = 340
    hud_box_height: int = 130
    hud_margin: int = 10

    # 4-tile debug view final size (post-resize)
    debug_tile_size: Tuple[int, int] = (1280, 720)


# ----- Top-level container -----
@dataclass(frozen=True)
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    dl: DLConfig = field(default_factory=DLConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


CONFIG = Config()
