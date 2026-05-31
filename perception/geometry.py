"""Pure math for steering and world projection.

This module MUST NOT import cv2 at top level — it is the unit-test surface.
Everything here takes numpy arrays and scalars and returns numpy arrays and
scalars. No frames, no model calls.

World frame convention: origin at the center of the near (bottom) edge of the
trapezoidal ROI. X right (meters), Y forward away from the car (meters).
Image frame: standard image coords, u right (pixels), v down (pixels).
"""
from __future__ import annotations

import math

import numpy as np


def _solve_homography(img_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """Direct linear solve for the 3x3 homography mapping image pixels -> world meters.

    Given 4 correspondences, builds an 8x8 system and solves with numpy.linalg.
    Returns a 3x3 matrix H such that (x, y, w) ~ H @ (u, v, 1).
    """
    if img_pts.shape != (4, 2) or world_pts.shape != (4, 2):
        raise ValueError(f"need 4x2 arrays, got {img_pts.shape} and {world_pts.shape}")
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        u, v = img_pts[i]
        x, y = world_pts[i]
        A[2 * i]     = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        A[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i]     = x
        b[2 * i + 1] = y
    h = np.linalg.solve(A, b)
    return np.array(
        [[h[0], h[1], h[2]],
         [h[3], h[4], h[5]],
         [h[6], h[7], 1.0]],
        dtype=np.float64,
    )


def build_road_homography(
    roi_top_left: tuple[int, int],
    roi_top_right: tuple[int, int],
    frame_wh: tuple[int, int],
    road_depth_m: float,
    lane_width_m: float,
) -> np.ndarray:
    """3x3 H mapping image pixel (u,v) -> world (X meters right, Y meters forward).

    Assumes a flat road. The ROI trapezoid's near edge is the lane's near edge
    (width = lane_width_m); its far edge is the lane's far edge at distance
    road_depth_m. This is the slice-1 calibration: no checkerboard, no UI.
    """
    W, H = frame_wh
    img_pts = np.array(
        [
            (0.0, float(H)),                                    # near-left
            (float(W), float(H)),                               # near-right
            (float(roi_top_right[0]), float(roi_top_right[1])), # far-right
            (float(roi_top_left[0]), float(roi_top_left[1])),   # far-left
        ],
        dtype=np.float64,
    )
    half_w = lane_width_m / 2.0
    world_pts = np.array(
        [
            (-half_w, 0.0),
            ( half_w, 0.0),
            ( half_w, road_depth_m),
            (-half_w, road_depth_m),
        ],
        dtype=np.float64,
    )
    return _solve_homography(img_pts, world_pts)


def pixel_to_world(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def world_to_pixel(H_inv: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H_inv @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def lane_center_x_at_y(left_fit: np.ndarray, right_fit: np.ndarray, y: float) -> float:
    lx = left_fit[0] * y * y + left_fit[1] * y + left_fit[2]
    rx = right_fit[0] * y * y + right_fit[1] * y + right_fit[2]
    return 0.5 * float(lx + rx)


def steering_angle_deg(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    frame_wh: tuple[int, int],
    y_eval_px: float,
    cte_weight: float = 0.7,
    heading_weight: float = 0.3,
) -> float:
    """Combined steering hint in degrees. Positive = steer right.

    Two terms:
      - Cross-track error: lane center offset from frame midline, normalized
        by frame half-width and converted to an angle via atan2.
      - Heading error: tangent dx/dy of the centerline at y_eval, converted
        via atan.

    The weights bias toward the cross-track term because it is the strongest
    signal in straight-road geometry; heading dominates only on curves.
    """
    W, _ = frame_wh
    a_avg = 0.5 * (left_fit[0] + right_fit[0])
    b_avg = 0.5 * (left_fit[1] + right_fit[1])

    center_x = lane_center_x_at_y(left_fit, right_fit, y_eval_px)
    cte_px = center_x - 0.5 * W

    tangent = 2.0 * a_avg * y_eval_px + b_avg

    cte_deg = math.degrees(math.atan2(cte_px, 0.5 * W))
    heading_deg = math.degrees(math.atan(tangent))

    return cte_weight * cte_deg + heading_weight * heading_deg


def lookahead_point(
    H: np.ndarray,
    H_inv: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    lookahead_m: float,
) -> tuple[float, float, float, float]:
    """Lane center at `lookahead_m` meters forward.

    Two-step solve:
      1. Project the world point (x=0, y=lookahead_m) into the image to get
         a row v_lookahead and a centered-road column u_centered.
      2. Evaluate the lane center polynomial at v_lookahead to get the actual
         pixel u, then re-project (u, v_lookahead) back to world to get the
         true world X (the cross-track error in meters).

    Returns (x_world, y_world, u_pixel, v_pixel). y_world is approximately
    lookahead_m by construction.
    """
    _, v_lookahead = world_to_pixel(H_inv, 0.0, lookahead_m)
    u_actual = lane_center_x_at_y(left_fit, right_fit, v_lookahead)
    x_world, y_world = pixel_to_world(H, u_actual, v_lookahead)
    return x_world, y_world, u_actual, v_lookahead
