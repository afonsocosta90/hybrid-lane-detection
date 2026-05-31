"""Pure-math contract tests for perception.geometry.

No torch, no cv2, no .mp4/.pth fixtures. Pins sign conventions, identity
round-trips, and monotonicity — the things that would silently regress.
"""
from __future__ import annotations

import numpy as np
import pytest

from perception import geometry

# Match the defaults in perception/configs/default.yaml so tests reflect
# the real deployed geometry.
ROI_TL = (400, 350)
ROI_TR = (1050, 350)
FRAME_WH = (1280, 720)
ROAD_DEPTH_M = 30.0
LANE_WIDTH_M = 3.5


@pytest.fixture
def H():
    return geometry.build_road_homography(
        ROI_TL, ROI_TR, FRAME_WH, ROAD_DEPTH_M, LANE_WIDTH_M,
    )


@pytest.fixture
def H_inv(H):
    return np.linalg.inv(H)


def test_homography_maps_corners_to_expected_world_points(H):
    half_w = LANE_WIDTH_M / 2.0
    expected = [
        ((0.0, FRAME_WH[1]),          (-half_w, 0.0)),
        ((FRAME_WH[0], FRAME_WH[1]),  ( half_w, 0.0)),
        (ROI_TR,                      ( half_w, ROAD_DEPTH_M)),
        (ROI_TL,                      (-half_w, ROAD_DEPTH_M)),
    ]
    for (u, v), (x_w, y_w) in expected:
        got = geometry.pixel_to_world(H, u, v)
        np.testing.assert_allclose(got, (x_w, y_w), atol=1e-9)


def test_world_to_pixel_round_trip(H, H_inv):
    # 4 corners (1m steps in world, sampled inside the road rectangle).
    for x_w, y_w in [(-1.0, 5.0), (0.0, 10.0), (1.0, 15.0), (0.5, 25.0)]:
        u, v = geometry.world_to_pixel(H_inv, x_w, y_w)
        x_back, y_back = geometry.pixel_to_world(H, u, v)
        np.testing.assert_allclose((x_back, y_back), (x_w, y_w), atol=1e-7)


def test_steering_angle_zero_on_perfectly_straight_lane():
    # Lane center sits exactly on the frame midline (640 = 1280/2).
    left = np.array([0.0, 0.0, 500.0])
    right = np.array([0.0, 0.0, 780.0])
    angle = geometry.steering_angle_deg(left, right, FRAME_WH, y_eval_px=670)
    assert abs(angle) < 0.05


def test_steering_angle_positive_when_lane_curves_right():
    # Positive 'a' on the second-order term: dx/dy > 0 at the evaluation row.
    # Centerline at y=670: a*y^2 + (cl+cr)/2 = 1e-3 * 670^2 + 640 = 1089 > 640.
    # Both terms positive — sign convention says positive = steer right.
    left = np.array([1e-3, 0.0, 500.0])
    right = np.array([1e-3, 0.0, 780.0])
    angle = geometry.steering_angle_deg(left, right, FRAME_WH, y_eval_px=670)
    assert angle > 0


def test_steering_angle_negative_when_lane_offset_left():
    # Centerline at x=500 (vs frame center 640): the lane is to the LEFT of
    # the car; the car should steer LEFT to recenter. Negative.
    left = np.array([0.0, 0.0, 360.0])
    right = np.array([0.0, 0.0, 640.0])
    angle = geometry.steering_angle_deg(left, right, FRAME_WH, y_eval_px=670)
    assert angle < 0


def test_lookahead_pixel_rises_as_lookahead_m_grows(H, H_inv):
    # Looking farther into world space corresponds to a higher row in the
    # image (smaller v). Pins the world-to-pixel direction.
    left = np.array([0.0, 0.0, 500.0])
    right = np.array([0.0, 0.0, 780.0])
    *_, v_near = geometry.lookahead_point(H, H_inv, left, right, lookahead_m=8.0)
    *_, v_far = geometry.lookahead_point(H, H_inv, left, right, lookahead_m=25.0)
    assert v_far < v_near


def test_lookahead_world_y_matches_requested_distance(H, H_inv):
    # The lane fit's centerline at x_image=640 is the IMAGE midline. The
    # world's x=0 axis is the trapezoid's center at each row, which differs
    # from x_image=640 wherever the trapezoid is asymmetric (e.g. near edge
    # center 640 but far edge center 725 in the default ROI). So we only
    # pin y_world here — x_world is a real CTE in meters and is bounded by
    # the lane width, not zero, on this geometry.
    left = np.array([0.0, 0.0, 500.0])
    right = np.array([0.0, 0.0, 780.0])
    x_w, y_w, _, _ = geometry.lookahead_point(H, H_inv, left, right, lookahead_m=15.0)
    assert abs(y_w - 15.0) < 1e-6
    assert abs(x_w) < LANE_WIDTH_M / 2.0


def test_lane_center_average_of_two_fits():
    left = np.array([0.0, 0.0, 100.0])
    right = np.array([0.0, 0.0, 300.0])
    assert geometry.lane_center_x_at_y(left, right, y=400.0) == 200.0
