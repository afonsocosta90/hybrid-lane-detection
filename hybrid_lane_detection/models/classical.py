import cv2
import numpy as np
import math

from ..config import CONFIG, ClassicalConfig
from ..utils.data_types import LaneData


class ClassicalBackend:
    def __init__(self, cfg: ClassicalConfig = CONFIG.classical):
        self.cfg = cfg
        self._left_ema = None
        self._right_ema = None

    def process(self, frame: np.ndarray):
        cfg = self.cfg
        h, w = frame.shape[:2]

        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, cfg.blur_kernel, 0)
        edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)

        # 2. Apply ROI Mask
        masked_edges = self._apply_roi_mask(edges)

        # 3. Line Detection
        lines = cv2.HoughLinesP(
            masked_edges,
            cfg.hough_rho,
            np.pi / cfg.hough_theta_divisor,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )

        if lines is None:
            return self._empty_data(), gray, masked_edges

        # 4. Separate and Filter by Angle
        l_pts, r_pts = self._separate_and_filter(lines, w // 2)

        if len(l_pts) < cfg.min_points_per_side or len(r_pts) < cfg.min_points_per_side:
            return self._empty_data(), gray, masked_edges

        # 5. Polynomial Fit (x = a*y^2 + b*y + c)
        l_fit = np.polyfit(l_pts[:, 1], l_pts[:, 0], 2)
        r_fit = np.polyfit(r_pts[:, 1], r_pts[:, 0], 2)

        # 6. Confidence on the *raw* fit (before smoothing) so a bad fit can
        #    be rejected without poisoning the EMA.
        raw_conf = self._calculate_confidence(l_fit, r_fit, h)

        # 7. Smooth coefficients across frames if the raw fit is trustworthy.
        if raw_conf >= cfg.min_trust:
            l_fit, r_fit = self._smooth(l_fit, r_fit)
            conf = raw_conf
        elif self._left_ema is not None:
            # Hold the last good fit for one frame; report the raw (low) conf
            # so downstream knows this frame was weak.
            l_fit, r_fit = self._left_ema, self._right_ema
            conf = raw_conf
        else:
            return self._empty_data(), gray, masked_edges

        return LaneData(l_fit, r_fit, conf, 'classical'), gray, masked_edges

    def _calculate_confidence(self, l_fit, r_fit, h):
        """Perspective-aware sanity check. Evaluates the two polynomials at
        a row near the car and a row near the horizon, then scores lane
        width, perspective convergence, and heading symmetry."""
        if l_fit is None or r_fit is None:
            return 0.0

        cfg = self.cfg
        y_near = h - cfg.y_near_offset_px
        y_far = int(h * cfg.y_far_factor)

        def x_at(fit, y):
            return fit[0] * y * y + fit[1] * y + fit[2]

        lx_near, rx_near = x_at(l_fit, y_near), x_at(r_fit, y_near)
        lx_far,  rx_far  = x_at(l_fit, y_far),  x_at(r_fit, y_far)

        width_near = rx_near - lx_near
        width_far = rx_far - lx_far

        # 1. Lane width near the car.
        if cfg.width_min_px <= width_near <= cfg.width_max_px:
            width_score = 1.0
        else:
            width_score = max(
                0.0,
                1.0 - abs(width_near - cfg.width_target_px) / cfg.width_falloff_px,
            )

        # 2. Perspective convergence: far row should be narrower than near.
        if width_near <= 0:
            convergence_score = 0.0
        else:
            ratio = width_far / width_near
            if cfg.convergence_ratio_min <= ratio <= cfg.convergence_ratio_max:
                convergence_score = 1.0
            else:
                convergence_score = max(
                    0.0,
                    1.0 - abs(ratio - cfg.convergence_ratio_target) * cfg.convergence_falloff,
                )

        # 3. Heading: tangent dx/dy = 2*a*y + b. At y_near the left tangent
        #    should be negative, the right positive, and they should mirror.
        tangent_l = 2.0 * l_fit[0] * y_near + l_fit[1]
        tangent_r = 2.0 * r_fit[0] * y_near + r_fit[1]
        if tangent_l < 0 and tangent_r > 0:
            mirror_err = abs(tangent_l + tangent_r) / (abs(tangent_l) + abs(tangent_r) + 1e-6)
            heading_score = max(0.0, 1.0 - mirror_err)
        else:
            heading_score = 0.0

        score = (
            cfg.width_weight * width_score
            + cfg.convergence_weight * convergence_score
            + cfg.heading_weight * heading_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _smooth(self, l_fit, r_fit):
        a = self.cfg.ema_alpha
        if self._left_ema is None:
            self._left_ema = l_fit
            self._right_ema = r_fit
        else:
            self._left_ema = a * l_fit + (1.0 - a) * self._left_ema
            self._right_ema = a * r_fit + (1.0 - a) * self._right_ema
        return self._left_ema, self._right_ema

    def _separate_and_filter(self, lines, midpoint):
        cfg = self.cfg
        l_pts, r_pts = [], []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            if cfg.min_angle_deg < angle < cfg.max_angle_deg:
                if (x1 + x2) / 2 < midpoint:
                    l_pts.extend([(x1, y1), (x2, y2)])
                else:
                    r_pts.extend([(x1, y1), (x2, y2)])
        return np.array(l_pts), np.array(r_pts)

    def _apply_roi_mask(self, img):
        h, w = img.shape[:2]
        mask = np.zeros_like(img)

        polygon = np.array([[
            (0, h),
            (w, h),
            self.cfg.roi_top_right,
            self.cfg.roi_top_left,
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def _empty_data(self):
        return LaneData(np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, 'classical')
