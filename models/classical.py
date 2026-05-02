import cv2
import numpy as np
import math
from utils.data_types import LaneData


class ClassicalBackend:
    # EMA smoothing factor for polynomial coefficients (0..1, larger = more responsive)
    EMA_ALPHA = 0.3
    # Raw confidence below this is treated as untrusted: the EMA is not updated.
    MIN_TRUST = 0.4

    def __init__(self):
        self._left_ema = None
        self._right_ema = None

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. Apply ROI Mask
        masked_edges = self._apply_roi_mask(edges)

        # 3. Line Detection
        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            return self._empty_data(), gray, masked_edges

        # 4. Separate and Filter by Angle
        l_pts, r_pts = self._separate_and_filter(lines, w // 2)

        if len(l_pts) < 5 or len(r_pts) < 5:
            return self._empty_data(), gray, masked_edges

        # 5. Polynomial Fit (x = a*y^2 + b*y + c)
        l_fit = np.polyfit(l_pts[:, 1], l_pts[:, 0], 2)
        r_fit = np.polyfit(r_pts[:, 1], r_pts[:, 0], 2)

        # 6. Confidence on the *raw* fit (before smoothing) so a bad fit can
        #    be rejected without poisoning the EMA.
        raw_conf = self._calculate_confidence(l_fit, r_fit, h)

        # 7. Smooth coefficients across frames if the raw fit is trustworthy.
        if raw_conf >= self.MIN_TRUST:
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
        """
        Perspective-aware sanity check. Evaluates the two polynomials at
        a row near the car and a row near the horizon, then scores:
        - lane width near the car (must be physically plausible),
        - perspective convergence (far row narrower than near row),
        - heading (left tangent points one way, right the other, mirrored).
        """
        if l_fit is None or r_fit is None:
            return 0.0

        y_near = h - 50
        y_far = int(h * 0.55)  # just below the ROI top in a 720-tall frame

        def x_at(fit, y):
            return fit[0] * y * y + fit[1] * y + fit[2]

        lx_near, rx_near = x_at(l_fit, y_near), x_at(r_fit, y_near)
        lx_far,  rx_far  = x_at(l_fit, y_far),  x_at(r_fit, y_far)

        width_near = rx_near - lx_near
        width_far = rx_far - lx_far

        # 1. Lane width near the car. Tuned for a 1280-wide dashcam.
        if 500 <= width_near <= 1000:
            width_score = 1.0
        else:
            width_score = max(0.0, 1.0 - abs(width_near - 750) / 400.0)

        # 2. Perspective convergence: far row should be narrower than near row.
        if width_near <= 0:
            convergence_score = 0.0
        else:
            ratio = width_far / width_near  # healthy: ~0.2-0.6
            if 0.1 <= ratio <= 0.7:
                convergence_score = 1.0
            else:
                convergence_score = max(0.0, 1.0 - abs(ratio - 0.4) * 2.0)

        # 3. Heading: tangent dx/dy = 2*a*y + b. At y_near the left tangent
        #    should be negative, the right positive, and they should mirror.
        tangent_l = 2.0 * l_fit[0] * y_near + l_fit[1]
        tangent_r = 2.0 * r_fit[0] * y_near + r_fit[1]
        if tangent_l < 0 and tangent_r > 0:
            mirror_err = abs(tangent_l + tangent_r) / (abs(tangent_l) + abs(tangent_r) + 1e-6)
            heading_score = max(0.0, 1.0 - mirror_err)
        else:
            heading_score = 0.0

        score = 0.5 * width_score + 0.25 * convergence_score + 0.25 * heading_score
        return float(np.clip(score, 0.0, 1.0))

    def _smooth(self, l_fit, r_fit):
        a = self.EMA_ALPHA
        if self._left_ema is None:
            self._left_ema = l_fit
            self._right_ema = r_fit
        else:
            self._left_ema = a * l_fit + (1.0 - a) * self._left_ema
            self._right_ema = a * r_fit + (1.0 - a) * self._right_ema
        return self._left_ema, self._right_ema

    def _separate_and_filter(self, lines, midpoint):
        l_pts, r_pts = [], []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            # Wide angle filter for Normal Perspective (30 to 150 degrees)
            if 30 < angle < 150:
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
            (1050, 350),
            (400, 350)
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def _empty_data(self):
        return LaneData(np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, 'classical')
