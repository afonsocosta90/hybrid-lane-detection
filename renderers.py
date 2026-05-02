import cv2
import numpy as np

class LaneRenderer:
    def __init__(self, render_y_top=500, render_conf_threshold=0.35):
        self.render_y_top = render_y_top
        self.render_conf_threshold = render_conf_threshold

    def render_classical_overlay(self, frame, data):
        if data.confidence < self.render_conf_threshold:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        plot_y = np.linspace(self.render_y_top, h - 1, 30)
        l_x = data.left_fit[0] * plot_y ** 2 + data.left_fit[1] * plot_y + data.left_fit[2]
        r_x = data.right_fit[0] * plot_y ** 2 + data.right_fit[1] * plot_y + data.right_fit[2]

        # Clamp so a noisy polynomial can't draw off-screen.
        l_x = np.clip(l_x, 0, w - 1)
        r_x = np.clip(r_x, 0, w - 1)

        pts_left = np.array([np.transpose(np.vstack([l_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
        return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

    def render_dl_overlay(self, frame, dl_result):
        if dl_result is None:
            return frame
        for lane in dl_result.lanes:
            for (x, y) in lane:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        return frame

    def draw_description(self, img, data, dl_active):
        w = img.shape[1]
        box_width = 340
        box_height = 130
        margin = 10
        tr_x1 = w - box_width - margin
        tr_y1 = margin
        tr_x2 = w - margin
        tr_y2 = margin + box_height

        overlay = img.copy()
        cv2.rectangle(overlay, (tr_x1, tr_y1), (tr_x2, tr_y2), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = tr_x1 + 10

        status_text = "LANE TRACKED" if data.confidence >= self.render_conf_threshold else "SEARCHING..."
        status_color = (0, 255, 0) if data.confidence >= self.render_conf_threshold else (0, 0, 255)
        cv2.putText(img, status_text, (text_x, tr_y1 + 30), font, 0.7, status_color, 2)
        cv2.putText(img, f"Classical conf: {data.confidence * 100:.1f}%", (text_x, tr_y1 + 60), font, 0.55, (0, 255, 0), 1)
        dl_label = "UFLD: ON  (red dots)" if dl_active else "UFLD: OFF (no weights)"
        dl_color = (0, 0, 255) if dl_active else (128, 128, 128)
        cv2.putText(img, dl_label, (text_x, tr_y1 + 90), font, 0.55, dl_color, 1)
        cv2.putText(img, "[space] pause   [q] quit", (text_x, tr_y1 + 118), font, 0.45, (200, 200, 200), 1)

        return img

    def render_prediction_tile(self, frame, classical_data, dl_result, dl_active):
        out = self.render_classical_overlay(frame, classical_data)
        out = self.render_dl_overlay(out, dl_result)
        out = self.draw_description(out, classical_data, dl_active)
        return out

    def create_debug_tile(self, frame, gray, edges, classical_data, dl_result, dl_active):
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        res_window = self.render_prediction_tile(frame.copy(), classical_data, dl_result, dl_active)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "1. ORIGINAL", (20, 40), font, 0.8, (0, 255, 0), 2)
        cv2.putText(gray_3ch, "2. GRAYSCALE + BLUR", (20, 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(edges_3ch, "3. CANNY EDGES + ROI", (20, 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(res_window, "4. CLASSICAL + UFLD", (20, 40), font, 0.8, (0, 255, 0), 2)

        top = np.hstack((frame, gray_3ch))
        bottom = np.hstack((edges_3ch, res_window))
        full_tile = np.vstack((top, bottom))

        return cv2.resize(full_tile, (1280, 720))