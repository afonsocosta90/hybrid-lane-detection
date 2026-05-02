import os
import cv2
import numpy as np

from models.classical import ClassicalBackend
from models.dl_backend import DLBackend

# Upper render bound: a bit below the ROI horizon (y=350) so we don't draw
# the carpet in a region where the quadratic fit is barely supported by data.
RENDER_Y_TOP = 500
RENDER_CONF_THRESHOLD = 0.35

# UFLD weights — drop a CULane ResNet-18 .pth in here. See weights/README.
DL_WEIGHTS_PATH = "weights/culane_18.pth"
DL_DATASET = "culane"
DL_BACKBONE = "18"
DL_DEVICE = "auto"  # auto -> DirectML -> CUDA -> CPU


def render_classical_overlay(frame, data):
    """Green carpet between the two polynomial fits when classical is confident."""
    if data.confidence < RENDER_CONF_THRESHOLD:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()
    plot_y = np.linspace(RENDER_Y_TOP, h - 1, 30)
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


def render_dl_overlay(frame, dl_result):
    """Red dots at each UFLD anchor point. Empty result is a no-op."""
    if dl_result is None:
        return frame
    for lane in dl_result.lanes:
        for (x, y) in lane:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame


def draw_description(img, data, dl_active):
    """Semi-transparent HUD box in the top-right corner."""
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

    status_text = "LANE TRACKED" if data.confidence >= RENDER_CONF_THRESHOLD else "SEARCHING..."
    status_color = (0, 255, 0) if data.confidence >= RENDER_CONF_THRESHOLD else (0, 0, 255)

    cv2.putText(img, status_text, (text_x, tr_y1 + 30), font, 0.7, status_color, 2)
    cv2.putText(img, f"Classical conf: {data.confidence * 100:.1f}%", (text_x, tr_y1 + 60), font, 0.55, (0, 255, 0), 1)
    dl_label = "UFLD: ON  (red dots)" if dl_active else "UFLD: OFF (no weights)"
    dl_color = (0, 0, 255) if dl_active else (128, 128, 128)
    cv2.putText(img, dl_label, (text_x, tr_y1 + 90), font, 0.55, dl_color, 1)
    cv2.putText(img, "[space] pause   [q] quit", (text_x, tr_y1 + 118), font, 0.45, (200, 200, 200), 1)

    return img


def render_prediction_tile(frame, classical_data, dl_result, dl_active):
    out = render_classical_overlay(frame, classical_data)
    out = render_dl_overlay(out, dl_result)
    out = draw_description(out, classical_data, dl_active)
    return out


def create_debug_tile(frame, gray, edges, classical_data, dl_result, dl_active):
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    res_window = render_prediction_tile(frame.copy(), classical_data, dl_result, dl_active)

    cv2.putText(frame, "1. ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(gray_3ch, "2. GRAYSCALE + BLUR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(edges_3ch, "3. CANNY EDGES + ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(res_window, "4. CLASSICAL + UFLD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    top = np.hstack((frame, gray_3ch))
    bottom = np.hstack((edges_3ch, res_window))
    full_tile = np.vstack((top, bottom))

    return cv2.resize(full_tile, (1280, 720))


def try_init_dl():
    """Attempt to load the UFLD backend. Returns None if weights are missing
    or the package can't be imported, so the program still runs classical-only."""
    if not os.path.isfile(DL_WEIGHTS_PATH) or os.path.getsize(DL_WEIGHTS_PATH) == 0:
        print(f"[warn] UFLD weights not found at '{DL_WEIGHTS_PATH}' — running classical only.")
        print("       See weights/README.md for download links.")
        return None
    try:
        return DLBackend(
            weights_path=DL_WEIGHTS_PATH,
            dataset=DL_DATASET,
            backbone=DL_BACKBONE,
            device=DL_DEVICE,
        )
    except Exception as e:
        print(f"[warn] Failed to init UFLD backend ({e}) — running classical only.")
        return None


def main():
    cap = cv2.VideoCapture("test-videos/test_video_1.mp4")
    classical = ClassicalBackend()
    dl = try_init_dl()
    dl_active = dl is not None

    is_paused = False

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break

            classical_data, gray, edges = classical.process(frame)
            dl_result = dl.process(frame) if dl_active else None

            debug_view = create_debug_tile(frame, gray, edges, classical_data, dl_result, dl_active)
            cv2.imshow("Lane Detection Pipeline", debug_view)

        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
            is_paused = not is_paused
            print("Paused" if is_paused else "Resuming...")
        elif key == ord('q'):
            break

        if is_paused:
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
