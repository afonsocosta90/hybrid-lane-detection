import os
import cv2

from config import CONFIG
from models.classical import ClassicalBackend
from models.dl_backend import DLBackend
from renderers import LaneRenderer


def try_init_dl():
    """Load the UFLD backend, or return None so the program runs classical-only."""
    cfg = CONFIG.dl
    if not os.path.isfile(cfg.weights_path) or os.path.getsize(cfg.weights_path) == 0:
        print(f"[warn] UFLD weights not found at '{cfg.weights_path}' — running classical only.")
        print("       See weights/README.md for download links.")
        return None
    try:
        return DLBackend(
            weights_path=cfg.weights_path,
            dataset=cfg.dataset,
            backbone=cfg.backbone,
            device=cfg.device,
        )
    except Exception as e:
        print(f"[warn] Failed to init UFLD backend ({e}) — running classical only.")
        return None


def main():
    cap = cv2.VideoCapture(CONFIG.video.path)
    classical = ClassicalBackend()
    dl = try_init_dl()
    renderer = LaneRenderer()
    dl_active = dl is not None

    is_paused = False

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break

            classical_data, gray, edges = classical.process(frame)
            dl_result = dl.process(frame) if dl_active else None

            debug_view = renderer.create_debug_tile(
                frame, gray, edges, classical_data, dl_result, dl_active
            )
            cv2.imshow(CONFIG.video.window_name, debug_view)

        key = cv2.waitKey(CONFIG.video.wait_key_ms) & 0xFF
        if key == ord(' '):
            is_paused = not is_paused
            print("Paused" if is_paused else "Resuming...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
