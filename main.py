import os
import cv2

from models.classical import ClassicalBackend
from models.dl_backend import DLBackend
from renderers import LaneRenderer

VIDEO_PATH = "test-videos/test_video_0.mp4"
WINDOW_NAME = "Lane Detection Pipeline"

# UFLD weights — drop a CULane ResNet-18 .pth in here. See weights/README.
DL_WEIGHTS_PATH = "weights/culane_18.pth"
DL_DATASET = "culane"
DL_BACKBONE = "18"
DL_DEVICE = "auto"  # auto -> DirectML -> CUDA -> CPU


def try_init_dl():
    """Load the UFLD backend, or return None so the program runs classical-only."""
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
    cap = cv2.VideoCapture(VIDEO_PATH)
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
            cv2.imshow(WINDOW_NAME, debug_view)

        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
            is_paused = not is_paused
            print("Paused" if is_paused else "Resuming...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
