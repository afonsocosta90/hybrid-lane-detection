"""Local demo runner. Loads a video, runs the hybrid detector, shows the
4-tile debug view. For library use see hybrid_lane_detection/lane_detector.py."""
import logging

import cv2

from hybrid_lane_detection import CONFIG, HybridLaneDetector


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    detector = HybridLaneDetector()
    cap = cv2.VideoCapture(CONFIG.video.path)
    is_paused = False

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.process(frame)
            cv2.imshow(CONFIG.video.window_name, detector.render_debug(frame, result))

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
