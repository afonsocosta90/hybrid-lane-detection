"""Pipeline runner. Loads a video, drives the orchestrator, displays via OverlayChain."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2

from perception.config import load_config
from perception.modules.lane import LaneModule
from perception.modules.objects import ObjectsModule
from perception.modules.steering import SteeringModule
from perception.pipeline import PerceptionPipeline
from perception.renderers import OverlayChain
from perception.renderers.hud_overlay import HudOverlay
from perception.renderers.lane_overlay import LaneOverlay
from perception.renderers.object_overlay import ObjectOverlay
from perception.renderers.steering_overlay import SteeringOverlay

DEFAULT_CONFIG = Path(__file__).resolve().parent / "perception" / "configs" / "default.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(os.environ.get("PERCEPTION_CONFIG", str(DEFAULT_CONFIG)))

    pipeline = PerceptionPipeline(
        [
            LaneModule(cfg),
            ObjectsModule(cfg.objects),
            SteeringModule(cfg.steering, cfg.classical, cfg.video),
        ]
    )
    chain = OverlayChain(
        [
            LaneOverlay(cfg.render),
            ObjectOverlay(),
            SteeringOverlay(),
            HudOverlay(),
        ]
    )

    cap = cv2.VideoCapture(cfg.video.path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {cfg.video.path!r}")

    fps_hint = cap.get(cv2.CAP_PROP_FPS) or None
    frame_idx = 0
    paused = False

    try:
        while cap.isOpened():
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                state = pipeline.step(frame, frame_idx, fps_hint=fps_hint)
                frame_idx += 1
                cv2.imshow(cfg.video.window_name, chain.render(state))

            key = cv2.waitKey(cfg.video.wait_key_ms) & 0xFF
            if key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resuming...")
            elif key == ord("q"):
                break
    finally:
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
