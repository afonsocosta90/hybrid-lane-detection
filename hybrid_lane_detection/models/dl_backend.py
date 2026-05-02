"""DL backend wrapper around the UFLD LaneDetector.

Keeps the orchestrator decoupled from the underlying inference package: if we
ever swap UFLD for something else, only this file changes.
"""

from ufld import LaneDetector, LaneResult, pick_device


class DLBackend:
    def __init__(
        self,
        weights_path: str,
        dataset: str = "culane",
        backbone: str = "18",
        device: str = "auto",
    ):
        self._detector = LaneDetector(
            weights_path=weights_path,
            dataset=dataset,
            backbone=backbone,
            device=pick_device(device),
        )

    def process(self, frame) -> LaneResult:
        return self._detector(frame)
