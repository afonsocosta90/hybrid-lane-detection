"""ObjectsModule — Ultralytics YOLO wrapper, COCO classes filtered to road agents.

Stage 0 (parallel with the lane module). The Detection dataclass carries a
track_id field that is always None in slice 1 — reserved for a tracker module
in slice 2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from perception.config import ObjectsConfig
from perception.contracts import PerceptionFrame, PerceptionState


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class ObjectsOutput:
    detections: tuple[Detection, ...]
    inference_ms: float


def _resolve_device(s: str) -> str | None:
    """Translate the config's device string to what Ultralytics expects.

    'auto' → None (Ultralytics picks). Anything else passes through verbatim
    so the user can force e.g. 'cuda:1' or 'cpu'.
    """
    if s.lower() == "auto":
        return None
    return s


class ObjectsModule:
    name: ClassVar[str] = "objects"
    stage: ClassVar[int] = 0

    def __init__(self, cfg: ObjectsConfig):
        self._cfg = cfg
        self._model = None
        self._class_ids: tuple[int, ...] = ()
        self._keep_set: frozenset[str] = frozenset(cfg.keep_classes)

    def warmup(self) -> None:
        from ultralytics import YOLO  # local import to keep config tests light

        model = YOLO(self._cfg.weights_path)
        device = _resolve_device(self._cfg.device)
        if device is not None:
            model.to(device)

        names: dict[int, str] = dict(model.names) if hasattr(model, "names") else {}
        wanted = self._keep_set
        ids = tuple(i for i, n in names.items() if n in wanted)
        if not ids:
            raise RuntimeError(
                f"none of {sorted(wanted)} present in YOLO model class names "
                f"({sorted(set(names.values()))[:10]}...)"
            )
        self._class_ids = ids

        # One dummy predict so frame 0 doesn't pay the JIT/allocate cost.
        dummy = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), dtype=np.uint8)
        model.predict(
            dummy,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
            classes=list(self._class_ids),
            imgsz=self._cfg.imgsz,
            verbose=False,
        )
        self._model = model

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> ObjectsOutput:
        assert self._model is not None, "ObjectsModule.process called before warmup"
        t0 = time.perf_counter()
        result = self._model.predict(
            frame.bgr,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
            classes=list(self._class_ids),
            imgsz=self._cfg.imgsz,
            verbose=False,
        )[0]

        names = result.names
        keep = self._keep_set
        dets: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for b in boxes:
                cls_idx = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                cls_name = names.get(cls_idx, str(cls_idx))
                if cls_name not in keep:
                    continue  # defensive — Ultralytics already filtered via classes=
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                dets.append(
                    Detection(
                        bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        class_name=cls_name,
                        confidence=conf,
                    )
                )
        return ObjectsOutput(
            detections=tuple(dets),
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )
