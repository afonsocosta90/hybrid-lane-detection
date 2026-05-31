"""Stage-parallel orchestrator.

Modules declare a stage int. Same-stage modules run in parallel inside a
ThreadPoolExecutor; later stages run after all earlier stages complete.

Failure policy:
- warmup() raises → propagated up, kills the process before any frame loops.
- process() raises → caught, recorded in state.errors[name], pipeline continues.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from typing import Iterable

import numpy as np

from perception.contracts import PerceptionFrame, PerceptionModule, PerceptionState

log = logging.getLogger(__name__)


def _timed_process(
    module: PerceptionModule, pframe: PerceptionFrame, state: PerceptionState
) -> tuple[object, float]:
    t0 = time.perf_counter()
    result = module.process(pframe, state)
    return result, (time.perf_counter() - t0) * 1000.0


class PerceptionPipeline:
    def __init__(self, modules: Iterable[PerceptionModule], max_workers: int | None = None):
        modules = list(modules)
        names = [m.name for m in modules]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate module names: {names}")

        self._stages: list[list[PerceptionModule]] = [
            list(group)
            for _, group in groupby(sorted(modules, key=lambda m: m.stage), key=lambda m: m.stage)
        ]
        widest = max((len(s) for s in self._stages), default=1)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or widest,
            thread_name_prefix="perception",
        )

        for m in modules:
            log.info("warmup: %s (stage %d)", m.name, m.stage)
            m.warmup()

    def step(self, bgr: np.ndarray, frame_idx: int, fps_hint: float | None = None) -> PerceptionState:
        pframe = PerceptionFrame(
            bgr=bgr,
            timestamp=time.perf_counter(),
            frame_idx=frame_idx,
            fps_hint=fps_hint,
        )
        state = PerceptionState(frame=pframe)

        for stage in self._stages:
            # Same-stage modules MUST be independent — they get no peer reads.
            # We collect results into stage-local dicts and only merge them back
            # into the shared state after the stage's barrier. Without this, a
            # racing peer could see a sibling's output non-deterministically.
            stage_outputs: dict[str, object] = {}
            stage_errors: dict[str, Exception] = {}
            stage_timings: dict[str, float] = {}
            futures = {self._pool.submit(_timed_process, m, pframe, state): m for m in stage}
            for fut, m in futures.items():
                try:
                    result, ms = fut.result()
                    stage_outputs[m.name] = result
                    stage_timings[m.name] = ms
                except Exception as exc:  # noqa: BLE001 — by design
                    stage_errors[m.name] = exc
                    log.exception("module %s failed on frame %d", m.name, frame_idx)
            state.outputs.update(stage_outputs)
            state.errors.update(stage_errors)
            state.module_timings_ms.update(stage_timings)
        return state

    def close(self) -> None:
        self._pool.shutdown(wait=True)
