"""Orchestrator contract: stage ordering and per-frame error isolation."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from perception.contracts import (
    PerceptionFrame,
    PerceptionState,
    StageDependencyError,
)
from perception.pipeline import PerceptionPipeline


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


class _Stage0Producer:
    name: ClassVar[str] = "producer"
    stage: ClassVar[int] = 0

    def warmup(self) -> None: ...

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> int:
        return 42


class _Stage1Reader:
    name: ClassVar[str] = "reader"
    stage: ClassVar[int] = 1

    def warmup(self) -> None: ...

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> int:
        upstream = state.outputs.get("producer")
        if upstream is None:
            raise StageDependencyError("producer output not available")
        return upstream * 2


class _Stage0Raiser:
    name: ClassVar[str] = "raiser"
    stage: ClassVar[int] = 0

    def warmup(self) -> None: ...

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> int:
        raise RuntimeError("boom")


class _WarmupRaiser:
    name: ClassVar[str] = "warmup_raiser"
    stage: ClassVar[int] = 0

    def warmup(self) -> None:
        raise RuntimeError("config bug surfaced")

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> int:
        return 0


def test_stage_ordering_lets_later_stages_read_earlier_outputs():
    pipeline = PerceptionPipeline([_Stage0Producer(), _Stage1Reader()])
    state = pipeline.step(_frame(), frame_idx=0)
    assert state.outputs["producer"] == 42
    assert state.outputs["reader"] == 84
    assert state.errors == {}
    pipeline.close()


def test_swapped_stages_break_dependency():
    # If the "reader" is registered at stage 0 alongside the producer, they
    # run in the same parallel group — the reader sees no producer output and
    # raises StageDependencyError, which the pipeline records (not fatal).
    class ReaderAtStage0(_Stage1Reader):
        stage: ClassVar[int] = 0

    pipeline = PerceptionPipeline([_Stage0Producer(), ReaderAtStage0()])
    state = pipeline.step(_frame(), frame_idx=0)
    assert "reader" in state.errors
    assert isinstance(state.errors["reader"], StageDependencyError)
    pipeline.close()


def test_module_error_does_not_block_sibling_or_pipeline():
    pipeline = PerceptionPipeline([_Stage0Producer(), _Stage0Raiser(), _Stage1Reader()])
    state = pipeline.step(_frame(), frame_idx=7)
    # Sibling stage-0 module still produced output.
    assert state.outputs["producer"] == 42
    # The raising module's error is recorded.
    assert "raiser" in state.errors
    assert isinstance(state.errors["raiser"], RuntimeError)
    assert "boom" in str(state.errors["raiser"])
    # Stage-1 reader still ran successfully (it depends on producer, not raiser).
    assert state.outputs["reader"] == 84
    pipeline.close()


def test_warmup_failure_is_fatal():
    with pytest.raises(RuntimeError, match="config bug surfaced"):
        PerceptionPipeline([_Stage0Producer(), _WarmupRaiser()])


def test_duplicate_module_names_rejected():
    class Dup(_Stage0Producer):
        pass

    with pytest.raises(ValueError, match="duplicate module names"):
        PerceptionPipeline([_Stage0Producer(), Dup()])


def test_frame_idx_propagates_to_state():
    pipeline = PerceptionPipeline([_Stage0Producer()])
    state = pipeline.step(_frame(), frame_idx=123, fps_hint=29.97)
    assert state.frame.frame_idx == 123
    assert state.frame.fps_hint == 29.97
    pipeline.close()


def test_module_timings_recorded_for_successful_modules():
    pipeline = PerceptionPipeline([_Stage0Producer(), _Stage1Reader()])
    state = pipeline.step(_frame(), frame_idx=0)
    assert "producer" in state.module_timings_ms
    assert "reader" in state.module_timings_ms
    assert state.module_timings_ms["producer"] >= 0.0
    # A raising module shouldn't appear in timings (only successful results do).
    pipeline.close()


def test_failed_module_absent_from_timings():
    pipeline = PerceptionPipeline([_Stage0Producer(), _Stage0Raiser()])
    state = pipeline.step(_frame(), frame_idx=0)
    assert "producer" in state.module_timings_ms
    assert "raiser" not in state.module_timings_ms
    assert "raiser" in state.errors
    pipeline.close()
