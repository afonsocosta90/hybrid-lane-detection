"""Module contract — the single source of truth.

Three types: PerceptionFrame (input), PerceptionState (cumulative output),
PerceptionModule (Protocol). Everything else in `perception/` imports from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from perception.modules.lane import LaneOutput
    from perception.modules.objects import ObjectsOutput
    from perception.modules.steering import SteeringOutput


class StageDependencyError(RuntimeError):
    """Raised when a stage-N module reads an output that wasn't produced.

    The renderer chain catches this and skips the dependent overlay; the HUD
    overlay surfaces it as a red badge so the failure is visible, not silent.
    """


class ModuleProcessError(RuntimeError):
    """Wraps an exception thrown by a module's process() call."""


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """Immutable per-frame input handed to every module."""

    bgr: np.ndarray
    timestamp: float
    frame_idx: int
    fps_hint: float | None = None


@dataclass(slots=True)
class PerceptionState:
    """Cumulative typed output for one frame. Built up stage by stage."""

    frame: PerceptionFrame
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)
    module_timings_ms: dict[str, float] = field(default_factory=dict)

    def lane(self) -> "LaneOutput":
        out = self.outputs.get("lane")
        if out is None:
            raise StageDependencyError(
                "lane output not available — is LaneModule registered at stage 0?"
            )
        return out

    def objects(self) -> "ObjectsOutput":
        out = self.outputs.get("objects")
        if out is None:
            raise StageDependencyError(
                "objects output not available — is ObjectsModule registered at stage 0?"
            )
        return out

    def steering(self) -> "SteeringOutput":
        out = self.outputs.get("steering")
        if out is None:
            raise StageDependencyError(
                "steering output not available — is SteeringModule registered at stage 1?"
            )
        return out


Output_co = TypeVar("Output_co", covariant=True)


@runtime_checkable
class PerceptionModule(Protocol, Generic[Output_co]):
    """The contract every module satisfies structurally.

    `name` and `stage` are ClassVar so the orchestrator can introspect them
    without constructing the module. Modules at the same stage MUST be
    independent (no reads from PerceptionState). Modules at stage N may read
    any output produced at stage < N via a typed accessor on PerceptionState.

    Failure policy:
      - `warmup()` raises  → process exits (fail loud at startup).
      - `process()` raises → caught by orchestrator, recorded in state.errors,
        next stage continues.
    """

    name: ClassVar[str]
    stage: ClassVar[int]

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> Output_co: ...

    def warmup(self) -> None: ...
