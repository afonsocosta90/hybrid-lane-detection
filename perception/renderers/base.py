"""Overlay Protocol + OverlayChain.

Each overlay reads only its module's slice of PerceptionState. The chain
catches StageDependencyError (raised by the typed accessors when a module
didn't produce output this frame) and skips that overlay silently.
"""
from __future__ import annotations

from typing import ClassVar, Iterable, Protocol, runtime_checkable

import numpy as np

from perception.contracts import PerceptionState, StageDependencyError


@runtime_checkable
class Overlay(Protocol):
    name: ClassVar[str]

    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray: ...


class OverlayChain:
    def __init__(self, overlays: Iterable[Overlay]):
        self._overlays = list(overlays)

    def render(self, state: PerceptionState) -> np.ndarray:
        canvas = state.frame.bgr.copy()
        for ov in self._overlays:
            try:
                canvas = ov.draw(canvas, state)
            except StageDependencyError:
                continue
        return canvas
