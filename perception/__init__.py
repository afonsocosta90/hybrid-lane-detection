"""Modular autonomous-perception pipeline.

See docs/ARCHITECTURE.md for the design contracts.
"""
from perception.config import AppConfig, load_config
from perception.contracts import (
    ModuleProcessError,
    PerceptionFrame,
    PerceptionModule,
    PerceptionState,
    StageDependencyError,
)

__all__ = [
    "AppConfig",
    "load_config",
    "PerceptionFrame",
    "PerceptionState",
    "PerceptionModule",
    "StageDependencyError",
    "ModuleProcessError",
]
