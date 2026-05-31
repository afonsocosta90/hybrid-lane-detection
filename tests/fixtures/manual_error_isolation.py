"""Manual verification: error isolation at runtime.

Monkey-patches ObjectsModule.process to raise every N-th frame, then runs main().
Expected behavior:
  - The demo keeps playing without crashing.
  - The HUD shows a red "!! objects: ERROR" badge on the bad frames.
  - The lane carpet, UFLD dots, and steering overlay continue to render.
  - When you press q, the process exits cleanly (code 0), no traceback escapes.

This file lives under tests/fixtures/ but is NOT collected by pytest (no test_
prefix) — run it directly with the rocm-pytorch python.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main  # noqa: E402
from perception.modules import objects as O  # noqa: E402

_real_process = O.ObjectsModule.process
_counter = {"n": 0}
_FAIL_EVERY = 30


def _flaky_process(self, frame, state):
    _counter["n"] += 1
    if _counter["n"] % _FAIL_EVERY == 0:
        raise RuntimeError(f"synthetic failure on frame {_counter['n']}")
    return _real_process(self, frame, state)


O.ObjectsModule.process = _flaky_process

if __name__ == "__main__":
    main.main()
