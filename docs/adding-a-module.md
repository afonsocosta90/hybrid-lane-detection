# Adding a module

This is the practical companion to [ARCHITECTURE.md](ARCHITECTURE.md). It walks through writing a new module from scratch and wiring it into the pipeline, using `SteeringModule` as the worked example because it shows both stage-1 (depends on lane output) and a clean separation between glue code and pure math.

## The five steps

1. Define the module's output type (a frozen dataclass).
2. Write the pure-logic helpers in `perception/geometry.py` (or a topic-specific file).
3. Write the module class — a Protocol implementation with `name`, `stage`, `warmup()`, `process()`.
4. Add a typed accessor on `PerceptionState`.
5. Write an `Overlay` for it. Wire both into `main.py`.

Then write tests for the pure-logic helpers. The module itself is glue — if the helpers are tested, the module is too.

---

## Worked example: `SteeringModule`

### Step 1 — output type

Outputs are always frozen dataclasses with `slots=True`. They live next to the module that produces them.

```python
# perception/modules/steering.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SteeringOutput:
    angle_deg: float                          # positive = steer right
    lookahead_world_xy: tuple[float, float]   # meters
    lookahead_pixel_uv: tuple[int, int]       # rendered position
    cross_track_error_m: float
    confidence: float                         # inherited from lane confidence
```

Rules of thumb for output types:

- `frozen=True, slots=True` always. Outputs are immutable per-frame snapshots.
- Tuples for vectors of known length (`tuple[float, float]`), not numpy arrays. Hashable, picklable, fast.
- Use plain types where possible. If you must hold a numpy array (e.g. coefficient vectors), document the shape in a docstring.
- Confidence belongs to the producer — don't try to centralize it.

### Step 2 — pure-logic helpers

Anything that's math goes in `perception/geometry.py` (or a sibling file in the same package). **No `cv2` import at module top.** That rule is what makes the helpers testable in under a second without a video.

```python
# perception/geometry.py — pure math, NO cv2
def lane_center_x_at_y(left_fit, right_fit, y):
    lx = left_fit[0]*y*y + left_fit[1]*y + left_fit[2]
    rx = right_fit[0]*y*y + right_fit[1]*y + right_fit[2]
    return 0.5 * (lx + rx)

def steering_angle_deg(left_fit, right_fit, frame_wh, y_eval_px):
    # ... combines cross-track error and heading error
    return ...

def build_road_homography(roi_tl, roi_tr, frame_wh, road_depth_m, lane_width_m):
    # ... solves 4-point homography via numpy.linalg.solve
    return ...

def lookahead_point(H, H_inv, left_fit, right_fit, lookahead_m):
    return ...
```

These are the functions the test suite will pin. They take numpy arrays and scalars and return numpy arrays and scalars. They never need a real frame.

### Step 3 — the module

The module itself is a 30-line Protocol implementation. The interesting code is in the helpers; the module is glue.

```python
# perception/modules/steering.py
from typing import ClassVar
import numpy as np

from perception.contracts import PerceptionFrame, PerceptionState
from perception import geometry
from perception.config import SteeringConfig, ClassicalConfig


class SteeringModule:
    name: ClassVar[str] = "steering"
    stage: ClassVar[int] = 1                      # reads state.lane()

    def __init__(self, cfg: SteeringConfig, classical_cfg: ClassicalConfig):
        self._cfg = cfg
        self._classical_cfg = classical_cfg
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None

    def warmup(self) -> None:
        # All knowable failures must surface here, before the camera opens.
        # In this case: can we solve the homography? Is the determinant non-degenerate?
        frame_wh = (1280, 720)                    # TODO: pull from VideoConfig once probed
        self._H = geometry.build_road_homography(
            self._classical_cfg.roi_top_left,
            self._classical_cfg.roi_top_right,
            frame_wh,
            road_depth_m=self._cfg.road_depth_m,
            lane_width_m=self._cfg.lane_width_m,
        )
        self._H_inv = np.linalg.inv(self._H)      # raises if singular — fail loud

    def process(self, frame: PerceptionFrame, state: PerceptionState) -> SteeringOutput:
        lane = state.lane()                       # typed accessor; raises if missing
        left = lane.classical.left_fit
        right = lane.classical.right_fit
        h, w = frame.bgr.shape[:2]

        y_eval = h - self._cfg.y_eval_offset_px
        angle = geometry.steering_angle_deg(left, right, (w, h), y_eval)
        x_world, y_world, u, v = geometry.lookahead_point(
            self._H, self._H_inv, left, right, self._cfg.lookahead_m,
        )
        cte_m = x_world                            # at lookahead_m forward, X offset IS the CTE
        return SteeringOutput(
            angle_deg=angle,
            lookahead_world_xy=(x_world, y_world),
            lookahead_pixel_uv=(int(u), int(v)),
            cross_track_error_m=cte_m,
            confidence=lane.classical.confidence,
        )
```

Things to notice:

- `name` and `stage` are `ClassVar`. The orchestrator reads them off the class without instantiating.
- `warmup()` does work that can fail. `np.linalg.inv` on a singular homography raises — that's the fail-loud entry point. Don't try/except it; let it propagate up.
- `process()` is short. All the math is in `geometry`.
- The typed accessor `state.lane()` is the only way to reach the lane output. If you write `state.outputs.get("lane")` you've defeated the contract.

### Step 4 — typed accessor

Add the accessor to `PerceptionState` in `perception/contracts.py`:

```python
# perception/contracts.py
class PerceptionState:
    # ... existing fields and accessors ...

    def steering(self) -> "SteeringOutput":
        out = self.outputs.get("steering")
        if out is None:
            raise StageDependencyError("steering output not available — is SteeringModule registered?")
        return out
```

One accessor per module. The accessor's `raise` is what makes downstream module bugs loud instead of silent.

### Step 5 — overlay + wire-up

```python
# perception/renderers/steering_overlay.py
import cv2
from typing import ClassVar
from perception.contracts import PerceptionState

class SteeringOverlay:
    name: ClassVar[str] = "steering"

    def draw(self, canvas, state: PerceptionState):
        s = state.steering()                       # raises StageDependencyError if missing
        u, v = s.lookahead_pixel_uv
        cv2.circle(canvas, (u, v), 8, (0, 255, 255), -1)
        cv2.putText(canvas, f"{s.angle_deg:+.1f}°", (u + 12, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return canvas
```

The overlay must do nothing more than read its slice of state and draw. **It must not** touch model code, raw detection logic, or other modules' state. If it needs information that isn't in `PerceptionState`, the fix is to put it there — not to reach across the contract.

Wire-up in `main.py`:

```python
pipeline = PerceptionPipeline([
    LaneModule(cfg),
    ObjectsModule(cfg.objects),
    SteeringModule(cfg.steering, cfg.classical),   # stage 1; reads lane output
])
chain = OverlayChain([
    LaneOverlay(cfg.render),
    ObjectOverlay(),
    SteeringOverlay(),                              # render after lane so the dot lands on the carpet
    HudOverlay(),
])
```

Order in `OverlayChain` is the draw order. Order in `PerceptionPipeline` doesn't matter — modules are sorted by stage internally.

---

## Tests for the new module

The module is glue, so the tests target the helpers. For `SteeringModule` that's `perception/geometry.py`:

```python
# tests/test_geometry.py
import numpy as np
from perception import geometry

def test_steering_zero_on_straight_lane():
    # x = 500 (left), x = 780 (right), frame width 1280 → center at 640 (frame mid)
    left  = np.array([0.0, 0.0, 500.0])
    right = np.array([0.0, 0.0, 780.0])
    angle = geometry.steering_angle_deg(left, right, frame_wh=(1280, 720), y_eval_px=670)
    assert abs(angle) < 0.1

def test_steering_curves_right():
    # positive a coefficient → centerline curves right at the evaluation row
    left  = np.array([1e-3, 0.0, 500.0])
    right = np.array([1e-3, 0.0, 780.0])
    angle = geometry.steering_angle_deg(left, right, frame_wh=(1280, 720), y_eval_px=670)
    assert angle > 0
```

The goal isn't to test every line of the module — it's to pin the *meaning* of the output (sign convention, monotonicity, identity round-trip). The module itself is small enough that a test for it would be a near-duplicate of the helper tests.

---

## A checklist before opening a PR

- [ ] Output type is a `@dataclass(frozen=True, slots=True)` in the module file.
- [ ] All math lives in a pure-logic file with no `cv2` import at module top.
- [ ] `warmup()` raises on every knowable failure (missing weights, singular matrix, bad config). No silent fallbacks.
- [ ] `process()` is glue — under ~30 lines, no math inline.
- [ ] `PerceptionState` has a typed accessor for the new output.
- [ ] The overlay reads only its own accessor; it imports nothing from `perception.modules`.
- [ ] Pure-logic tests pin sign conventions, identity round-trips, and monotonicity — the contract, not the implementation.
- [ ] The module is registered in `main.py` and the overlay is in `OverlayChain`.
- [ ] If you needed to add a config section, the pydantic model has range validators and a YAML default landed in `perception/configs/default.yaml`.
- [ ] [ARCHITECTURE.md](ARCHITECTURE.md) and the README's module table are updated.
