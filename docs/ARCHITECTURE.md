# Architecture

This document describes the design of the perception pipeline. It is the artifact a senior engineer would read on Day 1, and the artifact a video walkthrough would script from. Read [README.md](../README.md) for setup and what runs; read this for *why it's shaped this way*.

---

## Three design principles

1. **The architecture is the product, not the detections.** Anyone can call `yolo.track()`. What gets hired is the stuff around the model — typed contracts, validated config, isolated rendering, fail-loud startup, unit tests on pure logic. Treat the architecture as the deliverable.
2. **Pick a narrow vertical slice and resist scope creep.** The failure mode is ten half-modules with great architecture and no working end-to-end path. Slice 1 has exactly three modules: lane, objects, steering. Anything else is later.
3. **Every design decision must be defensible out loud.** Protocols over inheritance, pydantic over dataclasses, threads over multiprocessing, typed accessors over dict lookups — each is a deliberate trade-off, not a default. If you can't explain the rejected alternative in one sentence, the decision isn't made yet.

---

## The module contract

Everything hangs off three types in [`perception/contracts.py`](../perception/contracts.py).

### `PerceptionFrame` — the input

```python
@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    bgr: np.ndarray            # HxWx3 uint8, the raw frame
    timestamp: float           # perf_counter at capture
    frame_idx: int             # 0-based capture counter
    fps_hint: float | None     # source fps if known
```

Immutable. Handed to every module verbatim per frame. Modules **must not mutate** `bgr` — the renderer copies it before drawing.

### `PerceptionModule` — the Protocol

```python
@runtime_checkable
class PerceptionModule(Protocol, Generic[Output_co]):
    name: ClassVar[str]        # unique key in PerceptionState.outputs
    stage: ClassVar[int]       # 0, 1, 2, ...
    def process(self, frame: PerceptionFrame, state: "PerceptionState") -> Output_co: ...
    def warmup(self) -> None: ...
```

A module is anything that satisfies this Protocol structurally. No inheritance, no registration call. `name` and `stage` are `ClassVar` so the orchestrator can introspect them without constructing the module — the wiring is statically inspectable.

**Why Protocol over ABC.** An ABC forces every module to `import` and `inherit from` our base class. That couples a third-party tracker (installed from PyPI) to our types. Protocol lets a module satisfy the contract without ever importing it. It also keeps the existing `HybridLaneDetector` decoupled — we wrap it in a thin `LaneModule` adapter rather than modifying the lane library.

### `PerceptionState` — the cumulative output

```python
@dataclass(slots=True)
class PerceptionState:
    frame: PerceptionFrame
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)

    def lane(self) -> "LaneOutput": ...     # typed accessor, raises StageDependencyError if missing
    def objects(self) -> "ObjectsOutput": ...
    def steering(self) -> "SteeringOutput": ...
```

The typed accessors are the key ergonomic decision. A stage-1 module never writes `state.outputs.get("lane")`; it writes `state.lane()`. If the lane module didn't fire this frame, the accessor raises `StageDependencyError`, the dependent module fails cleanly, and the renderer chain catches the same exception and skips the dependent overlay.

This pattern keeps a `None`-broadcast bug from silently propagating from a missing module into steering math into the rendered frame.

---

## Stages and the orchestrator

Modules declare an integer `stage` at the class level. The orchestrator ([`perception/pipeline.py`](../perception/pipeline.py)) groups by stage and runs each group in parallel.

```
stage 0: { lane, objects }      ← run in parallel ThreadPoolExecutor
            │
            ▼
stage 1: { steering }           ← reads state.lane()
            │
            ▼
stage 2: { ... future ... }
```

### Rules

- Modules at the same stage **must be independent** — they may not read `PerceptionState.outputs`.
- Modules at stage N may read any output produced by a module at stage < N.
- Stage groups are barriers. The orchestrator waits for every module in stage N to return before starting stage N+1.

The orchestrator enforces the independence rule: stage-N outputs are buffered into a stage-local dict and only merged into `PerceptionState.outputs` *after* the stage barrier. A racing same-stage peer that tries to read its sibling's output sees the prior-stage view only — never the sibling's in-flight result. This turns a documented rule into a structural guarantee.

### Concurrency choice

`ThreadPoolExecutor`, size = widest stage. Justification:

- **Threads over multiprocessing.** `torch.nn.Module.forward` releases the GIL inside its CUDA / DirectML / CPU kernel. So does OpenCV's `Canny`, `HoughLinesP`, etc. Threads get real parallelism without pickling an HD frame or duplicating model weights in VRAM. Multiprocessing would do the opposite — copy the frame and the model per worker per step.
- **Threads over asyncio.** Asyncio would force `async def process` on every module author. That's a hostile contract for compute-bound CV code that already releases the GIL. Sync `process()` with thread parallelism at stage boundaries is the lowest-friction shape for the people writing modules.

### Failure policy — fail loud, then fail soft

| Phase     | If a module raises | Why                                                                                                                          |
|-----------|--------------------|------------------------------------------------------------------------------------------------------------------------------|
| `warmup`  | Process exits      | Missing weights, wrong device, degenerate homography — these are config bugs. They must surface before the camera opens, not three minutes into a video. |
| `process` | Caught, recorded   | A flaky YOLO frame at minute 4 shouldn't blow away a lane carpet that's been rendering for 4 minutes. The pipeline records the exception in `state.errors[name]` and continues. Dependent modules raise `StageDependencyError` via the typed accessor; the renderer chain skips that overlay. The HUD overlay shows a red badge so failures are visible, not silent. |

This is the single most important design rule. Everything testable in this codebase exists to make these two policies enforceable.

---

## Rendering

The renderer is a chain of `Overlay` Protocols ([`perception/renderers/base.py`](../perception/renderers/base.py)). Each overlay reads only its module's slice of state.

```python
@runtime_checkable
class Overlay(Protocol):
    name: ClassVar[str]
    def draw(self, canvas: np.ndarray, state: PerceptionState) -> np.ndarray: ...

class OverlayChain:
    def render(self, state: PerceptionState) -> np.ndarray:
        canvas = state.frame.bgr.copy()
        for ov in self._overlays:
            try:
                canvas = ov.draw(canvas, state)
            except StageDependencyError:
                continue            # the module didn't fire this frame; skip silently
        return canvas
```

- `LaneOverlay` reads `state.lane()`. Wraps the existing `LaneRenderer` methods — no rewrite.
- `ObjectOverlay` reads `state.objects()`. Draws bounding boxes + labels.
- `SteeringOverlay` reads `state.steering()`. Draws a steering-wheel indicator and the lookahead dot.
- `HudOverlay` reads `state.errors` plus per-module timing. It's the only overlay allowed to look across modules — it's the dashboard.

The renderer never imports a detection backend. It never touches model code. If an overlay needs information, it must be in `PerceptionState`. This is the rule that lets us unit-test the pure logic without OpenCV in the loop.

---

## Configuration

[`perception/config.py`](../perception/config.py) is a tree of pydantic v2 `BaseModel`s. The tree is loaded from [`perception/configs/default.yaml`](../perception/configs/default.yaml) by `load_config(path) -> AppConfig`, which is called once at `main.py` startup.

### What pydantic gives us that frozen dataclasses don't

- **Range validators on individual fields** (`conf_threshold: float = Field(0.35, ge=0.0, le=1.0)`). Wrong YAML → `ValidationError` with the exact field path.
- **Cross-field invariants** (e.g. the classical scoring weights must sum to 1.0 — this lives in a `@field_validator` instead of a comment).
- **Free JSON serialization** for telemetry (`model_dump_json()`).
- **One source of truth on disk.** Diffs are config diffs, not code diffs. A Dockerized run mounts a different `--config` without rebuilding.

### Validation passes

1. **`load_config()`** — pydantic structural + range + cross-field validation. Fails before any module is constructed.
2. **`Module.warmup()`** — disk / device / homography checks pydantic can't see. Fails before the camera opens.

By the time `pipeline.step()` runs for the first time, every reachable failure path has either fired or is permanently gated.

### Why not stay on `@dataclass(frozen=True)`

Dataclasses give types but not validation. The "classical weights must sum to 1.0" invariant currently lives in a comment ([`hybrid_lane_detection/config.py:72`](../hybrid_lane_detection/config.py)) and is enforced nowhere. Pydantic makes it an enforced validator — a corrupted YAML can't reach the hot path.

### Why YAML over a Python config file

- An ops person can retune for a new dashcam without a code change or a redeploy.
- A reviewer sees a config diff in the PR, not a code diff.
- A Docker run mounts a different YAML; same image, different behavior.

The cost is a `pyyaml` dependency and a five-line loader. Worth it.

### What the legacy config still does

`hybrid_lane_detection/config.py` is preserved unchanged. `LaneModule.__init__` reads the pydantic `ClassicalConfig` and constructs the legacy frozen-dataclass `ClassicalConfig` once. From that point on, the classical backend reads the legacy type. This isolates the migration — the lane library can be pip-installed standalone, and its consumers don't have to take a pydantic dependency.

---

## Why two packages

The repo ships two Python packages:

- `hybrid_lane_detection/` — the lane library. Pip-installable on its own. Exports `HybridLaneDetector`, `LaneData`, `LaneRenderer`. Has no idea the perception pipeline exists.
- `perception/` — the application. Owns the orchestrator, the modules, the pydantic config, the overlay chain. Imports `hybrid_lane_detection` as a dependency.

Justification: the lane work is genuinely reusable — someone could install the library in a notebook and call `HybridLaneDetector` for ad-hoc analysis. Forcing them to import the pipeline runtime (pydantic, ThreadPoolExecutor, YAML, ultralytics) for that use case would be hostile. The two-package shape makes the boundary literal instead of just notional.

The risk: it can read as over-architecture for a small repo. Mitigation: the talk track ("the lane detector is a library; the perception pipeline is the app that consumes it") is genuine and defensible. If, after a few months, the split doesn't pull its weight, fold them into one package — that's a 30-minute refactor, not a regret.

---

## Testing strategy

Tests live in `tests/` and target pure-logic surfaces only. No torch, no cv2.VideoCapture, no .mp4 / .pth fixtures. Total runtime stays under 1 second.

| Test                                       | Pins                                                                                |
|--------------------------------------------|-------------------------------------------------------------------------------------|
| `test_homography_roundtrip`                | `pixel_to_world(world_to_pixel(...))` returns identity on the 4 ROI corners.        |
| `test_steering_zero_on_straight_lane`      | Perfectly straight synthetic fits → angle ≈ 0°.                                     |
| `test_steering_curves_right`               | Positive `a` coefficient → angle > 0°. Pins sign convention.                         |
| `test_lookahead_monotonic`                 | Increasing `road_depth_m` pushes lookahead pixel higher in the image (smaller `v`). |
| `test_pipeline_stage_ordering`             | Stage-1 module can read stage-0 output; swapping stages raises `StageDependencyError`. |
| `test_pipeline_module_error_isolated`      | A raising stage-0 module doesn't block its sibling; `state.errors` populated; dependent stage-1 module raises cleanly; renderer skips overlay. |
| `test_config_weights_must_sum_to_one`      | Bad YAML → `ValidationError` at `load_config()`.                                     |

What we **don't** test: the YOLO model itself (Ultralytics' problem), the UFLD inference (the `ufld` package's problem), `cv2.imshow` (you have eyes).

---

## What's explicitly out of slice 1

Each of these is a defensible slice-2 follow-up. They get said "no" to in PR review.

- Multi-object tracking (ByteTrack/StrongSORT). `Detection.track_id` exists in the dataclass but is always `None`.
- 4-point calibration tool. Homography reuses existing ROI corners + assumed lane width.
- Bird's-eye-view visualization tile.
- Lane-departure / collision warnings.
- Web dashboard / FastAPI / telemetry sink.
- Multi-camera, geofenced zones, recording-to-disk.
- Replacing UFLD or the classical backend.
- Asyncio.

---

## Honest weakest joints

- **The homography assumption.** ROI corners were tuned for one 1280×720 dashcam. Reusing them as ground-plane correspondences inherits that tuning. The "X meters ahead" number will be wrong by tens of percent on a different camera. `road_depth_m` and `lane_width_m` are first-class config knobs to make the assumption visible. Slice 2: a real calibration tool.
- **Stage-0 latency stacking.** ThreadPool runs lane (~15ms) and YOLO (80–200ms CPU) in parallel, but `step()` blocks on the slower one. The lane work is wasted concurrency. Acceptable for slice 1 (the target is contract correctness, not FPS). The contract supports a future per-module frequency override (run YOLO every 3 frames) without breaking anything — that's a stage attribute, not a contract change.
- **Ultralytics is GPLv3.** Fine for a portfolio. Worth knowing if this ever turns commercial.
- **Two-package risk.** Could read as over-architecture. Mitigation: the talk track. Fallback: fold into one package — 30 minutes, no regret.
