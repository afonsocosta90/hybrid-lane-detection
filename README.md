# Perception Pipeline

A modular, typed-contract autonomous-perception pipeline for road video. Hybrid lane detection (classical CV + UFLD v2) is one module; object detection and steering estimation are stacked on top through the same contract. The design goal is **production engineering quality, not detection accuracy** — typed contracts, validated config, fail-loud startup, renderer isolation, pure-logic unit tests.

The thing that makes this repo worth reading isn't `yolo.track()` or polynomial lane fits — it's everything around them.

![pipeline overview](docs/img/pipeline.png) <!-- placeholder; add later -->

---

## What runs per frame

| Module     | Stage | Output                                                | Status      |
|------------|-------|-------------------------------------------------------|-------------|
| `lane`     | 0     | `LaneOutput(classical, dl, dl_active)`                | shipped     |
| `objects`  | 0     | `ObjectsOutput(detections, inference_ms)`             | slice 1     |
| `steering` | 1     | `SteeringOutput(angle_deg, lookahead_xy, ...)`        | slice 1     |

Stage 0 modules run in parallel inside a `ThreadPoolExecutor`; stage 1 modules read whatever stage 0 produced via typed accessors. The overlay chain composes the final visualization — each overlay reads only its module's slice of state.

```
┌──────────────────────┐                ┌───────────────────────┐
│   cv2.VideoCapture   │ ──── frame ──▶ │ PerceptionPipeline    │
└──────────────────────┘                │  ┌──────┐ ┌────────┐  │
                                        │  │ lane │ │objects │  │ stage 0 (parallel)
                                        │  └───┬──┘ └────┬───┘  │
                                        │      ▼         ▼      │
                                        │  ┌──────────────────┐ │
                                        │  │     steering     │ │ stage 1
                                        │  └────────┬─────────┘ │
                                        └───────────┼───────────┘
                                                    ▼
                                        ┌──────────────────────┐
                                        │   OverlayChain       │
                                        │  lane ▶ objects ▶    │
                                        │  steering ▶ hud      │
                                        └──────────┬───────────┘
                                                   ▼
                                            cv2.imshow
```

---

## Project structure

```
perception/                 # NEW — the application
  contracts.py              # PerceptionModule Protocol, PerceptionFrame, PerceptionState, errors
  pipeline.py               # PerceptionPipeline (stage-parallel orchestrator)
  geometry.py               # Pure math: homography, steering, lookahead (NO cv2)
  config.py                 # Pydantic v2 BaseModels + load_config(yaml)
  configs/default.yaml      # All tunables on disk
  modules/
    lane.py                 # Adapter wrapping hybrid_lane_detection.HybridLaneDetector
    objects.py              # Ultralytics YOLO (cars / people / cyclists / motorbikes / trucks)
    steering.py             # Stage-1 module: angle + N-meters-ahead lookahead
  renderers/
    base.py                 # Overlay Protocol + OverlayChain
    lane_overlay.py
    object_overlay.py
    steering_overlay.py
    hud_overlay.py          # FPS, per-module ms, error badges

hybrid_lane_detection/      # The lane library — preserved as-is, pip-installable on its own
  lane_detector.py
  config.py
  renderers.py
  models/
    classical.py
    dl_backend.py
  utils/data_types.py

tests/
  test_geometry.py          # Pure-math (no torch / cv2 fixtures)
  test_pipeline.py          # Contract + stage ordering + error isolation
  test_config.py            # Pydantic validators fail loudly

main.py                     # Thin entry: load_config → PerceptionPipeline → loop
docs/
  ARCHITECTURE.md           # Read this if you want to understand the design
  adding-a-module.md        # Walkthrough for extending the pipeline
```

### Two packages, one app

`hybrid_lane_detection/` is **unchanged** and stays a standalone pip-installable library — you can `pip install` it on its own and use `HybridLaneDetector` without any of the orchestration code. `perception/` is the application that consumes it as one module among several. The library doesn't know the application exists. This is a deliberate split; see [ARCHITECTURE.md § Why two packages](docs/ARCHITECTURE.md#why-two-packages).

---

## Requirements

- **Python 3.11** (the `torch-directml` wheel for AMD on Windows is 3.11-only)
- A dashcam clip in `test-videos/` (configurable)
- A UFLD CULane / Tusimple checkpoint in `weights/` (optional — lane runs classical-only if missing)
- A YOLO checkpoint in `weights/` (e.g. `yolov8n.pt` — auto-downloaded by Ultralytics on first run)

## Setup

```powershell
git clone https://github.com/afonsocosta90/hybrid-lane-detection.git
cd hybrid-lane-detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install torch for your hardware (pick ONE):
pip install torch-directml                                                       # AMD on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121             # NVIDIA
pip install torch                                                                # CPU only

pip install -e .[dev]
```

## Running

```powershell
python main.py                                              # uses perception/configs/default.yaml
$env:PERCEPTION_CONFIG = "perception/configs/highway.yaml"  # override per-run
python main.py
```

Controls:

| Key     | Action          |
|---------|-----------------|
| `Space` | Pause / resume  |
| `q`     | Quit            |

## Tests

```powershell
pytest tests/ -v
```

The test suite is pure-logic — no `torch` model loads, no `cv2.VideoCapture` calls, no `.mp4` or `.pth` fixtures. Should run under 1 second.

---

## Configuration

All tunables live in [`perception/configs/default.yaml`](perception/configs/default.yaml). Pydantic validates at load time, before any module is constructed:

- Out-of-range value → `ValidationError` with the field path.
- Invariant violation (e.g. classical scoring weights must sum to 1.0) → custom validator raises.
- File-on-disk checks (does `weights_path` exist?) happen in each module's `warmup()` for a second fail-loud pass before the camera opens.

The legacy `hybrid_lane_detection/config.py` is preserved — `LaneModule` converts the pydantic `ClassicalConfig` into the legacy frozen dataclass once at construction so the classical backend never sees pydantic.

---

## Architecture in one paragraph

A `PerceptionModule` is anything that implements a `Protocol` with `name`, `stage`, `warmup()`, and `process(frame, state) -> Output`. The orchestrator groups modules by stage, runs each stage's modules in parallel, and collects outputs into a typed `PerceptionState`. Stage-N modules read prior outputs via typed accessors (`state.lane()`) that raise if the dependency wasn't produced. Failures in `warmup()` kill the process (fail-loud); failures in `process()` are recorded in `state.errors` and skipped (soft). Rendering is a chain of `Overlay` Protocols — each overlay reads only its module's output. Config is pydantic v2 over YAML, validated at startup. For the rest, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Lane module (the original work)

The lane module is a thin adapter over [`HybridLaneDetector`](hybrid_lane_detection/lane_detector.py), which runs two backends side-by-side on every frame:

- **Classical CV** — Canny + Hough + 2nd-order polyfit + EMA smoothing, with a perspective-aware confidence score. Renders a green carpet between the two fitted lane curves.
- **UFLD v1** — PyTorch inference via the [fast-lane-detection](https://github.com/afonsocosta90/fast-lane-detection) package. Renders red anchor-row points per detected lane.

The classical pipeline:

1. Grayscale → Gaussian blur → Canny edges
2. Trapezoidal ROI mask focused on the road
3. Probabilistic Hough transform → line segments
4. Angle filter (30°–150°) and left/right split by image midpoint
5. 2nd-order polyfit `x = ay² + by + c` per side
6. **Confidence score** — perspective-aware sanity check on lane width, perspective convergence, and heading symmetry
7. **EMA smoothing** of polynomial coefficients across frames; bad fits don't poison the EMA

The UFLD backend is a thin wrapper around `ufld.LaneDetector`. If `weights/culane_18.pth` is missing, the lane module gracefully runs classical-only and the HUD shows `UFLD: OFF`.

### UFLD weights

| Dataset  | Backbone  | Filename          | Download                                                              |
|----------|-----------|-------------------|-----------------------------------------------------------------------|
| CULane   | ResNet-18 | `culane_18.pth`   | https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view |
| Tusimple | ResNet-18 | `tusimple_18.pth` | https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view |

---

## Object-detection module

[Ultralytics YOLO](https://docs.ultralytics.com) (v8 / v11). COCO classes filtered to `{car, truck, person, bicycle, motorcycle}` — the same five classes that matter for a driving scene. Class filtering is enforced twice (via `classes=` to Ultralytics for NMS short-circuit and defensively in the post-processing comprehension) so `COCO_KEEP` is the single source of truth.

Device selection reuses `ufld.pick_device` so DirectML → CUDA → CPU preference matches the lane backend.

Ultralytics is GPLv3. Fine for a portfolio. Worth flagging if this ever turns commercial.

---

## Steering module

Two outputs per frame:

- **Steering angle (deg)** — combination of cross-track error (lane center offset from frame midline) and heading error (tangent of the centerline at an evaluation row). Positive = steer right.
- **Lookahead point** — the lane center at *N meters ahead*, rendered in image space, derived from a flat-road homography built from the existing ROI corners + assumed lane width (3.5 m).

The math lives in [`perception/geometry.py`](perception/geometry.py) as pure functions with no `cv2` import. That file is the unit-test surface.

**The homography assumption is the weakest joint in slice 1.** The ROI corners were tuned for one 1280×720 dashcam; reusing them as ground-plane correspondences inherits that tuning. "X meters ahead" will be wrong by tens of percent on a different camera. `road_depth_m` and `lane_width_m` are first-class config knobs to make this auditable. A 4-point calibration tool is slice-2 work.

---

## Concurrency model

Stage-0 modules (lane, objects) run in parallel via `ThreadPoolExecutor`. Stage-1 modules (steering) run after, reading state through typed accessors.

Threads — not multiprocessing — because both `torch.nn.Module.forward` and OpenCV's Canny/Hough release the GIL. Threads share GPU context and zero-copy numpy buffers; multiprocessing would force per-process weights in VRAM and pickle HD frames every step.

Per-frame wall time on a parallel stage = `max(lane_ms, objects_ms)`, not their sum. The HUD logs per-module ms so the speedup is visible.

---

## Roadmap

Slice 1 — this README:
- [x] PerceptionModule Protocol + PerceptionState typed accessors
- [x] Stage-parallel ThreadPoolExecutor orchestrator
- [x] Pydantic v2 config from YAML, fail-loud at startup
- [x] Lane module (adapter over existing HybridLaneDetector)
- [x] Objects module (Ultralytics YOLO, COCO-filtered)
- [x] Steering module (angle + N-meters-ahead)
- [x] Overlay chain (lane / objects / steering / HUD)
- [x] Unit tests on pure logic (geometry, pipeline contract, config validators)

Slice 2 and beyond:
- [ ] Multi-object tracking (ByteTrack) — populate `Detection.track_id`
- [ ] 4-point camera calibration tool (replace the ROI-corner homography)
- [ ] Bird's-eye-view visualization tile
- [ ] Per-module run frequency (skip-frames knob — e.g. run YOLO every 3 frames)
- [ ] Dockerfile + GitHub Actions CI

---

## Credits

- UFLDv1 architecture & weights: Qin, Wang, Li — *Ultra Fast Structure-aware Deep Lane Detection* (ECCV 2020)
- Inference packaging: [fast-lane-detection](https://github.com/afonsocosta90/fast-lane-detection)
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics

## License

Personal / educational project. No license declared yet.
