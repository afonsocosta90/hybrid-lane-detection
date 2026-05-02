# Hybrid Lane Detection

A real-time lane detection pipeline that combines a fast **classical computer vision** path (Canny + Hough + 2nd-order polynomial fit with EMA smoothing) with a **deep-learning fallback** ([UFLDv2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2) via ONNX Runtime). The classical backend runs every frame; the DL backend kicks in when classical confidence drops below a gate threshold.

The 4-tile debug view shows the full pipeline live:

```
+----------------+----------------+
|  1. ORIGINAL   |  2. GRAY+BLUR  |
+----------------+----------------+
|  3. CANNY+ROI  |  4. PREDICTION |
+----------------+----------------+
```

## Project structure

```
hybrid-lane-detection/
├── main.py                 # System orchestrator (the "Loop") + debug HUD
├── config.py               # Camera intrinsics & warp points (WIP)
├── utils/
│   └── data_types.py       # LaneData dataclass (bridge between backends)
├── models/
│   ├── classical.py        # Canny / Hough / Polyfit + EMA + confidence
│   └── dl_backend.py       # UFLDv2 wrapper (ONNX Runtime) — WIP
├── weights/
│   └── ufld_v2_res18.onnx  # Trained UFLDv2 weights (not in repo)
└── test-videos/            # Local dashcam clips (not in repo)
```

## Requirements

- Python 3.11+ (3.14 wheels are pinned in [requirements.txt](requirements.txt))
- A webcam or dashcam clip in `test-videos/`

## Setup

```powershell
# 1. Clone
git clone https://github.com/<your-user>/hybrid-lane-detection.git
cd hybrid-lane-detection

# 2. Create a virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

For GPU inference, swap `onnxruntime` for `onnxruntime-gpu` (requires CUDA 12.x).

## Running

Drop a dashcam clip into `test-videos/` (default expected name: `test_video_1.mp4`) and run:

```powershell
python main.py
```

Controls:

| Key       | Action               |
| --------- | -------------------- |
| `Space`   | Pause / resume       |
| `q`       | Quit                 |

## How it works

### Classical backend ([models/classical.py](models/classical.py))

1. Grayscale → Gaussian blur → Canny edges
2. Trapezoidal ROI mask focused on the road
3. Probabilistic Hough transform → line segments
4. Angle filter (30°–150°) and left/right split by image midpoint
5. 2nd-order polyfit `x = ay² + by + c` per side
6. **Confidence score** — perspective-aware sanity check on:
   - Lane width near the car (must be physically plausible)
   - Perspective convergence (far row narrower than near row)
   - Heading symmetry (left/right tangent mirrored)
7. **EMA smoothing** of polynomial coefficients across frames; bad fits don't poison the EMA

### DL backend ([models/dl_backend.py](models/dl_backend.py))

UFLDv2 ONNX wrapper — currently a placeholder, planned to run only when classical confidence drops below the gate.

### Bridge ([utils/data_types.py](utils/data_types.py))

Both backends emit a `LaneData` dataclass (left/right polynomial fits + confidence + source tag) so downstream code stays agnostic to which path produced the result.

## Roadmap

- [ ] Implement UFLDv2 ONNX wrapper in `dl_backend.py`
- [ ] Wire the confidence gate in `main.py` to switch backends
- [ ] Bird's-eye-view warp in `config.py` for curvature & lane-offset metrics
- [ ] Per-frame timing overlay in the HUD

## License

Personal / educational project. No license declared yet.
