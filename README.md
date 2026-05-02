# Hybrid Lane Detection

Real-time lane detection that runs **two backends concurrently** on every frame and renders both for direct comparison:

- **Classical CV** — Canny + Hough + 2nd-order polyfit + EMA smoothing, with a perspective-aware confidence score. Renders a green carpet between the two fitted lane curves.
- **UFLD (Ultra-Fast Lane Detection)** — PyTorch inference via the [fast-lane-detection](https://github.com/afonsocosta90/fast-lane-detection) package. Renders red anchor-row points per detected lane.

The 4-tile debug view shows the full pipeline:

```
+----------------+----------------+
|  1. ORIGINAL   |  2. GRAY+BLUR  |
+----------------+----------------+
|  3. CANNY+ROI  |  4. CLASSICAL  |
|                |     + UFLD     |
+----------------+----------------+
```

## Project structure

```
hybrid-lane-detection/
├── main.py                 # System orchestrator + dual-backend rendering
├── config.py               # Camera intrinsics & warp points (WIP)
├── utils/
│   └── data_types.py       # LaneData dataclass (classical output)
├── models/
│   ├── classical.py        # Canny / Hough / Polyfit + EMA + confidence
│   └── dl_backend.py       # Thin wrapper over ufld.LaneDetector
├── weights/                # UFLD checkpoints (not in repo — see weights/README)
└── test-videos/            # Local dashcam clips (not in repo)
```

## Requirements

- **Python 3.11** (the `torch-directml` wheel for AMD on Windows is 3.11-only)
- A webcam or dashcam clip in `test-videos/`
- A UFLD CULane / Tusimple checkpoint in `weights/`

## Setup

```powershell
# 1. Clone
git clone https://github.com/afonsocosta90/hybrid-lane-detection.git
cd hybrid-lane-detection

# 2. Create a Python 3.11 virtualenv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install torch for your hardware (pick ONE):
pip install torch-directml                                                       # AMD on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121             # NVIDIA
pip install torch                                                                # CPU only

# 4. Install the rest
pip install -r requirements.txt
```

## Weights

The UFLD backend needs a pretrained checkpoint. Place it at `weights/culane_18.pth`:

| Dataset  | Backbone  | Filename             | Download                                                              |
|----------|-----------|----------------------|-----------------------------------------------------------------------|
| CULane   | ResNet-18 | `culane_18.pth`      | https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view |
| Tusimple | ResNet-18 | `tusimple_18.pth`    | https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view |

If the weights file is missing, the program still runs — the UFLD path is skipped and only the classical overlay is drawn. The HUD says `UFLD: OFF`.

To switch dataset/backbone, edit the constants at the top of [main.py](main.py).

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

Thin wrapper around `ufld.LaneDetector` from the [fast-lane-detection](https://github.com/afonsocosta90/fast-lane-detection) package. UFLDv1 (Qin et al., ECCV 2020) — runs on AMD via DirectML, NVIDIA via CUDA, or CPU. Returns up to 4 lanes as lists of `(x, y)` pixel points sampled at fixed row anchors.

### Concurrency

Both backends run sequentially on the same frame inside the main loop. There's no threading yet — frame time is roughly `classical_time + UFLD_time`. The visualization draws both overlays on the prediction tile so you can directly compare where they agree and disagree.

## Roadmap

- [x] Dual-backend concurrent rendering (classical + UFLD)
- [ ] Bird's-eye-view warp in `config.py` for curvature & lane-offset metrics
- [ ] Per-frame timing overlay in the HUD
- [ ] Optional: move UFLD onto a worker thread to decouple frame rate

## Credits

- UFLDv1 architecture & weights: Qin, Wang, Li — *Ultra Fast Structure-aware Deep Lane Detection* (ECCV 2020)
- Inference packaging: [fast-lane-detection](https://github.com/afonsocosta90/fast-lane-detection)

## License

Personal / educational project. No license declared yet.
