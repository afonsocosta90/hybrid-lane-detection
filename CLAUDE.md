# CLAUDE.md — Project conventions

## What this project is

A modular autonomous-perception pipeline for road video. Hybrid lane detection is one module among several (lane, objects, steering today; tracking, calibration, BEV later). The portfolio goal is **production engineering quality, not detection accuracy** — the architecture is the deliverable.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design contracts and [docs/adding-a-module.md](docs/adding-a-module.md) for the extension guide.

## Project conventions

These are non-negotiable. Each is a deliberate choice with a defensible alternative rejected.

- **Protocols over inheritance.** New modules satisfy `PerceptionModule` structurally. No ABC, no required base class. Lets third-party code (e.g. a tracker from PyPI) satisfy the contract without importing our types.
- **Pydantic v2 over frozen dataclasses for config.** Field validators + cross-field invariants run at `load_config()` startup. Wrong YAML can't reach the hot path. The legacy `hybrid_lane_detection/config.py` (frozen dataclass) is preserved unchanged — the lane module adapts pydantic → legacy at construction.
- **YAML on disk, not Python config.** Ops can retune a different dashcam without a code change. Diffs are config diffs.
- **Renderer never touches detection logic.** Overlays read only their module's slice of `PerceptionState` via typed accessors. The `Overlay` Protocol can import `cv2`; it cannot import anything from `perception.modules`.
- **Pure math in `perception/geometry.py` with no `cv2` import at module top.** That file is the unit-test surface. Tests must run under 1s with no torch / no `.mp4` / no `.pth` fixtures.
- **Fail loud at startup, fail soft at runtime.** `warmup()` raises kill the process — wrong weights, no device, singular homography are config bugs and surface before the camera opens. `process()` raises are caught and recorded in `state.errors[name]`; the pipeline continues; downstream modules raise `StageDependencyError` via the typed accessor; the renderer chain catches and skips the dependent overlay; the HUD overlay renders a red badge.
- **Threads, not multiprocessing, not asyncio.** Both torch and OpenCV release the GIL in their hot kernels — threads get real parallelism with shared GPU context and zero pickling. Multiprocessing duplicates model weights in VRAM and pickles HD frames per step. Asyncio forces `async def process` on module authors, hostile to compute-bound code.
- **Two packages, one app.** `hybrid_lane_detection/` is a standalone pip-installable library. `perception/` is the application that consumes it. Library doesn't know the app exists.
- **Outputs are `@dataclass(frozen=True, slots=True)`.** Immutable per-frame snapshots. Tuples for vectors of known length. Confidence belongs to the producer.
- **No comments explaining *what*.** Comments only when the *why* is non-obvious — a hidden constraint, a workaround, a surprising invariant. Well-named identifiers do the rest.

## Vertical-slice discipline

Slice 1 is three modules: lane, objects, steering. Everything else is later. The failure mode for this kind of repo is ten half-modules with great architecture and no working end-to-end path. When tempted to add tracking / BEV / calibration UI / dashboards mid-slice — say no. They get their own slice.

## Workflow preferences

- **Docs first.** For non-trivial work, update README / ARCHITECTURE.md / adding-a-module.md before writing implementation code. Writing the docs is a design test — anything awkward to describe in prose is awkward to write in code.
- **No `Co-Authored-By: Claude` trailer on commits.**
- **PowerShell on Windows.** This is a Windows 11 / pwsh environment. Use `$env:VAR`, not `$VAR`. Use the `Bash` tool when a POSIX flow is genuinely simpler.
