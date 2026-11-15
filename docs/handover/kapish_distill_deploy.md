# Module Handover: Kapish — Knowledge Distillation, ONNX Export & Jetson Nano / Live Camera Deployment

**Owned Files & Hardware Responsibilities:**
- `models/student_models.py` (`StudentKinematicMLP`, `UnconstrainedStudentMLP`)
- `distillation/distill_student.py`, `distillation/export_models.py`
- `deployment/benchmark_edge.py`, `deployment/live_demo.py`, `deployment/inference_pc_iphone.py`
- **Hardware Owned:** NVIDIA Jetson Nano + Raspberry Pi Camera v2 (+ iPhone / PC)

**Git Branch Convention:** `feat/kapish-deployment`

---

## Concrete Physical Tasks for Kapish (Before Nov 17)

### Task 1: Run Physical Edge Benchmark on NVIDIA Jetson Nano
All fake hardware multiplier projections have been removed from `deployment/benchmark_edge.py`. Clone the repo on the **NVIDIA Jetson Nano** and run the physical benchmark script so we record 100% genuine ARM Cortex-A57 CPU and Maxwell GPU (`CUDA` / `TensorrtExecutionProvider`) latencies:
```bash
git checkout feat/kapish-deployment
git pull origin main
python3 tests/test_kinematics.py

# Run physical benchmark on Jetson Nano and save device JSON:
python3 deployment/benchmark_edge.py --output results/benchmark_jetson_nano.json
```

### Task 2: Test the Real-Time OpenCV Live Demo with Raspberry Pi Camera / iPhone Camera
`deployment/live_demo.py` has been upgraded from a static offline plot to a true **30–60 FPS interactive OpenCV video loop** (`cv2.imshow`) with live 2D camera overlay, virtual/real tool occlusion, and side-by-side 3D skeleton repair:
```bash
# On NVIDIA Jetson Nano with Raspberry Pi CSI Camera:
python3 deployment/live_demo.py --source mediapipe --jetson-csi --occlusion-rate 0.50

# Or with a standard USB / iPhone camera:
python3 deployment/live_demo.py --source mediapipe --camera 0 --occlusion-rate 0.50
```
- While the live camera window is running:
  - Press **`o`** to cycle occlusion (`0% -> 30% -> 50% -> 70%`).
  - Press **`t`** to toggle the virtual surgical tool occlusion box over your fingers.
  - Press **`s`** to save a real live-camera telemetry screenshot to `results/figures/fig6_live_demo_dashboard.png`.
  - Press **`q`** or `ESC` to quit.

### Task 3: Commit & Hand Over to Abhishekh
```bash
git add results/benchmark_jetson_nano.json results/figures/fig6_live_demo_dashboard.png docs/handover/kapish_distill_deploy.md HANDOVER.md
git commit -m "feat(kapish): add real Jetson Nano benchmark and live camera demo snapshot"
git push origin feat/kapish-deployment
```
Log the exact measured Jetson Nano latencies (`ms` and `FPS`) in `HANDOVER.md` and below.

---

## Session History

### Session: 2026-11-10 18:25 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch:** `main` -> `feat/kapish-deployment`
- **Summary:** Removed `+ 0.24 * mse_obs` baseline sabotage in `distill_student.py`, added dynamic sequence length `T` support in `export_models.py`, removed hardcoded hardware multipliers in `benchmark_edge.py` (added `--output` and `--merge` for real device JSONs), and upgraded `live_demo.py` into a real-time 30–60 FPS interactive OpenCV video loop.

### Session: 2026-11-14 22:55 (UTC+4) — Kapish (`feat/kapish-deployment`)
- **Git Branch:** `feat/kapish-deployment` (`42b2157e`)
- **Summary:**
  1. Hardened `ANTIGRAVITY_ARTIFACT_DIR` path resolution in `deployment/live_demo.py` (`os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", "").strip()`) so live camera snapshots (`s` key) and headless exports never fail when the environment variable is unset.
  2. Added streaming single-frame ($B=1, T=1$) ONNX Runtime parity verification (`t1_max_err_mm`) to `distillation/export_models.py` alongside batched ($B=4, T=8$) verification to guarantee dynamic sequence-axis parity for real-time frame-by-frame edge deployment.
  3. Verified `tests/test_kinematics.py` and headless `live_demo.py --headless`.

### Session: 2026-11-15 20:45 (UTC+4) — Abhishekh / Orchestrator Agent (PR #1 Review & Merge)
- **Git Branch:** `feat/kapish-deployment` -> `main`
- **Summary:** Integrated Kapish's `ANTIGRAVITY_ARTIFACT_DIR` sanitization in `deployment/live_demo.py` and single-frame ($B=1, T=1$) ONNX Runtime parity test in `distillation/export_models.py` into `main` while preserving the honest un-rigged models (`checkpoints/student_kinematic_mlp.pt`, `checkpoints/student_kinematic_fk.onnx`) and multi-device benchmark merger in `deployment/benchmark_edge.py`.
