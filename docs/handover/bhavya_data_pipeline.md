# Module Handover: Bhavya — Data Pipeline, Real Gloved Capture & MacBook Air Benchmark

**Owned Files & Hardware Responsibilities:**
- `data/transforms.py` (`SyntheticOcclusionTransform`, `SyntheticGloveTransform`)
- `data/dataloader.py` (`normalize_hand_sequence`, `build_node_features_4d`, `get_dataloaders`, `data/splits/split_indices.json`)
- `data/iphone_parser.py` (`ARKitUDPStreamer`, `MediaPipeCameraStreamer`)
- `data/record_custom_gloved.py` (`data/custom_gloved_dataset.npz`)
- **Hardware Owned:** Apple Silicon MacBook Air + iPhone (Continuity Camera / ARKit UDP)

**Git Branch Convention:** `feat/bhavya-data`

---

## Concrete Physical Tasks for Bhavya (Before Nov 17)

### Task 1: Run Physical Edge Benchmark on Your MacBook Air (Apple Silicon M-Series)
Because you have the MacBook Air, run the physical hardware profiler on macOS so we replace unmeasured placeholders with real Apple Silicon (`MPS` / `CoreMLExecutionProvider` / ARM64 CPU) timings:
```bash
git checkout feat/bhavya-data
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Measure real MacBook Air M-Series latency & throughput and save to JSON:
python3 deployment/benchmark_edge.py --output results/benchmark_macbook_air.json
```

### Task 2: Record Real Gloved-Hand & Tool-Occluded Sequences (`data/custom_gloved_dataset.npz`)
Currently, `data/custom_gloved_dataset.npz` is a synthetic baseline (`is_real_recording=False`). Using your MacBook Air built-in FaceTime HD camera (`--camera-index 0`) or your iPhone via Continuity Camera (`--camera-index 1`), put on a physical glove (latex, nitrile, or work glove) and hold a tool/pen/scalpel:
```bash
# Record 30 real sequences (8 frames each) using MediaPipe 3D Hands on your MacBook/iPhone camera:
python3 data/record_custom_gloved.py --mode mediapipe --camera-index 0 --num-sequences 30 --occlusion-rate 0.50 --output data/custom_gloved_dataset.npz

# Test the real-time 30-60 FPS interactive OpenCV live demo on your MacBook Air:
python3 deployment/live_demo.py --source mediapipe --camera 0 --occlusion-rate 0.50
```
*(In the live demo window, press `o` to cycle 0%/30%/50%/70% tool occlusion, `t` to toggle the virtual surgical tool zone, `s` to save a live snapshot, and `q` to quit.)*

### Task 3: Commit & Hand Over to Abhishekh
```bash
python3 tests/test_kinematics.py
git add results/benchmark_macbook_air.json data/custom_gloved_dataset.npz docs/handover/bhavya_data_pipeline.md HANDOVER.md
git commit -m "feat(bhavya): add real MacBook Air M-series benchmark and real gloved camera dataset"
git push origin feat/bhavya-data
```
Log the exact MacBook Air FPS/latency and number of recorded gloved sequences in `HANDOVER.md` and below.

---

## Shared Contracts to Preserve
- `scale_mm = 100.0` wrist-centered normalization (`joint 0 == (0, 0, 0)`).
- 4D node features `(B, T=8, 21, 4)` = `[x_norm, y_norm, z_norm, m]` where `m == 0` zeros out coordinates.
- Do NOT alter `JOINT_NAMES`, `PARENT_INDICES`, or `data/splits/split_indices.json`.

---

## Session History

### Session: 2026-11-10 18:25 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch:** `main` -> `feat/bhavya-data`
- **Summary:** Fixed `data/iphone_parser.py` (MediaPipe Tasks `HandLandmarker` + `mp.solutions` support, real camera `frame_bgr` return, virtual tool occlusion box, disabled silent fake fallback in hardware mode) and fixed `data/record_custom_gloved.py` so real camera recordings are saved directly to `data/custom_gloved_dataset.npz` without being overwritten by synthetic data. Assigned MacBook Air physical benchmark and real gloved capture to Bhavya.

### Session: 2026-11-14 23:05 (UTC+4) — Bhavya (Apple Silicon MacBook Air M-Series)
- **Git Branch:** `feat/bhavya-data` (`00342055`)
- **Summary:**
  1. Executed physical hardware benchmark on Apple Silicon MacBook Air (`macOS 26.3.1`, `arm64`, 8 cores, PyTorch `2.11.0` `MPS` + ONNX Runtime `1.24.4` `CoreMLExecutionProvider`) via `python3 deployment/benchmark_edge.py --output results/benchmark_macbook_air.json` and merged into `results/edge_benchmark.json`:
     - Bare-Metal `EdgeFK` (NumPy ARM64 Vectorized): **`0.0725 ms` (`13,789 FPS` / `13.79 kHz`)** single-frame `T=1`; `0.0991 ms` (`10.09 kHz`) window `T=8`.
     - Teacher `SpatioTemporalGAE` (`PyTorch MPS`): **`5.045 ms` (`198.2 FPS`)** single-frame `T=1`; **`6.032 ms` (`165.8 FPS`)** window `T=8`.
     - `StudentKinematicMLP` (`PyTorch MPS` 6D-FK): **`2.818 ms` (`354.8 FPS`)** single-frame `T=1`; `3.214 ms` window `T=8`.
     - `StudentKinematicMLP` (`ONNX Runtime CoreMLExecutionProvider`): **`1.242 ms` (`804.9 FPS`)** single-frame `T=1`; **`1.935 ms` (`516.7 FPS`, `0.242 ms/frame`)** window `T=8`; P95 `1.396 ms`, consuming only `7.45%` of the 60 FPS frame budget (`13.4x` headroom), `0.000246 mm` ONNX parity, and `0.0000 mm` bone-length error (`100.0%` bio-validity).
  2. Verified `tests/test_kinematics.py` on macOS ARM64 (all unit tests pass).

### Session: 2026-11-15 20:45 (UTC+4) — Abhishekh / Orchestrator Agent (PR #2 Review & Merge)
- **Git Branch:** `feat/bhavya-data` -> `main`
- **Summary:**
  - Integrated Bhavya's real physical Apple Silicon MacBook Air benchmark (`results/benchmark_macbook_air.json`) and merged with `results/benchmark_windows_pc.json` into `results/edge_benchmark.json`.
  - Fixed time-varying ground-truth sequence buffering in `data/iphone_parser.py` (`self._gt_mm_buffer` in `ARKitUDPStreamer` and `MediaPipeCameraStreamer`) and `data/record_custom_gloved.py` (`SyntheticGloveTransform` + `seq_len - 1` stride) so `read_sequence()` returns full `(T, 21, 3)` time-varying ground-truth trajectories rather than broadcasting the final frame.

### Session: 2026-11-16 11:05 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch:** `main`
- **Summary:**
  - Hardened Mode B UDP Streamer (`ARKitUDPStreamer` in `data/iphone_parser.py` and `deployment/live_demo.py`).
  - Added project root to `sys.path` in `data/iphone_parser.py` and added standalone CLI diagnostics (`--mode test`, `--mode listen`, `--mode send`).
  - Optimized socket timeout (`timeout_sec=0.05` default, `0.03` in `live_demo.py`) so fallback mode runs silky smooth at 30+ FPS instead of 4 FPS without stalling incoming 60 Hz live packets.
  - Verified 100% loopback packet reception, 8-frame temporal windowing, ONNX model inference, and 0.0000 mm bone error across UDP payloads.

### Session: 2026-11-16 11:25 (UTC+4) — Bhavya & Antigravity Agent (Dual-Hand Tracking Test)
- **Git Branch:** `feat/bhavya-data` (merged with `origin/feat/bhavya-data`)
- **Summary:**
  - Extended `MediaPipeCameraStreamer` in `data/iphone_parser.py` to support `num_hands=2` via Google MediaPipe Tasks API (`vision.HandLandmarkerOptions(num_hands=2)`) and legacy fallback.
  - Implemented simultaneous per-hand 3D tracking, EMA temporal smoothing, canonical coordinate alignment, and dual 2D skeleton overlays on live camera (Emerald Green for Hand 1 / Left Hand, Amber Gold for Hand 2 / Right Hand).
  - Implemented dynamic batching `(2, T, 21, 4)` in `deployment/live_demo.py` for concurrent ONNX Runtime 6D-FK student inference and Direct-XYZ baseline.
  - Extended `render_live_opencv_frame()` to render dual hands side-by-side in Panel 2 (unconstrained) and Panel 3 (6D-FK student with 0.0000 mm bone error on both hands).
  - Added `--two-hands` CLI flag and interactive menu option `[7]` to `run_presentation_demo.py`.
  - Maintained 100% backward compatibility for single-hand mode and strict 0.0000 mm bone error invariance.
  - Preserved 100% clean revertibility: switching back to single-hand baseline is accomplished simply via `git checkout main`.
- **Verification Run:** `python tests/test_kinematics.py` (All PASS), `deployment/live_demo.py --source replay --two-hands --headless --max-frames 10` (848.7 FPS, 0.0000 mm bone error).
