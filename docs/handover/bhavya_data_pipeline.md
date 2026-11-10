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
