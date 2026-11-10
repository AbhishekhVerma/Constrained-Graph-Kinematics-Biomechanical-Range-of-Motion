# Module Handover: Kapish — Knowledge Distillation, ONNX Export & Multi-Hardware Deployment

**Owned Files:**
- `models/student_models.py` (`StudentKinematicMLP`, `UnconstrainedStudentMLP`, `TemporalMixingBlock1D`, `ResidualMLPBlock`)
- `distillation/distill_student.py`, `distillation/export_models.py`
- `deployment/benchmark_edge.py`, `deployment/live_demo.py`, `deployment/inference_pc_iphone.py`

**Git Branch Convention:** `feat/kapish-*`

---

## Current Module State
- **`StudentKinematicMLP` (`78,398` params, `0.30 MB` PyTorch / `433.2 KB` ONNX):**
  - Distilled from `checkpoints/teacher_st_gae.pt`, achieving `6.64x` fewer parameters and `137.3x` fewer FLOPs/frame (`0.163 MFLOPs` vs `22.43 MFLOPs`).
  - Exported to `checkpoints/student_kinematic_fk.onnx` with `0.000145 mm` max PyTorch-vs-ONNX Runtime parity error.
  - Held-out test MPJPE: `4.08 mm` (30% occ), `5.05 mm` (50% occ), `8.36 mm` (70% occ); `0.0000 mm` bone error; `100.0%` bio-valid.
  - Measured Windows CPU ONNX Runtime latency: `0.964 ms` (`1,037 FPS`) for `T=1`, `0.709 ms` for `T=8`.
- **Live Demo & Dashboard (`deployment/live_demo.py`):**
  - Tested in `--source replay` mode (`1.065 ms`, `939.3 FPS`, snapshot saved to `results/figures/fig6_live_demo_dashboard.png`).
  - Supports `--source mediapipe` (MacBook webcam, iPhone Continuity Camera, Jetson Nano + RPi CSI Camera via `--jetson-csi`) and `--source udp` (`--port 5005`).

## Upstream / Downstream Contracts
- **Expects from Abhishekh:** Trained `checkpoints/teacher_st_gae.pt` and `training/losses.py` (`DistillationLoss`).
- **Expects from Vrushank:** ONNX-exportable `BiomechanicalRoMClamper` and `DifferentiableFK`.
- **Expects from Bhavya:** Normalized `(B, T=8, 21, 4)` inputs from `data/dataloader.py` and `data/iphone_parser.py`.

## Next Tasks for Kapish's Agent
1. **Run natively on MacBook Air (Apple Silicon):**
   ```bash
   python3 deployment/benchmark_edge.py
   python3 deployment/live_demo.py --source mediapipe --camera 0
   ```
   Verify `CoreMLExecutionProvider` / `MPS` timings and test iPhone Continuity Camera (`--camera 1`) and ARKit UDP (`--source udp --port 5005`).
2. **Run natively on NVIDIA Jetson Nano + Raspberry Pi CSI Camera:**
   ```bash
   python3 deployment/benchmark_edge.py
   python3 deployment/live_demo.py --source mediapipe --jetson-csi
   ```
   Verify `TensorrtExecutionProvider` / `CUDAExecutionProvider` timings and GStreamer `nvarguscamerasrc` capture.
3. Commit measured `results/edge_benchmark.json` logs from the MacBook Air and Jetson Nano back to `HANDOVER.md` so Abhishekh can embed real device measurements into `Final_Project_Report.pdf` before Nov 17!

---

## Session History

### Session: 2026-11-10 18:05 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch & Base Commit:** `main` (initial repo setup)
- **Objective:** Establish distilled ONNX student model, benchmark suite, and dual-mode live demo.
- **Commands & Tests Executed:** `distillation/export_models.py`, `deployment/benchmark_edge.py`, `deployment/live_demo.py --source replay` -> All passed (`0.964 ms` ONNX CPU, `0.000145 mm` parity).
- **Handover to Next Agent:** Pull `main` on MacBook Air and Jetson Nano, create `feat/kapish-hardware-validation`, run live benchmarks/camera demos, and log measured device FPS/latencies in `HANDOVER.md`.
