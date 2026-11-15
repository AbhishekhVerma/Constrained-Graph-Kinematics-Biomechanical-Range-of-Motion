# Central Handover Ledger (`HANDOVER.md`)

**Project:** Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry  
**Course:** Foundations of Data Science (FDS) | **Target Presentation Date:** November 17  
**Repository Lead:** Abhishekh | **Teammates:** Bhavya, Vrushank, Kapish

> **Rule for All Agents & Teammates:** Read this file at the start of every work session and append a handover entry below before pushing or opening a PR. See [`AGENTS.md`](AGENTS.md) for non-negotiable invariants.

---

## 1. Current Project Snapshot

| Component | Status | Verified Output / Artifact | Owner |
| :--- | :--- | :--- | :--- |
| **Canonical 21-Joint Topology & FreiHAND Streamer** | Verified | `data/dataset_ingestion.py` (HTTP range-streamed 32,560 FreiHAND poses), `data/splits/split_indices.json` (960 train / 240 test) | Bhavya / Vrushank |
| **3D Occlusion & Glove Transforms** | Verified | `data/transforms.py` (30%/50%/70% clipping + 12% stiffness + AR(1) jitter) | Bhavya |
| **Dual-Mode Live Capture & Custom Recorder** | Verified | `data/iphone_parser.py`, `data/record_custom_gloved.py`, `data/custom_gloved_dataset.npz` | Bhavya |
| **6D Rotation Math, RoM Clamping, Calibration & Edge FK** | Verified | `models/kinematics.py`, `tests/test_kinematics.py` (`0.000011 mm` max bone error, `100%` bio-valid, dynamic palm calibration) | Vrushank |
| **Teacher ST-GAE & Spatial-Only Ablation** | Trained & Verified | `checkpoints/teacher_st_gae.pt` (520,189 params), `checkpoints/teacher_spatial_only.pt` | Abhishekh |
| **Distilled Student MLP & Direct-XYZ Baseline** | Trained & Exported | `checkpoints/student_kinematic_mlp.pt` (78,398 params), `checkpoints/student_kinematic_fk.onnx` (433 KB) | Kapish |
| **Ablation Metrics & 300-DPI Figures** | Generated | `results/metrics_summary.json`, `results/figures/fig0`–`fig7`, `results/figures/live_demo_telemetry.gif` | Abhishekh / Kapish |
| **Cross-Hardware Benchmark & Live Demo** | Measured on Windows PC & MacBook Air M-Series (Jetson Nano Pending) | `deployment/benchmark_edge.py`, `deployment/live_demo.py`, `run_presentation_demo.py`, `results/edge_benchmark.json`, `results/benchmark_macbook_air.json`, `results/benchmark_windows_pc.json` | Kapish / Bhavya |
| **Literature Review, Slides (11) & IEEE Report** | Finalized | `docs/literature_review.md`, `docs/final_presentation_slides.md`, `Final_Presentation_Slides.pdf`, `Final_Project_Report.pdf` | Team |

---

## 2. Verified Baseline Metrics (Do Not Regress Below These)

### Held-Out 20% Test Split on Hybrid Real FreiHAND + Clinical MANO (50% Surgical Tool Occlusion + Glove Stiffness/Jitter)
- **Corrupted Input:** MPJPE `52.08 mm` | Occluded MPJPE `100.29 mm` | PA-MPJPE `38.59 mm` | Bone Error `19.96 mm` | Bio-Validity `0.0%` | Jitter `25.67 mm/frame^2`
- **Unconstrained Student MLP (Direct XYZ, Un-Rigged Baseline):** MPJPE `9.95 mm` | Occluded `14.27 mm` | PA-MPJPE `8.18 mm` | Bone Error `2.31 mm` | Bio-Validity `29.95%` | Jitter `11.51 mm/frame^2`
- **Spatial-Only GAT + 6D DFK (No GRU Ablation):** MPJPE `8.41 mm` | Occluded `12.45 mm` | PA-MPJPE `6.84 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `13.26 mm/frame^2`
- **Teacher ST-GAE (Full GAT + GRU + 6D DFK):** MPJPE `6.51 mm` | Occluded `9.09 mm` | PA-MPJPE `5.00 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `7.81 mm/frame^2`
- **Distilled Student MLP + Edge 6D FK (Ours):** MPJPE `7.04 mm` | Occluded `10.02 mm` | PA-MPJPE `5.50 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `8.27 mm/frame^2`

### Custom Gloved Benchmark (`data/custom_gloved_dataset.npz`, 50% Occlusion)
- **Teacher ST-GAE:** MPJPE `6.37 mm` | **Distilled Student MLP (Ours):** MPJPE `7.01 mm`, Bone Error `0.0000 mm`, Bio-Validity `100.0%` (vs. `UnconstrainedStudentMLP`: `9.71 mm`, Bone Error `2.25 mm`, Bio-Validity `30.73%`)

### ONNX Host & Edge Hardware Runtimes (`results/edge_benchmark.json`)
- **Apple MacBook Air (M-Series ARM64 + `CoreMLExecutionProvider` / `MPS`, Measured by Bhavya):**
  - Bare-Metal NumPy `EdgeFK`: **`0.0725 ms` (`13.79 kHz` / `13,789 FPS`)** single-frame `T=1`; `0.0991 ms` (`10.09 kHz`) window `T=8`.
  - Teacher `SpatioTemporalGAE` (`PyTorch MPS`): **`5.045 ms` (`198.2 FPS`)** single-frame `T=1`; **`6.032 ms` (`165.8 FPS`)** window `T=8`.
  - Distilled `StudentKinematicMLP` (`ONNX CoreMLExecutionProvider`): **`1.242 ms` (`804.9 FPS`)** single-frame `T=1`, **`1.935 ms` (`516.7 FPS`, `0.242 ms/frame`)** window `T=8`, **`13.4x` 60 FPS real-time headroom**, `0.000246 mm` ONNX parity, `0.0000 mm` bone error.
- **Windows 11 PC Workstation (`CPUExecutionProvider` on x86_64 24-Core CPU, Measured by Abhishekh):**
  - Bare-Metal NumPy `EdgeFK`: **`0.199 ms` (`5.01 kHz`)** single-frame `T=1`; `0.288 ms` window `T=8`.
  - Teacher `SpatioTemporalGAE` (`PyTorch CPU`): **`22.094 ms` (`45.3 FPS`)** single-frame `T=1`; **`29.340 ms` (`34.1 FPS`)** window `T=8`.
  - Distilled `StudentKinematicMLP` (`ONNX CPUExecutionProvider`): **`1.792 ms` (`557.9 FPS`)** single-frame `T=1`, **`2.546 ms` (`392.8 FPS`, `0.318 ms/frame`)** window `T=8`, **`9.3x` 60 FPS real-time headroom**, `0.000141 mm` ONNX parity, `0.0000 mm` bone error.
- **Physical Hardware Status:** Windows PC x86_64 CPU and Apple MacBook Air M-Series (`MPS` / `CoreMLExecutionProvider`) are physically measured and merged; NVIDIA Jetson Nano and Raspberry Pi 5/4B remain flagged `PENDING_PHYSICAL_MEASUREMENT` for Kapish.

---

## 3. Active Remaining Work Before Nov 17 Presentation

1. **Physical Hardware Benchmarks & Gloved Recording (Teammates):**
   - **Bhavya (MacBook Air M-Series):** Clone repo on MacBook Air, run `python deployment/benchmark_edge.py --output results/benchmark_macbook_air.json`, record gloved sequences with built-in FaceTime / iPhone Continuity Camera (`python data/record_custom_gloved.py --mode mediapipe`), and commit.
   - **Kapish (NVIDIA Jetson Nano + Raspberry Pi CSI Camera):** Run `python deployment/benchmark_edge.py --output results/benchmark_jetson_nano.json` on Jetson Nano, test CSI camera live demo (`python deployment/live_demo.py --source mediapipe --jetson-csi`), and commit.
   - **Vrushank (FreiHAND Ingestion & Kinematics Parity):** Verify real FreiHAND annotation loading (`python data/dataset_ingestion.py --download-freihand`), run kinematics unit tests (`python tests/test_kinematics.py`).
   - **Abhishekh (Repository Lead):** Merge hardware benchmark JSONs (`python deployment/benchmark_edge.py --merge results/benchmark_windows_pc.json results/benchmark_macbook_air.json results/benchmark_jetson_nano.json`), finalize presentation slides and IEEE report, and compile `Final_Project_Report.pdf`.

---

## 4. Shared Tensor & API Contracts

```text
Raw / Camera / UDP Input (mm)
   └─> normalize_hand_sequence(..., scale_mm=100.0) + build_node_features_4d(...)
   └─> Tensor x: float32 shape (B, T=8, 21, 4) = [x_norm, y_norm, z_norm, m]
         │
         ├─> Teacher: SpatioTemporalGAE(x) / SpatialOnlyGAE(x)
         │      returns {"positions_mm": (B, T, 21, 3), "rot6d": (B, T, 21, 6), "attn_weights": ...}
         │
         └─> Student: StudentKinematicMLP(x)  [or ONNX checkpoints/student_kinematic_fk.onnx]
                returns {"positions_mm": (B, T, 21, 3), "rot6d": (B, T, 21, 6)}
```

- **Important:** `data/processed/*.npz` is excluded in `.gitignore` (~44 MB). Any script calling `ingest_or_generate_dataset()` will auto-generate it locally on first run, or run `python data/dataset_ingestion.py`.

---

## 5. Chronological Agent Handover Log

| Date (UTC+4) | Teammate / Agent | Branch / Commit | Summary of Changes | Verification Run | Open Handoff / Next Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-10 | Abhishekh (Orchestrator Agent) | `main` | Initialized Git repository, `.gitignore`, `.gitattributes`, CI workflow, PR template, `AGENTS.md`, `HANDOVER.md`, and 4 module handover docs in `docs/handover/`. | `tests/test_kinematics.py` | Initial repo setup. |
| 2026-11-10 | Abhishekh (Orchestrator Agent) | `main` | Removed baseline rigging; integrated official Google MediaPipe Tasks HandLandmarker; updated live recording, edge benchmark `--merge`, and 60 FPS OpenCV HUD demo; retrained Teacher & Student models honestly; generated 300-DPI publication figures. | `test_kinematics.py`, `train_teacher.py`, `distill_student.py`, `export_models.py`, `evaluate_and_plot.py`, `benchmark_edge.py`, `live_demo.py` | Bhavya runs MacBook Air benchmarks; Kapish runs Jetson Nano benchmarks; Vrushank checks FreiHAND; Abhishekh merges results. |
| 2026-11-10 | Abhishekh (Teacher & Eval Lead) | `feat/abhishekh-teacher` | Updated `docs/final_presentation_slides.md` and `docs/Final_Project_Report.md` with honest empirical metrics; recompiled `Final_Project_Report.pdf`; verified all kinematics tests; updated handover docs. | `tests/test_kinematics.py` (All PASS), `docs/build_report_pdf.py` (PDF OK) | Bhavya benchmarks MacBook Air M-Series; Kapish benchmarks Jetson Nano; Vrushank verifies FreiHAND; Abhishekh merges benchmarks. |
| 2026-11-14 | Vrushank (Kinematics & Public Data) | `feat/vrushank-kinematics` | Synced branch with main; extracted real FreiHAND `training_xyz.json` (32,560 poses) via Freiburg HTTP range stream; ingested 1,200 real sequences with analytical IK -> 6D -> FK harmonization (max bone err $3.81 \times 10^{-6}\text{ mm}$); added dynamic palm-scale calibration to `models/kinematics.py` and test to `tests/test_kinematics.py`. | `tests/test_kinematics.py` (100% PASS), `data/dataset_ingestion.py` | Real FreiHAND data ready in `data/raw/` & `data/processed/`. Abhishekh can retrain models on harmonized dataset if desired. |
| 2026-11-14 | Kapish (Student & Edge Deploy) | `feat/kapish-deployment` | Hardened `ANTIGRAVITY_ARTIFACT_DIR` path resolution in `deployment/live_demo.py`; added single-frame ($B=1, T=1$) ONNX Runtime parity test (`t1_max_err_mm`) in `distillation/export_models.py`. | `tests/test_kinematics.py`, `distillation/export_models.py`, `deployment/live_demo.py` | Physical Jetson Nano benchmark (`results/benchmark_jetson_nano.json`) ready to run when hardware is connected. |
| 2026-11-14 | Bhavya (Data & MacBook Air) | `feat/bhavya-data` | Executed physical Apple Silicon MacBook Air M-Series benchmark (`results/benchmark_macbook_air.json`): ONNX `CoreMLExecutionProvider` `1.242 ms` (`804.9 FPS`) single-frame `T=1`, `1.935 ms` (`516.7 FPS`) window `T=8`, `0.0725 ms` (`13.79 kHz`) NumPy `EdgeFK`, and `6.032 ms` Teacher `MPS`. | `tests/test_kinematics.py` (macOS ARM64 PASS), `deployment/benchmark_edge.py` | Merged into `results/edge_benchmark.json` by Abhishekh. |
| 2026-11-15 | Abhishekh (Orchestrator Agent) | `main` | Reviewed & merged PR #1 (`feat/kapish-deployment`), PR #2 (`feat/bhavya-data`), and PR #3 (`feat/vrushank-kinematics`); fixed time-varying GT sequence buffering in `data/iphone_parser.py` & `data/record_custom_gloved.py`; merged Windows PC + MacBook Air benchmarks into `results/edge_benchmark.json`; regenerated figures, slides, and `Final_Project_Report.pdf`. | `tests/test_kinematics.py` (7/7 PASS), `export_models.py`, `evaluate_and_plot.py`, `build_report_pdf.py` | All 4 branches unified and pushed to `origin`. |
| 2026-11-15 | Abhishekh (Orchestrator Agent) | `main` | Streamed 32,560 real FreiHAND 3D poses (`data/raw/freihand_xyz.json`), built 50/50 Hybrid Real FreiHAND + Clinical MANO dataset across all 5 seeds + `custom_gloved_dataset.npz`, retrained Teacher & Student models, tightened biomechanical validity threshold (`5.0 mm` / `15 deg`), added `fig0_block_diagram.png` + `fig7_fds_granular_and_gesture_utility.png` + `run_presentation_demo.py` + `live_demo_telemetry.gif`, and compiled `Final_Project_Report.pdf` & `Final_Presentation_Slides.pdf`. | `tests/test_kinematics.py` (7/7 PASS), `train_teacher.py`, `distill_student.py`, `export_models.py`, `evaluate_and_plot.py`, `build_report_pdf.py`, `build_presentation_deck.py` | Ready for Nov 17 Presentation (`python run_presentation_demo.py`). |
