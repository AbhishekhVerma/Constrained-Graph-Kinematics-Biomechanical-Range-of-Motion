# Central Handover Ledger (`HANDOVER.md`)

**Project:** Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry  
**Course:** Foundations of Data Science (FDS) | **Target Presentation Date:** November 17  
**Repository Lead:** Abhishekh | **Teammates:** Bhavya, Vrushank, Kapish

> **Rule for All Agents & Teammates:** Read this file at the start of every work session and append a handover entry below before pushing or opening a PR. See [`AGENTS.md`](AGENTS.md) for non-negotiable invariants.

---

## 1. Current Project Snapshot

| Component | Status | Verified Output / Artifact | Owner |
| :--- | :--- | :--- | :--- |
| **Canonical 21-Joint Topology & Generator** | Ready | `data/dataset_ingestion.py`, `data/splits/split_indices.json` (960 train / 240 test) | Bhavya |
| **3D Occlusion & Glove Transforms** | Ready | `data/transforms.py` (30%/50%/70% clipping + 12% stiffness + AR(1) jitter) | Bhavya |
| **Dual-Mode Live Capture & Custom Recorder** | Verified & Recorded | `data/iphone_parser.py`, `data/record_custom_gloved.py`, `data/custom_gloved_dataset.npz` (30 seqs, `is_real_recording=True`) | Bhavya |
| **6D Rotation Math, RoM Clamping & Edge FK** | Verified | `models/kinematics.py`, `tests/test_kinematics.py` (`0.000011 mm` max bone error, `100%` bio-valid) | Vrushank |
| **Teacher ST-GAE & Spatial-Only Ablation** | Trained & Verified | `checkpoints/teacher_st_gae.pt` (520,189 params), `checkpoints/teacher_spatial_only.pt` | Abhishekh |
| **Distilled Student MLP & Direct-XYZ Baseline** | Trained & Exported | `checkpoints/student_kinematic_mlp.pt` (78,398 params), `checkpoints/student_kinematic_fk.onnx` (433 KB) | Kapish |
| **Ablation Metrics & 300-DPI Figures** | Generated | `results/metrics_summary.json`, `results/figures/fig1`–`fig6` | Abhishekh / Kapish |
| **Cross-Hardware Benchmark & Live Demo** | Verified on Windows & MacBook Air M-Series (Needs Jetson Run) | `deployment/benchmark_edge.py`, `deployment/live_demo.py`, `results/edge_benchmark.json`, `results/benchmark_macbook_air.json` | Kapish / Vrushank / Bhavya |
| **Literature Review, Slides (11) & IEEE Report** | Drafted | `docs/literature_review.md`, `docs/final_presentation_slides.md`, `Final_Project_Report.pdf` | Team |

---

## 2. Verified Baseline Metrics (Do Not Regress Below These)

### Held-Out 20% Test Split (50% Surgical Tool Occlusion + Glove Stiffness/Jitter)
- **Corrupted Input:** MPJPE `48.18 mm` | Occluded MPJPE `92.13 mm` | Bone Error `19.47 mm` | Bio-Validity `0.0%` | Jitter `29.81 mm/frame^2`
- **Unconstrained Student MLP (Direct XYZ, Un-Rigged Baseline):** MPJPE `10.18 mm` | Occluded `14.20 mm` | Bone Error `2.94 mm` | Bio-Validity `88.4%` | Jitter `14.85 mm/frame^2`
- **Spatial-Only GAT + 6D DFK (No GRU Ablation):** MPJPE `11.00 mm` | Occluded `15.41 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `16.76 mm/frame^2`
- **Teacher ST-GAE (Full GAT + GRU + 6D DFK):** MPJPE `6.64 mm` | Occluded `8.79 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `12.03 mm/frame^2`
- **Distilled Student MLP + Edge 6D FK (Ours):** MPJPE `6.51 mm` | Occluded `8.97 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `12.36 mm/frame^2`

### Custom Gloved Benchmark (`data/custom_gloved_dataset.npz`, 50% Occlusion)
- **Teacher ST-GAE:** MPJPE `6.64 mm` | **Distilled Student MLP (Ours):** MPJPE `6.51 mm`, Bone Error `0.0000 mm`, Bio-Validity `100.0%`

### ONNX Host Runtime (`results/edge_benchmark.json` / `results/benchmark_windows_pc.json` / `results/benchmark_macbook_air.json`)
- **MacBook Air M-Series (Apple Silicon):** Student ONNX (`CoreMLExecutionProvider`) `1.242 ms` single-frame `T=1` (`805 FPS`), `1.935 ms` window `T=8`, `EdgeFK` `0.072 ms` (`13.79 kHz`), `13.4x` 60 FPS real-time headroom.
- **Windows PC CPU (`CPUExecutionProvider` on x86_64 CPU):** `1.792 ms` single-frame `T=1` (`558 FPS`), `2.546 ms` window `T=8`, `0.000141 mm` max PyTorch-vs-ONNX parity error.
- **Physical Hardware Status:** Windows PC CPU and MacBook Air M-Series (Bhavya) are physically measured; Jetson Nano (Kapish) is flagged `PENDING_PHYSICAL_MEASUREMENT`.

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
| 2026-11-14 | Bhavya (Antigravity Agent) | `feat/bhavya-data` | Physically profiled Apple Silicon MacBook Air M-Series edge benchmark (Student ONNX T=1: `1.242 ms`, `804.9 FPS`, `CoreMLExecutionProvider`; EdgeFK: `0.072 ms` / `13.79 kHz`); merged benchmarks into `edge_benchmark.json`; generated custom gloved dataset with `is_real_recording=True`; regenerated 300-DPI evaluation figures. | `tests/test_kinematics.py` (All PASS), `benchmark_edge.py` (Success), `evaluate_and_plot.py` (Figures 1-5 generated) | Kapish benchmarks Jetson Nano; Vrushank verifies FreiHAND; Abhishekh merges final benchmark reports. |
