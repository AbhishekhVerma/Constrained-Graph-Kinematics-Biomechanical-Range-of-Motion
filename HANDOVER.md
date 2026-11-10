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
| **Dual-Mode Live Capture & Custom Recorder** | Code Ready (Needs Live Hardware Test) | `data/iphone_parser.py`, `data/record_custom_gloved.py`, `data/custom_gloved_dataset.npz` | Bhavya |
| **6D Rotation Math, RoM Clamping & Edge FK** | Verified | `models/kinematics.py`, `tests/test_kinematics.py` (`0.000011 mm` max bone error, `100%` bio-valid) | Vrushank |
| **Teacher ST-GAE & Spatial-Only Ablation** | Trained & Verified | `checkpoints/teacher_st_gae.pt` (520,189 params), `checkpoints/teacher_spatial_only.pt` | Abhishekh |
| **Distilled Student MLP & Direct-XYZ Baseline** | Trained & Exported | `checkpoints/student_kinematic_mlp.pt` (78,398 params), `checkpoints/student_kinematic_fk.onnx` (433 KB) | Kapish |
| **Ablation Metrics & 300-DPI Figures** | Generated | `results/metrics_summary.json`, `results/figures/fig1`–`fig6` | Abhishekh / Kapish |
| **Cross-Hardware Benchmark & Live Demo** | Tested on Windows Replay (Needs MacBook/Jetson Run) | `deployment/benchmark_edge.py`, `deployment/live_demo.py`, `results/edge_benchmark.json` | Kapish / Vrushank |
| **Literature Review, Slides (11) & IEEE Report** | Drafted | `docs/literature_review.md`, `docs/final_presentation_slides.md`, `Final_Project_Report.pdf` | Team |

---

## 2. Verified Baseline Metrics (Do Not Regress Below These)

### Held-Out 20% Test Split (50% Surgical Tool Occlusion + Glove Stiffness/Jitter)
- **Corrupted Input:** MPJPE `48.18 mm` | Occluded MPJPE `92.13 mm` | Bone Error `19.47 mm` | Bio-Validity `0.0%` | Jitter `29.81 mm/frame^2`
- **Unconstrained Student MLP (Direct XYZ):** MPJPE `15.03 mm` | Occluded `25.11 mm` | Bone Error `5.85 mm` | Bio-Validity `8.7%` | Jitter `15.17 mm/frame^2`
- **Spatial-Only GAT + 6D DFK (No GRU):** MPJPE `6.20 mm` | Occluded `9.03 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `15.96 mm/frame^2`
- **Teacher ST-GAE (Full GAT + GRU + 6D DFK):** MPJPE `3.79 mm` | Occluded `4.82 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `12.31 mm/frame^2`
- **Distilled Student MLP + Edge 6D FK (Ours):** MPJPE `5.05 mm` | Occluded `6.90 mm` | Bone Error `0.0000 mm` | Bio-Validity `100.0%` | Jitter `12.68 mm/frame^2`

### Custom Gloved Benchmark (`data/custom_gloved_dataset.npz`, 50% Occlusion)
- **Teacher ST-GAE:** MPJPE `3.60 mm` | **Distilled Student MLP (Ours):** MPJPE `4.85 mm`, Bone Error `0.0000 mm`, Bio-Validity `100.0%`

### ONNX Host Runtime (`results/edge_benchmark.json`)
- **Student ONNX (`CPUExecutionProvider`):** `0.964 ms` single-frame `T=1` (`1,037 FPS`), `0.709 ms` window `T=8`, `0.000145 mm` max PyTorch-vs-ONNX parity error.

---

## 3. Active Remaining Work Before Nov 17 Presentation

1. **Real-World Hardware Runs & Recording (All Teammates):**
   - **Bhavya / Abhishekh:** Record real gloved + scalpel/tool gestures using `data/record_custom_gloved.py --mode mediapipe` (and/or `--mode udp` from iPhone) to replace/augment the synthetic benchmark in `data/custom_gloved_dataset.npz` with real captured footage, then re-run `training/evaluate_and_plot.py`.
   - **Bhavya:** Optionally ingest real FreiHAND / HO-3D raw JSON coordinates into `data/raw/` and run real-vs-synthetic cross-validation.
   - **Kapish / Vrushank:** Run `deployment/benchmark_edge.py` natively on the **MacBook Air (Apple Silicon)** and **NVIDIA Jetson Nano (+ Raspberry Pi CSI Camera)** to replace projected numbers in `results/edge_benchmark.json` and the report with directly measured device numbers!
   - **All:** Test `deployment/live_demo.py --source mediapipe --camera 0` (MacBook / Continuity Camera) and `--jetson-csi` (Jetson Nano), and rehearse the 11-slide deck in `docs/final_presentation_slides.md`.

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
| 2026-11-10 | Abhishekh (Orchestrator Agent) | `main` (Initial Repo Setup) | Initialized Git repository, `.gitignore`, `.gitattributes`, CI workflow, PR template, `AGENTS.md`, `HANDOVER.md`, and 4 module handover docs in `docs/handover/`. Baseline codebase, checkpoints, 300-DPI figures, 11-slide deck, and `Final_Project_Report.pdf` are already built and verified. | `tests/test_kinematics.py`, `evaluate_and_plot.py`, `benchmark_edge.py`, `live_demo.py --source replay` | Push `main` to GitHub remote; teammates clone repo, create `feat/<name>-*` branches, run real hardware recordings/benchmarks on MacBook Air, iPhone, and Jetson Nano + RPi Camera, and log results here. |
