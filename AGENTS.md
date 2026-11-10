# Repository Agent & Collaboration Contract (`AGENTS.md`)

This file governs **all AI coding agents** (Antigravity, Claude, Codex, Cursor, Copilot) and human contributors working on **Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry**.

---

## 1. Mandatory Session Protocol (Read First, Write Last)

Every agent session MUST follow this exact workflow so all four teammates stay synchronized across machines (Windows PC, Apple Silicon MacBook Air, NVIDIA Jetson Nano, Raspberry Pi, iPhone):

### At Session Start (Before Editing Any File)
1. **Pull latest changes** from `main` (or your feature branch):
   ```bash
   git fetch origin
   git status
   ```
2. **Read `HANDOVER.md`** at the repository root to review current integration status, recent commits, active blockers, and downstream notes.
3. **Read your module's handover document** in `docs/handover/`:
   - **Abhishekh (Teacher ST-GAE & Evaluation):** `docs/handover/abhishekh_teacher_eval.md`
   - **Bhavya (Data Pipeline & Domain Adaptation):** `docs/handover/bhavya_data_pipeline.md`
   - **Vrushank (6D Rotations & Edge FK):** `docs/handover/vrushank_kinematics.md`
   - **Kapish (Student Distillation & Edge Deployment):** `docs/handover/kapish_distill_deploy.md`
4. **Check if your changes touch a shared interface** (Section 3 below). Never silently alter tensor shapes, joint index ordering, or file paths owned by another module.

### Before Ending a Session (Mandatory Handover Update)
1. **Run verification tests** matching what you changed:
   - Always run: `python tests/test_kinematics.py`
   - If data/models/training changed: re-run `distillation/export_models.py`, `training/evaluate_and_plot.py`, and `deployment/benchmark_edge.py`.
2. **Update your module handover file** (`docs/handover/<teammate>_<module>.md`) using `docs/handover/TEMPLATE_HANDOVER.md`.
3. **Append one row to the Handover Log in `HANDOVER.md`** documenting: date/time, teammate & agent, branch/commit, files modified, commands & test results observed, any interface notes, and next steps for teammates.
4. **Commit on your feature branch** (`feat/<teammate>-<short-topic>`) and open a Pull Request to `main` using `.github/PULL_REQUEST_TEMPLATE.md`. Never force-push `main`.

---

## 2. Team Ownership & Git Branching Map

| Teammate | Role | Owned Paths | Branch Prefix | Handover File |
| :--- | :--- | :--- | :--- | :--- |
| **Abhishekh** | Teacher ST-GAE & Evaluation | `models/teacher_st_gae.py`, `training/`, `docs/` | `feat/abhishekh-*` | `docs/handover/abhishekh_teacher_eval.md` |
| **Bhavya** | Data Pipeline & 3D Domain Adaptation | `data/` | `feat/bhavya-*` | `docs/handover/bhavya_data_pipeline.md` |
| **Vrushank** | 6D Continuous Rotations & Edge FK | `models/kinematics.py`, `tests/test_kinematics.py` | `feat/vrushank-*` | `docs/handover/vrushank_kinematics.md` |
| **Kapish** | Student Distillation & Hardware Deployment | `models/student_models.py`, `distillation/`, `deployment/` | `feat/kapish-*` | `docs/handover/kapish_distill_deploy.md` |

Cross-module edits require noting the change in **both** affected handover files and in `HANDOVER.md`.

---

## 3. Non-Negotiable Technical Invariants (Shared Contracts)

Do NOT break these contracts unless all four module owners agree and update every downstream script:

1. **Canonical 21-Joint MANO Skeleton Ordering (`0..20`)** (`data/dataset_ingestion.py` & `models/kinematics.py`):
   - `0`: `WRIST` (root)
   - `1..4`: `THUMB_CMC`, `THUMB_MCP`, `THUMB_IP`, `THUMB_TIP`
   - `5..8`: `INDEX_MCP`, `INDEX_PIP`, `INDEX_DIP`, `INDEX_TIP`
   - `9..12`: `MIDDLE_MCP`, `MIDDLE_PIP`, `MIDDLE_DIP`, `MIDDLE_TIP`
   - `13..16`: `RING_MCP`, `RING_PIP`, `RING_DIP`, `RING_TIP`
   - `17..20`: `PINKY_MCP`, `PINKY_PIP`, `PINKY_DIP`, `PINKY_TIP`
   - `PARENT_INDICES = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]`
2. **Root Normalization & Units**:
   - Network **inputs** are wrist-centered (`joint 0 == (0, 0, 0)`) and scaled by `DEFAULT_PALM_SCALE_MM = 100.0` mm (`data/dataloader.py`: `normalize_hand_sequence`).
   - Node feature tensor shape is **`(B, T, 21, 4)`** containing `[x_norm, y_norm, z_norm, m]`, where `m in {0.0, 1.0}` is the binary visibility mask (`m = 0.0` zeros out coordinates; wrist `0` remains `m = 1.0`).
   - Network **outputs** (`positions_mm`) are in **millimeters** with wrist at `(0, 0, 0)` and shape **`(B, T, 21, 3)`**.
3. **6D Continuous Rotation Convention**:
   - Models (`SpatioTemporalGAE`, `SpatialOnlyGAE`, `StudentKinematicMLP`) predict `rot6d` of shape `(B, T, 21, 6)` -> clamped via `BiomechanicalRoMClamper` -> converted to `SO(3)` via Gram-Schmidt (`rot6d_to_matrix`) -> traversed via `DifferentiableFK` / `EdgeFK`.
   - Bone lengths MUST match `CANONICAL_BONE_LENGTHS_MM` (`0.0000 mm` error on FK outputs).
   - Pure PyTorch ops only inside forward graphs exported to ONNX (no `torch-geometric` C++ scatter extensions inside ONNX-exported Student graphs).
4. **Strict 80/20 Train/Test Split**:
   - Split indices are persisted in `data/splits/split_indices.json` (`960` train / `240` test). Never evaluate on training indices or overwrite the split silently.
5. **ONNX & Hardware Compatibility**:
   - `checkpoints/student_kinematic_fk.onnx` must remain compatible with `onnxruntime` CPU, macOS CoreML/MPS, and Jetson Nano TensorRT/CUDA.
6. **Academic Writing Style (`docs/`)**:
   - Keep presentation slides and IEEE report prose direct, factual, and free of AI writing clichés (avoid words such as *delve, testament, tapestry, catastrophically, robust, crucial, vital, moreover, furthermore, underscore, seamlessly*).

---

## 4. Environment & Quick Verification Commands

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\python.exe tests/test_kinematics.py
  ```
- **macOS (MacBook Air) / Linux (Jetson Nano / Raspberry Pi):**
  ```bash
  python3 tests/test_kinematics.py
  ```
- **Regenerate Processed Training Data (ignored in `.gitignore` to keep git fast):**
  ```bash
  python data/dataset_ingestion.py
  ```
