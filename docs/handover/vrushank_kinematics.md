# Module Handover: Vrushank — 6D Continuous Rotations, Public Dataset Ingestion & Fair Ablations

**Owned Files & Responsibilities:**
- `models/kinematics.py` (`rot6d_to_matrix`, `matrix_to_rot6d`, `BiomechanicalRoMClamper`, `check_biomechanical_validity`, `DifferentiableFK`, `positions_to_rot6d`, `EdgeFK`)
- `data/dataset_ingestion.py` (Public `FreiHAND` / `HO-3D` 3D annotation ingestion & IK->FK harmonization into 6D rotations)
- `tests/test_kinematics.py`

**Git Branch Convention:** `feat/vrushank-kinematics`

---

## Concrete Tasks for Vrushank (Before Nov 17)

### Task 1: Ingest Real Public 3D Hand Annotations (`FreiHAND` / `HO-3D`) via 6D IK-FK Harmonization
Previously, `_try_load_real_annotations()` in `data/dataset_ingestion.py` was not wired into the saved `.npz` files, so training only used procedural MANO trajectories.
1. Pull the latest `main` into `feat/vrushank-kinematics`:
   ```bash
   git checkout feat/vrushank-kinematics
   git pull origin main
   ```
2. Download or place the lightweight official **FreiHAND** 3D XYZ annotation file (`training_xyz.json` / `evaluation_xyz.json`, ~4 MB) into `data/raw/` using the built-in downloader:
   ```bash
   python data/dataset_ingestion.py --download-freihand
   ```
   *(Or place `training_xyz.json` directly into `data/raw/` and run `python data/dataset_ingestion.py`.)*
3. Verify that `ingest_or_generate_dataset()` maps the real 3D MANO joints through `positions_to_rot6d` -> `DifferentiableFK` (`0.0000 mm` bone-length error) and reports `"data_source": "real_freihand_annotations"`.

### Task 2: Verify 6D Kinematics & Fair Baseline Integrity (`models/kinematics.py`, `tests/test_kinematics.py`)
1. Run the full kinematics and ONNX parity suite on your machine:
   ```bash
   python tests/test_kinematics.py
   ```
2. Optional enhancement for live camera demos: In `models/kinematics.py`, verify/tune live palm-scale calibration (scaling `CANONICAL_BONE_LENGTHS_MM` by a scalar factor $\alpha$ estimated from visible `WRIST -> MIDDLE_MCP` length during the first unoccluded frames of a camera stream while keeping relative bone lengths locked to `0.0000 mm` variance).
3. Commit and push to `feat/vrushank-kinematics`, logging your dataset ingestion stats and test results in `HANDOVER.md` and below so Abhishekh can re-run the final training on the real FreiHAND trajectories!

---

## Shared Contracts to Preserve
- `CANONICAL_BONE_LENGTHS_MM`, `REST_BONE_DIRECTIONS`, `ANATOMICAL_MIN_LIMITS_RAD`, `ANATOMICAL_MAX_LIMITS_RAD`, and `PARENT_INDICES` must stay identical between `data/dataset_ingestion.py` and `models/kinematics.py`.
- Pure PyTorch tensor ops only in `rot6d_to_matrix`, `BiomechanicalRoMClamper`, and `DifferentiableFK` so `torch.onnx.export` succeeds without C++ extensions.

---

## Session History

### Session: 2026-11-10 18:25 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch:** `main` -> `feat/vrushank-kinematics`
- **Summary:** Fixed `data/dataset_ingestion.py` so `_try_load_real_annotations()` actually processes real `FreiHAND`/`HO-3D` 3D JSON/NPY annotations in `data/raw/` and added `--download-freihand`. Removed artificial handicaps from `SpatialOnlyGAE` and `UnconstrainedStudentMLP`. Assigned public dataset ingestion & kinematics verification to Vrushank (swapped MacBook Air physical benchmark to Bhavya).
