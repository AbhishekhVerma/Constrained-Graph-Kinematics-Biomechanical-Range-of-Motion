# Module Handover: Vrushank — 6D Continuous Rotations & Edge Forward Kinematics

**Owned Files:**
- `models/kinematics.py` (`rot6d_to_matrix`, `matrix_to_rot6d`, `BiomechanicalRoMClamper`, `check_biomechanical_validity`, `DifferentiableFK`, `positions_to_rot6d`, `EdgeFK`)
- `tests/test_kinematics.py`

**Git Branch Convention:** `feat/vrushank-*`

---

## Current Module State
- **`rot6d_to_matrix`:** Pure PyTorch Gram-Schmidt 6D-to-`SO(3)` mapping (100% ONNX-exportable, `SO(3)` orthogonality error `2.79e-6`, determinant error `4.17e-7`).
- **`BiomechanicalRoMClamper` & `check_biomechanical_validity`:** `tanh` soft-clamping guarantees **100.0% biomechanical validity** even under extreme random inputs.
- **`DifferentiableFK` & `EdgeFK`:** Hierarchical Forward Kinematics along `PARENT_INDICES` with **`0.000011 mm`** max bone-length error and **`0.000023 mm`** PyTorch-vs-NumPy parity error.
- **Bare-Metal NumPy `EdgeFK` Speed (Windows x86_64 host):** `0.162 ms` single-frame (`6,192 FPS` / `6.19 kHz`), `0.233 ms` window `T=8`.

## Upstream / Downstream Contracts
- **Synced with Bhavya (`data/dataset_ingestion.py`):** Must keep `CANONICAL_BONE_LENGTHS_MM`, `REST_BONE_DIRECTIONS`, `ANATOMICAL_MIN_LIMITS_RAD`, `ANATOMICAL_MAX_LIMITS_RAD`, and `PARENT_INDICES` identical (currently imported from `data.dataset_ingestion` with exact fallback constants).
- **Provides to Abhishekh & Kapish:**
  - `DifferentiableFK(rot6d)` and `BiomechanicalRoMClamper(rot6d)` taking `(..., 21, 6)` and returning `(..., 21, 3)` in millimeters.
  - Pure PyTorch ops only (no custom C++ extensions, no dynamic control flow that breaks `torch.onnx.export`).

## Next Tasks for Vrushank's Agent
1. Verify `tests/test_kinematics.py` and `EdgeFK` latency directly on the **MacBook Air (Apple Silicon)** and **NVIDIA Jetson Nano** (ARM64).
2. Optional research/paper enhancement: Test per-subject palm-scale calibration (dynamically scaling `CANONICAL_BONE_LENGTHS_MM` using a live user's visible `WRIST -> MIDDLE_MCP` bone length during the first 10 frames of `live_demo.py`) without breaking the `0.0000 mm` normalized bone-length guarantee.
3. Always run `python tests/test_kinematics.py` before committing any math updates.

---

## Session History

### Session: 2026-11-10 18:05 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch & Base Commit:** `main` (initial repo setup)
- **Objective:** Establish verified `models/kinematics.py` and unit tests.
- **Commands & Tests Executed:** `.\.venv\Scripts\python.exe tests/test_kinematics.py` -> Exit code 0, all assertions passed (`0.000011 mm` bone error, `100.0%` bio-validity).
- **Handover to Next Agent:** Create `feat/vrushank-*` before modifying any kinematics code, keep `torch.onnx.export` compatibility intact, and log every benchmark or math update in `HANDOVER.md`.
