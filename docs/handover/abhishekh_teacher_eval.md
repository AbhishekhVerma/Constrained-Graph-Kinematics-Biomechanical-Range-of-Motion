# Module Handover: Abhishekh — Teacher ST-GAE, Ablations & Documentation

**Owned Files:**
- `models/teacher_st_gae.py` (`MultiHeadGraphAttentionLayer`, `SpatioTemporalGAE`, `SpatialOnlyGAE`)
- `training/losses.py` (`MPJPELoss`, `Rotation6DLoss`, `TemporalSmoothnessLoss`, `DistillationLoss`)
- `training/utils.py` (`compute_mpjpe_mm`, `compute_pa_mpjpe_mm`, `compute_bone_length_error_mm`, `compute_temporal_jitter_mm_s2`, `compute_biomechanical_validity_pct`)
- `training/train_teacher.py`, `training/evaluate_and_plot.py`
- `docs/literature_review.md`, `docs/final_presentation_slides.md`, `docs/Final_Project_Report.md`, `docs/build_report_pdf.py`

**Git Branch Convention:** `feat/abhishekh-*`

---

## Current Module State
- **Teacher `SpatioTemporalGAE` (`520,189` params, `1.99 MB`):** Trained and saved at `checkpoints/teacher_st_gae.pt`.
  - Held-out test MPJPE: `3.33 mm` (30% occ), `3.79 mm` (50% occ), `6.35 mm` (70% occ); `0.0000 mm` bone error; `100.0%` bio-valid; `12.31 mm/frame^2` temporal jitter at 50% occ.
- **Ablation `SpatialOnlyGAE` (`491,198` params):** Trained and saved at `checkpoints/teacher_spatial_only.pt`.
  - Held-out test MPJPE: `4.67 mm` (30% occ), `6.20 mm` (50% occ), `10.54 mm` (70% occ); `15.96 mm/frame^2` temporal jitter at 50% occ.
- **Outputs & Figures:** `results/metrics_summary.json` and `results/figures/fig1`–`fig5` generated at 300 DPI.

## Upstream / Downstream Contracts
- **Expects from Bhavya (`data/dataloader.py`):** Batch dicts with normalized 4D features `(B, T=8, 21, 4)`, `gt_positions_mm` `(B, T, 21, 3)`, `gt_rot6d` `(B, T, 21, 6)`, and `visibility_mask` `(B, T, 21)`.
- **Expects from Vrushank (`models/kinematics.py`):** `BiomechanicalRoMClamper`, `DifferentiableFK`, `get_hand_adjacency_matrix`, and `check_biomechanical_validity`.
- **Provides to Kapish (`distillation/distill_student.py`):** `checkpoints/teacher_st_gae.pt` outputting `{"positions_mm", "rot6d", "attn_weights"}`.

## Next Tasks for Abhishekh's Agent
1. Re-run `training/evaluate_and_plot.py` once Bhavya adds real recorded gloved/tool samples to `data/custom_gloved_dataset.npz` or real FreiHAND/HO-3D JSONs to `data/raw/`.
2. Update `docs/final_presentation_slides.md` and `docs/Final_Project_Report.md` (and rebuild `Final_Project_Report.pdf` via `python docs/build_report_pdf.py`) with real MacBook Air & Jetson Nano hardware numbers from Kapish.

---

## Session History

### Session: 2026-11-10 18:05 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch & Base Commit:** `main` (initial repo setup)
- **Objective:** Establish Teacher ST-GAE baseline, ablation pipeline, figures, slides, and handover docs.
- **Commands & Tests Executed:** `tests/test_kinematics.py`, `training/evaluate_and_plot.py` -> All passed, checkpoints and 300-DPI figures verified.
- **Handover to Next Agent:** Pull `main`, create `feat/abhishekh-*` before making edits, and update this file + `HANDOVER.md` after any re-training or report rebuild.
