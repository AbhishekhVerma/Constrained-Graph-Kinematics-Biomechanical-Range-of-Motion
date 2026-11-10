# Module Handover: Bhavya — Data Pipeline & 3D Domain Adaptation

**Owned Files:**
- `data/dataset_ingestion.py` (Canonical 21-joint MANO topology, `CANONICAL_BONE_LENGTHS_MM`, `ingest_or_generate_dataset`)
- `data/transforms.py` (`SyntheticOcclusionTransform`, `SyntheticGloveTransform`)
- `data/dataloader.py` (`normalize_hand_sequence`, `build_node_features_4d`, `get_dataloaders`, `data/splits/split_indices.json`)
- `data/iphone_parser.py` (`ARKitUDPStreamer`, `MediaPipeCameraStreamer`, `build_jetson_csi_pipeline`)
- `data/record_custom_gloved.py` (`data/custom_gloved_dataset.npz`)

**Git Branch Convention:** `feat/bhavya-*`

---

## Current Module State
- Canonical 21-joint MANO ordering, `PARENT_INDICES`, and `CANONICAL_BONE_LENGTHS_MM` are locked and shared with `models/kinematics.py`.
- Deterministic `data/splits/split_indices.json` enforces a strict **80/20 split** (`960` train / `240` test sequences).
- `data/processed/*.npz` is excluded from Git (`.gitignore`); running `python data/dataset_ingestion.py` (or any dataloader call) regenerates it automatically.
- `data/custom_gloved_dataset.npz` (`120` sequences, `(120, 8, 21, 4)`) is currently a synthetic benchmark ready for real gloved recordings.

## Upstream / Downstream Contracts
- **Provides to Everyone:**
  - `scale_mm = 100.0` wrist-centered normalization (`joint 0 == (0, 0, 0)`).
  - 4D node features `(B, T=8, 21, 4)` = `[x_norm, y_norm, z_norm, m]` where `m == 0` zeros out coordinates.
  - `ARKitUDPStreamer` (UDP port `5005`) and `MediaPipeCameraStreamer` returning the exact same `(1, 8, 21, 4)` normalized window format for `deployment/live_demo.py`.
- **Warning:** Do NOT change `JOINT_NAMES`, `PARENT_INDICES`, or `split_indices.json` without coordinating across all four teammates!

## Next Tasks for Bhavya's Agent
1. Test `data/iphone_parser.py` with live hardware:
   - `python data/record_custom_gloved.py --mode mediapipe --camera 0` (wearing real surgical/latex or work gloves and holding a tool/pen/scalpel).
   - Optionally test `--mode udp --port 5005` with iPhone ARKit streaming.
2. Save newly recorded real gloved trajectories to `data/custom_gloved_dataset.npz` (or a new file `data/custom_gloved_real.npz` and wire it cleanly without breaking existing replay compatibility), and note the exact sequence count & gesture types in `HANDOVER.md`.
3. Optionally place FreiHAND `freihand_xyz.json` / HO-3D annotations in `data/raw/` and verify `ingest_or_generate_dataset()` ingests them cleanly.

---

## Session History

### Session: 2026-11-10 18:05 (UTC+4) — Abhishekh / Orchestrator Agent
- **Git Branch & Base Commit:** `main` (initial repo setup)
- **Objective:** Initialize Bhavya's `data/` module and handover instructions.
- **Commands & Tests Executed:** `data/` module verified end-to-end by Data Pipeline Engineer & Training Engineer (`960/240` split, `custom_gloved_dataset.npz` generated).
- **Handover to Next Agent:** On your machine, create `feat/bhavya-real-gloved-capture`, install `requirements.txt` (including `mediapipe`), capture real gloved gestures, and log the updated dataset stats in `HANDOVER.md`.
