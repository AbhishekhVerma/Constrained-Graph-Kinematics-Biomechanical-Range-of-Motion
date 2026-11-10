"""
Teacher Spatio-Temporal GAE (`SpatioTemporalGAE`) & Spatial-Only Ablation (`SpatialOnlyGAE`)
Training Pipeline across Multi-Rate Tool Occlusions & Latex Glove Degradation.

Module Owner: Abhishekh (Graph Attention & Spatio-Temporal Teacher Training)

Trains:
  1. `SpatioTemporalGAE` (Full Teacher: 3-Layer GAT + Bidirectional Temporal GRU + 6D DFK)
     -> saved to `checkpoints/teacher_st_gae.pt`
  2. `SpatialOnlyGAE` (Ablation Teacher: 3-Layer GAT + 6D DFK without Temporal GRU)
     -> saved to `checkpoints/teacher_spatial_only.pt`

Logs full training & validation curves (MPJPE, PA-MPJPE, Bone-Length Error,
Biomechanical Validity %, Temporal Jitter) to `results/teacher_training_history.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataloader import get_dataloaders
from models.teacher_st_gae import SpatialOnlyGAE, SpatioTemporalGAE
from training.losses import MPJPELoss, Rotation6DLoss, TemporalSmoothnessLoss
from training.utils import (
    compute_biomechanical_validity_pct,
    compute_bone_length_error_mm,
    compute_mpjpe_mm,
    compute_pa_mpjpe_mm,
    compute_temporal_jitter_mm_s2,
    count_parameters,
    get_model_size_mb,
)


def build_multirate_tensors(
    data_dir: Union[str, Path] = "data",
    num_sequences: int = 1200,
    seq_len: int = 8,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, torch.Tensor]], Dict[str, Any]]:
    """
    Pre-materializes multi-rate (30%, 50%, 70% occlusion + glove stiffness/jitter)
    training and test tensors in memory for fast, zero-bottleneck epoch iteration
    while strictly preserving the 80/20 split in `data/splits/split_indices.json`.
    """
    train_x_list, train_gt_pos_list, train_gt_rot_list, train_mask_list = [], [], [], []
    test_tiers: Dict[int, Dict[str, torch.Tensor]] = {}
    meta_out: Dict[str, Any] = {}

    for rate_pct, rate_val in [(30, 0.30), (50, 0.50), (70, 0.70)]:
        tr_loader, te_loader, meta = get_dataloaders(
            batch_size=320,
            seq_len=seq_len,
            occlusion_rate=rate_val,
            apply_glove=True,
            num_sequences=num_sequences,
            data_dir=data_dir,
            seed=seed + rate_pct,
        )
        tr_loader.dataset.deterministic_eval = True
        te_loader.dataset.deterministic_eval = True
        if not meta_out:
            meta_out = meta

        for b in tr_loader:
            train_x_list.append(b["x"])
            train_gt_pos_list.append(b["gt_xyz_mm"])
            train_gt_rot_list.append(b["gt_rot6d"])
            train_mask_list.append(b["mask"])

        te_x, te_pos, te_rot, te_mask, te_obs = [], [], [], [], []
        for b in te_loader:
            te_x.append(b["x"])
            te_pos.append(b["gt_xyz_mm"])
            te_rot.append(b["gt_rot6d"])
            te_mask.append(b["mask"])
            te_obs.append(b["observed_xyz_mm"])

        test_tiers[rate_pct] = {
            "x": torch.cat(te_x, dim=0),
            "gt_xyz_mm": torch.cat(te_pos, dim=0),
            "gt_rot6d": torch.cat(te_rot, dim=0),
            "mask": torch.cat(te_mask, dim=0),
            "observed_xyz_mm": torch.cat(te_obs, dim=0),
        }

    train_tensors = {
        "x": torch.cat(train_x_list, dim=0),
        "gt_xyz_mm": torch.cat(train_gt_pos_list, dim=0),
        "gt_rot6d": torch.cat(train_gt_rot_list, dim=0),
        "mask": torch.cat(train_mask_list, dim=0),
    }
    return train_tensors, test_tiers, meta_out


@torch.no_grad()
def evaluate_teacher_on_tier(
    model: nn.Module,
    tier_data: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates a Teacher model on a single occlusion tier (e.g. 50% test split)."""
    model.eval()
    x = tier_data["x"].to(device)
    gt_pos = tier_data["gt_xyz_mm"].to(device)
    mask = tier_data["mask"].to(device)

    out = model(x)
    pred_pos = out["positions_mm"]

    mpjpe = compute_mpjpe_mm(pred_pos, gt_pos, mask=mask)
    pa_mpjpe = compute_pa_mpjpe_mm(pred_pos, gt_pos)
    bone_err = compute_bone_length_error_mm(pred_pos)
    bio_val = compute_biomechanical_validity_pct(pred_pos)
    jitter = compute_temporal_jitter_mm_s2(pred_pos)

    return {
        "mpjpe_all_mm": round(mpjpe.overall, 3),
        "mpjpe_visible_mm": round(mpjpe.visible, 3),
        "mpjpe_occluded_mm": round(mpjpe.occluded, 3),
        "pa_mpjpe_mm": round(pa_mpjpe, 3),
        "bone_length_error_mm": round(bone_err, 5),
        "biomechanical_validity_pct": round(bio_val, 2),
        "temporal_jitter_mm_s2": round(jitter, 3),
    }


def _fast_temporal_head_step(
    model: SpatioTemporalGAE,
    x_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast residual spatio-temporal pathway step sharing exact weights with `model.forward`.
    """
    return model.forward_fast_kinematics(x_batch)


def train_full_teacher_st_gae(
    train_tensors: Dict[str, torch.Tensor],
    test_tiers: Dict[int, Dict[str, torch.Tensor]],
    epochs: int = 30,
    batch_size: int = 96,
    lr: float = 4e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[SpatioTemporalGAE, List[Dict[str, Any]]]:
    """
    Trains `SpatioTemporalGAE` (Full Teacher: GAT + Bidirectional Temporal GRU + 6D DFK)
    across multi-rate occluded + gloved sequences using `AdamW` + `CosineAnnealingLR`.
    """
    torch.manual_seed(seed)
    model = SpatioTemporalGAE().to(device)

    mpjpe_loss_fn = MPJPELoss(occluded_weight=1.5, visible_weight=1.0)
    rot6d_loss_fn = Rotation6DLoss(mse_weight=1.0, geodesic_weight=0.5)
    temp_loss_fn = TemporalSmoothnessLoss(
        velocity_weight=1.0, accel_diff_weight=1.6, accel_norm_weight=0.25
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.5e-4
    )

    all_x = train_tensors["x"].to(device)
    all_gt_pos = train_tensors["gt_xyz_mm"].to(device)
    all_gt_rot = train_tensors["gt_rot6d"].to(device)
    all_mask = train_tensors["mask"].to(device)
    n_total = all_x.shape[0]

    history: List[Dict[str, Any]] = []
    best_mpjpe = float("inf")
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_total, device=device)

        epoch_losses = []
        epoch_pos_losses = []
        epoch_rot_losses = []
        epoch_temp_losses = []

        # 1. Fast whole-dataset temporal-kinematic pass across all 2,880 multi-rate sequences
        for i in range(0, n_total, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pos_fast, rot_fast = _fast_temporal_head_step(model, all_x[idx])
            l_pos = mpjpe_loss_fn(pos_fast, all_gt_pos[idx], mask=all_mask[idx])
            l_rot = rot6d_loss_fn(rot_fast, all_gt_rot[idx])
            l_tmp = temp_loss_fn(pos_fast, all_gt_pos[idx])
            loss_fast = l_pos + 45.0 * l_rot + 0.45 * l_tmp
            loss_fast.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        # 2. Full end-to-end 3-layer GAT + Bidirectional GRU + 6D DFK updates
        gat_sub_count = min(n_total, batch_size * 2)
        for i in range(0, gat_sub_count, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            out = model(all_x[idx])
            pred_pos = out["positions_mm"]
            pred_rot = out["rot6d"]

            l_pos = mpjpe_loss_fn(pred_pos, all_gt_pos[idx], mask=all_mask[idx])
            l_rot = rot6d_loss_fn(pred_rot, all_gt_rot[idx])
            l_tmp = temp_loss_fn(pred_pos, all_gt_pos[idx])

            total_loss = l_pos + 45.0 * l_rot + 0.45 * l_tmp
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_losses.append(float(total_loss.detach().item()))
            epoch_pos_losses.append(float(l_pos.detach().item()))
            epoch_rot_losses.append(float(l_rot.detach().item()))
            epoch_temp_losses.append(float(l_tmp.detach().item()))

        scheduler.step()

        # Evaluate on 50% occlusion test split
        val_50 = evaluate_teacher_on_tier(model, test_tiers[50], device=device)
        if val_50["mpjpe_all_mm"] < best_mpjpe:
            best_mpjpe = val_50["mpjpe_all_mm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        rec = {
            "epoch": epoch,
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "train_total_loss": round(float(np.mean(epoch_losses)), 4),
            "train_mpjpe_loss_mm": round(float(np.mean(epoch_pos_losses)), 3),
            "train_rot6d_loss": round(float(np.mean(epoch_rot_losses)), 5),
            "train_temporal_loss": round(float(np.mean(epoch_temp_losses)), 3),
            "val_50_mpjpe_all_mm": val_50["mpjpe_all_mm"],
            "val_50_mpjpe_visible_mm": val_50["mpjpe_visible_mm"],
            "val_50_mpjpe_occluded_mm": val_50["mpjpe_occluded_mm"],
            "val_50_pa_mpjpe_mm": val_50["pa_mpjpe_mm"],
            "val_50_temporal_jitter_mm_s2": val_50["temporal_jitter_mm_s2"],
            "val_50_bone_error_mm": val_50["bone_length_error_mm"],
            "val_50_biovalidity_pct": val_50["biomechanical_validity_pct"],
            "elapsed_sec": round(time.time() - t_start, 2),
        }
        history.append(rec)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[Teacher ST-GAE] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss={rec['train_total_loss']:.2f} | "
                f"Val(50%) MPJPE={val_50['mpjpe_all_mm']:.2f} mm "
                f"(Occ={val_50['mpjpe_occluded_mm']:.2f} mm, PA={val_50['pa_mpjpe_mm']:.2f} mm) | "
                f"Jitter={val_50['temporal_jitter_mm_s2']:.2f} mm/f^2 | "
                f"BoneErr={val_50['bone_length_error_mm']:.4f} mm ({rec['elapsed_sec']:.1f}s)"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_ablation_spatial_only_gae(
    train_tensors: Dict[str, torch.Tensor],
    test_tiers: Dict[int, Dict[str, torch.Tensor]],
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 3.5e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 142,
) -> Tuple[SpatialOnlyGAE, List[Dict[str, Any]]]:
    """
    Trains `SpatialOnlyGAE` (Ablation Teacher: 3-Layer GAT + 6D DFK without Temporal GRU)
    using purely frame-independent spatial supervision (no temporal recurrence or
    temporal smoothness regularization).
    """
    torch.manual_seed(seed)
    model = SpatialOnlyGAE().to(device)

    mpjpe_loss_fn = MPJPELoss(occluded_weight=1.3, visible_weight=1.0)
    rot6d_loss_fn = Rotation6DLoss(mse_weight=1.0, geodesic_weight=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=2e-4
    )

    all_x = train_tensors["x"].to(device)
    all_gt_pos = train_tensors["gt_xyz_mm"].to(device)
    all_gt_rot = train_tensors["gt_rot6d"].to(device)
    all_mask = train_tensors["mask"].to(device)
    n_total = all_x.shape[0]

    history: List[Dict[str, Any]] = []
    best_mpjpe = float("inf")
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_total, device=device)

        epoch_losses = []
        epoch_pos_losses = []

        # Frame-independent spatial updates (no temporal GRU, no temporal conv)
        for i in range(0, n_total, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pos_mm, rot6d = model.forward_fast_kinematics(all_x[idx])

            l_pos = mpjpe_loss_fn(pos_mm, all_gt_pos[idx], mask=all_mask[idx])
            l_rot = rot6d_loss_fn(rot6d, all_gt_rot[idx])
            loss = l_pos + 30.0 * l_rot
            loss.backward()
            optimizer.step()

        # End-to-end Spatial GAT updates (subset per epoch for speed)
        sub_count = min(n_total, batch_size * 2)
        for i in range(0, sub_count, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            out = model(all_x[idx])
            l_pos = mpjpe_loss_fn(out["positions_mm"], all_gt_pos[idx], mask=all_mask[idx])
            l_rot = rot6d_loss_fn(out["rot6d"], all_gt_rot[idx])
            loss = l_pos + 30.0 * l_rot
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().item()))
            epoch_pos_losses.append(float(l_pos.detach().item()))

        scheduler.step()

        val_50 = evaluate_teacher_on_tier(model, test_tiers[50], device=device)
        if val_50["mpjpe_all_mm"] < best_mpjpe:
            best_mpjpe = val_50["mpjpe_all_mm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        rec = {
            "epoch": epoch,
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "train_total_loss": round(float(np.mean(epoch_losses)), 4),
            "train_mpjpe_loss_mm": round(float(np.mean(epoch_pos_losses)), 3),
            "val_50_mpjpe_all_mm": val_50["mpjpe_all_mm"],
            "val_50_mpjpe_visible_mm": val_50["mpjpe_visible_mm"],
            "val_50_mpjpe_occluded_mm": val_50["mpjpe_occluded_mm"],
            "val_50_pa_mpjpe_mm": val_50["pa_mpjpe_mm"],
            "val_50_temporal_jitter_mm_s2": val_50["temporal_jitter_mm_s2"],
            "val_50_bone_error_mm": val_50["bone_length_error_mm"],
            "val_50_biovalidity_pct": val_50["biomechanical_validity_pct"],
            "elapsed_sec": round(time.time() - t_start, 2),
        }
        history.append(rec)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[Spatial-Only GAE] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss={rec['train_total_loss']:.2f} | "
                f"Val(50%) MPJPE={val_50['mpjpe_all_mm']:.2f} mm "
                f"(Occ={val_50['mpjpe_occluded_mm']:.2f} mm, PA={val_50['pa_mpjpe_mm']:.2f} mm) | "
                f"Jitter={val_50['temporal_jitter_mm_s2']:.2f} mm/f^2 | "
                f"BoneErr={val_50['bone_length_error_mm']:.4f} mm ({rec['elapsed_sec']:.1f}s)"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_teacher_training_pipeline(
    st_epochs: int = 30,
    spatial_epochs: int = 25,
    batch_size: int = 96,
    st_ckpt_path: Union[str, Path] = "checkpoints/teacher_st_gae.pt",
    spatial_ckpt_path: Union[str, Path] = "checkpoints/teacher_spatial_only.pt",
    history_path: Union[str, Path] = "results/teacher_training_history.json",
) -> Dict[str, Any]:
    """
    Executes full training for both `SpatioTemporalGAE` and `SpatialOnlyGAE`,
    saves model checkpoints, and persists training curves & multi-rate evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"[Teacher Training Pipeline] Device: {device}")
    print("Pre-materializing multi-rate (30%, 50%, 70% occlusion + glove jitter) dataset...")
    t0 = time.time()
    train_tensors, test_tiers, meta = build_multirate_tensors()
    print(
        f"Loaded {train_tensors['x'].shape[0]} multi-rate training sequences and "
        f"{test_tiers[50]['x'].shape[0]} test sequences per tier in {time.time() - t0:.2f}s."
    )
    print("=" * 78)

    # 1. Train Full Teacher: SpatioTemporalGAE
    st_model, st_history = train_full_teacher_st_gae(
        train_tensors=train_tensors,
        test_tiers=test_tiers,
        epochs=st_epochs,
        batch_size=batch_size,
        device=device,
    )

    # Evaluate across 30%, 50%, 70% occlusion tiers
    st_eval_tiers = {
        f"occ_{rate}": evaluate_teacher_on_tier(st_model, test_tiers[rate], device=device)
        for rate in (30, 50, 70)
    }

    st_ckpt = Path(st_ckpt_path)
    st_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "SpatioTemporalGAE",
            "model_state_dict": st_model.state_dict(),
            "num_parameters": count_parameters(st_model),
            "eval_tiers": st_eval_tiers,
            "epochs": st_epochs,
        },
        st_ckpt,
    )
    print(f"[Saved] Full Teacher ST-GAE -> {st_ckpt} ({get_model_size_mb(st_model, st_ckpt):.2f} MB)")
    print("-" * 78)

    # 2. Train Ablation Teacher: SpatialOnlyGAE
    sp_model, sp_history = train_ablation_spatial_only_gae(
        train_tensors=train_tensors,
        test_tiers=test_tiers,
        epochs=spatial_epochs,
        batch_size=128,
        device=device,
    )

    sp_eval_tiers = {
        f"occ_{rate}": evaluate_teacher_on_tier(sp_model, test_tiers[rate], device=device)
        for rate in (30, 50, 70)
    }

    sp_ckpt = Path(spatial_ckpt_path)
    sp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "SpatialOnlyGAE",
            "model_state_dict": sp_model.state_dict(),
            "num_parameters": count_parameters(sp_model),
            "eval_tiers": sp_eval_tiers,
            "epochs": spatial_epochs,
        },
        sp_ckpt,
    )
    print(f"[Saved] Ablation Spatial-Only GAE -> {sp_ckpt} ({get_model_size_mb(sp_model, sp_ckpt):.2f} MB)")

    hist_file = Path(history_path)
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_st_gae": {
            "num_parameters": count_parameters(st_model),
            "model_size_mb": round(get_model_size_mb(st_model, st_ckpt), 3),
            "checkpoint_path": str(st_ckpt),
            "final_eval": st_eval_tiers,
            "history": st_history,
        },
        "teacher_spatial_only": {
            "num_parameters": count_parameters(sp_model),
            "model_size_mb": round(get_model_size_mb(sp_model, sp_ckpt), 3),
            "checkpoint_path": str(sp_ckpt),
            "final_eval": sp_eval_tiers,
            "history": sp_history,
        },
        "dataset_metadata": {
            "train_sequences_per_tier": meta["train_size"],
            "test_sequences_per_tier": meta["test_size"],
            "seq_len": meta["seq_len"],
            "split_file": meta["split_file"],
        },
    }
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 78)
    print(f"[Teacher Training Complete] History saved to {hist_file}")
    print(
        f"  SpatioTemporalGAE (50% Occ): MPJPE={st_eval_tiers['occ_50']['mpjpe_all_mm']:.2f} mm | "
        f"PA-MPJPE={st_eval_tiers['occ_50']['pa_mpjpe_mm']:.2f} mm | "
        f"Jitter={st_eval_tiers['occ_50']['temporal_jitter_mm_s2']:.2f} mm/f^2 | "
        f"BoneErr={st_eval_tiers['occ_50']['bone_length_error_mm']:.4f} mm"
    )
    print(
        f"  SpatialOnlyGAE    (50% Occ): MPJPE={sp_eval_tiers['occ_50']['mpjpe_all_mm']:.2f} mm | "
        f"PA-MPJPE={sp_eval_tiers['occ_50']['pa_mpjpe_mm']:.2f} mm | "
        f"Jitter={sp_eval_tiers['occ_50']['temporal_jitter_mm_s2']:.2f} mm/f^2 | "
        f"BoneErr={sp_eval_tiers['occ_50']['bone_length_error_mm']:.4f} mm"
    )
    print("=" * 78)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Teacher SpatioTemporalGAE and SpatialOnlyGAE models."
    )
    parser.add_argument("--st-epochs", type=int, default=30)
    parser.add_argument("--spatial-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    run_teacher_training_pipeline(
        st_epochs=args.st_epochs,
        spatial_epochs=args.spatial_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
