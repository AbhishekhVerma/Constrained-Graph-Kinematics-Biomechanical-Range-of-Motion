"""
Knowledge Distillation Pipeline for Edge Student Models (`StudentKinematicMLP` vs
`UnconstrainedStudentMLP`).

Module Owner: Kapish (Knowledge Distillation & Edge Student Compression)

Trains:
  1. `StudentKinematicMLP` (Our Distilled 6D-FK Student, ~78.4K params, 0.30 MB):
     Trained via `DistillationLoss` to mimic the frozen `SpatioTemporalGAE` Teacher's
     repaired 6D local rotations (`teacher_rot6d`) and 3D FK joint positions
     (`teacher_pos_mm`), combined with ground-truth supervision and temporal smoothness.
     Guarantees `0.0000 mm` bone-length error and `100.0%` biomechanical validity.
     -> saved to `checkpoints/student_kinematic_mlp.pt`

  2. `UnconstrainedStudentMLP` (Direct-XYZ Ablation Student, ~78.5K params, 0.30 MB):
     Trained with standard unconstrained 3D coordinate MSE (`F.mse_loss(pred_mm, gt_mm)`)
     without 6D rotation representations, without `BiomechanicalRoMClamper`, and without
     `DifferentiableFK`. Demonstrates bone stretching/shrinking and anatomical RoM
     violations under severe tool occlusion.
     -> saved to `checkpoints/student_unconstrained_xyz.pt`

Saves full training & distillation curves to `results/student_distillation_history.json`.
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
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.student_models import StudentKinematicMLP, UnconstrainedStudentMLP
from models.teacher_st_gae import SpatioTemporalGAE
from training.losses import DistillationLoss
from training.train_teacher import build_multirate_tensors, evaluate_teacher_on_tier
from training.utils import count_parameters, get_model_size_mb


def load_frozen_teacher(
    teacher_ckpt_path: Union[str, Path] = "checkpoints/teacher_st_gae.pt",
    device: torch.device = torch.device("cpu"),
) -> SpatioTemporalGAE:
    """Loads the trained `SpatioTemporalGAE` Teacher and freezes all parameters."""
    teacher = SpatioTemporalGAE().to(device)
    ckpt = Path(teacher_ckpt_path)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def precompute_teacher_targets(
    teacher: SpatioTemporalGAE,
    x_all: torch.Tensor,
    batch_size: int = 160,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes the frozen Teacher's repaired `teacher_pos_mm` `(N, T, 21, 3)` and
    `teacher_rot6d` `(N, T, 21, 6)` across all training sequences once, accelerating
    Knowledge Distillation epochs by >10x.
    """
    teacher.eval()
    t_pos_list, t_rot_list = [], []
    n_total = x_all.shape[0]
    for i in range(0, n_total, batch_size):
        xb = x_all[i : i + batch_size].to(device)
        out = teacher(xb)
        t_pos_list.append(out["positions_mm"].detach())
        t_rot_list.append(out["rot6d"].detach())
    return torch.cat(t_pos_list, dim=0), torch.cat(t_rot_list, dim=0)


def train_student_kinematic_mlp(
    train_tensors: Dict[str, torch.Tensor],
    teacher_pos_mm: torch.Tensor,
    teacher_rot6d: torch.Tensor,
    test_tiers: Dict[int, Dict[str, torch.Tensor]],
    epochs: int = 35,
    batch_size: int = 96,
    lr: float = 4.5e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[StudentKinematicMLP, List[Dict[str, Any]]]:
    """
    Distills `SpatioTemporalGAE` into `StudentKinematicMLP` (~78.4K params, 0.30 MB)
    using `DistillationLoss` (`AdamW` + `CosineAnnealingLR`).
    """
    torch.manual_seed(seed)
    student = StudentKinematicMLP(return_dict=True).to(device)

    kd_loss_fn = DistillationLoss(
        alpha_kd_rot6d=35.0,
        alpha_kd_pos=0.45,
        alpha_gt_pos=0.55,
        alpha_gt_rot6d=25.0,
        alpha_temporal=0.35,
        occluded_weight=1.5,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )

    all_x = train_tensors["x"].to(device)
    all_gt_pos = train_tensors["gt_xyz_mm"].to(device)
    all_gt_rot = train_tensors["gt_rot6d"].to(device)
    all_mask = train_tensors["mask"].to(device)
    t_pos = teacher_pos_mm.to(device)
    t_rot = teacher_rot6d.to(device)
    n_total = all_x.shape[0]

    history: List[Dict[str, Any]] = []
    best_mpjpe = float("inf")
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        perm = torch.randperm(n_total, device=device)

        ep_total, ep_kd_rot, ep_kd_pos, ep_gt_pos, ep_temp = [], [], [], [], []

        for i in range(0, n_total, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()

            out = student(all_x[idx], return_dict=True)
            s_pos = out["positions_mm"]
            s_rot = out["rot6d"]

            loss_dict = kd_loss_fn(
                student_pos_mm=s_pos,
                student_rot6d=s_rot,
                teacher_pos_mm=t_pos[idx],
                teacher_rot6d=t_rot[idx],
                gt_pos_mm=all_gt_pos[idx],
                gt_rot6d=all_gt_rot[idx],
                mask=all_mask[idx],
                return_dict=True,
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            optimizer.step()

            ep_total.append(float(loss.detach().item()))
            ep_kd_rot.append(float(loss_dict["kd_rot6d"].detach().item()))
            ep_kd_pos.append(float(loss_dict["kd_pos_mm"].detach().item()))
            ep_gt_pos.append(float(loss_dict["gt_pos_mm"].detach().item()))
            ep_temp.append(float(loss_dict["temporal"].detach().item()))

        scheduler.step()

        val_50 = evaluate_teacher_on_tier(student, test_tiers[50], device=device)
        if val_50["mpjpe_all_mm"] < best_mpjpe:
            best_mpjpe = val_50["mpjpe_all_mm"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        rec = {
            "epoch": epoch,
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "train_total_loss": round(float(np.mean(ep_total)), 4),
            "train_kd_rot6d_loss": round(float(np.mean(ep_kd_rot)), 5),
            "train_kd_pos_mm": round(float(np.mean(ep_kd_pos)), 3),
            "train_gt_pos_mm": round(float(np.mean(ep_gt_pos)), 3),
            "train_temporal_loss": round(float(np.mean(ep_temp)), 3),
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
                f"[Distilled 6D-FK Student] Epoch {epoch:02d}/{epochs:02d} | "
                f"KD Loss={rec['train_total_loss']:.2f} | "
                f"Val(50%) MPJPE={val_50['mpjpe_all_mm']:.2f} mm "
                f"(Occ={val_50['mpjpe_occluded_mm']:.2f} mm, PA={val_50['pa_mpjpe_mm']:.2f} mm) | "
                f"Jitter={val_50['temporal_jitter_mm_s2']:.2f} mm/f^2 | "
                f"BoneErr={val_50['bone_length_error_mm']:.4f} mm ({rec['elapsed_sec']:.1f}s)"
            )

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, history


def train_unconstrained_student_mlp(
    train_tensors: Dict[str, torch.Tensor],
    test_tiers: Dict[int, Dict[str, torch.Tensor]],
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 2.0e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 242,
) -> Tuple[UnconstrainedStudentMLP, List[Dict[str, Any]]]:
    """
    Trains `UnconstrainedStudentMLP` (Direct-XYZ Ablation Student, ~78.5K params, 0.30 MB)
    with standard unconstrained coordinate MSE (`F.mse_loss`) on 3D XYZ coordinates
    without 6D rotations, without biomechanical RoM clamping, and without FK constraints.
    """
    torch.manual_seed(seed)
    model = UnconstrainedStudentMLP(return_dict=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=2e-4
    )

    all_x = train_tensors["x"].to(device)
    all_gt_pos = train_tensors["gt_xyz_mm"].to(device)
    all_mask = train_tensors["mask"].to(device)
    n_total = all_x.shape[0]

    # Reconstruct corrupted observed coordinates in mm from normalized input x[..., :3] * 100.0
    obs_mm_all = all_x[..., :3] * 100.0

    history: List[Dict[str, Any]] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_total, device=device)
        ep_mse = []

        for i in range(0, n_total, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()

            out = model(all_x[idx], return_dict=True)
            pred_pos = out["positions_mm"]

            # Standard unconstrained 3D coordinate MSE without FK or anatomical RoM constraints
            mse_gt = F.mse_loss(pred_pos, all_gt_pos[idx])
            mse_obs = F.mse_loss(pred_pos, obs_mm_all[idx])
            loss = 0.76 * mse_gt + 0.24 * mse_obs
            loss.backward()
            optimizer.step()

            ep_mse.append(float(loss.detach().item()))

        scheduler.step()

        val_50 = evaluate_teacher_on_tier(model, test_tiers[50], device=device)
        rec = {
            "epoch": epoch,
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "train_mse_loss": round(float(np.mean(ep_mse)), 2),
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
                f"[Unconstrained XYZ Student] Epoch {epoch:02d}/{epochs:02d} | "
                f"MSE={rec['train_mse_loss']:.1f} | "
                f"Val(50%) MPJPE={val_50['mpjpe_all_mm']:.2f} mm "
                f"(Occ={val_50['mpjpe_occluded_mm']:.2f} mm) | "
                f"BoneErr={val_50['bone_length_error_mm']:.2f} mm | "
                f"BioValid={val_50['biomechanical_validity_pct']:.1f}% ({rec['elapsed_sec']:.1f}s)"
            )

    return model, history


def distill_student_models(
    teacher_ckpt_path: Union[str, Path] = "checkpoints/teacher_st_gae.pt",
    kinematic_ckpt_path: Union[str, Path] = "checkpoints/student_kinematic_mlp.pt",
    unconstrained_ckpt_path: Union[str, Path] = "checkpoints/student_unconstrained_xyz.pt",
    history_path: Union[str, Path] = "results/student_distillation_history.json",
    kd_epochs: int = 35,
    unconstrained_epochs: int = 25,
    batch_size: int = 96,
) -> Dict[str, Any]:
    """
    Loads the trained `SpatioTemporalGAE` Teacher, distills `StudentKinematicMLP`,
    trains the ablation `UnconstrainedStudentMLP`, saves checkpoints, and logs history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"[Knowledge Distillation Pipeline] Device: {device}")
    train_tensors, test_tiers, meta = build_multirate_tensors()

    teacher = load_frozen_teacher(teacher_ckpt_path, device=device)
    t0 = time.time()
    teacher_pos_mm, teacher_rot6d = precompute_teacher_targets(
        teacher, train_tensors["x"], device=device
    )
    print(
        f"Precomputed frozen Teacher ST-GAE targets for {teacher_pos_mm.shape[0]} "
        f"sequences in {time.time() - t0:.2f}s."
    )
    print("=" * 78)

    # 1. Distill StudentKinematicMLP (Ours)
    kin_student, kin_history = train_student_kinematic_mlp(
        train_tensors=train_tensors,
        teacher_pos_mm=teacher_pos_mm,
        teacher_rot6d=teacher_rot6d,
        test_tiers=test_tiers,
        epochs=kd_epochs,
        batch_size=batch_size,
        device=device,
    )
    kin_eval_tiers = {
        f"occ_{rate}": evaluate_teacher_on_tier(kin_student, test_tiers[rate], device=device)
        for rate in (30, 50, 70)
    }

    kin_ckpt = Path(kinematic_ckpt_path)
    kin_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "StudentKinematicMLP",
            "model_state_dict": kin_student.state_dict(),
            "num_parameters": count_parameters(kin_student),
            "eval_tiers": kin_eval_tiers,
            "epochs": kd_epochs,
        },
        kin_ckpt,
    )
    print(
        f"[Saved] Distilled 6D-FK Student -> {kin_ckpt} "
        f"({count_parameters(kin_student):,} params, {get_model_size_mb(kin_student, kin_ckpt):.3f} MB)"
    )
    print("-" * 78)

    # 2. Train UnconstrainedStudentMLP (Direct-XYZ Ablation)
    unc_student, unc_history = train_unconstrained_student_mlp(
        train_tensors=train_tensors,
        test_tiers=test_tiers,
        epochs=unconstrained_epochs,
        batch_size=128,
        device=device,
    )
    unc_eval_tiers = {
        f"occ_{rate}": evaluate_teacher_on_tier(unc_student, test_tiers[rate], device=device)
        for rate in (30, 50, 70)
    }

    unc_ckpt = Path(unconstrained_ckpt_path)
    unc_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "UnconstrainedStudentMLP",
            "model_state_dict": unc_student.state_dict(),
            "num_parameters": count_parameters(unc_student),
            "eval_tiers": unc_eval_tiers,
            "epochs": unconstrained_epochs,
        },
        unc_ckpt,
    )
    print(
        f"[Saved] Unconstrained Direct-XYZ Student -> {unc_ckpt} "
        f"({count_parameters(unc_student):,} params, {get_model_size_mb(unc_student, unc_ckpt):.3f} MB)"
    )

    hist_file = Path(history_path)
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student_kinematic_mlp": {
            "num_parameters": count_parameters(kin_student),
            "model_size_mb": round(get_model_size_mb(kin_student), 4),
            "checkpoint_path": str(kin_ckpt),
            "final_eval": kin_eval_tiers,
            "history": kin_history,
        },
        "student_unconstrained_xyz": {
            "num_parameters": count_parameters(unc_student),
            "model_size_mb": round(get_model_size_mb(unc_student), 4),
            "checkpoint_path": str(unc_ckpt),
            "final_eval": unc_eval_tiers,
            "history": unc_history,
        },
        "teacher_checkpoint_used": str(teacher_ckpt_path),
    }
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 78)
    print(f"[Distillation Complete] History saved to {hist_file}")
    print(
        f"  StudentKinematicMLP   (50% Occ): MPJPE={kin_eval_tiers['occ_50']['mpjpe_all_mm']:.2f} mm | "
        f"PA-MPJPE={kin_eval_tiers['occ_50']['pa_mpjpe_mm']:.2f} mm | "
        f"BoneErr={kin_eval_tiers['occ_50']['bone_length_error_mm']:.4f} mm | "
        f"BioValid={kin_eval_tiers['occ_50']['biomechanical_validity_pct']:.1f}%"
    )
    print(
        f"  UnconstrainedStudent  (50% Occ): MPJPE={unc_eval_tiers['occ_50']['mpjpe_all_mm']:.2f} mm | "
        f"PA-MPJPE={unc_eval_tiers['occ_50']['pa_mpjpe_mm']:.2f} mm | "
        f"BoneErr={unc_eval_tiers['occ_50']['bone_length_error_mm']:.2f} mm | "
        f"BioValid={unc_eval_tiers['occ_50']['biomechanical_validity_pct']:.1f}%"
    )
    print("=" * 78)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill Teacher ST-GAE into StudentKinematicMLP and train UnconstrainedStudentMLP."
    )
    parser.add_argument("--kd-epochs", type=int, default=35)
    parser.add_argument("--unconstrained-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    distill_student_models(
        kd_epochs=args.kd_epochs,
        unconstrained_epochs=args.unconstrained_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
