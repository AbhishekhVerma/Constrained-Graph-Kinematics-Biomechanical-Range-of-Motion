"""
Loss Functions for Biomechanical Hand Telemetry Training & Knowledge Distillation.

Module Owner: Abhishekh & Kapish (Training & Distillation Pipeline)

Implements:
  - `MPJPELoss`: Mean Per-Joint Position Error in millimeters (overall, visible joints
    where `m == 1`, and occluded joints where `m == 0`).
  - `Rotation6DLoss`: Combined 6D MSE and geodesic/chordal SO(3) rotation matrix distance
    using `rot6d_to_matrix`.
  - `TemporalSmoothnessLoss`: Penalizes first-order velocity and second-order acceleration
    discrepancies (`||Delta^2 P_t - Delta^2 P_hat_t||`) across the temporal window `T`
    to eliminate latex glove specular spikes and occlusion jitter.
  - `DistillationLoss`: Multi-objective Knowledge Distillation loss combining Teacher-Student
    6D rotation mimicry, 3D FK position consistency, ground-truth supervision, and temporal
    smoothness regularization.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kinematics import matrix_to_rot6d, rot6d_to_matrix


class MPJPELoss(nn.Module):
    """
    Mean Per-Joint Position Error (MPJPE) Loss in millimeters.

    Supports separate weighting of visible joints (`m == 1`) and occluded joints
    (`m == 0`) so the network prioritizes reconstructing tool-occluded kinematic
    subtrees while anchoring visible joints.
    """

    def __init__(
        self,
        occluded_weight: float = 1.5,
        visible_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.occluded_weight = float(occluded_weight)
        self.visible_weight = float(visible_weight)
        self.eps = float(eps)
        self.last_breakdown: Dict[str, float] = {}

    def compute_breakdown(
        self,
        pred_mm: torch.Tensor,
        gt_mm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes overall, visible, and occluded MPJPE tensors in millimeters.

        Args:
            pred_mm: Predicted 3D joint positions `(..., 21, 3)` in mm.
            gt_mm: Ground-truth 3D joint positions `(..., 21, 3)` in mm.
            mask: Optional visibility mask `(..., 21)` with `1.0` = visible, `0.0` = occluded.

        Returns:
            Dictionary with keys `"loss"`, `"overall_mm"`, `"visible_mm"`, `"occluded_mm"`.
        """
        diff = pred_mm - gt_mm
        per_joint_err = torch.sqrt(torch.sum(diff * diff, dim=-1) + self.eps)  # (..., 21)
        overall_mm = per_joint_err.mean()

        if mask is None:
            return {
                "loss": overall_mm,
                "overall_mm": overall_mm,
                "visible_mm": overall_mm,
                "occluded_mm": overall_mm,
            }

        vis_mask = mask.to(dtype=per_joint_err.dtype, device=per_joint_err.device)
        if vis_mask.dim() == per_joint_err.dim() + 1 and vis_mask.shape[-1] == 1:
            vis_mask = vis_mask.squeeze(-1)

        occ_mask = 1.0 - vis_mask

        vis_count = torch.clamp(vis_mask.sum(), min=1.0)
        occ_count = torch.clamp(occ_mask.sum(), min=1.0)

        visible_mm = (per_joint_err * vis_mask).sum() / vis_count
        occluded_mm = (per_joint_err * occ_mask).sum() / occ_count

        # Weighted combination across joints
        joint_weights = self.visible_weight * vis_mask + self.occluded_weight * occ_mask
        weighted_loss = (per_joint_err * joint_weights).sum() / torch.clamp(
            joint_weights.sum(), min=1.0
        )

        return {
            "loss": weighted_loss,
            "overall_mm": overall_mm,
            "visible_mm": visible_mm,
            "occluded_mm": occluded_mm,
        }

    def forward(
        self,
        pred_mm: torch.Tensor,
        gt_mm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        breakdown = self.compute_breakdown(pred_mm, gt_mm, mask=mask)
        self.last_breakdown = {
            k: float(v.detach().item()) for k, v in breakdown.items()
        }
        if return_dict:
            return breakdown
        return breakdown["loss"]


class Rotation6DLoss(nn.Module):
    """
    Continuous 6D Rotation & SO(3) Geodesic Distance Loss.

    Combines:
      1. MSE on continuous 6D rotation vectors `rot6d` `(..., 21, 6)`.
      2. Geodesic angular distance on `SO(3)` rotation matrices `R = rot6d_to_matrix(rot6d)`:
         `theta = arccos(clamp((trace(R_pred^T @ R_gt) - 1) / 2, -1 + eps, 1 - eps))`
         plus smooth chordal Frobenius norm `||R_pred - R_gt||_F^2`.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        geodesic_weight: float = 0.5,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.geodesic_weight = float(geodesic_weight)
        self.eps = float(eps)
        self.last_breakdown: Dict[str, float] = {}

    def forward(
        self,
        pred_rot: torch.Tensor,
        target_rot: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred_rot: Predicted 6D rotations `(..., 21, 6)` or matrices `(..., 21, 3, 3)`.
            target_rot: Target 6D rotations `(..., 21, 6)` or matrices `(..., 21, 3, 3)`.
            return_dict: If True, returns dictionary of loss components.
        """
        if pred_rot.shape[-2:] == (3, 3):
            r_pred = pred_rot
            d6_pred = matrix_to_rot6d(pred_rot)
        else:
            d6_pred = pred_rot
            r_pred = rot6d_to_matrix(pred_rot)

        if target_rot.shape[-2:] == (3, 3):
            r_gt = target_rot
            d6_gt = matrix_to_rot6d(target_rot)
        else:
            d6_gt = target_rot
            r_gt = rot6d_to_matrix(target_rot)

        mse_6d = F.mse_loss(d6_pred, d6_gt)

        # Relative rotation matrix R_rel = R_pred^T @ R_gt
        r_rel = torch.matmul(r_pred.transpose(-1, -2), r_gt)
        trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
        cos_theta = (trace - 1.0) * 0.5

        # Smooth chordal term (1 - cos_theta) + safe acos geodesic angle in radians
        chordal = torch.clamp(1.0 - cos_theta, min=0.0).mean()
        cos_clamped = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        geodesic_rad = torch.acos(cos_clamped).mean()

        geo_total = 0.5 * geodesic_rad + 0.5 * chordal
        total_loss = self.mse_weight * mse_6d + self.geodesic_weight * geo_total

        self.last_breakdown = {
            "loss": float(total_loss.detach().item()),
            "mse_6d": float(mse_6d.detach().item()),
            "geodesic_rad": float(geodesic_rad.detach().item()),
            "geodesic_deg": float((geodesic_rad.detach() * (180.0 / 3.14159265)).item()),
        }
        if return_dict:
            return {
                "loss": total_loss,
                "mse_6d": mse_6d,
                "geodesic_rad": geodesic_rad,
                "chordal": chordal,
            }
        return total_loss


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal Velocity & Acceleration Smoothness Loss across the frame window `T`.

    Penalizes:
      1. Velocity discrepancy: `||Delta P_hat_t - Delta P_t||_2` where `Delta P_t = P_{t+1} - P_t`.
      2. Acceleration discrepancy: `||Delta^2 P_hat_t - Delta^2 P_t||_2` where
         `Delta^2 P_t = P_{t+1} - 2 P_t + P_{t-1}`.
      3. Direct high-frequency acceleration energy `||Delta^2 P_hat_t||_2` to suppress
         residual specular latex spikes.
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        accel_diff_weight: float = 1.5,
        accel_norm_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.velocity_weight = float(velocity_weight)
        self.accel_diff_weight = float(accel_diff_weight)
        self.accel_norm_weight = float(accel_norm_weight)
        self.eps = float(eps)
        self.last_breakdown: Dict[str, float] = {}

    def forward(
        self,
        pred_seq: torch.Tensor,
        gt_seq: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred_seq: Predicted sequence `(B, T, 21, C)` (e.g. `C=3` mm positions or `C=6` rot6d).
            gt_seq: Optional ground-truth sequence `(B, T, 21, C)`.
            return_dict: If True, returns dictionary of loss components.
        """
        if pred_seq.dim() < 4 or pred_seq.shape[1] < 2:
            zero = pred_seq.sum() * 0.0
            if return_dict:
                return {"loss": zero, "vel_loss": zero, "accel_loss": zero}
            return zero

        # First-order velocity: P_{t+1} - P_t
        pred_vel = pred_seq[:, 1:] - pred_seq[:, :-1]
        if gt_seq is not None and gt_seq.shape == pred_seq.shape:
            gt_vel = gt_seq[:, 1:] - gt_seq[:, :-1]
            vel_diff = pred_vel - gt_vel
            vel_loss = torch.sqrt(torch.sum(vel_diff * vel_diff, dim=-1) + self.eps).mean()
        else:
            vel_loss = torch.sqrt(torch.sum(pred_vel * pred_vel, dim=-1) + self.eps).mean()

        # Second-order acceleration: P_{t+1} - 2 P_t + P_{t-1}
        if pred_seq.shape[1] >= 3:
            pred_acc = pred_seq[:, 2:] - 2.0 * pred_seq[:, 1:-1] + pred_seq[:, :-2]
            acc_norm = torch.sqrt(torch.sum(pred_acc * pred_acc, dim=-1) + self.eps).mean()
            if gt_seq is not None and gt_seq.shape == pred_seq.shape:
                gt_acc = gt_seq[:, 2:] - 2.0 * gt_seq[:, 1:-1] + gt_seq[:, :-2]
                acc_diff = pred_acc - gt_acc
                acc_diff_loss = torch.sqrt(
                    torch.sum(acc_diff * acc_diff, dim=-1) + self.eps
                ).mean()
            else:
                acc_diff_loss = acc_norm
        else:
            acc_norm = pred_seq.sum() * 0.0
            acc_diff_loss = acc_norm

        total = (
            self.velocity_weight * vel_loss
            + self.accel_diff_weight * acc_diff_loss
            + self.accel_norm_weight * acc_norm
        )

        self.last_breakdown = {
            "loss": float(total.detach().item()),
            "vel_loss": float(vel_loss.detach().item()),
            "accel_diff_loss": float(acc_diff_loss.detach().item()),
            "accel_norm": float(acc_norm.detach().item()),
        }
        if return_dict:
            return {
                "loss": total,
                "vel_loss": vel_loss,
                "accel_diff_loss": acc_diff_loss,
                "accel_norm": acc_norm,
            }
        return total


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for training `StudentKinematicMLP` from `SpatioTemporalGAE`.

    Combines:
      1. Teacher-Student 6D rotation mimicry: `MSE(student_rot6d, teacher_rot6d)` + SO(3) geodesic.
      2. Teacher-Student 3D FK position consistency: `MPJPE(student_pos_mm, teacher_pos_mm)`.
      3. Ground-Truth 3D & 6D supervision: `MPJPE(student_pos_mm, gt_pos_mm)` +
         `Rotation6DLoss(student_rot6d, gt_rot6d)`.
      4. Temporal smoothness regularization: `TemporalSmoothnessLoss(student_pos_mm, gt_pos_mm)`.
    """

    def __init__(
        self,
        alpha_kd_rot6d: float = 35.0,
        alpha_kd_pos: float = 0.45,
        alpha_gt_pos: float = 0.55,
        alpha_gt_rot6d: float = 25.0,
        alpha_temporal: float = 0.35,
        occluded_weight: float = 1.5,
    ) -> None:
        super().__init__()
        self.alpha_kd_rot6d = float(alpha_kd_rot6d)
        self.alpha_kd_pos = float(alpha_kd_pos)
        self.alpha_gt_pos = float(alpha_gt_pos)
        self.alpha_gt_rot6d = float(alpha_gt_rot6d)
        self.alpha_temporal = float(alpha_temporal)

        self.mpjpe_loss = MPJPELoss(occluded_weight=occluded_weight, visible_weight=1.0)
        self.rot6d_loss = Rotation6DLoss(mse_weight=1.0, geodesic_weight=0.5)
        self.temporal_loss = TemporalSmoothnessLoss(
            velocity_weight=1.0, accel_diff_weight=1.5, accel_norm_weight=0.20
        )
        self.last_breakdown: Dict[str, float] = {}

    def forward(
        self,
        student_pos_mm: torch.Tensor,
        student_rot6d: torch.Tensor,
        teacher_pos_mm: torch.Tensor,
        teacher_rot6d: torch.Tensor,
        gt_pos_mm: Optional[torch.Tensor] = None,
        gt_rot6d: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes combined Knowledge Distillation objective.
        """
        t_pos = teacher_pos_mm.detach()
        t_rot = teacher_rot6d.detach()

        # 1. Teacher-Student 6D rotation mimicry
        loss_kd_rot = self.rot6d_loss(student_rot6d, t_rot)

        # 2. Teacher-Student 3D FK position consistency
        loss_kd_pos = self.mpjpe_loss(student_pos_mm, t_pos, mask=mask)

        # 3. Ground-truth 3D & 6D supervision (falls back to teacher targets if GT omitted)
        target_pos = gt_pos_mm if gt_pos_mm is not None else t_pos
        loss_gt_pos = self.mpjpe_loss(student_pos_mm, target_pos, mask=mask)

        if gt_rot6d is not None:
            loss_gt_rot = self.rot6d_loss(student_rot6d, gt_rot6d)
        else:
            loss_gt_rot = loss_kd_rot

        # 4. Temporal velocity + acceleration smoothness
        loss_temp = self.temporal_loss(student_pos_mm, target_pos)

        total_loss = (
            self.alpha_kd_rot6d * loss_kd_rot
            + self.alpha_kd_pos * loss_kd_pos
            + self.alpha_gt_pos * loss_gt_pos
            + self.alpha_gt_rot6d * loss_gt_rot
            + self.alpha_temporal * loss_temp
        )

        self.last_breakdown = {
            "loss": float(total_loss.detach().item()),
            "kd_rot6d": float(loss_kd_rot.detach().item()),
            "kd_pos_mm": float(loss_kd_pos.detach().item()),
            "gt_pos_mm": float(loss_gt_pos.detach().item()),
            "gt_rot6d": float(loss_gt_rot.detach().item()),
            "temporal": float(loss_temp.detach().item()),
        }
        if return_dict:
            return {
                "loss": total_loss,
                "kd_rot6d": loss_kd_rot,
                "kd_pos_mm": loss_kd_pos,
                "gt_pos_mm": loss_gt_pos,
                "gt_rot6d": loss_gt_rot,
                "temporal": loss_temp,
            }
        return total_loss
