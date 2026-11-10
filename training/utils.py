"""
Biomechanical Hand Evaluation Metrics & Profiling Utilities.

Module Owner: Abhishekh & Kapish (Evaluation & Visualization Pipeline)

Implements:
  - `compute_mpjpe_mm(pred_mm, gt_mm, mask=None)`: Overall, visible-joint (`m == 1`),
    and occluded-joint (`m == 0`) Mean Per-Joint Position Error in millimeters.
  - `compute_pa_mpjpe_mm(pred_mm, gt_mm)`: Procrustes-Aligned MPJPE in millimeters
    via batched Singular Value Decomposition (SVD) with reflection correction.
  - `compute_bone_length_error_mm(pred_mm)`: Mean absolute error of all 20 bone
    segments relative to `CANONICAL_BONE_LENGTHS_MM[1:]`.
  - `compute_temporal_jitter_mm_s2(pred_mm)`: Mean frame-to-frame second-order
    acceleration norm `||P_{t+1} - 2 P_t + P_{t-1}||_2` in `mm/frame^2`.
  - `compute_biomechanical_validity_pct(pred_mm)`: Percentage of poses satisfying
    anatomical bending, dorsal hyperextension, and bone integrity limits using
    `check_biomechanical_validity` from `models/kinematics.py`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from models.kinematics import (
    CANONICAL_BONE_LENGTHS_MM,
    NUM_JOINTS,
    PARENT_INDICES,
    check_biomechanical_validity,
)


class MPJPEResult(float):
    """
    A `float` subclass representing overall MPJPE in millimeters that ALSO provides
    dictionary-style and attribute access to `"overall"`, `"visible"`, and `"occluded"`
    MPJPE in millimeters.

    Because `isinstance(res, float)` is True:
      - Arithmetic (`res + 1.0`), formatting (`f"{res:.2f}"`), and `json.dumps` work natively.
      - Dictionary access (`res["overall"]`, `res["visible"]`, `res["occluded"]`) and
        attributes (`res.overall`, `res.visible`, `res.occluded`) also work seamlessly.
    """

    def __new__(
        cls,
        overall_mm: float,
        visible_mm: float,
        occluded_mm: float,
    ) -> "MPJPEResult":
        obj = super().__new__(cls, float(overall_mm))
        obj.overall = float(overall_mm)
        obj.overall_mm = float(overall_mm)
        obj.all = float(overall_mm)
        obj.visible = float(visible_mm)
        obj.visible_mm = float(visible_mm)
        obj.occluded = float(occluded_mm)
        obj.occluded_mm = float(occluded_mm)
        obj._dict = {
            "overall": float(overall_mm),
            "overall_mm": float(overall_mm),
            "all": float(overall_mm),
            "visible": float(visible_mm),
            "visible_mm": float(visible_mm),
            "occluded": float(occluded_mm),
            "occluded_mm": float(occluded_mm),
        }
        return obj

    def __getitem__(self, key: str) -> float:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def to_dict(self) -> Dict[str, float]:
        return dict(self._dict)


def _to_tensor(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.detach().to(dtype=torch.float32)
    return torch.as_tensor(arr, dtype=torch.float32)


def compute_mpjpe_mm(
    pred_mm: Union[torch.Tensor, np.ndarray],
    gt_mm: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> MPJPEResult:
    """
    Computes Mean Per-Joint Position Error (MPJPE) in millimeters across all joints,
    visible joints (`mask == 1`), and occluded joints (`mask == 0`).

    Args:
        pred_mm: Predicted 3D positions `(..., 21, 3)` in millimeters.
        gt_mm: Ground-truth 3D positions `(..., 21, 3)` in millimeters.
        mask: Optional visibility mask `(..., 21)` with `1.0` = visible, `0.0` = occluded.

    Returns:
        `MPJPEResult` (a `float` equal to `overall_mm` with `.overall`, `.visible`,
        `.occluded` attributes and dict-style access).
    """
    pred = _to_tensor(pred_mm)
    gt = _to_tensor(gt_mm).to(pred.device)

    per_joint_err = torch.norm(pred - gt, p=2, dim=-1)  # (..., 21)
    overall_val = float(per_joint_err.mean().item())

    if mask is None:
        return MPJPEResult(overall_val, overall_val, overall_val)

    vis = _to_tensor(mask).to(pred.device)
    if vis.dim() == per_joint_err.dim() + 1 and vis.shape[-1] == 1:
        vis = vis.squeeze(-1)

    occ = 1.0 - vis
    vis_sum = float(vis.sum().item())
    occ_sum = float(occ.sum().item())

    vis_val = (
        float((per_joint_err * vis).sum().item() / vis_sum)
        if vis_sum > 0.0
        else overall_val
    )
    occ_val = (
        float((per_joint_err * occ).sum().item() / occ_sum)
        if occ_sum > 0.0
        else overall_val
    )

    return MPJPEResult(overall_val, vis_val, occ_val)


def compute_pa_mpjpe_mm(
    pred_mm: Union[torch.Tensor, np.ndarray],
    gt_mm: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Computes Procrustes-Aligned MPJPE (PA-MPJPE) in millimeters via batched SVD.

    Aligns each predicted 21-joint hand pose `(..., 21, 3)` to its corresponding
    ground-truth pose via optimal translation, isotropic scale, and proper 3D
    rotation `R in SO(3)` before computing the per-joint Euclidean distance.

    Args:
        pred_mm: Predicted 3D joint positions `(..., 21, 3)` in mm.
        gt_mm: Ground-truth 3D joint positions `(..., 21, 3)` in mm.

    Returns:
        Mean PA-MPJPE in millimeters (`float`).
    """
    pred = _to_tensor(pred_mm).reshape(-1, NUM_JOINTS, 3).cpu().numpy()
    gt = _to_tensor(gt_mm).reshape(-1, NUM_JOINTS, 3).cpu().numpy()

    mu_pred = np.mean(pred, axis=1, keepdims=True)  # (N, 1, 3)
    mu_gt = np.mean(gt, axis=1, keepdims=True)      # (N, 1, 3)

    pred_c = pred - mu_pred  # (N, 21, 3)
    gt_c = gt - mu_gt        # (N, 21, 3)

    var_pred = np.sum(pred_c * pred_c, axis=(1, 2))  # (N,)

    # Cross-covariance matrix H = pred_c^T @ gt_c of shape (N, 3, 3)
    H = np.matmul(np.transpose(pred_c, (0, 2, 1)), gt_c)

    U, S, Vt = np.linalg.svd(H)  # U: (N, 3, 3), S: (N, 3), Vt: (N, 3, 3)

    # Ensure proper rotation (det(R) = +1) without reflection
    R = np.matmul(U, Vt)
    det_r = np.linalg.det(R)  # (N,)

    sign_diag = np.ones((pred.shape[0], 3), dtype=np.float32)
    sign_diag[det_r < 0.0, 2] = -1.0

    U_fixed = U * sign_diag[:, None, :]
    R_proper = np.matmul(U_fixed, Vt)  # (N, 3, 3)

    trace_s = np.sum(S * sign_diag, axis=-1)  # (N,)
    scale = np.where(var_pred > 1e-6, trace_s / np.maximum(var_pred, 1e-6), 1.0)
    scale = scale[:, None, None]

    pred_aligned = scale * np.matmul(pred_c, R_proper) + mu_gt  # (N, 21, 3)
    err = np.linalg.norm(pred_aligned - gt, axis=-1)            # (N, 21)
    return float(np.mean(err))


def compute_bone_length_error_mm(
    pred_mm: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Computes the Mean Absolute Error (in millimeters) of all 20 bone lengths
    relative to `CANONICAL_BONE_LENGTHS_MM[1:]`.

    For models with `DifferentiableFK` (`SpatioTemporalGAE`, `SpatialOnlyGAE`,
    `StudentKinematicMLP`), this error is `0.0000 mm` to floating-point precision.

    Args:
        pred_mm: 3D hand positions `(..., 21, 3)` in millimeters.

    Returns:
        Mean absolute bone-length error in millimeters (`float`).
    """
    pos = _to_tensor(pred_mm)
    target_lengths = torch.as_tensor(
        CANONICAL_BONE_LENGTHS_MM[1:], dtype=torch.float32, device=pos.device
    )

    bone_lengths = []
    for j in range(1, NUM_JOINTS):
        p = PARENT_INDICES[j]
        length_j = torch.norm(pos[..., j, :] - pos[..., p, :], p=2, dim=-1)
        bone_lengths.append(length_j)

    actual_lengths = torch.stack(bone_lengths, dim=-1)  # (..., 20)
    abs_err = torch.abs(actual_lengths - target_lengths)
    return float(abs_err.mean().item())


def compute_temporal_jitter_mm_s2(
    pred_mm: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Computes the mean frame-to-frame second-order acceleration norm
    `||P_{t+1} - 2 P_t + P_{t-1}||_2` in `mm/frame^2` across the temporal window `T`.

    Args:
        pred_mm: Sequence of 3D joint positions `(B, T, 21, 3)` or `(T, 21, 3)` in mm.

    Returns:
        Mean temporal acceleration jitter in `mm/frame^2` (`float`).
    """
    pos = _to_tensor(pred_mm)
    if pos.dim() == 3 and pos.shape[-2:] == (NUM_JOINTS, 3):
        pos = pos.unsqueeze(0)  # (1, T, 21, 3)

    if pos.dim() < 4 or pos.shape[1] < 3:
        return 0.0

    accel = pos[:, 2:] - 2.0 * pos[:, 1:-1] + pos[:, :-2]  # (B, T-2, 21, 3)
    accel_norm = torch.norm(accel, p=2, dim=-1)             # (B, T-2, 21)
    return float(accel_norm.mean().item())


def compute_biomechanical_validity_pct(
    pred_mm: Union[torch.Tensor, np.ndarray],
    max_bending_deg: float = 125.0,
    max_hyperextension_deg: float = 20.0,
    bone_length_tol_mm: float = 12.0,
) -> float:
    """
    Evaluates the percentage `[0.0, 100.0]` of 3D hand poses that satisfy all
    anatomical range-of-motion, dorsal hyperextension, and bone-length constraints
    via `check_biomechanical_validity` from `models.kinematics`.
    """
    res = check_biomechanical_validity(
        pred_mm,
        max_bending_deg=max_bending_deg,
        max_hyperextension_deg=max_hyperextension_deg,
        bone_length_tol_mm=bone_length_tol_mm,
        check_bone_lengths=True,
        return_details=True,
    )
    return float(res["valid_pct"])


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in `model`."""
    return int(sum(p.numel() for p in model.parameters()))


def get_model_size_mb(
    model: nn.Module,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> float:
    """
    Returns model size in megabytes (MB) based on FP32 parameter + buffer bytes
    (or on-disk file size if `checkpoint_path` is provided and exists).
    """
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return float(Path(checkpoint_path).stat().st_size) / (1024.0 * 1024.0)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return float(param_bytes + buffer_bytes) / (1024.0 * 1024.0)


def measure_inference_latency_ms(
    model: nn.Module,
    input_shape: tuple = (1, 8, 21, 4),
    device: Union[str, torch.device] = "cpu",
    num_warmup: int = 15,
    num_runs: int = 80,
) -> Dict[str, float]:
    """
    Measures single-sequence and per-frame inference latency in milliseconds (ms)
    and throughput in frames per second (FPS).
    """
    model.eval()
    model.to(device)
    dummy_x = torch.randn(*input_shape, dtype=torch.float32, device=device)
    seq_len = input_shape[1] if len(input_shape) == 4 else 1

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_x)

        timings_ms = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_x)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    arr_ms = np.array(timings_ms, dtype=np.float64)
    seq_mean_ms = float(np.mean(arr_ms))
    seq_p50_ms = float(np.median(arr_ms))
    frame_ms = seq_p50_ms / float(max(seq_len, 1))
    fps = 1000.0 / max(frame_ms, 1e-6)

    return {
        "sequence_latency_ms": round(seq_p50_ms, 4),
        "sequence_mean_ms": round(seq_mean_ms, 4),
        "per_frame_latency_ms": round(frame_ms, 4),
        "fps": round(fps, 1),
    }
