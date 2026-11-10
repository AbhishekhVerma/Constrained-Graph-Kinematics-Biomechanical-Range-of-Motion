"""
Lightweight Edge Student Models for Knowledge Distillation & Ablation Analysis.

Module Owner: Kapish (Edge Student Architecture & ONNX Exportability)

Key Components:
  - `StudentKinematicMLP`: Ultra-fast edge student (~78K parameters, ~0.31 MB FP32)
    taking `(B, T, 21, 4)` (or `(B, 21, 4)`), applying depthwise-separable temporal
    1D mixing + Residual MLP blocks with SiLU activations, predicting 6D local
    rotations `(B, T, 21, 6)`, clamping via `BiomechanicalRoMClamper`, and passing
    through `DifferentiableFK` to output `"positions_mm"` and `"rot6d"`.
    100% ONNX-exportable (`torch.onnx.export`) with zero custom C++ operators.
  - `UnconstrainedStudentMLP`: Ablation baseline with the exact same MLP backbone
    and parameter budget (~78K parameters), but whose final layer directly regresses
    unconstrained 3D joint coordinates `(B, T, 21, 3)` in millimeters with NO 6D
    rotation representation and NO Forward Kinematics layer (demonstrating stretchy
    bones and anatomical failure under occlusion).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.kinematics import (
    ANATOMICAL_MAX_LIMITS_RAD,
    ANATOMICAL_MIN_LIMITS_RAD,
    NUM_JOINTS,
    REST_JOINT_POSITIONS_MM,
    BiomechanicalRoMClamper,
    DifferentiableFK,
    euler_xyz_to_matrix_torch,
    matrix_to_rot6d,
)


def _build_neutral_pose_rot6d() -> torch.Tensor:
    """Returns canonical neutral hand posture in 6D space `(21, 6)`."""
    min_t = torch.as_tensor(ANATOMICAL_MIN_LIMITS_RAD, dtype=torch.float32)
    max_t = torch.as_tensor(ANATOMICAL_MAX_LIMITS_RAD, dtype=torch.float32)
    neutral_euler = min_t + 0.20 * (max_t - min_t)
    neutral_euler[0] = 0.0
    neutral_mat = euler_xyz_to_matrix_torch(neutral_euler)
    return matrix_to_rot6d(neutral_mat)


class TemporalMixingBlock1D(nn.Module):
    """
    Fast, ONNX-exportable 1D temporal mixing block across the `T` frame window.

    Uses a depthwise 1D convolution (`kernel_size=3, padding=1`) followed by a
    pointwise linear projection, LayerNorm, and SiLU activation with a residual
    skip connection. Operates identically on any sequence length `T >= 1`.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.02) -> None:
        super().__init__()
        self.dw_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.pw_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Tensor of shape `(B, T, hidden_dim)`.

        Returns:
            Tensor of shape `(B, T, hidden_dim)`.
        """
        # Transpose to (B, D, T) for 1D temporal convolution, then back to (B, T, D)
        h_conv = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)
        h_mix = self.dropout(self.act(self.pw_linear(h_conv)))
        return self.norm(h + h_mix)


class ResidualMLPBlock(nn.Module):
    """
    Two-layer Residual MLP block with LayerNorm and SiLU activations.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.02) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(self.fc1(h)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return self.act2(h + out)


class StudentKinematicMLP(nn.Module):
    """
    Lightweight Edge Student Model with Differentiable Forward Kinematics (~78K params, ~0.31 MB).

    Pipeline:
      1. Flattens per-frame 21-joint input `(B, T, 21, 4)` -> `(B, T, 84)` and projects
         to `hidden_dim = 128`.
      2. Applies `TemporalMixingBlock1D` across the `T` time window to suppress latex
         sensor jitter and bridge occluded frames without recurrent state overhead.
      3. Processes features through a `ResidualMLPBlock` with SiLU activations.
      4. Projects to 6D joint rotations `(B, T, 21, 6)`, enforces anatomical joint
         limits via `BiomechanicalRoMClamper`, and traverses `DifferentiableFK` to
         guarantee `0.0000 mm` bone-length error and 100% biomechanically valid poses.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        num_res_blocks: int = 1,
        dropout: float = 0.02,
        palm_scale_mm: float = 100.0,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.palm_scale_mm = float(palm_scale_mm)
        self.return_dict = bool(return_dict)

        rest_pos_norm = torch.as_tensor(
            REST_JOINT_POSITIONS_MM / self.palm_scale_mm, dtype=torch.float32
        )
        self.register_buffer("rest_pos_norm", rest_pos_norm)
        self.register_buffer("neutral_rot6d", _build_neutral_pose_rot6d())

        frame_in_dim = NUM_JOINTS * in_channels  # 21 * 4 = 84
        self.input_proj = nn.Sequential(
            nn.Linear(frame_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.temporal_mixer = TemporalMixingBlock1D(hidden_dim=hidden_dim, dropout=dropout)

        self.res_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_res_blocks)
            ]
        )

        self.rot6d_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, NUM_JOINTS * 6),
        )
        nn.init.normal_(self.rot6d_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rot6d_head[-1].bias)

        self.rom_clamper = BiomechanicalRoMClamper()
        self.fk_layer = DifferentiableFK()

    def _encode_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Runs shared input normalization, temporal mixing, and Residual MLP blocks."""
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_time = True
        if x.shape[-1] == 3:
            ones = torch.ones_like(x[..., :1])
            x = torch.cat((x, ones), dim=-1)

        xyz = x[..., :3]
        vis = x[..., 3:4]

        # Pure tensor scale check (ONNX-safe, no .item() calls)
        max_coord = torch.amax(torch.abs(xyz), dim=(-2, -1), keepdim=True)
        xyz_norm = torch.where(max_coord > 10.0, xyz / self.palm_scale_mm, xyz)

        # Replace occluded joint coordinates (m == 0) with canonical rest pose priors
        xyz_imputed = xyz_norm * vis + self.rest_pos_norm * (1.0 - vis)
        x_clean = torch.cat((xyz_imputed, vis), dim=-1)  # (B, T, 21, 4)

        bsz, seq_len = x_clean.shape[0], x_clean.shape[1]
        flat_in = x_clean.reshape(bsz, seq_len, NUM_JOINTS * self.in_channels)

        h = self.input_proj(flat_in)        # (B, T, hidden_dim)
        h = self.temporal_mixer(h)          # (B, T, hidden_dim)
        for block in self.res_blocks:
            h = block(h)                    # (B, T, hidden_dim)

        return h, squeeze_time

    def forward(
        self,
        x: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of `StudentKinematicMLP`.

        Args:
            x: Input tensor of shape `(B, T, 21, 4)` or `(B, 21, 4)`.
            return_dict: Optional override; if False, returns `(positions_mm, rot6d)` tuple.

        Returns:
            Dictionary `{"positions_mm": positions_mm, "rot6d": rot6d}` (or tuple if `return_dict=False`).
        """
        h, squeeze_time = self._encode_backbone(x)
        bsz, seq_len = h.shape[0], h.shape[1]

        raw_rot6d = self.rot6d_head(h).view(bsz, seq_len, NUM_JOINTS, 6) + self.neutral_rot6d
        rot6d = self.rom_clamper(raw_rot6d)
        positions_mm = self.fk_layer(rot6d)

        if squeeze_time:
            positions_mm = positions_mm.squeeze(1)
            rot6d = rot6d.squeeze(1)

        use_dict = self.return_dict if return_dict is None else return_dict
        if not use_dict:
            return positions_mm, rot6d

        return {
            "positions_mm": positions_mm,
            "rot6d": rot6d,
        }


class UnconstrainedStudentMLP(nn.Module):
    """
    Unconstrained Direct-XYZ Student MLP Ablation Baseline (~78K parameters).

    Shares the exact same input projection, `TemporalMixingBlock1D`, and `ResidualMLPBlock`
    backbone as `StudentKinematicMLP` with a matched parameter budget, but its output head
    directly regresses unconstrained 3D joint coordinates `(B, T, 21, 3)` in millimeters
    with NO 6D rotation representation, NO `BiomechanicalRoMClamper`, and NO `DifferentiableFK`.

    Used in the ablation study to demonstrate how unconstrained coordinate regression
    suffers from bone length stretching/shrinking and anatomically impossible joint
    angles under tool occlusion.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        bottleneck_dim: int = 86,
        num_res_blocks: int = 1,
        dropout: float = 0.02,
        palm_scale_mm: float = 100.0,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.palm_scale_mm = float(palm_scale_mm)
        self.return_dict = bool(return_dict)

        rest_pos_mm = torch.as_tensor(REST_JOINT_POSITIONS_MM, dtype=torch.float32)
        self.register_buffer("rest_pos_mm", rest_pos_mm)
        self.register_buffer("rest_pos_norm", rest_pos_mm / self.palm_scale_mm)

        frame_in_dim = NUM_JOINTS * in_channels  # 84
        self.input_proj = nn.Sequential(
            nn.Linear(frame_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.temporal_mixer = TemporalMixingBlock1D(hidden_dim=hidden_dim, dropout=dropout)

        self.res_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_res_blocks)
            ]
        )

        # Bottleneck dimension (86) chosen so total parameter count matches StudentKinematicMLP (~78.4K)
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, NUM_JOINTS * 3),
        )
        nn.init.normal_(self.xyz_head[-1].weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.xyz_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of `UnconstrainedStudentMLP`.

        Args:
            x: Input tensor of shape `(B, T, 21, 4)` or `(B, 21, 4)`.
            return_dict: Optional override; if False, returns `positions_mm` Tensor directly.

        Returns:
            Dictionary `{"positions_mm": positions_mm}` (or Tensor if `return_dict=False`).
        """
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_time = True
        if x.shape[-1] == 3:
            ones = torch.ones_like(x[..., :1])
            x = torch.cat((x, ones), dim=-1)

        xyz = x[..., :3]
        vis = x[..., 3:4]

        max_coord = torch.amax(torch.abs(xyz), dim=(-2, -1), keepdim=True)
        xyz_norm = torch.where(max_coord > 10.0, xyz / self.palm_scale_mm, xyz)

        # Note: Direct-XYZ baseline receives the raw masked coordinates without FK constraints
        x_in = torch.cat((xyz_norm * vis, vis), dim=-1)
        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        flat_in = x_in.reshape(bsz, seq_len, NUM_JOINTS * self.in_channels)

        h = self.input_proj(flat_in)
        h = self.temporal_mixer(h)
        for block in self.res_blocks:
            h = block(h)

        # Predict direct 3D coordinate offsets in mm (scaled by palm_scale_mm around rest pose)
        delta_mm = self.xyz_head(h).view(bsz, seq_len, NUM_JOINTS, 3) * self.palm_scale_mm
        positions_mm = self.rest_pos_mm + delta_mm
        # Anchor wrist (joint 0) at origin (0, 0, 0)
        positions_mm = positions_mm - positions_mm[..., 0:1, :]

        if squeeze_time:
            positions_mm = positions_mm.squeeze(1)

        use_dict = self.return_dict if return_dict is None else return_dict
        if not use_dict:
            return positions_mm

        return {
            "positions_mm": positions_mm,
        }
