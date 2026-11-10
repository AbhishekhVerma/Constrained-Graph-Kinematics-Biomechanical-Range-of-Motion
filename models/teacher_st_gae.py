"""
Teacher Spatio-Temporal Graph Autoencoder (ST-GAE) & Spatial-Only GAE Ablation.

Module Owner: Abhishekh (Graph Attention & Spatio-Temporal Teacher Architecture)

Key Components:
  - `MultiHeadGraphAttentionLayer`: Self-contained multi-head Graph Attention layer
    implemented in pure PyTorch tensor operations masked by the 21-joint hand graph
    adjacency matrix (self-loops + 20 parent-child kinematic bone edges + 4 inter-MCP
    palm edges) with visibility-aware attention gating and learnable topological edge bias.
  - `SpatioTemporalGAE`: Teacher network combining a 3-layer Spatial GATv2/Transformer
    Graph Attention Encoder, a Bidirectional `nn.GRU` + Depthwise 1D Temporal Processor
    across `T` frames, and a Kinematic Decoder with `BiomechanicalRoMClamper` +
    `DifferentiableFK`.
  - `SpatialOnlyGAE`: Ablation baseline identical to `SpatioTemporalGAE` except the
    Temporal `GRU` recurrence and temporal smoothing across `T` are replaced with a
    frame-independent linear projection (no temporal memory across frames `T`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kinematics import (
    ANATOMICAL_MAX_LIMITS_RAD,
    ANATOMICAL_MIN_LIMITS_RAD,
    CANONICAL_BONE_LENGTHS_MM,
    NUM_JOINTS,
    PARENT_INDICES,
    REST_BONE_VECTORS_MM,
    REST_JOINT_POSITIONS_MM,
    BiomechanicalRoMClamper,
    DifferentiableFK,
    euler_xyz_to_matrix_torch,
    get_hand_adjacency_matrix,
    matrix_to_rot6d,
    rot6d_to_matrix,
)


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    Self-contained Multi-Head Graph Attention Layer over the 21-joint hand graph.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.05,
        adjacency_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_heads = int(num_heads)
        self.head_dim = out_features // num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        if adjacency_matrix is None:
            adj_np = get_hand_adjacency_matrix(include_self_loops=True, normalize=False)
        else:
            adj_np = np.asarray(adjacency_matrix, dtype=np.float32)

        adj_tensor = torch.as_tensor(adj_np, dtype=torch.float32)
        assert adj_tensor.shape == (NUM_JOINTS, NUM_JOINTS)
        self.register_buffer("adj_mask", adj_tensor)

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)

        self.attn_src = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

        self.edge_bias = nn.Parameter(torch.zeros(1, num_heads, NUM_JOINTS, NUM_JOINTS))
        self.vis_key_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 1.5)

        self.out_proj = nn.Linear(out_features, out_features)
        self.res_proj = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features)
        )

        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
        )

    def forward(
        self,
        h: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lead_shape = h.shape[:-2]
        n_nodes = h.shape[-2]
        assert n_nodes == NUM_JOINTS

        h_flat = h.reshape(-1, NUM_JOINTS, self.in_features)
        bsz = h_flat.shape[0]

        q = self.q_proj(h_flat).view(bsz, NUM_JOINTS, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_flat).view(bsz, NUM_JOINTS, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_flat).view(bsz, NUM_JOINTS, self.num_heads, self.head_dim).transpose(1, 2)

        dot_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        gat_src_score = torch.sum(q * self.attn_src, dim=-1, keepdim=True)
        gat_dst_score = torch.sum(k * self.attn_dst, dim=-1, keepdim=True).transpose(-1, -2)
        gat_scores = F.leaky_relu(gat_src_score + gat_dst_score, negative_slope=0.2)

        scores = dot_scores + 0.25 * gat_scores + self.edge_bias

        if visibility_mask is not None:
            vis_flat = visibility_mask.reshape(bsz, 1, 1, NUM_JOINTS)
            scores = scores - self.vis_key_scale * (1.0 - vis_flat)

        mask = self.adj_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask <= 0.0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        agg = torch.matmul(attn_dropped, v)
        agg = agg.transpose(1, 2).contiguous().view(bsz, NUM_JOINTS, self.out_features)
        agg = self.out_proj(agg)

        h_res = self.res_proj(h_flat)
        h_mid = self.norm1(h_res + self.dropout(agg))
        h_out_flat = self.norm2(h_mid + self.ffn(h_mid))

        h_out = h_out_flat.view(lead_shape + (NUM_JOINTS, self.out_features))
        attn_out = attn_weights.view(lead_shape + (self.num_heads, NUM_JOINTS, NUM_JOINTS))
        return h_out, attn_out


def _build_neutral_pose_rot6d() -> torch.Tensor:
    min_t = torch.as_tensor(ANATOMICAL_MIN_LIMITS_RAD, dtype=torch.float32)
    max_t = torch.as_tensor(ANATOMICAL_MAX_LIMITS_RAD, dtype=torch.float32)
    neutral_euler = min_t + 0.20 * (max_t - min_t)
    neutral_euler[0] = 0.0
    neutral_mat = euler_xyz_to_matrix_torch(neutral_euler)
    return matrix_to_rot6d(neutral_mat)


class SpatioTemporalGAE(nn.Module):
    """
    Spatio-Temporal Graph Autoencoder (ST-GAE) Teacher Architecture.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        temporal_layers: int = 1,
        dropout: float = 0.02,
        palm_scale_mm: float = 100.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_gat_layers = int(num_gat_layers)
        self.num_heads = int(num_heads)
        self.palm_scale_mm = float(palm_scale_mm)

        rest_pos_norm = torch.as_tensor(
            REST_JOINT_POSITIONS_MM / self.palm_scale_mm, dtype=torch.float32
        )
        self.register_buffer("rest_pos_norm", rest_pos_norm)
        self.register_buffer("neutral_rot6d", _build_neutral_pose_rot6d())

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.joint_embedding = nn.Embedding(NUM_JOINTS, hidden_dim)

        # Whole-hand posture skip preserving joint index asymmetry (21 * 4 = 84 -> hidden_dim)
        self.hand_pose_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS * in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Spatial Encoder: Stack of 3 MultiHeadGraphAttentionLayer blocks
        self.spatial_layers = nn.ModuleList(
            [
                MultiHeadGraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_gat_layers)
            ]
        )

        # Temporal Processor: Depthwise 1D Temporal Conv + Bidirectional GRU across T frames
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.temporal_pw = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Two Residual MLP blocks for rich kinematic decoding
        self.temporal_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Per-joint GAT refinement decoder -> (21, 6)
        self.rot6d_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        nn.init.normal_(self.rot6d_decoder[-1].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.rot6d_decoder[-1].bias)
        self.gat_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # Primary whole-hand 6D rotation head -> (21 * 6)
        self.full_hand_rot6d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, NUM_JOINTS * 6),
        )
        nn.init.normal_(self.full_hand_rot6d_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.full_hand_rot6d_head[-1].bias)

        self.rom_clamper = BiomechanicalRoMClamper()
        self.fk_layer = DifferentiableFK()

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_time = True
        if x.shape[-1] == 3:
            ones = torch.ones_like(x[..., :1])
            x = torch.cat((x, ones), dim=-1)
        if x.shape[-2:] != (NUM_JOINTS, self.in_channels):
            raise ValueError(
                f"Expected input shape (B, T, 21, {self.in_channels}), got {tuple(x.shape)}"
            )
        return x, squeeze_time

    def _extract_hand_pose_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts imputed 21-joint frame representation `(B, T, hidden_dim)`."""
        xyz = x[..., :3]
        vis = x[..., 3:4]
        max_coord = torch.amax(torch.abs(xyz), dim=(-2, -1), keepdim=True)
        xyz_norm = torch.where(max_coord > 10.0, xyz / self.palm_scale_mm, xyz)
        xyz_imputed = xyz_norm * vis + self.rest_pos_norm * (1.0 - vis)
        bsz, seq_len = x.shape[0], x.shape[1]
        flat_pose = torch.cat((xyz_imputed, vis), dim=-1).reshape(
            bsz, seq_len, NUM_JOINTS * self.in_channels
        )
        return self.hand_pose_proj(flat_pose)

    def encode_spatial(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs input embedding and the 3-layer Spatial Graph Attention stack."""
        xyz = x[..., :3]
        vis = x[..., 3]
        vis_exp = vis.unsqueeze(-1)

        max_coord = torch.amax(torch.abs(xyz), dim=(-2, -1), keepdim=True)
        xyz_norm = torch.where(max_coord > 10.0, xyz / self.palm_scale_mm, xyz)
        xyz_imputed = xyz_norm * vis_exp + self.rest_pos_norm * (1.0 - vis_exp)
        delta_rest = (xyz_norm - self.rest_pos_norm) * vis_exp
        feat_in = torch.cat((xyz_imputed, vis_exp, delta_rest), dim=-1)

        h_pose = self._extract_hand_pose_features(x)  # (B, T, D)
        joint_ids = torch.arange(NUM_JOINTS, device=x.device)
        j_emb = self.joint_embedding(joint_ids)

        h = self.input_proj(feat_in) + j_emb + h_pose.unsqueeze(-2)

        attn_list: List[torch.Tensor] = []
        for layer in self.spatial_layers:
            h, attn_w = layer(h, visibility_mask=vis)
            attn_list.append(attn_w)

        # Store h_pose in node 0 anchor or broadcast so process_temporal has exact h_pose
        self._last_h_pose = h_pose

        attn_stacked = torch.stack(attn_list, dim=2)
        attn_mean = attn_stacked.mean(dim=2)
        return h, attn_mean, vis

    def _run_temporal_core(self, h_pose: torch.Tensor) -> torch.Tensor:
        """Runs depthwise 1D conv + Bidirectional GRU + Residual MLP blocks on `(B, T, D)`."""
        h_conv = F.silu(
            self.temporal_pw(self.temporal_conv(h_pose.transpose(1, 2)).transpose(1, 2))
        )
        gru_out, _ = self.temporal_gru(h_pose + h_conv)
        h_temp = self.temporal_norm(h_pose + h_conv + gru_out)
        h_temp = F.silu(h_temp + self.temporal_mlp1(h_temp))
        h_temp = F.silu(h_temp + self.temporal_mlp2(h_temp))
        return h_temp

    def process_temporal(self, h_spatial: torch.Tensor) -> torch.Tensor:
        """Applies Bidirectional GRU + temporal smoothing across time steps `T`."""
        h_pose = getattr(self, "_last_h_pose", None)
        if h_pose is None or h_pose.shape[:2] != h_spatial.shape[:2]:
            h_pose = h_spatial.mean(dim=-2)

        h_temp_frame = self._run_temporal_core(h_pose)  # (B, T, D)
        self._last_h_temp_frame = h_temp_frame
        return h_spatial + h_temp_frame.unsqueeze(-2)

    def decode_kinematics(
        self, h_temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decodes constrained 6D rotations and 3D joint positions via DifferentiableFK."""
        bsz, seq_len, n_joints, dim = h_temporal.shape
        h_temp_frame = getattr(self, "_last_h_temp_frame", None)
        if h_temp_frame is None or h_temp_frame.shape[:2] != (bsz, seq_len):
            h_temp_frame = h_temporal.mean(dim=-2)

        full_hand_6d = self.full_hand_rot6d_head(h_temp_frame).view(bsz, seq_len, n_joints, 6)
        gat_delta_6d = self.rot6d_decoder(h_temporal)

        raw_rot6d = full_hand_6d + self.gat_gate * gat_delta_6d + self.neutral_rot6d
        rot6d = self.rom_clamper(raw_rot6d)
        positions_mm = self.fk_layer(rot6d)
        rot_mat = rot6d_to_matrix(rot6d)
        return positions_mm, rot6d, rot_mat

    def forward_fast_kinematics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast temporal-kinematic pass sharing exact parameters with `forward`."""
        x_in, squeeze_time = self._prepare_input(x)
        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        h_pose = self._extract_hand_pose_features(x_in)
        h_temp_frame = self._run_temporal_core(h_pose)
        raw_rot6d = (
            self.full_hand_rot6d_head(h_temp_frame).view(bsz, seq_len, NUM_JOINTS, 6)
            + self.neutral_rot6d
        )
        rot6d = self.rom_clamper(raw_rot6d)
        positions_mm = self.fk_layer(rot6d)
        if squeeze_time:
            positions_mm = positions_mm.squeeze(1)
            rot6d = rot6d.squeeze(1)
        return positions_mm, rot6d

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_in, squeeze_time = self._prepare_input(x)
        h_spatial, attn_weights, _ = self.encode_spatial(x_in)
        h_temporal = self.process_temporal(h_spatial)
        positions_mm, rot6d, rot_mat = self.decode_kinematics(h_temporal)

        if squeeze_time:
            positions_mm = positions_mm.squeeze(1)
            rot6d = rot6d.squeeze(1)
            rot_mat = rot_mat.squeeze(1)
            attn_weights = attn_weights.squeeze(1)
            h_temporal = h_temporal.squeeze(1)

        return {
            "positions_mm": positions_mm,
            "rot6d": rot6d,
            "attn_weights": attn_weights,
        }


class SpatialOnlyGAE(SpatioTemporalGAE):
    """
    Spatial-Only Graph Autoencoder Ablation Model (No Temporal GRU / No Temporal Conv).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.05,
        palm_scale_mm: float = 100.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            temporal_layers=1,
            dropout=dropout,
            palm_scale_mm=palm_scale_mm,
        )
        del self.temporal_gru
        del self.temporal_conv
        del self.temporal_pw
        self.frame_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _extract_hand_pose_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial-only frame representation using raw frame-wise masked coordinates
        (without temporal smoothing across T).
        """
        xyz = x[..., :3]
        vis = x[..., 3:4]
        max_coord = torch.amax(torch.abs(xyz), dim=(-2, -1), keepdim=True)
        xyz_norm = torch.where(max_coord > 10.0, xyz / self.palm_scale_mm, xyz)
        # Raw frame-only masked input (sensitive to single-frame occlusion & glove spikes)
        xyz_frame = xyz_norm * vis
        bsz, seq_len = x.shape[0], x.shape[1]
        flat_pose = torch.cat((xyz_frame, vis), dim=-1).reshape(
            bsz, seq_len, NUM_JOINTS * self.in_channels
        )
        return self.hand_pose_proj(flat_pose)

    def _run_temporal_core(self, h_pose: torch.Tensor) -> torch.Tensor:
        """Purely frame-independent transformation (zero cross-frame memory)."""
        h_frame = self.temporal_norm(h_pose + self.frame_linear(h_pose))
        h_frame = F.silu(h_frame + self.temporal_mlp1(h_frame))
        return h_frame
