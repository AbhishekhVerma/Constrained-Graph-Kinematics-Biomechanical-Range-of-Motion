"""
Differentiable Forward Kinematics, Continuous 6D Rotations, Biomechanical Range-of-Motion
Clamping, and Bare-Metal NumPy Edge FK for the 21-Joint MANO Hand Skeleton.

Module Owner: Vrushank (Kinematics & Biomechanical Constraints)

Key Components:
  - `rot6d_to_matrix`: Gram-Schmidt continuous 6D-to-SO(3) conversion (Zhou et al., CVPR 2019)
    using pure PyTorch tensor ops (`F.normalize`, `torch.cross`, `torch.stack`), 100% ONNX-exportable.
  - `matrix_to_rot6d`: Extracts first two columns `(..., 6)` of a 3x3 rotation matrix.
  - `BiomechanicalRoMClamper`: Smooth `tanh` soft-clamping of local joint rotations to
    anatomical flexion/abduction/twist limits so joints never hyperextend backward.
  - `check_biomechanical_validity`: Evaluates inter-segment bending angles, hyperextension
    violations, and bone-length integrity on any 3D hand skeleton `(..., 21, 3)`.
  - `DifferentiableFK`: Exact hierarchical Forward Kinematics along `PARENT_INDICES` with
    0.0000 mm bone-length variance and closed-form inverse `positions_to_rot6d` / `analytical_ik_6d`.
  - `EdgeFK`: Pure NumPy implementation of `rot6d_to_matrix` + kinematic tree traversal for
    CPU / edge deployment verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from data.dataset_ingestion import (
        ANATOMICAL_MAX_LIMITS_RAD,
        ANATOMICAL_MIN_LIMITS_RAD,
        CANONICAL_BONE_LENGTHS_MM,
        HAND_EDGES,
        JOINT_NAMES,
        KINEMATIC_CHAINS,
        KINEMATIC_TREE_EDGES,
        NUM_BONES,
        NUM_JOINTS,
        PALM_CONNECTIONS,
        PARENT_INDICES,
        REST_BONE_DIRECTIONS,
        REST_BONE_VECTORS_MM,
        REST_JOINT_POSITIONS_MM,
        get_hand_adjacency_matrix,
    )
except ImportError:
    NUM_JOINTS = 21
    NUM_BONES = 20
    JOINT_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    PARENT_INDICES = [
        -1,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
    ]
    KINEMATIC_CHAINS = {
        "thumb": [0, 1, 2, 3, 4],
        "index": [0, 5, 6, 7, 8],
        "middle": [0, 9, 10, 11, 12],
        "ring": [0, 13, 14, 15, 16],
        "pinky": [0, 17, 18, 19, 20],
    }
    CANONICAL_BONE_LENGTHS_MM = np.array(
        [
            0.0,
            34.0, 46.0, 30.0, 20.0,
            66.0, 40.0, 24.0, 17.0,
            68.0, 42.0, 27.0, 18.0,
            62.0, 39.0, 25.0, 17.0,
            54.0, 31.0, 19.0, 16.0,
        ],
        dtype=np.float32,
    )
    _RAW_REST_DIRECTIONS = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.78, 0.58, 0.23],
            [-0.58, 0.78, 0.23],
            [-0.42, 0.89, 0.17],
            [-0.35, 0.92, 0.17],
            [-0.27, 0.96, 0.02],
            [-0.06, 0.998, 0.0],
            [-0.04, 0.999, 0.0],
            [-0.03, 0.999, 0.0],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.24, 0.97, 0.04],
            [0.05, 0.998, 0.02],
            [0.03, 0.999, 0.01],
            [0.02, 0.999, 0.01],
            [0.46, 0.88, 0.09],
            [0.10, 0.994, 0.04],
            [0.07, 0.997, 0.03],
            [0.05, 0.998, 0.02],
        ],
        dtype=np.float32,
    )
    REST_BONE_DIRECTIONS = np.zeros_like(_RAW_REST_DIRECTIONS, dtype=np.float32)
    for _j in range(1, NUM_JOINTS):
        _norm = float(np.linalg.norm(_RAW_REST_DIRECTIONS[_j]))
        REST_BONE_DIRECTIONS[_j] = _RAW_REST_DIRECTIONS[_j] / max(_norm, 1e-8)
    REST_BONE_VECTORS_MM = (
        CANONICAL_BONE_LENGTHS_MM[:, None] * REST_BONE_DIRECTIONS
    ).astype(np.float32)
    REST_JOINT_POSITIONS_MM = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for _j in range(1, NUM_JOINTS):
        REST_JOINT_POSITIONS_MM[_j] = (
            REST_JOINT_POSITIONS_MM[PARENT_INDICES[_j]] + REST_BONE_VECTORS_MM[_j]
        )
    KINEMATIC_TREE_EDGES = [(PARENT_INDICES[j], j) for j in range(1, NUM_JOINTS)]
    PALM_CONNECTIONS = [(1, 5), (5, 9), (9, 13), (13, 17)]
    HAND_EDGES = []
    for _u, _v in KINEMATIC_TREE_EDGES + PALM_CONNECTIONS:
        HAND_EDGES.append([_u, _v])
        HAND_EDGES.append([_v, _u])

    def get_hand_adjacency_matrix(
        include_self_loops: bool = True, normalize: bool = False
    ) -> np.ndarray:
        adj = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
        for u, v in HAND_EDGES:
            adj[u, v] = 1.0
        if include_self_loops:
            np.fill_diagonal(adj, 1.0)
        if normalize:
            deg = adj.sum(axis=-1, keepdims=True)
            deg_inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
            adj = deg_inv_sqrt * adj * np.transpose(deg_inv_sqrt)
        return adj

    _deg = np.pi / 180.0
    ANATOMICAL_MIN_LIMITS_RAD = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    ANATOMICAL_MAX_LIMITS_RAD = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    ANATOMICAL_MIN_LIMITS_RAD[0] = np.array([-40.0, -45.0, -25.0]) * _deg
    ANATOMICAL_MAX_LIMITS_RAD[0] = np.array([+40.0, +45.0, +25.0]) * _deg
    ANATOMICAL_MIN_LIMITS_RAD[1] = np.array([-15.0, -15.0, -15.0]) * _deg
    ANATOMICAL_MAX_LIMITS_RAD[1] = np.array([+25.0, +20.0, +25.0]) * _deg
    ANATOMICAL_MIN_LIMITS_RAD[2] = np.array([-15.0, -10.0, -45.0]) * _deg
    ANATOMICAL_MAX_LIMITS_RAD[2] = np.array([+55.0, +35.0, +20.0]) * _deg
    ANATOMICAL_MIN_LIMITS_RAD[3] = np.array([0.0, -5.0, -12.0]) * _deg
    ANATOMICAL_MAX_LIMITS_RAD[3] = np.array([+60.0, +10.0, +12.0]) * _deg
    ANATOMICAL_MIN_LIMITS_RAD[4] = np.array([-10.0, -2.0, -2.0]) * _deg
    ANATOMICAL_MAX_LIMITS_RAD[4] = np.array([+80.0, +2.0, +2.0]) * _deg
    for _j in (5, 9):
        ANATOMICAL_MIN_LIMITS_RAD[_j] = np.array([-2.0, -2.0, -2.0]) * _deg
        ANATOMICAL_MAX_LIMITS_RAD[_j] = np.array([+2.0, +2.0, +2.0]) * _deg
    for _j in (13, 17):
        ANATOMICAL_MIN_LIMITS_RAD[_j] = np.array([0.0, -3.0, -4.0]) * _deg
        ANATOMICAL_MAX_LIMITS_RAD[_j] = np.array([+10.0, +3.0, +4.0]) * _deg
    for _j in (6, 10, 14, 18):
        ANATOMICAL_MIN_LIMITS_RAD[_j] = np.array([-15.0, -5.0, -20.0]) * _deg
        ANATOMICAL_MAX_LIMITS_RAD[_j] = np.array([+90.0, +5.0, +20.0]) * _deg
    for _j in (7, 11, 15, 19):
        ANATOMICAL_MIN_LIMITS_RAD[_j] = np.array([0.0, -2.0, -2.0]) * _deg
        ANATOMICAL_MAX_LIMITS_RAD[_j] = np.array([+110.0, +2.0, +2.0]) * _deg
    for _j in (8, 12, 16, 20):
        ANATOMICAL_MIN_LIMITS_RAD[_j] = np.array([0.0, -2.0, -2.0]) * _deg
        ANATOMICAL_MAX_LIMITS_RAD[_j] = np.array([+80.0, +2.0, +2.0]) * _deg


# ---------------------------------------------------------------------------
# 1. Continuous 6D Rotation Representation (Zhou et al., CVPR 2019)
# ---------------------------------------------------------------------------


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts continuous 6D rotation representation `(..., 6)` to orthogonal 3x3
    rotation matrices `(..., 3, 3)` in SO(3) via Gram-Schmidt orthogonalization
    (Zhou et al., CVPR 2019).

    Uses pure PyTorch tensor ops (`F.normalize`, `torch.cross`, `torch.stack`)
    so it is 100% ONNX-exportable without custom C++ extensions.

    Args:
        d6: Tensor of shape `(..., 6)` containing the two unnormalized 3D column vectors.

    Returns:
        matrix: Orthogonal rotation matrices of shape `(..., 3, 3)` with `det(R) = +1`.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    # Safe fallback to canonical X-axis if a1 is near-zero so output is always in SO(3)
    e1 = torch.zeros_like(a1)
    e1[..., 0] = 1.0
    a1_norm = torch.norm(a1, p=2, dim=-1, keepdim=True)
    a1_safe = torch.where(a1_norm < 1e-6, e1, a1)
    b1 = F.normalize(a1_safe, p=2, dim=-1, eps=1e-8)

    # Gram-Schmidt projection: remove component of a2 along b1
    proj = torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2_unnorm = a2 - proj

    # Safe fallback if a2 is collinear with b1 or zero
    e2 = torch.zeros_like(a2)
    e2[..., 1] = 1.0
    e3 = torch.zeros_like(a2)
    e3[..., 2] = 1.0
    alt_axis = torch.where(torch.abs(b1[..., 1:2]) < 0.9, e2, e3)
    alt_orth = alt_axis - torch.sum(b1 * alt_axis, dim=-1, keepdim=True) * b1
    b2_norm = torch.norm(b2_unnorm, p=2, dim=-1, keepdim=True)
    b2_safe = torch.where(b2_norm < 1e-6, alt_orth, b2_unnorm)
    b2 = F.normalize(b2_safe, p=2, dim=-1, eps=1e-8)

    # Right-handed third column b3 = b1 x b2
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts the continuous 6D rotation representation `(..., 6)` from the first
    two columns of a 3x3 rotation matrix `(..., 3, 3)`.

    Args:
        matrix: Rotation matrix tensor of shape `(..., 3, 3)`.

    Returns:
        d6: Continuous 6D rotation tensor of shape `(..., 6)`.
    """
    col0 = matrix[..., :, 0]
    col1 = matrix[..., :, 1]
    return torch.cat((col0, col1), dim=-1)


# ---------------------------------------------------------------------------
# 2. Differentiable Euler <-> SO(3) Conversion & Biomechanical RoM Clamper
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix_torch(euler_rad: torch.Tensor) -> torch.Tensor:
    """
    Converts Euler angles `(..., 3)` `[rx_flexion, ry_twist, rz_abduction]` in radians
    into 3x3 rotation matrices `(..., 3, 3)` via `R = Rz(rz) @ Ry(ry) @ Rx(rx)`.
    Matches `data.dataset_ingestion.euler_xyz_to_matrix` exactly and is 100% ONNX-exportable.
    """
    rx = euler_rad[..., 0]
    ry = euler_rad[..., 1]
    rz = euler_rad[..., 2]

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx

    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx

    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    row0 = torch.stack((r00, r01, r02), dim=-1)
    row1 = torch.stack((r10, r11, r12), dim=-1)
    row2 = torch.stack((r20, r21, r22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _safe_angle_from_sin_cos(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes angle in `[-pi, pi]` from `(sin, cos)` using ONNX-standard ops
    (`sqrt`, `acos`, `clamp`, `where`) with a smooth C1 polynomial transition near 0
    so gradients at `angle = 0` are finite and exact (`d(theta)/d(s) = 1.0`).
    """
    norm = torch.sqrt(s * s + c * c + 1e-8)
    sn = s / norm
    cn = torch.clamp(c / norm, -0.99999, 0.99999)
    sign_s = torch.where(sn >= 0.0, torch.ones_like(sn), -torch.ones_like(sn))
    acos_angle = sign_s * torch.acos(cn)
    # Smooth small-angle expansion when cos(theta) > 0.9995 (|theta| < 1.8 deg)
    small_angle = sn * (1.0 + (1.0 - cn) * 0.3333333)
    return torch.where(cn > 0.9995, small_angle, acos_angle)


def matrix_to_euler_xyz_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts Euler angles `(..., 3)` `[rx_flexion, ry_twist, rz_abduction]` in radians
    from rotation matrix `R = Rz(rz) @ Ry(ry) @ Rx(rx)` of shape `(..., 3, 3)`.
    Uses pure ONNX-traceable operations with smooth gradients everywhere.
    """
    sy = torch.clamp(-matrix[..., 2, 0], -0.99999, 0.99999)
    ry = torch.asin(sy)
    cy = torch.clamp(torch.cos(ry), min=1e-6)

    sx = matrix[..., 2, 1] / cy
    cx = matrix[..., 2, 2] / cy
    rx = _safe_angle_from_sin_cos(sx, cx)

    sz = matrix[..., 1, 0] / cy
    cz = matrix[..., 0, 0] / cy
    rz = _safe_angle_from_sin_cos(sz, cz)

    return torch.stack((rx, ry, rz), dim=-1)


class BiomechanicalRoMClamper(nn.Module):
    """
    Enforces anatomical Range-of-Motion (RoM) constraints on predicted 6D joint
    rotations `(..., 21, 6)` or local rotation matrices `(..., 21, 3, 3)` prior to
    Forward Kinematics traversal.

    Constrains relative flexion (`rx`), axial twist (`ry`), and abduction/adduction (`rz`)
    for all 21 joints using smooth `torch.tanh` soft-clamping around each joint's
    anatomical RoM interval `[ANATOMICAL_MIN_LIMITS_RAD, ANATOMICAL_MAX_LIMITS_RAD]`.
    This guarantees that finger joints (MCP, PIP, DIP, IP) never hyperextend backward
    or splay into unnatural configurations (100% biomechanically valid poses).
    """

    def __init__(
        self,
        min_limits_rad: Optional[Union[np.ndarray, torch.Tensor]] = None,
        max_limits_rad: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min_limits_rad is None:
            min_limits_rad = ANATOMICAL_MIN_LIMITS_RAD
        if max_limits_rad is None:
            max_limits_rad = ANATOMICAL_MAX_LIMITS_RAD

        min_t = torch.as_tensor(min_limits_rad, dtype=torch.float32)
        max_t = torch.as_tensor(max_limits_rad, dtype=torch.float32)
        assert min_t.shape == (NUM_JOINTS, 3) and max_t.shape == (NUM_JOINTS, 3)

        center_t = 0.5 * (min_t + max_t)
        half_range_t = torch.clamp(0.5 * (max_t - min_t), min=1e-4)

        self.register_buffer("min_limits_rad", min_t)
        self.register_buffer("max_limits_rad", max_t)
        self.register_buffer("center_rad", center_t)
        self.register_buffer("half_range_rad", half_range_t)
        self.temperature: float = float(max(temperature, 1e-3))

    def clamp_euler(self, euler_rad: torch.Tensor) -> torch.Tensor:
        """
        Smoothly clamps Euler angles `(..., 21, 3)` into `[min_limits_rad, max_limits_rad]`
        via `center + half_range * tanh((euler - center) / (half_range * temperature))`.
        """
        scaled = (euler_rad - self.center_rad) / (
            self.half_range_rad * self.temperature + 1e-6
        )
        return self.center_rad + self.half_range_rad * torch.tanh(scaled)

    def clamp_matrix(self, rot_mat: torch.Tensor) -> torch.Tensor:
        """
        Clamps local rotation matrices `(..., 21, 3, 3)` to anatomical RoM limits.
        """
        euler_raw = matrix_to_euler_xyz_torch(rot_mat)
        euler_clamped = self.clamp_euler(euler_raw)
        return euler_xyz_to_matrix_torch(euler_clamped)

    def forward(self, rot_input: torch.Tensor) -> torch.Tensor:
        """
        Applies smooth biomechanical RoM clamping to predicted 6D rotations `(..., 21, 6)`
        or rotation matrices `(..., 21, 3, 3)`.

        Args:
            rot_input: Tensor of shape `(..., 21, 6)` or `(..., 21, 3, 3)`.

        Returns:
            Clamped rotation tensor of the exact same shape as `rot_input`.
        """
        if rot_input.shape[-2:] == (3, 3):
            return self.clamp_matrix(rot_input)

        if rot_input.shape[-1] != 6:
            raise ValueError(
                f"Expected rot_input last dim 6 or (3, 3), got shape {tuple(rot_input.shape)}"
            )

        rot_mat = rot6d_to_matrix(rot_input)
        clamped_mat = self.clamp_matrix(rot_mat)
        return matrix_to_rot6d(clamped_mat)


# ---------------------------------------------------------------------------
# 3. Biomechanical Pose Validity Checker (for any 3D Hand Skeleton)
# ---------------------------------------------------------------------------

# 15 consecutive parent-bone -> child-bone articulation triplets (grandparent, parent, child)
ARTICULATION_TRIPLETS: List[Tuple[int, int, int]] = [
    # Thumb chain
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index chain
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle chain
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring chain
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky chain
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

# Distal hinge triplets (PIP and DIP joints of the 4 fingers + Thumb IP)
HINGE_TRIPLETS: List[Tuple[int, int, int]] = [
    (2, 3, 4),    # Thumb IP
    (5, 6, 7), (6, 7, 8),       # Index PIP, DIP
    (9, 10, 11), (10, 11, 12),  # Middle PIP, DIP
    (13, 14, 15), (14, 15, 16), # Ring PIP, DIP
    (17, 18, 19), (18, 19, 20), # Pinky PIP, DIP
]


class BiomechanicalValidityResult(torch.Tensor):
    """
    A boolean `torch.Tensor` representing per-pose biomechanical validity `(...,)`
    that ALSO exposes dictionary-style keys and attributes for detailed metrics:
      - `res.all()`, `res.float().mean()` -> works natively as a boolean Tensor!
      - `res["valid_pct"]` / `res.valid_pct` -> float percentage `[0.0, 100.0]`
      - `res["valid_ratio"]` / `res.valid_ratio` -> float ratio `[0.0, 1.0]`
      - `res["is_valid"]` / `res.is_valid` -> bool whether all poses are valid
      - `res["valid_mask"]` / `res.valid_mask` -> boolean Tensor of shape `(...,)`
      - `res["bending_angles_deg"]` -> Tensor of shape `(..., 15)`
      - `res["max_bone_error_mm"]` -> float max bone length deviation in mm
    """

    @staticmethod
    def __new__(
        cls,
        valid_mask: torch.Tensor,
        details: Dict[str, Any],
    ) -> "BiomechanicalValidityResult":
        obj = torch.Tensor._make_subclass(cls, valid_mask, False)
        obj._details = details
        return obj

    @property
    def valid_pct(self) -> float:
        return float(self._details["valid_pct"])

    @property
    def valid_ratio(self) -> float:
        return float(self._details["valid_ratio"])

    @property
    def is_valid(self) -> bool:
        return bool(self._details["is_valid"])

    @property
    def valid_mask(self) -> torch.Tensor:
        return self._details["valid_mask"]

    @property
    def bending_angles_deg(self) -> torch.Tensor:
        return self._details["bending_angles_deg"]

    @property
    def max_bone_error_mm(self) -> float:
        return float(self._details["max_bone_error_mm"])

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._details[key]
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return key in self._details
        return super().__contains__(key)

    def keys(self):
        return self._details.keys()

    def items(self):
        return self._details.items()

    def values(self):
        return self._details.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self._details.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._details)


def check_biomechanical_validity(
    positions_3d: Union[torch.Tensor, np.ndarray],
    max_bending_deg: float = 125.0,
    max_hyperextension_deg: float = 20.0,
    bone_length_tol_mm: float = 12.0,
    check_bone_lengths: bool = True,
    return_details: bool = False,
) -> Union[BiomechanicalValidityResult, Dict[str, Any]]:
    """
    Evaluates anatomical validity on ANY 3D hand skeleton `positions_3d` of shape `(..., 21, 3)`
    so that `% Biomechanically Valid Poses` can be measured identically on both unconstrained
    Direct-XYZ regression baselines and constrained Forward Kinematics models.

    Checks three anatomical criteria per hand pose:
      1. **Inter-Segment Bending Cone**: At all 15 phalangeal/metacarpal articulations
         `(g -> p -> i)`, the inter-segment bending angle `arccos(u_parent . u_child)`
         must not exceed `max_bending_deg` (125 deg, preventing folded-back broken fingers).
      2. **No Dorsal Hyperextension**: Relative to the palmar flexion cone defined by the
         palm plane `(Wrist -> Index MCP)` x `(Wrist -> Pinky MCP)`, finger phalanges must
         not hyperextend dorsally backward beyond `max_hyperextension_deg` (20 deg).
      3. **Anatomical Bone Integrity**: No bone segment may collapse (`< 5 mm`) or deviate
         from canonical human bone lengths by more than `bone_length_tol_mm` (when coordinates
         are in mm scale).

    Args:
        positions_3d: Tensor or ndarray of shape `(..., 21, 3)`.
        max_bending_deg: Maximum allowed inter-segment flexion angle in degrees.
        max_hyperextension_deg: Maximum allowed dorsal backward hyperextension in degrees.
        bone_length_tol_mm: Maximum allowed bone length stretch/compression in mm.
        check_bone_lengths: Whether to enforce bone length consistency.
        return_details: If True, returns a plain `dict`; otherwise returns a
            `BiomechanicalValidityResult` (which is both a boolean `torch.Tensor`
            AND supports dict-style key lookup).

    Returns:
        `BiomechanicalValidityResult` (boolean Tensor of shape `(...,)` with dict access)
        or `dict` if `return_details=True`.
    """
    pos = torch.as_tensor(positions_3d, dtype=torch.float32)
    if pos.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"Expected positions_3d shape (..., 21, 3), got {tuple(pos.shape)}")

    # 1. Compute bone vectors and bone lengths for all 20 bones (j = 1..20)
    bone_vecs_list: List[torch.Tensor] = []
    bone_lens_list: List[torch.Tensor] = []
    for j in range(1, NUM_JOINTS):
        p = PARENT_INDICES[j]
        v = pos[..., j, :] - pos[..., p, :]
        l = torch.norm(v, p=2, dim=-1)
        bone_vecs_list.append(v)
        bone_lens_list.append(l)

    bone_lens = torch.stack(bone_lens_list, dim=-1)  # (..., 20)
    non_collapsed = (bone_lens > 1e-2).all(dim=-1)   # (...,)

    # Check if coordinates are in millimeter scale (mean palm length > 10.0) or normalized scale
    canonical_lens = torch.as_tensor(
        CANONICAL_BONE_LENGTHS_MM[1:], dtype=torch.float32, device=pos.device
    )
    mean_scale = float(bone_lens.detach().mean().item()) if bone_lens.numel() > 0 else 1.0
    if mean_scale > 5.0:
        bone_err_mm = torch.abs(bone_lens - canonical_lens)
    else:
        # Normalized palm scale (~100 mm = 1.0)
        bone_err_mm = torch.abs(bone_lens * 100.0 - canonical_lens)

    max_bone_err_per_pose = torch.max(bone_err_mm, dim=-1).values  # (...,)
    if check_bone_lengths:
        bone_valid = non_collapsed & (max_bone_err_per_pose <= bone_length_tol_mm)
    else:
        bone_valid = non_collapsed

    # 2. Compute inter-segment bending angles at all 15 articulation triplets
    bending_angles_list: List[torch.Tensor] = []
    for g, p, c in ARTICULATION_TRIPLETS:
        v_parent = F.normalize(pos[..., p, :] - pos[..., g, :], p=2, dim=-1, eps=1e-8)
        v_child = F.normalize(pos[..., c, :] - pos[..., p, :], p=2, dim=-1, eps=1e-8)
        cos_theta = torch.clamp(torch.sum(v_parent * v_child, dim=-1), -1.0, 1.0)
        angle_deg = torch.acos(cos_theta) * (180.0 / float(np.pi))
        bending_angles_list.append(angle_deg)

    bending_angles_deg = torch.stack(bending_angles_list, dim=-1)  # (..., 15)
    bending_valid = (bending_angles_deg <= max_bending_deg).all(dim=-1)

    # 3. Check dorsal hyperextension relative to the palmar normal
    # In our canonical frame: Index MCP (5) is at (-X, +Y), Pinky MCP (17) is at (+X, +Y),
    # so (P_17 - P_0) x (P_5 - P_0) points in the +Z palmar direction.
    v_index_mc = pos[..., 5, :] - pos[..., 0, :]
    v_pinky_mc = pos[..., 17, :] - pos[..., 0, :]
    palm_normal = F.normalize(torch.cross(v_pinky_mc, v_index_mc, dim=-1), p=2, dim=-1, eps=1e-8)

    hyperext_angles_list: List[torch.Tensor] = []
    for g, p, c in HINGE_TRIPLETS:
        if g in (5, 9, 13, 17):
            v_mc = F.normalize(pos[..., g, :] - pos[..., 0, :], p=2, dim=-1, eps=1e-8)
            v_parent = F.normalize(pos[..., p, :] - pos[..., g, :], p=2, dim=-1, eps=1e-8)
            v_child = F.normalize(pos[..., c, :] - pos[..., p, :], p=2, dim=-1, eps=1e-8)
            # Transverse hinge axis of the finger: palm_normal x v_mc
            lateral_axis = F.normalize(
                torch.cross(palm_normal, v_mc, dim=-1), p=2, dim=-1, eps=1e-8
            )
            # Local palmar flexion direction perpendicular to v_parent
            palmar_dir = F.normalize(
                torch.cross(v_parent, lateral_axis, dim=-1), p=2, dim=-1, eps=1e-8
            )
            sin_dorsal = torch.clamp(-torch.sum(v_child * palmar_dir, dim=-1), 0.0, 1.0)
            hyperext_deg = torch.asin(sin_dorsal) * (180.0 / float(np.pi))
            hyperext_angles_list.append(hyperext_deg)

    if hyperext_angles_list:
        hyperext_deg_tensor = torch.stack(hyperext_angles_list, dim=-1)
        hyperext_valid = (hyperext_deg_tensor <= max_hyperextension_deg).all(dim=-1)
    else:
        hyperext_deg_tensor = torch.zeros_like(bending_angles_deg[..., :1])
        hyperext_valid = torch.ones_like(bending_valid)

    valid_mask = bone_valid & bending_valid & hyperext_valid
    valid_ratio = float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 1.0
    valid_pct = valid_ratio * 100.0
    max_bone_err_val = (
        float(max_bone_err_per_pose.max().item()) if max_bone_err_per_pose.numel() > 0 else 0.0
    )

    details: Dict[str, Any] = {
        "valid_mask": valid_mask,
        "valid_pose_mask": valid_mask,
        "is_valid": bool(valid_mask.all().item()),
        "valid_ratio": valid_ratio,
        "valid_pct": valid_pct,
        "valid_percentage": valid_pct,
        "bending_angles_deg": bending_angles_deg,
        "hyperextension_deg": hyperext_deg_tensor,
        "bone_error_mm": bone_err_mm,
        "max_bone_error_mm": max_bone_err_val,
        "bone_valid": bone_valid,
        "bending_valid": bending_valid,
        "hyperext_valid": hyperext_valid,
    }

    if return_details:
        return details
    return BiomechanicalValidityResult(valid_mask, details)


# ---------------------------------------------------------------------------
# 4. Closed-Form Analytical Inverse Kinematics (3D Positions <-> 6D Rotations)
# ---------------------------------------------------------------------------


def _rodrigues_align_unit_vectors(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact minimal-angle 3x3 rotation matrix `R` in SO(3) such that
    `R @ u = v` for unit 3D vectors `u` and `v` of shape `(..., 3)`:
        `R = I + [k]_x + [k]_x^2 / (1 + u · v)` where `k = u x v`.
    Exact to floating-point machine precision (`< 1e-6`) with safe 180-deg handling.
    """
    u_unit = F.normalize(u, p=2, dim=-1, eps=1e-8)
    v_unit = F.normalize(v, p=2, dim=-1, eps=1e-8)

    k = torch.cross(u_unit, v_unit, dim=-1)  # (..., 3)
    c = torch.sum(u_unit * v_unit, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)

    kx = k[..., 0]
    ky = k[..., 1]
    kz = k[..., 2]
    zero = torch.zeros_like(kx)

    # Skew-symmetric matrix K = [k]_x of shape (..., 3, 3)
    row0 = torch.stack((zero, -kz, ky), dim=-1)
    row1 = torch.stack((kz, zero, -kx), dim=-1)
    row2 = torch.stack((-ky, kx, zero), dim=-1)
    K = torch.stack((row0, row1, row2), dim=-2)

    K2 = torch.matmul(K, K)
    eye = torch.eye(3, dtype=u.dtype, device=u.device).expand_as(K)

    denom = torch.clamp(1.0 + c, min=1e-6)
    R_rodrigues = eye + K + K2 / denom

    # Fallback for antiparallel vectors (c < -0.9999): 180-degree rotation around orthogonal axis
    e_x = torch.zeros_like(u_unit)
    e_x[..., 0] = 1.0
    e_y = torch.zeros_like(u_unit)
    e_y[..., 1] = 1.0
    ortho_seed = torch.where(torch.abs(u_unit[..., 0:1]) < 0.9, e_x, e_y)
    axis_180 = F.normalize(
        ortho_seed - torch.sum(ortho_seed * u_unit, dim=-1, keepdim=True) * u_unit,
        p=2,
        dim=-1,
        eps=1e-8,
    )
    R_180 = 2.0 * torch.matmul(axis_180.unsqueeze(-1), axis_180.unsqueeze(-2)) - eye

    return torch.where(c < -0.9999, R_180, R_rodrigues)


def positions_to_rot6d(
    positions_3d: torch.Tensor,
    rest_bone_vectors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Closed-form analytical Inverse Kinematics converting 3D hand joint coordinates
    `positions_3d` of shape `(..., 21, 3)` into hierarchical local 6D rotations
    `rot6d` of shape `(..., 21, 6)` such that `DifferentiableFK(rot6d)` reconstructs
    any skeleton with canonical bone lengths to `0.0000 mm` error.

    For each joint `j = 1..20` with parent `p = PARENT_INDICES[j]`:
      1. Computes unit direction `d_j = normalize(P_j - P_p)` in global space.
      2. Transforms `d_j` into parent `p`'s local coordinate frame: `d_local = R_global[p]^T @ d_j`.
      3. Computes `R_local[j]` via exact Rodrigues alignment mapping canonical unit bone
         vector `u_j` to `d_local`, and updates `R_global[j] = R_global[p] @ R_local[j]`.

    Args:
        positions_3d: Tensor of shape `(..., 21, 3)`.
        rest_bone_vectors: Optional canonical rest bone vectors `(21, 3)`.

    Returns:
        rot6d: Tensor of shape `(..., 21, 6)`.
    """
    pos = torch.as_tensor(positions_3d, dtype=torch.float32)
    if rest_bone_vectors is None:
        rest_vecs = torch.as_tensor(
            REST_BONE_VECTORS_MM, dtype=pos.dtype, device=pos.device
        )
    else:
        rest_vecs = torch.as_tensor(
            rest_bone_vectors, dtype=pos.dtype, device=pos.device
        )

    batch_shape = pos.shape[:-2]
    eye = torch.eye(3, dtype=pos.dtype, device=pos.device)
    root_rot = eye.expand(batch_shape + (3, 3))

    global_rots: List[torch.Tensor] = [root_rot]
    local_rots: List[torch.Tensor] = [root_rot]

    for j in range(1, NUM_JOINTS):
        p = PARENT_INDICES[j]
        p_global = global_rots[p]  # (..., 3, 3)

        vec_obs = pos[..., j, :] - pos[..., p, :]
        d_global = F.normalize(vec_obs, p=2, dim=-1, eps=1e-8)  # (..., 3)

        # Express target bone direction in parent p's local frame: R_global[p]^T @ d_global
        d_local = torch.matmul(p_global.transpose(-1, -2), d_global.unsqueeze(-1)).squeeze(-1)

        u_rest = F.normalize(rest_vecs[j], p=2, dim=-1, eps=1e-8).expand_as(d_local)
        r_local_j = _rodrigues_align_unit_vectors(u_rest, d_local)
        r_global_j = torch.matmul(p_global, r_local_j)

        local_rots.append(r_local_j)
        global_rots.append(r_global_j)

    local_rot_tensor = torch.stack(local_rots, dim=-3)  # (..., 21, 3, 3)
    return matrix_to_rot6d(local_rot_tensor)


# ---------------------------------------------------------------------------
# 5. Differentiable Forward Kinematics Layer (PyTorch + ONNX Exportable)
# ---------------------------------------------------------------------------


class DifferentiableFK(nn.Module):
    """
    Differentiable Forward Kinematics layer for the canonical 21-joint MANO hand skeleton.

    Stores canonical rest bone vectors `rest_bone_vectors` of shape `(21, 3)` whose norms
    equal the exact canonical bone lengths in millimeters (`CANONICAL_BONE_LENGTHS_MM`),
    with wrist joint `0` anchored at `(0, 0, 0)`.

    Given local 6D rotations `rot6d` of shape `(..., 21, 6)` (or local rotation matrices
    `(..., 21, 3, 3)`):
      1. Converts `rot6d` -> `R_local` in SO(3) via `rot6d_to_matrix`.
      2. Traverses joints `j = 0..20` in topological order along `PARENT_INDICES`:
           `R_global[0] = R_local[0]`, `P[0] = (0, 0, 0)`
           `R_global[j] = R_global[parent(j)] @ R_local[j]`
           `P[j] = P[parent(j)] + R_global[j] @ rest_bone_vectors[j]`
      3. Guarantees `||P[j] - P[parent(j)]||_2 == L_j` to floating-point precision
         (`0.0000 mm` bone-length error) for any input in SO(3).
    """

    def __init__(
        self,
        rest_bone_vectors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bone_lengths_mm: Optional[Union[np.ndarray, torch.Tensor]] = None,
        rest_directions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        apply_rom_clamp: bool = False,
    ) -> None:
        super().__init__()
        if rest_bone_vectors is not None:
            rest_vecs = torch.as_tensor(rest_bone_vectors, dtype=torch.float32)
            assert rest_vecs.shape == (NUM_JOINTS, 3)
            lengths = torch.norm(rest_vecs, p=2, dim=-1)
            dirs = torch.zeros_like(rest_vecs)
            dirs[1:] = F.normalize(rest_vecs[1:], p=2, dim=-1, eps=1e-8)
        else:
            lengths_np = (
                CANONICAL_BONE_LENGTHS_MM
                if bone_lengths_mm is None
                else np.asarray(bone_lengths_mm, dtype=np.float32)
            )
            dirs_np = (
                REST_BONE_DIRECTIONS
                if rest_directions is None
                else np.asarray(rest_directions, dtype=np.float32)
            )
            lengths = torch.as_tensor(lengths_np, dtype=torch.float32)
            dirs = torch.as_tensor(dirs_np, dtype=torch.float32)
            rest_vecs = lengths.unsqueeze(-1) * dirs

        self.register_buffer("rest_bone_vectors", rest_vecs)
        self.register_buffer("bone_lengths_mm", lengths)
        self.register_buffer("rest_directions", dirs)
        self.parent_indices: List[int] = list(PARENT_INDICES)
        self.apply_rom_clamp: bool = bool(apply_rom_clamp)
        self.rom_clamper = BiomechanicalRoMClamper()

    def forward(
        self,
        rot6d_or_matrix: torch.Tensor,
        root_position: Optional[torch.Tensor] = None,
        clamp_rom: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Computes 3D joint positions `(..., 21, 3)` in millimeters from local 6D rotations
        `(..., 21, 6)` or local 3x3 rotation matrices `(..., 21, 3, 3)`.

        Uses pure functional list accumulation + `torch.stack` (zero in-place tensor
        mutation) so `torch.onnx.export` produces a clean, static ONNX graph.
        """
        do_clamp = self.apply_rom_clamp if clamp_rom is None else clamp_rom

        if rot6d_or_matrix.shape[-2:] == (NUM_JOINTS, 6):
            rot_in = self.rom_clamper(rot6d_or_matrix) if do_clamp else rot6d_or_matrix
            local_rots = rot6d_to_matrix(rot_in)
        elif rot6d_or_matrix.shape[-3:] == (NUM_JOINTS, 3, 3):
            local_rots = (
                self.rom_clamper.clamp_matrix(rot6d_or_matrix)
                if do_clamp
                else rot6d_or_matrix
            )
        else:
            raise ValueError(
                f"Expected input shape (..., 21, 6) or (..., 21, 3, 3), got {tuple(rot6d_or_matrix.shape)}"
            )

        batch_shape = local_rots.shape[:-3]
        if root_position is None:
            pos_0 = torch.zeros(
                batch_shape + (3,), dtype=local_rots.dtype, device=local_rots.device
            )
        else:
            pos_0 = root_position

        global_rots: List[torch.Tensor] = [local_rots[..., 0, :, :]]
        positions: List[torch.Tensor] = [pos_0]

        for j in range(1, NUM_JOINTS):
            p = self.parent_indices[j]
            r_global_j = torch.matmul(global_rots[p], local_rots[..., j, :, :])
            global_rots.append(r_global_j)

            bone_vec_j = self.rest_bone_vectors[j]  # (3,)
            offset_j = torch.matmul(r_global_j, bone_vec_j.unsqueeze(-1)).squeeze(-1)
            positions.append(positions[p] + offset_j)

        return torch.stack(positions, dim=-2)

    def positions_to_rot6d(self, positions_3d: torch.Tensor) -> torch.Tensor:
        """
        Analytical inverse kinematics mapping 3D joint positions `(..., 21, 3)`
        to local 6D rotations `(..., 21, 6)`.
        """
        return positions_to_rot6d(positions_3d, rest_bone_vectors=self.rest_bone_vectors)

    def analytical_ik_6d(self, positions_3d: torch.Tensor) -> torch.Tensor:
        """Alias for `positions_to_rot6d`."""
        return self.positions_to_rot6d(positions_3d)

    def compute_bone_lengths(self, positions_3d: torch.Tensor) -> torch.Tensor:
        """
        Computes the 20 bone lengths `(..., 20)` in mm for any 3D hand pose `(..., 21, 3)`.
        """
        lengths = []
        for j in range(1, NUM_JOINTS):
            p = self.parent_indices[j]
            lengths.append(torch.norm(positions_3d[..., j, :] - positions_3d[..., p, :], p=2, dim=-1))
        return torch.stack(lengths, dim=-1)

    def compute_bone_length_error_mm(self, positions_3d: torch.Tensor) -> torch.Tensor:
        """
        Computes absolute bone length error `(..., 20)` in mm against `self.bone_lengths_mm[1:]`.
        """
        actual = self.compute_bone_lengths(positions_3d)
        target = self.bone_lengths_mm[1:]
        return torch.abs(actual - target)


# ---------------------------------------------------------------------------
# 6. Pure NumPy EdgeFK for Bare-Metal / CPU Edge Deployment Verification
# ---------------------------------------------------------------------------


class EdgeFK:
    """
    Pure NumPy implementation of `rot6d_to_matrix`, `matrix_to_rot6d`, smooth
    biomechanical RoM clamping, and hierarchical Forward Kinematics traversal
    for bare-metal CPU / ONNXRuntime edge inference without PyTorch overhead.
    """

    def __init__(
        self,
        rest_bone_vectors: Optional[np.ndarray] = None,
        bone_lengths_mm: Optional[np.ndarray] = None,
        rest_directions: Optional[np.ndarray] = None,
        apply_rom_clamp: bool = False,
    ) -> None:
        if rest_bone_vectors is not None:
            self.rest_bone_vectors = np.asarray(rest_bone_vectors, dtype=np.float32)
            self.bone_lengths_mm = np.linalg.norm(self.rest_bone_vectors, axis=-1).astype(np.float32)
            dirs = np.zeros_like(self.rest_bone_vectors, dtype=np.float32)
            dirs[1:] = self.rest_bone_vectors[1:] / np.maximum(
                self.bone_lengths_mm[1:, None], 1e-8
            )
            self.rest_directions = dirs
        else:
            self.bone_lengths_mm = (
                CANONICAL_BONE_LENGTHS_MM.copy()
                if bone_lengths_mm is None
                else np.asarray(bone_lengths_mm, dtype=np.float32)
            )
            self.rest_directions = (
                REST_BONE_DIRECTIONS.copy()
                if rest_directions is None
                else np.asarray(rest_directions, dtype=np.float32)
            )
            self.rest_bone_vectors = (
                self.bone_lengths_mm[:, None] * self.rest_directions
            ).astype(np.float32)

        self.parent_indices: List[int] = list(PARENT_INDICES)
        self.apply_rom_clamp: bool = bool(apply_rom_clamp)
        self.min_limits_rad = ANATOMICAL_MIN_LIMITS_RAD.copy()
        self.max_limits_rad = ANATOMICAL_MAX_LIMITS_RAD.copy()
        self.center_rad = 0.5 * (self.min_limits_rad + self.max_limits_rad)
        self.half_range_rad = np.maximum(
            0.5 * (self.max_limits_rad - self.min_limits_rad), 1e-4
        )

    @staticmethod
    def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
        """
        Pure NumPy Gram-Schmidt 6D-to-SO(3) conversion matching `rot6d_to_matrix` in PyTorch.
        """
        d6_arr = np.asarray(d6, dtype=np.float32)
        a1 = d6_arr[..., 0:3]
        a2 = d6_arr[..., 3:6]

        e1 = np.zeros_like(a1)
        e1[..., 0] = 1.0
        a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
        a1_safe = np.where(a1_norm < 1e-6, e1, a1)
        b1 = a1_safe / np.maximum(np.linalg.norm(a1_safe, axis=-1, keepdims=True), 1e-8)

        proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2_unnorm = a2 - proj

        e2 = np.zeros_like(a2)
        e2[..., 1] = 1.0
        e3 = np.zeros_like(a2)
        e3[..., 2] = 1.0
        alt_axis = np.where(np.abs(b1[..., 1:2]) < 0.9, e2, e3)
        alt_orth = alt_axis - np.sum(b1 * alt_axis, axis=-1, keepdims=True) * b1
        b2_norm = np.linalg.norm(b2_unnorm, axis=-1, keepdims=True)
        b2_safe = np.where(b2_norm < 1e-6, alt_orth, b2_unnorm)
        b2 = b2_safe / np.maximum(np.linalg.norm(b2_safe, axis=-1, keepdims=True), 1e-8)

        b3 = np.cross(b1, b2, axis=-1)
        return np.stack((b1, b2, b3), axis=-1).astype(np.float32)

    @staticmethod
    def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
        """Extracts first two columns `(..., 6)` of a 3x3 rotation matrix in NumPy."""
        mat = np.asarray(matrix, dtype=np.float32)
        return np.concatenate((mat[..., :, 0], mat[..., :, 1]), axis=-1).astype(np.float32)

    def clamp_rom(self, rot6d: np.ndarray) -> np.ndarray:
        """Applies smooth `tanh` biomechanical RoM clamping in pure NumPy."""
        R = self.rot6d_to_matrix(rot6d)
        sy = np.clip(-R[..., 2, 0], -0.99999, 0.99999)
        ry = np.arcsin(sy)
        cy = np.maximum(np.cos(ry), 1e-6)
        rx = np.arctan2(R[..., 2, 1] / cy, R[..., 2, 2] / cy)
        rz = np.arctan2(R[..., 1, 0] / cy, R[..., 0, 0] / cy)
        euler = np.stack((rx, ry, rz), axis=-1)

        scaled = (euler - self.center_rad) / (self.half_range_rad + 1e-6)
        euler_clamped = self.center_rad + self.half_range_rad * np.tanh(scaled)

        rx_c, ry_c, rz_c = euler_clamped[..., 0], euler_clamped[..., 1], euler_clamped[..., 2]
        cx, sx = np.cos(rx_c), np.sin(rx_c)
        cy, sy = np.cos(ry_c), np.sin(ry_c)
        cz, sz = np.cos(rz_c), np.sin(rz_c)

        r00 = cz * cy
        r01 = cz * sy * sx - sz * cx
        r02 = cz * sy * cx + sz * sx
        r10 = sz * cy
        r11 = sz * sy * sx + cz * cx
        r12 = sz * sy * cx - cz * sx
        r20 = -sy
        r21 = cy * sx
        r22 = cy * cx

        R_clamped = np.stack(
            [
                np.stack([r00, r01, r02], axis=-1),
                np.stack([r10, r11, r12], axis=-1),
                np.stack([r20, r21, r22], axis=-1),
            ],
            axis=-2,
        ).astype(np.float32)
        return self.matrix_to_rot6d(R_clamped)

    def forward(
        self,
        rot6d_or_matrix: np.ndarray,
        root_position: Optional[np.ndarray] = None,
        clamp_rom: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Pure NumPy kinematic tree traversal producing `(..., 21, 3)` 3D joint positions in mm.
        """
        arr = np.asarray(rot6d_or_matrix, dtype=np.float32)
        do_clamp = self.apply_rom_clamp if clamp_rom is None else clamp_rom

        if arr.shape[-2:] == (NUM_JOINTS, 6):
            rot_in = self.clamp_rom(arr) if do_clamp else arr
            local_rots = self.rot6d_to_matrix(rot_in)
        elif arr.shape[-3:] == (NUM_JOINTS, 3, 3):
            if do_clamp:
                rot_6d = self.clamp_rom(self.matrix_to_rot6d(arr))
                local_rots = self.rot6d_to_matrix(rot_6d)
            else:
                local_rots = arr
        else:
            raise ValueError(
                f"Expected input shape (..., 21, 6) or (..., 21, 3, 3), got {arr.shape}"
            )

        batch_shape = local_rots.shape[:-3]
        positions = np.zeros(batch_shape + (NUM_JOINTS, 3), dtype=np.float32)
        if root_position is not None:
            positions[..., 0, :] = root_position

        global_rots: List[Optional[np.ndarray]] = [None] * NUM_JOINTS
        global_rots[0] = local_rots[..., 0, :, :]

        for j in range(1, NUM_JOINTS):
            p = self.parent_indices[j]
            r_global_j = np.matmul(global_rots[p], local_rots[..., j, :, :])
            global_rots[j] = r_global_j
            offset_j = np.einsum("...ik,k->...i", r_global_j, self.rest_bone_vectors[j])
            positions[..., j, :] = positions[..., p, :] + offset_j

        return positions

    def __call__(
        self,
        rot6d_or_matrix: np.ndarray,
        root_position: Optional[np.ndarray] = None,
        clamp_rom: Optional[bool] = None,
    ) -> np.ndarray:
        return self.forward(rot6d_or_matrix, root_position=root_position, clamp_rom=clamp_rom)

    def compute_bone_lengths(self, positions_3d: np.ndarray) -> np.ndarray:
        """Computes 20 bone lengths `(..., 20)` in mm for a NumPy skeleton `(..., 21, 3)`."""
        pos = np.asarray(positions_3d, dtype=np.float32)
        lengths = [
            np.linalg.norm(pos[..., j, :] - pos[..., self.parent_indices[j], :], axis=-1)
            for j in range(1, NUM_JOINTS)
        ]
        return np.stack(lengths, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Dynamic Palm-Scale Calibration Utilities (Live Camera Adaptation)
# ---------------------------------------------------------------------------

def estimate_palm_scale_factor(
    positions_3d: Union[np.ndarray, torch.Tensor],
    visibility_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    min_scale: float = 0.65,
    max_scale: float = 1.45,
) -> float:
    """
    Estimates an anatomical palm-scale factor alpha from unoccluded metacarpal
    landmarks (WRIST -> MIDDLE_MCP and WRIST -> INDEX_MCP).

    Used during the first unoccluded frames of live camera streams to calibrate
    canonical bone lengths to the user's specific hand geometry while strictly
    locking bone-length error to 0.0000 mm.

    Args:
        positions_3d: Hand joint coordinates `(..., 21, 3)` in millimeters.
        visibility_mask: Optional binary visibility mask `(..., 21)`.
        min_scale: Minimum valid anatomical scale factor (default 0.65).
        max_scale: Maximum valid anatomical scale factor (default 1.45).

    Returns:
        Clamped scalar scale factor alpha (e.g. 1.0 = canonical ~185 mm hand).
    """
    if isinstance(positions_3d, torch.Tensor):
        pts = positions_3d.detach().cpu().numpy()
    else:
        pts = np.asarray(positions_3d, dtype=np.float32)

    pts = pts.reshape(-1, NUM_JOINTS, 3)

    if visibility_mask is not None:
        if isinstance(visibility_mask, torch.Tensor):
            mask = visibility_mask.detach().cpu().numpy()
        else:
            mask = np.asarray(visibility_mask, dtype=np.float32)
        mask = mask.reshape(-1, NUM_JOINTS)
    else:
        mask = np.ones((pts.shape[0], NUM_JOINTS), dtype=np.float32)

    scales: List[float] = []
    anchor_pairs = [
        (0, 5, CANONICAL_BONE_LENGTHS_MM[5]),  # Wrist -> Index MCP
        (0, 9, CANONICAL_BONE_LENGTHS_MM[9]),  # Wrist -> Middle MCP
    ]

    for p_idx, c_idx, canon_len in anchor_pairs:
        valid_frames = (mask[:, p_idx] > 0.5) & (mask[:, c_idx] > 0.5)
        if np.any(valid_frames):
            diff = pts[valid_frames, c_idx, :] - pts[valid_frames, p_idx, :]
            observed_len = np.linalg.norm(diff, axis=-1)
            good = observed_len > 10.0
            if np.any(good):
                scales.append(float(np.median(observed_len[good]) / canon_len))

    if len(scales) == 0:
        return 1.0

    raw_scale = float(np.mean(scales))
    return float(np.clip(raw_scale, min_scale, max_scale))


def calibrate_bone_lengths(
    scale_factor: float = 1.0,
    base_lengths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns dynamically calibrated bone lengths in mm preserving canonical human proportions.

    Args:
        scale_factor: Scalar alpha estimated by `estimate_palm_scale_factor`.
        base_lengths: Optional base bone lengths `(21,)` in mm (defaults to `CANONICAL_BONE_LENGTHS_MM`).

    Returns:
        Calibrated bone lengths `(21,)` with exact relative ratios preserved.
    """
    if base_lengths is None:
        base = CANONICAL_BONE_LENGTHS_MM.copy()
    else:
        base = np.asarray(base_lengths, dtype=np.float32).copy()
    return (base * float(scale_factor)).astype(np.float32)
