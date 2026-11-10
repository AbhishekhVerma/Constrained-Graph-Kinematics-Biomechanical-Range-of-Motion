"""
Biomechanical Hand Skeleton Topology & Articulated Trajectory Ingestion/Generation.

Defines the canonical 21-joint MANO hand skeleton, anatomical bone lengths in millimeters,
rest-pose bone unit vectors, kinematic tree + inter-finger graph edges for GAT message
passing, and the deterministic Biomechanical Articulated Hand Trajectory Generator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# 1. Canonical 21-Joint MANO Hand Topology
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 21
NUM_BONES: int = 20

JOINT_NAMES: List[str] = [
    "WRIST",       # 0
    "THUMB_CMC",   # 1
    "THUMB_MCP",   # 2
    "THUMB_IP",    # 3
    "THUMB_TIP",   # 4
    "INDEX_MCP",   # 5
    "INDEX_PIP",   # 6
    "INDEX_DIP",   # 7
    "INDEX_TIP",   # 8
    "MIDDLE_MCP",  # 9
    "MIDDLE_PIP",  # 10
    "MIDDLE_DIP",  # 11
    "MIDDLE_TIP",  # 12
    "RING_MCP",    # 13
    "RING_PIP",    # 14
    "RING_DIP",    # 15
    "RING_TIP",    # 16
    "PINKY_MCP",   # 17
    "PINKY_PIP",   # 18
    "PINKY_DIP",   # 19
    "PINKY_TIP",   # 20
]

JOINT_NAME_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(JOINT_NAMES)}

PARENT_INDICES: List[int] = [
    -1,              # 0: Wrist (root)
    0, 1, 2, 3,      # 1-4: Thumb (CMC, MCP, IP, TIP)
    0, 5, 6, 7,      # 5-8: Index (MCP, PIP, DIP, TIP)
    0, 9, 10, 11,    # 9-12: Middle (MCP, PIP, DIP, TIP)
    0, 13, 14, 15,   # 13-16: Ring (MCP, PIP, DIP, TIP)
    0, 17, 18, 19,   # 17-20: Pinky (MCP, PIP, DIP, TIP)
]

KINEMATIC_CHAINS: Dict[str, List[int]] = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

FINGER_SUBTREES: Dict[str, List[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

DISTAL_TIP_INDICES: List[int] = [4, 8, 12, 16, 20]
MCP_JOINTS: List[int] = [2, 5, 9, 13, 17]
PIP_JOINTS: List[int] = [3, 6, 10, 14, 18]
DIP_JOINTS: List[int] = [4, 7, 11, 15, 19]

# ---------------------------------------------------------------------------
# 2. Canonical Human Anatomical Bone Lengths (mm) & Rest Skeleton Directions
# ---------------------------------------------------------------------------
# Index 0 is the root (0.0 mm); index j in 1..20 is the length of bone (PARENT_INDICES[j] -> j).
# Anatomical ranges:
#   Palm metacarpals: ~45-68 mm (thumb CMC offset 34 mm, thumb MC 46 mm)
#   Proximal phalanges: ~28-42 mm
#   Middle phalanges: ~18-28 mm
#   Distal phalanges: ~15-20 mm

CANONICAL_BONE_LENGTHS_MM: np.ndarray = np.array(
    [
        0.0,   # 0: WRIST (root)
        # Thumb (1-4)
        34.0,  # 0 -> 1: Wrist to Thumb CMC
        46.0,  # 1 -> 2: Thumb CMC to MCP (Metacarpal)
        30.0,  # 2 -> 3: Thumb MCP to IP (Proximal phalanx)
        20.0,  # 3 -> 4: Thumb IP to TIP (Distal phalanx)
        # Index (5-8)
        66.0,  # 0 -> 5: Wrist to Index MCP (Metacarpal)
        40.0,  # 5 -> 6: Index MCP to PIP (Proximal phalanx)
        24.0,  # 6 -> 7: Index PIP to DIP (Middle phalanx)
        17.0,  # 7 -> 8: Index DIP to TIP (Distal phalanx)
        # Middle (9-12)
        68.0,  # 0 -> 9: Wrist to Middle MCP (Metacarpal)
        42.0,  # 9 -> 10: Middle MCP to PIP (Proximal phalanx)
        27.0,  # 10 -> 11: Middle PIP to DIP (Middle phalanx)
        18.0,  # 11 -> 12: Middle DIP to TIP (Distal phalanx)
        # Ring (13-16)
        62.0,  # 0 -> 13: Wrist to Ring MCP (Metacarpal)
        39.0,  # 13 -> 14: Ring MCP to PIP (Proximal phalanx)
        25.0,  # 14 -> 15: Ring PIP to DIP (Middle phalanx)
        17.0,  # 15 -> 16: Ring DIP to TIP (Distal phalanx)
        # Pinky (17-20)
        54.0,  # 0 -> 17: Wrist to Pinky MCP (Metacarpal)
        31.0,  # 17 -> 18: Pinky MCP to PIP (Proximal phalanx)
        19.0,  # 18 -> 19: Pinky PIP to DIP (Middle phalanx)
        16.0,  # 19 -> 20: Pinky DIP to TIP (Distal phalanx)
    ],
    dtype=np.float32,
)

# 20-element view for callers indexing bones 0..19 (corresponding to joints 1..20)
BONE_LENGTHS_20_MM: np.ndarray = CANONICAL_BONE_LENGTHS_MM[1:].copy()

# Coordinate frame for rest pose:
#   +Y: Distal direction (along extended middle finger)
#   +X: Lateral direction across palm (Thumb at -X, Pinky at +X)
#   +Z: Palmar/anterior direction (positive X-axis rotation bends fingers toward +Z into palm)
_RAW_REST_DIRECTIONS = np.array(
    [
        [0.0, 0.0, 0.0],     # 0: WRIST
        # Thumb: angled radially (-X), distally (+Y), and slightly palmar (+Z)
        [-0.78, 0.58, 0.23], # 0 -> 1: Thumb CMC
        [-0.58, 0.78, 0.23], # 1 -> 2: Thumb MCP
        [-0.42, 0.89, 0.17], # 2 -> 3: Thumb IP
        [-0.35, 0.92, 0.17], # 3 -> 4: Thumb TIP
        # Index: slightly radial (-X), primarily distal (+Y)
        [-0.27, 0.96, 0.02], # 0 -> 5: Index MCP
        [-0.06, 0.998, 0.0], # 5 -> 6: Index PIP
        [-0.04, 0.999, 0.0], # 6 -> 7: Index DIP
        [-0.03, 0.999, 0.0], # 7 -> 8: Index TIP
        # Middle: straight distal (+Y)
        [0.00, 1.00, 0.00],  # 0 -> 9: Middle MCP
        [0.00, 1.00, 0.00],  # 9 -> 10: Middle PIP
        [0.00, 1.00, 0.00],  # 10 -> 11: Middle DIP
        [0.00, 1.00, 0.00],  # 11 -> 12: Middle TIP
        # Ring: slightly ulnar (+X), primarily distal (+Y)
        [0.24, 0.97, 0.04],  # 0 -> 13: Ring MCP
        [0.05, 0.998, 0.02], # 13 -> 14: Ring PIP
        [0.03, 0.999, 0.01], # 14 -> 15: Ring DIP
        [0.02, 0.999, 0.01], # 15 -> 16: Ring TIP
        # Pinky: ulnar (+X), distal (+Y), slight palmar arch (+Z)
        [0.46, 0.88, 0.09],  # 0 -> 17: Pinky MCP
        [0.10, 0.994, 0.04], # 17 -> 18: Pinky PIP
        [0.07, 0.997, 0.03], # 18 -> 19: Pinky DIP
        [0.05, 0.998, 0.02], # 19 -> 20: Pinky TIP
    ],
    dtype=np.float32,
)

# Normalize each non-zero row to exact unit length
REST_BONE_DIRECTIONS: np.ndarray = np.zeros_like(_RAW_REST_DIRECTIONS, dtype=np.float32)
for _j in range(1, NUM_JOINTS):
    _norm = float(np.linalg.norm(_RAW_REST_DIRECTIONS[_j]))
    REST_BONE_DIRECTIONS[_j] = _RAW_REST_DIRECTIONS[_j] / max(_norm, 1e-8)

REST_BONE_VECTORS_MM: np.ndarray = (
    CANONICAL_BONE_LENGTHS_MM[:, None] * REST_BONE_DIRECTIONS
).astype(np.float32)


def compute_rest_joint_positions_mm() -> np.ndarray:
    """Computes the 3D coordinates (21, 3) of the canonical open/rest hand in mm."""
    pos = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for j in range(1, NUM_JOINTS):
        p = PARENT_INDICES[j]
        pos[j] = pos[p] + REST_BONE_VECTORS_MM[j]
    return pos


REST_JOINT_POSITIONS_MM: np.ndarray = compute_rest_joint_positions_mm()

# Reference palm scale (mm) used for root normalization (~100 mm)
DEFAULT_PALM_SCALE_MM: float = 100.0

# ---------------------------------------------------------------------------
# 3. Graph Edges (Kinematic Tree + Inter-Finger Palm Connections)
# ---------------------------------------------------------------------------

KINEMATIC_TREE_EDGES: List[Tuple[int, int]] = [
    (PARENT_INDICES[j], j) for j in range(1, NUM_JOINTS)
]

PALM_CONNECTIONS: List[Tuple[int, int]] = [
    (1, 5),   # Thumb CMC <-> Index MCP
    (5, 9),   # Index MCP <-> Middle MCP
    (9, 13),  # Middle MCP <-> Ring MCP
    (13, 17), # Ring MCP <-> Pinky MCP
]

# Undirected + directed kinematic tree edges + inter-finger palm connections
_UNDIRECTED_BASE_EDGES: List[Tuple[int, int]] = KINEMATIC_TREE_EDGES + PALM_CONNECTIONS
HAND_EDGES: List[List[int]] = []
for _u, _v in _UNDIRECTED_BASE_EDGES:
    HAND_EDGES.append([_u, _v])
    HAND_EDGES.append([_v, _u])

HAND_EDGES_NP: np.ndarray = np.array(HAND_EDGES, dtype=np.int64).T  # shape (2, 48)


def get_hand_edge_index(include_self_loops: bool = True) -> np.ndarray:
    """Returns graph edge_index of shape (2, E) as int64 numpy array."""
    edges = list(HAND_EDGES)
    if include_self_loops:
        for i in range(NUM_JOINTS):
            edges.append([i, i])
    return np.array(edges, dtype=np.int64).T


def get_hand_adjacency_matrix(
    include_self_loops: bool = True, normalize: bool = False
) -> np.ndarray:
    """Returns (21, 21) float32 adjacency matrix for the canonical hand graph."""
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


# ---------------------------------------------------------------------------
# 4. Anatomical Joint Angle Limits (Radians)
# ---------------------------------------------------------------------------
# Euler convention per joint j: [flexion_x, twist_y, abduction_z] in radians.
# Wait: In our FK chain:
#   pos[j] = pos[p] + R_global[j] @ rest_vec[j], where R_global[j] = R_global[p] @ R_local[j].
# Therefore:
#   - j = 0: Wrist root rotation
#   - j in {5, 9, 13, 17}: Metacarpal bones (rigid palm / slight carpal arch)
#   - j in {6, 10, 14, 18}: Proximal phalanges (driven by MCP flexion + abduction)
#   - j in {7, 11, 15, 19}: Middle phalanges (driven by PIP hinge flexion)
#   - j in {8, 12, 16, 20}: Distal phalanges (driven by DIP hinge flexion)
#   - j in {1, 2, 3, 4}: Thumb articulation chain


def _build_anatomical_limits_rad() -> Tuple[np.ndarray, np.ndarray]:
    """Builds (min_limits_rad, max_limits_rad) arrays of shape (21, 3)."""
    deg = np.pi / 180.0
    min_lim = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    max_lim = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    # 0: Wrist root orientation
    min_lim[0] = np.array([-40.0, -45.0, -25.0], dtype=np.float32) * deg
    max_lim[0] = np.array([+40.0, +45.0, +25.0], dtype=np.float32) * deg

    # 1: Thumb CMC base offset
    min_lim[1] = np.array([-15.0, -15.0, -15.0], dtype=np.float32) * deg
    max_lim[1] = np.array([+25.0, +20.0, +25.0], dtype=np.float32) * deg

    # 2: Thumb Metacarpal (CMC articulation: flexion, pronation, opposition/abduction)
    min_lim[2] = np.array([-15.0, -10.0, -45.0], dtype=np.float32) * deg
    max_lim[2] = np.array([+55.0, +35.0, +20.0], dtype=np.float32) * deg

    # 3: Thumb Proximal Phalanx (Thumb MCP articulation)
    min_lim[3] = np.array([0.0, -5.0, -12.0], dtype=np.float32) * deg
    max_lim[3] = np.array([+60.0, +10.0, +12.0], dtype=np.float32) * deg

    # 4: Thumb Distal Phalanx (Thumb IP hinge articulation)
    min_lim[4] = np.array([-10.0, -2.0, -2.0], dtype=np.float32) * deg
    max_lim[4] = np.array([+80.0, +2.0, +2.0], dtype=np.float32) * deg

    # Palm Metacarpals: 5 (Index), 9 (Middle) rigid; 13 (Ring), 17 (Pinky) slight palmar cupping
    for j in (5, 9):
        min_lim[j] = np.array([-2.0, -2.0, -2.0], dtype=np.float32) * deg
        max_lim[j] = np.array([+2.0, +2.0, +2.0], dtype=np.float32) * deg
    for j in (13, 17):
        min_lim[j] = np.array([0.0, -3.0, -4.0], dtype=np.float32) * deg
        max_lim[j] = np.array([+10.0, +3.0, +4.0], dtype=np.float32) * deg

    # Finger MCP joints -> rotate proximal phalanges (6, 10, 14, 18)
    for j in (6, 10, 14, 18):
        min_lim[j] = np.array([-15.0, -5.0, -20.0], dtype=np.float32) * deg
        max_lim[j] = np.array([+90.0, +5.0, +20.0], dtype=np.float32) * deg

    # Finger PIP joints -> rotate middle phalanges (7, 11, 15, 19) - strict hinge [0, 110 deg]
    for j in (7, 11, 15, 19):
        min_lim[j] = np.array([0.0, -2.0, -2.0], dtype=np.float32) * deg
        max_lim[j] = np.array([+110.0, +2.0, +2.0], dtype=np.float32) * deg

    # Finger DIP joints -> rotate distal phalanges (8, 12, 16, 20) - strict hinge [0, 80 deg]
    for j in (8, 12, 16, 20):
        min_lim[j] = np.array([0.0, -2.0, -2.0], dtype=np.float32) * deg
        max_lim[j] = np.array([+80.0, +2.0, +2.0], dtype=np.float32) * deg

    return min_lim, max_lim


ANATOMICAL_MIN_LIMITS_RAD, ANATOMICAL_MAX_LIMITS_RAD = _build_anatomical_limits_rad()


# ---------------------------------------------------------------------------
# 5. SO(3) / 6D Rotation & Forward Kinematics Utilities (NumPy)
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix(euler_rad: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (..., 3) [rx_flexion, ry_twist, rz_abduction] in radians
    into 3x3 rotation matrices (..., 3, 3) via R = Rz(rz) @ Ry(ry) @ Rx(rx).
    """
    rx = euler_rad[..., 0]
    ry = euler_rad[..., 1]
    rz = euler_rad[..., 2]

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # R = Rz @ Ry @ Rx
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx

    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx

    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    mat = np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )
    return mat.astype(np.float32)


def matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    """
    Extracts continuous 6D rotation representation (..., 6) from 3x3 rotation
    matrices (..., 3, 3) following Zhou et al. (CVPR 2019) using the first two columns.
    """
    col0 = rot_mat[..., :, 0]  # (..., 3)
    col1 = rot_mat[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Reconstructs orthogonal 3x3 rotation matrices (..., 3, 3) in SO(3) from
    6D representations (..., 6) via Gram-Schmidt orthogonalization.
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2_unnorm = a2 - proj
    b2 = b2_unnorm / np.maximum(np.linalg.norm(b2_unnorm, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def forward_kinematics_numpy(
    local_rotations: np.ndarray,
    bone_lengths_mm: np.ndarray = CANONICAL_BONE_LENGTHS_MM,
    rest_directions: np.ndarray = REST_BONE_DIRECTIONS,
    root_positions_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Exact Forward Kinematics over the 21-joint MANO hand kinematic tree.

    Args:
        local_rotations: (..., 21, 3, 3) local joint rotation matrices in SO(3).
        bone_lengths_mm: (21,) canonical bone lengths in millimeters.
        rest_directions: (21, 3) canonical unit bone directions.
        root_positions_mm: Optional (..., 3) wrist root position in mm (defaults to 0).

    Returns:
        positions_mm: (..., 21, 3) 3D joint positions in millimeters.
    """
    batch_shape = local_rotations.shape[:-3]
    positions = np.zeros(batch_shape + (NUM_JOINTS, 3), dtype=np.float32)
    if root_positions_mm is not None:
        positions[..., 0, :] = root_positions_mm

    global_rots: List[Optional[np.ndarray]] = [None] * NUM_JOINTS
    global_rots[0] = local_rotations[..., 0, :, :]

    rest_vecs = (bone_lengths_mm[:, None] * rest_directions).astype(np.float32)

    for j in range(1, NUM_JOINTS):
        p = PARENT_INDICES[j]
        p_rot = global_rots[p]
        j_rot = np.matmul(p_rot, local_rotations[..., j, :, :])
        global_rots[j] = j_rot

        # Rotate rest bone vector (3,) -> (..., 3)
        vec = rest_vecs[j]
        offset = np.einsum("...ik,k->...i", j_rot, vec)
        positions[..., j, :] = positions[..., p, :] + offset

    return positions


# ---------------------------------------------------------------------------
# 6. Biomechanical Articulated Hand Trajectory Generator
# ---------------------------------------------------------------------------

GESTURE_NAMES: List[str] = [
    "scalpel_pinch",
    "needle_driver_rotation",
    "power_drill_grip",
    "finger_flexion_waves",
    "precision_opposition",
    "flat_palpation",
]


def _clamp_to_anatomical_limits(euler_rad: np.ndarray) -> np.ndarray:
    """Clamps (..., 21, 3) Euler angles strictly to anatomical range-of-motion."""
    return np.clip(
        euler_rad,
        ANATOMICAL_MIN_LIMITS_RAD,
        ANATOMICAL_MAX_LIMITS_RAD,
    ).astype(np.float32)


def _synthesize_single_sequence_angles(
    gesture_id: int,
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthesizes a smooth, anatomically valid sequence of Euler angles (T, 21, 3)
    in radians for a given surgical/industrial gesture archetype.
    """
    deg = np.pi / 180.0
    angles = np.zeros((seq_len, NUM_JOINTS, 3), dtype=np.float32)

    # Normalized time phase t in [0, 1] with randomized starting phase & speed
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    freq = float(rng.uniform(0.65, 1.35))
    t_space = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    wave = np.sin(phase0 + 2.0 * np.pi * freq * t_space)          # [-1, 1]
    half_wave = 0.5 * (1.0 + wave)                                # [0, 1]
    cos_wave = np.cos(phase0 + 2.0 * np.pi * freq * t_space)      # [-1, 1]

    # Subtle natural wrist motion common to all gestures
    wrist_pitch = (rng.uniform(-10.0, 12.0) + 6.0 * wave) * deg
    wrist_roll = (rng.uniform(-12.0, 12.0) + 5.0 * cos_wave) * deg
    wrist_yaw = (rng.uniform(-8.0, 8.0) + 3.0 * wave) * deg
    angles[:, 0, 0] = wrist_pitch
    angles[:, 0, 1] = wrist_roll
    angles[:, 0, 2] = wrist_yaw

    if gesture_id == 0:
        # 1. Scalpel Pinch (Tripod precision pinch: Thumb + Index + Middle oppose;
        #    Ring + Pinky flexed into stabilized support)
        pinch_close = 0.45 + 0.45 * half_wave  # [0.45, 0.90]
        # Thumb opposition & flexion
        angles[:, 1, :] = np.array([12.0, 10.0, 14.0]) * deg
        angles[:, 2, 0] = (28.0 + 14.0 * pinch_close) * deg
        angles[:, 2, 1] = 20.0 * deg
        angles[:, 2, 2] = (-26.0 - 8.0 * pinch_close) * deg
        angles[:, 3, 0] = (24.0 + 16.0 * pinch_close) * deg
        angles[:, 4, 0] = (18.0 + 22.0 * pinch_close) * deg

        # Index & Middle precision flexion
        for mcp_j, pip_j, dip_j, abd_deg in [(6, 7, 8, -4.0), (10, 11, 12, 3.0)]:
            mcp_flex = (38.0 + 22.0 * pinch_close + rng.uniform(-3.0, 3.0)) * deg
            pip_flex = (46.0 + 26.0 * pinch_close + rng.uniform(-3.0, 3.0)) * deg
            dip_flex = 0.68 * pip_flex
            angles[:, mcp_j, 0] = mcp_flex
            angles[:, mcp_j, 2] = abd_deg * deg
            angles[:, pip_j, 0] = pip_flex
            angles[:, dip_j, 0] = dip_flex

        # Ring & Pinky tucked
        for mcp_j, pip_j, dip_j, base_deg in [(14, 15, 16, 62.0), (18, 19, 20, 70.0)]:
            angles[:, mcp_j, 0] = (base_deg + 8.0 * pinch_close) * deg
            angles[:, pip_j, 0] = (base_deg + 16.0 + 8.0 * pinch_close) * deg
            angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]
        angles[:, 13, 0] = 5.0 * deg
        angles[:, 17, 0] = 8.0 * deg

    elif gesture_id == 1:
        # 2. Needle Driver Rotation (Suturing pronation/supination wrist sweep +
        #    index stabilizing shaft, thumb/ring in finger loops)
        angles[:, 0, 1] = (rng.uniform(-10.0, 10.0) + 32.0 * wave) * deg  # Pronation sweep
        angles[:, 0, 0] = (8.0 + 10.0 * cos_wave) * deg
        ratchet = 0.5 + 0.5 * cos_wave

        # Thumb in upper ring loop
        angles[:, 1, :] = np.array([10.0, 8.0, 12.0]) * deg
        angles[:, 2, 0] = (24.0 + 12.0 * ratchet) * deg
        angles[:, 2, 1] = 16.0 * deg
        angles[:, 2, 2] = -22.0 * deg
        angles[:, 3, 0] = (28.0 + 10.0 * ratchet) * deg
        angles[:, 4, 0] = (20.0 + 15.0 * ratchet) * deg

        # Index extended along instrument for precision guidance
        angles[:, 6, 0] = (14.0 + 8.0 * half_wave) * deg
        angles[:, 6, 2] = -5.0 * deg
        angles[:, 7, 0] = (18.0 + 8.0 * half_wave) * deg
        angles[:, 8, 0] = 0.65 * angles[:, 7, 0]

        # Middle, Ring, Pinky flexed around handle / lower ring loop
        for mcp_j, pip_j, dip_j, flex_b in [
            (10, 11, 12, 48.0),
            (14, 15, 16, 58.0),
            (18, 19, 20, 64.0),
        ]:
            angles[:, mcp_j, 0] = (flex_b + 12.0 * ratchet) * deg
            angles[:, pip_j, 0] = (flex_b + 18.0 + 12.0 * ratchet) * deg
            angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]
        angles[:, 13, 0] = 6.0 * deg
        angles[:, 17, 0] = 8.0 * deg

    elif gesture_id == 2:
        # 3. Power Drill Grip (Tight cylindrical grasp on Middle/Ring/Pinky,
        #    Thumb wrapped in opposition, Index squeezing trigger)
        trigger_pull = half_wave  # [0, 1]
        # Thumb opposition wrap
        angles[:, 1, :] = np.array([15.0, 12.0, 15.0]) * deg
        angles[:, 2, 0] = 38.0 * deg
        angles[:, 2, 1] = 24.0 * deg
        angles[:, 2, 2] = -32.0 * deg
        angles[:, 3, 0] = (38.0 + 8.0 * trigger_pull) * deg
        angles[:, 4, 0] = (35.0 + 10.0 * trigger_pull) * deg

        # Index trigger squeeze
        angles[:, 6, 0] = (32.0 + 36.0 * trigger_pull) * deg
        angles[:, 7, 0] = (40.0 + 45.0 * trigger_pull) * deg
        angles[:, 8, 0] = 0.70 * angles[:, 7, 0]

        # Middle, Ring, Pinky firm handle wrap
        for mcp_j, pip_j, dip_j, base_m, base_p in [
            (10, 11, 12, 68.0, 86.0),
            (14, 15, 16, 72.0, 92.0),
            (18, 19, 20, 76.0, 95.0),
        ]:
            jitter = rng.uniform(-3.0, 3.0)
            angles[:, mcp_j, 0] = (base_m + jitter + 4.0 * trigger_pull) * deg
            angles[:, pip_j, 0] = (base_p + jitter + 4.0 * trigger_pull) * deg
            angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]
        angles[:, 13, 0] = 7.0 * deg
        angles[:, 17, 0] = 9.0 * deg

    elif gesture_id == 3:
        # 4. Finger Flexion Waves (Sequential biomechanical wave across digits
        #    with anatomical DIP = 0.68 * PIP coupling and abduction fan)
        finger_specs = [
            (6, 7, 8, 0.0, -8.0),
            (10, 11, 12, 0.45, 0.0),
            (14, 15, 16, 0.90, +6.0),
            (18, 19, 20, 1.35, +12.0),
        ]
        for mcp_j, pip_j, dip_j, lag, spread_deg in finger_specs:
            f_wave = 0.5 * (
                1.0 + np.sin(phase0 + 2.0 * np.pi * freq * t_space - lag)
            )
            angles[:, mcp_j, 0] = (10.0 + 68.0 * f_wave) * deg
            angles[:, mcp_j, 2] = spread_deg * (1.0 - 0.65 * f_wave) * deg
            angles[:, pip_j, 0] = (12.0 + 86.0 * f_wave) * deg
            angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]

        # Thumb wave
        t_wave = 0.5 * (1.0 + np.sin(phase0 + 2.0 * np.pi * freq * t_space + 0.3))
        angles[:, 1, 0] = (8.0 + 10.0 * t_wave) * deg
        angles[:, 2, 0] = (10.0 + 32.0 * t_wave) * deg
        angles[:, 2, 2] = (-10.0 - 20.0 * t_wave) * deg
        angles[:, 3, 0] = (10.0 + 38.0 * t_wave) * deg
        angles[:, 4, 0] = (8.0 + 48.0 * t_wave) * deg

    elif gesture_id == 4:
        # 5. Precision Opposition (Thumb-to-fingertip opposition targeting one
        #    of the four fingers while non-target digits stay semi-open)
        target_finger = int(rng.integers(0, 4))  # 0:Index, 1:Middle, 2:Ring, 3:Pinky
        opp = 0.35 + 0.60 * half_wave

        # Thumb moves deeper across palm for ring/pinky targets
        target_scale = 0.65 + 0.12 * target_finger
        angles[:, 1, :] = np.array([14.0, 12.0, 16.0]) * target_scale * deg
        angles[:, 2, 0] = (22.0 + 26.0 * opp * target_scale) * deg
        angles[:, 2, 1] = (16.0 + 12.0 * opp) * deg
        angles[:, 2, 2] = (-18.0 - 22.0 * opp * target_scale) * deg
        angles[:, 3, 0] = (18.0 + 28.0 * opp) * deg
        angles[:, 4, 0] = (16.0 + 32.0 * opp) * deg

        finger_triplets = [
            (6, 7, 8, -4.0),
            (10, 11, 12, 0.0),
            (14, 15, 16, 4.0),
            (18, 19, 20, 8.0),
        ]
        for f_idx, (mcp_j, pip_j, dip_j, abd) in enumerate(finger_triplets):
            if f_idx == target_finger:
                angles[:, mcp_j, 0] = (28.0 + 36.0 * opp) * deg
                angles[:, mcp_j, 2] = -abd * 0.5 * deg
                angles[:, pip_j, 0] = (34.0 + 42.0 * opp) * deg
                angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]
            else:
                dist = abs(f_idx - target_finger)
                symp = max(0.15, 0.45 - 0.12 * dist) * opp
                angles[:, mcp_j, 0] = (12.0 + 24.0 * symp) * deg
                angles[:, mcp_j, 2] = abd * deg
                angles[:, pip_j, 0] = (14.0 + 28.0 * symp) * deg
                angles[:, dip_j, 0] = 0.68 * angles[:, pip_j, 0]
        angles[:, 13, 0] = 5.0 * opp * deg
        angles[:, 17, 0] = 8.0 * opp * deg

    else:
        # 6. Flat Palpation (Open surgical palpation with gentle rhythmic
        #    MCP/PIP depression pulses in Index/Middle/Ring)
        pulse = half_wave
        angles[:, 1, 0] = 6.0 * deg
        angles[:, 2, 0] = (12.0 + 8.0 * pulse) * deg
        angles[:, 2, 2] = -12.0 * deg
        angles[:, 3, 0] = (10.0 + 8.0 * pulse) * deg
        angles[:, 4, 0] = (8.0 + 8.0 * pulse) * deg

        for mcp_j, pip_j, dip_j, p_scale in [
            (6, 7, 8, 1.0),
            (10, 11, 12, 1.1),
            (14, 15, 16, 0.9),
            (18, 19, 20, 0.6),
        ]:
            angles[:, mcp_j, 0] = (8.0 + 22.0 * p_scale * pulse) * deg
            angles[:, pip_j, 0] = (10.0 + 18.0 * p_scale * pulse) * deg
            angles[:, dip_j, 0] = 0.65 * angles[:, pip_j, 0]

    # Add smooth, small physiological micro-variation and clamp strictly to RoM limits
    micro_noise = rng.normal(0.0, 0.6 * deg, size=angles.shape).astype(np.float32)
    # Zero out noise on locked hinge axes (PIP/DIP abduction & twist)
    for hinge_j in [4, 7, 8, 11, 12, 15, 16, 19, 20]:
        micro_noise[:, hinge_j, 1:] = 0.0
    for palm_j in [5, 9]:
        micro_noise[:, palm_j, :] = 0.0

    angles = _clamp_to_anatomical_limits(angles + micro_noise)
    return angles


def download_freihand_xyz(raw_dir: Union[str, Path] = "data/raw") -> Optional[Path]:
    """
    Downloads lightweight FreiHAND 3D joint coordinate annotations into `data/raw/`.
    Tries public research mirrors for FreiHAND XYZ annotations.
    """
    import urllib.request
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_path = out_dir / "freihand_xyz.json"
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[dataset_ingestion] FreiHAND annotations already present at: {dest_path}")
        return dest_path

    # Public mirror URLs for FreiHAND evaluation/training 3D keypoints
    mirrors = [
        "https://raw.githubusercontent.com/lmb-freiburg/freihand/master/evaluation_xyz.json",
        "https://huggingface.co/datasets/datasets/resolve/main/freihand_evaluation_xyz.json",
    ]
    for url in mirrors:
        try:
            print(f"[dataset_ingestion] Attempting download from: {url}")
            urllib.request.urlretrieve(url, dest_path)
            if dest_path.exists() and dest_path.stat().st_size > 1000:
                print(f"[dataset_ingestion] Successfully downloaded FreiHAND annotations: {dest_path}")
                return dest_path
        except Exception as err:
            print(f"[dataset_ingestion] Mirror failed ({url}): {err}")

    print("[dataset_ingestion] Could not download FreiHAND automatically. Place 'freihand_xyz.json' in 'data/raw/'.")
    return None


def _try_load_real_annotations(
    raw_dir: Path,
    num_sequences: int,
    seq_len: int,
) -> Optional[np.ndarray]:
    """
    Checks `data/raw/` for FreiHAND or HO-3D 3D XYZ JSON annotations.
    Returns (N_real, seq_len, 21, 3) in millimeters if present, else None.
    """
    candidates = [
        raw_dir / "freihand_evaluation_xyz.json",
        raw_dir / "freihand_training_xyz.json",
        raw_dir / "freihand_xyz.json",
        raw_dir / "ho3d_xyz.json",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 100:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                arr = np.asarray(raw_data, dtype=np.float32)
                if arr.ndim == 3 and arr.shape[1:] == (NUM_JOINTS, 3):
                    # Convert meters to millimeters if values are small (< 2.0)
                    if np.max(np.abs(arr)) < 2.5:
                        arr = arr * 1000.0
                    # Center at wrist (joint 0)
                    arr = arr - arr[:, 0:1, :]
                    n_avail = arr.shape[0] // seq_len
                    if n_avail > 0:
                        n_take = min(n_avail, num_sequences)
                        return arr[: n_take * seq_len].reshape(
                            n_take, seq_len, NUM_JOINTS, 3
                        )
            except Exception:
                continue
    return None


def ingest_or_generate_dataset(
    data_dir: Union[str, Path] = "data",
    num_sequences: int = 1200,
    seq_len: int = 8,
    seed: int = 42,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Checks for real FreiHAND/HO-3D 3D annotations in `data/raw/` and runs the
    deterministic Biomechanical Articulated Hand Trajectory Generator to produce
    (N, T, 21, 3) ground-truth 3D positions in millimeters along with exact
    local rotation matrices (N, T, 21, 3, 3) and 6D rotations (N, T, 21, 6).

    Every ground-truth frame strictly obeys human joint angle limits and exact
    canonical bone lengths.

    Args:
        data_dir: Base directory for data storage (default: "data").
        num_sequences: Number of temporal hand sequences N (default: 1200).
        seq_len: Sequence length T in frames (default: 8).
        seed: Deterministic random seed (default: 42).
        force_regenerate: If True, regenerates even if cached .npz exists.

    Returns:
        Dictionary containing:
            - "positions_mm" / "xyz_mm": (N, T, 21, 3) float32 in mm
            - "rotations_mat" / "R_local": (N, T, 21, 3, 3) float32 in SO(3)
            - "rotations_6d" / "rot6d": (N, T, 21, 6) float32
            - "euler_angles_rad" / "joint_angles": (N, T, 21, 3) float32
            - "gesture_labels": (N,) int64 in 0..5
            - "gesture_names": List[str]
            - "bone_lengths_mm": (21,) float32
            - "rest_directions": (21, 3) float32
            - "parent_indices": (21,) int64
    """
    base_dir = Path(data_dir)
    raw_dir = base_dir / "raw"
    proc_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    cache_file = proc_dir / f"biomechanical_hand_N{num_sequences}_T{seq_len}_seed{seed}.npz"
    if cache_file.exists() and not force_regenerate:
        try:
            loaded = np.load(cache_file, allow_pickle=False)
            positions_mm = loaded["positions_mm"].astype(np.float32)
            rotations_mat = loaded["rotations_mat"].astype(np.float32)
            rotations_6d = loaded["rotations_6d"].astype(np.float32)
            euler_angles_rad = loaded["euler_angles_rad"].astype(np.float32)
            gesture_labels = loaded["gesture_labels"].astype(np.int64)
            if positions_mm.shape == (num_sequences, seq_len, NUM_JOINTS, 3):
                return {
                    "positions_mm": positions_mm,
                    "xyz_mm": positions_mm,
                    "joints_3d": positions_mm,
                    "rotations_mat": rotations_mat,
                    "R_local": rotations_mat,
                    "rotations_6d": rotations_6d,
                    "rot6d": rotations_6d,
                    "euler_angles_rad": euler_angles_rad,
                    "joint_angles": euler_angles_rad,
                    "gesture_labels": gesture_labels,
                    "gesture_names": list(GESTURE_NAMES),
                    "bone_lengths_mm": CANONICAL_BONE_LENGTHS_MM.copy(),
                    "rest_directions": REST_BONE_DIRECTIONS.copy(),
                    "parent_indices": np.array(PARENT_INDICES, dtype=np.int64),
                }
        except Exception:
            pass

    real_annotations = _try_load_real_annotations(raw_dir, num_sequences=num_sequences, seq_len=seq_len)
    if real_annotations is not None and real_annotations.shape[0] >= 10:
        import torch
        from models.kinematics import positions_to_rot6d, rot6d_to_matrix

        print(f"[dataset_ingestion] Loaded {real_annotations.shape[0]} real 3D hand sequences from {raw_dir}")
        t_real = torch.from_numpy(real_annotations)
        rot6d_t = positions_to_rot6d(t_real)
        rot_mat_t = rot6d_to_matrix(rot6d_t)
        rotations_mat = rot_mat_t.numpy()
        rotations_6d = rot6d_t.numpy()
        positions_mm = forward_kinematics_numpy(
            rotations_mat,
            bone_lengths_mm=CANONICAL_BONE_LENGTHS_MM,
            rest_directions=REST_BONE_DIRECTIONS,
        )
        all_euler = np.zeros_like(positions_mm)
        gesture_labels = np.zeros((real_annotations.shape[0],), dtype=np.int64)
        data_source = "real_annotations"
    else:
        rng = np.random.default_rng(seed)
        all_euler = np.zeros((num_sequences, seq_len, NUM_JOINTS, 3), dtype=np.float32)
        gesture_labels = np.zeros((num_sequences,), dtype=np.int64)

        num_gestures = len(GESTURE_NAMES)
        for i in range(num_sequences):
            g_id = i % num_gestures
            gesture_labels[i] = g_id
            all_euler[i] = _synthesize_single_sequence_angles(g_id, seq_len, rng)

        rotations_mat = euler_xyz_to_matrix(all_euler)       # (N, T, 21, 3, 3)
        rotations_6d = matrix_to_rot6d(rotations_mat)        # (N, T, 21, 6)

        positions_mm = forward_kinematics_numpy(
            rotations_mat,
            bone_lengths_mm=CANONICAL_BONE_LENGTHS_MM,
            rest_directions=REST_BONE_DIRECTIONS,
        )
        data_source = "procedural_mano_simulation"

    # Cache to disk for fast subsequent loads
    try:
        np.savez_compressed(
            cache_file,
            positions_mm=positions_mm,
            rotations_mat=rotations_mat,
            rotations_6d=rotations_6d,
            euler_angles_rad=all_euler,
            gesture_labels=gesture_labels,
            data_source=data_source,
        )
    except Exception:
        pass

    return {
        "positions_mm": positions_mm,
        "xyz_mm": positions_mm,
        "joints_3d": positions_mm,
        "rotations_mat": rotations_mat,
        "R_local": rotations_mat,
        "rotations_6d": rotations_6d,
        "rot6d": rotations_6d,
        "euler_angles_rad": all_euler,
        "joint_angles": all_euler,
        "gesture_labels": gesture_labels,
        "gesture_names": list(GESTURE_NAMES),
        "bone_lengths_mm": CANONICAL_BONE_LENGTHS_MM.copy(),
        "rest_directions": REST_BONE_DIRECTIONS.copy(),
        "parent_indices": np.array(PARENT_INDICES, dtype=np.int64),
        "data_source": data_source,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest or generate canonical MANO hand datasets.")
    parser.add_argument("--download-freihand", action="store_true", help="Download official FreiHAND 3D annotations into data/raw/.")
    parser.add_argument("--num-sequences", type=int, default=1200, help="Number of sequences to generate.")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length in frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-regenerate", action="store_true", help="Force dataset regeneration.")
    args = parser.parse_args()

    if args.download_freihand:
        download_freihand_xyz("data/raw")

    ds = ingest_or_generate_dataset(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
        force_regenerate=args.force_regenerate,
    )
    pos = ds["positions_mm"]
    p_idx = PARENT_INDICES
    bone_lens = [
        np.linalg.norm(pos[:, :, j, :] - pos[:, :, p_idx[j], :], axis=-1).mean()
        for j in range(1, NUM_JOINTS)
    ]
    print(f"Dataset shape: {pos.shape} | Source: {ds.get('data_source', 'unknown')}")
    print(f"Max bone length error (mm): {np.max(np.abs(np.array(bone_lens) - CANONICAL_BONE_LENGTHS_MM[1:])):.6e}")
