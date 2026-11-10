"""
Kinematics, Teacher Spatio-Temporal GAE, and Student MLP Models.

Exposes all primary classes and utilities across:
  - `models.kinematics` (Vrushank's Module)
  - `models.teacher_st_gae` (Abhishekh's Module)
  - `models.student_models` (Kapish's Module)
"""

from models.kinematics import (
    ANATOMICAL_MAX_LIMITS_RAD,
    ANATOMICAL_MIN_LIMITS_RAD,
    CANONICAL_BONE_LENGTHS_MM,
    JOINT_NAMES,
    NUM_BONES,
    NUM_JOINTS,
    PARENT_INDICES,
    REST_BONE_DIRECTIONS,
    REST_BONE_VECTORS_MM,
    REST_JOINT_POSITIONS_MM,
    BiomechanicalRoMClamper,
    BiomechanicalValidityResult,
    DifferentiableFK,
    EdgeFK,
    check_biomechanical_validity,
    euler_xyz_to_matrix_torch,
    matrix_to_euler_xyz_torch,
    matrix_to_rot6d,
    positions_to_rot6d,
    rot6d_to_matrix,
)
from models.student_models import (
    ResidualMLPBlock,
    StudentKinematicMLP,
    TemporalMixingBlock1D,
    UnconstrainedStudentMLP,
)
from models.teacher_st_gae import (
    MultiHeadGraphAttentionLayer,
    SpatialOnlyGAE,
    SpatioTemporalGAE,
)

__all__ = [
    "NUM_JOINTS",
    "NUM_BONES",
    "JOINT_NAMES",
    "PARENT_INDICES",
    "CANONICAL_BONE_LENGTHS_MM",
    "REST_BONE_DIRECTIONS",
    "REST_BONE_VECTORS_MM",
    "REST_JOINT_POSITIONS_MM",
    "ANATOMICAL_MIN_LIMITS_RAD",
    "ANATOMICAL_MAX_LIMITS_RAD",
    "rot6d_to_matrix",
    "matrix_to_rot6d",
    "euler_xyz_to_matrix_torch",
    "matrix_to_euler_xyz_torch",
    "BiomechanicalRoMClamper",
    "BiomechanicalValidityResult",
    "check_biomechanical_validity",
    "positions_to_rot6d",
    "DifferentiableFK",
    "EdgeFK",
    "MultiHeadGraphAttentionLayer",
    "SpatioTemporalGAE",
    "SpatialOnlyGAE",
    "TemporalMixingBlock1D",
    "ResidualMLPBlock",
    "StudentKinematicMLP",
    "UnconstrainedStudentMLP",
]
