"""
Knowledge Distillation and ONNX/TensorRT/CoreML Export Routines.

Module Owner: Kapish (Edge Student Architecture, Distillation & ONNX Deployment)
"""

from distillation.distill_student import (
    distill_student_models,
    load_frozen_teacher,
    precompute_teacher_targets,
    train_student_kinematic_mlp,
    train_unconstrained_student_mlp,
)
from distillation.export_models import (
    export_all_student_models,
    export_single_student_to_onnx,
)

__all__ = [
    "distill_student_models",
    "load_frozen_teacher",
    "precompute_teacher_targets",
    "train_student_kinematic_mlp",
    "train_unconstrained_student_mlp",
    "export_all_student_models",
    "export_single_student_to_onnx",
]
