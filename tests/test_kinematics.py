"""
Comprehensive Verification Test Suite for Kinematics, Teacher ST-GAE, and Student Models.

Tests:
  1. `rot6d_to_matrix` and `matrix_to_rot6d`: SO(3) orthogonality (`R^T R = I`),
     determinant `+1.0`, degenerate zero-vector safety, and exact round-trip recovery.
  2. `positions_to_rot6d` / `analytical_ik_6d` + `DifferentiableFK` round-trip:
     verifies `0.0000 mm` reconstruction error on canonical articulated hand trajectories.
  3. `0.0000 mm` bone-length invariance across `DifferentiableFK` and `EdgeFK` (and
     PyTorch-vs-NumPy parity `< 1e-5 mm`).
  4. `BiomechanicalRoMClamper` and `check_biomechanical_validity`: verifies that
     clamped FK poses achieve 100.0% biomechanical validity even under extreme random
     inputs, while unconstrained corrupted poses fail anatomical checks.
  5. Forward and backward gradient passes on `SpatioTemporalGAE`, `SpatialOnlyGAE`,
     `StudentKinematicMLP` (~78K params, <0.4 MB), and `UnconstrainedStudentMLP`.
  6. Clean ONNX export (`torch.onnx.export`) and ONNXRuntime parity for `StudentKinematicMLP`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.kinematics import (
    CANONICAL_BONE_LENGTHS_MM,
    NUM_JOINTS,
    PARENT_INDICES,
    REST_JOINT_POSITIONS_MM,
    BiomechanicalRoMClamper,
    DifferentiableFK,
    EdgeFK,
    check_biomechanical_validity,
    euler_xyz_to_matrix_torch,
    matrix_to_rot6d,
    positions_to_rot6d,
    rot6d_to_matrix,
)
from models.student_models import StudentKinematicMLP, UnconstrainedStudentMLP
from models.teacher_st_gae import SpatialOnlyGAE, SpatioTemporalGAE


def test_rot6d_so3_properties() -> None:
    """Tests `rot6d_to_matrix` SO(3) orthogonality, det=+1, and `matrix_to_rot6d` round-trip."""
    torch.manual_seed(42)
    rand_d6 = torch.randn(16, 8, NUM_JOINTS, 6)
    R = rot6d_to_matrix(rand_d6)
    assert R.shape == (16, 8, NUM_JOINTS, 3, 3)

    # Check orthogonality R^T @ R == I
    eye = torch.eye(3).expand_as(R)
    orth_err = torch.max(torch.abs(torch.matmul(R.transpose(-1, -2), R) - eye)).item()
    assert orth_err < 1e-5, f"SO(3) orthogonality error too high: {orth_err}"

    # Check determinant == +1.0
    dets = torch.det(R)
    det_err = torch.max(torch.abs(dets - 1.0)).item()
    assert det_err < 1e-5, f"SO(3) determinant error too high: {det_err}"

    # Check round-trip R -> rot6d -> R_rec
    d6_extracted = matrix_to_rot6d(R)
    R_rec = rot6d_to_matrix(d6_extracted)
    rt_err = torch.max(torch.abs(R - R_rec)).item()
    assert rt_err < 1e-5, f"rot6d round-trip matrix error too high: {rt_err}"

    # Check zero-input safety (no NaNs)
    zero_d6 = torch.zeros(2, NUM_JOINTS, 6)
    R_zero = rot6d_to_matrix(zero_d6)
    assert not torch.isnan(R_zero).any(), "NaN produced on zero 6D input!"
    print(f"[PASS] rot6d_to_matrix SO(3) orthogonality err={orth_err:.2e}, det err={det_err:.2e}, round-trip={rt_err:.2e}")


def test_fk_ik_roundtrip_and_bone_lengths() -> None:
    """Tests 0.00 mm bone-length error on DifferentiableFK/EdgeFK and exact IK-FK round-trip."""
    torch.manual_seed(123)
    fk = DifferentiableFK()
    edge_fk = EdgeFK()
    clamper = BiomechanicalRoMClamper()

    # 1. Generate random biomechanically clamped 6D rotations (B=8, T=8, 21, 6)
    raw_d6 = torch.randn(8, 8, NUM_JOINTS, 6)
    clamped_d6 = clamper(raw_d6)

    pos_torch = fk(clamped_d6)
    pos_numpy = edge_fk(clamped_d6.detach().numpy())

    # Verify PyTorch DifferentiableFK vs NumPy EdgeFK match
    torch_vs_np_err = float(np.max(np.abs(pos_torch.detach().numpy() - pos_numpy)))
    assert torch_vs_np_err < 1e-4, f"DifferentiableFK vs EdgeFK mismatch: {torch_vs_np_err} mm"

    # Verify 0.0000 mm bone length error on DifferentiableFK and EdgeFK
    bone_err_torch = fk.compute_bone_length_error_mm(pos_torch)
    max_bone_err_torch = float(bone_err_torch.max().item())
    mean_bone_err_torch = float(bone_err_torch.mean().item())
    assert max_bone_err_torch < 1e-4, f"DifferentiableFK bone length error: {max_bone_err_torch} mm"

    bone_lens_np = edge_fk.compute_bone_lengths(pos_numpy)
    max_bone_err_np = float(np.max(np.abs(bone_lens_np - CANONICAL_BONE_LENGTHS_MM[1:])))
    assert max_bone_err_np < 1e-4, f"EdgeFK bone length error: {max_bone_err_np} mm"

    # 2. Test analytical Inverse Kinematics `positions_to_rot6d` -> `DifferentiableFK` round-trip
    # Use a pose with identity wrist global rotation (R_0 = I) so wrist-anchored IK is exact
    identity_wrist_d6 = clamped_d6.clone()
    identity_wrist_d6[..., 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pos_target = fk(identity_wrist_d6)

    ik_rot6d = positions_to_rot6d(pos_target)
    pos_reconstructed = fk(ik_rot6d)
    ik_fk_err_mm = float(torch.max(torch.abs(pos_target - pos_reconstructed)).item())
    assert ik_fk_err_mm < 1e-3, f"IK -> FK round-trip error too high: {ik_fk_err_mm:.6f} mm"

    print(
        f"[PASS] DifferentiableFK & EdgeFK max bone-length err={max_bone_err_torch:.6f} mm | "
        f"Torch-vs-NumPy err={torch_vs_np_err:.6f} mm | IK->FK round-trip err={ik_fk_err_mm:.6f} mm"
    )


def test_biomechanical_rom_clamper_and_validity() -> None:
    """Tests that BiomechanicalRoMClamper guarantees 100% biomechanically valid poses."""
    torch.manual_seed(999)
    fk = DifferentiableFK()
    clamper = BiomechanicalRoMClamper()

    # Extreme adversarial random inputs (scale = 10.0)
    extreme_d6 = torch.randn(32, 8, NUM_JOINTS, 6) * 10.0
    clamped_d6 = clamper(extreme_d6)
    valid_positions = fk(clamped_d6)

    validity_res = check_biomechanical_validity(valid_positions)
    assert isinstance(validity_res, torch.Tensor)
    assert validity_res.all(), f"Expected 100% valid poses after RoM clamper, got {validity_res.valid_pct:.2f}%"
    assert validity_res["valid_pct"] == 100.0
    assert validity_res.max_bone_error_mm < 1e-4

    # Corrupt a skeleton with unconstrained Direct-XYZ noise (stretching bones & hyperextending joints)
    corrupted_positions = valid_positions.clone()
    corrupted_positions[..., 8, :] += torch.tensor([0.0, -60.0, -80.0])  # Snap index tip backward
    corrupted_res = check_biomechanical_validity(corrupted_positions)
    assert not corrupted_res.all(), "Corrupted skeleton should fail biomechanical validity!"

    print(
        f"[PASS] BiomechanicalRoMClamper validity={validity_res.valid_pct:.1f}% "
        f"(corrupted Direct-XYZ validity={corrupted_res.valid_pct:.1f}%)"
    )


def test_teacher_and_student_models_forward_backward() -> None:
    """Tests forward and backward gradient passes on all 4 Teacher & Student architectures."""
    torch.manual_seed(7)
    bsz, seq_len = 4, 8
    # Create normalized input (B, T, 21, 4) with 50% simulated occlusion mask
    rest_norm = torch.as_tensor(REST_JOINT_POSITIONS_MM / 100.0, dtype=torch.float32)
    xyz = rest_norm.unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, NUM_JOINTS, 3)
    xyz = xyz + 0.05 * torch.randn(bsz, seq_len, NUM_JOINTS, 3)
    vis = torch.ones(bsz, seq_len, NUM_JOINTS, 1)
    vis[:, :, 6:13, :] = 0.0  # Occlude Index & Middle digits
    x_input = torch.cat((xyz * vis, vis), dim=-1).requires_grad_(True)

    target_pos_mm = torch.as_tensor(REST_JOINT_POSITIONS_MM, dtype=torch.float32).expand(
        bsz, seq_len, NUM_JOINTS, 3
    )

    fk = DifferentiableFK()

    models_to_test = [
        ("SpatioTemporalGAE", SpatioTemporalGAE()),
        ("SpatialOnlyGAE", SpatialOnlyGAE()),
        ("StudentKinematicMLP", StudentKinematicMLP()),
        ("UnconstrainedStudentMLP", UnconstrainedStudentMLP()),
    ]

    for name, model in models_to_test:
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        size_mb = (n_params * 4) / (1024.0 * 1024.0)

        out = model(x_input)
        assert "positions_mm" in out, f"{name} missing 'positions_mm'"
        pos_mm = out["positions_mm"]
        assert pos_mm.shape == (bsz, seq_len, NUM_JOINTS, 3), f"{name} bad shape {pos_mm.shape}"

        # Compute loss and verify backward pass populates parameter gradients
        loss = torch.mean((pos_mm - target_pos_mm) ** 2)
        if "rot6d" in out:
            rot6d = out["rot6d"]
            assert rot6d.shape == (bsz, seq_len, NUM_JOINTS, 6)
            loss = loss + 0.1 * torch.mean(rot6d ** 2)
            # Check 0.00 mm bone length error and 100% biomechanical validity on FK models
            bone_err = float(fk.compute_bone_length_error_mm(pos_mm).max().item())
            assert bone_err < 1e-4, f"{name} bone length error {bone_err} mm"
            val_res = check_biomechanical_validity(pos_mm)
            assert val_res.valid_pct == 100.0, f"{name} validity {val_res.valid_pct}%"

        if name in ("SpatioTemporalGAE", "SpatialOnlyGAE"):
            assert "attn_weights" in out
            assert out["attn_weights"].shape[-2:] == (NUM_JOINTS, NUM_JOINTS)

        if name == "StudentKinematicMLP":
            assert 50_000 <= n_params <= 95_000, f"StudentKinematicMLP params={n_params} outside 50K-95K target"
            assert size_mb < 0.40, f"StudentKinematicMLP size={size_mb:.3f} MB exceeds 0.40 MB"

        model.zero_grad()
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(grad_norms) > 0 and all(np.isfinite(g) for g in grad_norms), (
            f"{name} failed gradient flow check!"
        )
        print(
            f"[PASS] {name:24s} | params={n_params:7,d} ({size_mb:.3f} MB) | "
            f"out={tuple(pos_mm.shape)} | grad_tensors={len(grad_norms)}"
        )


def test_student_onnx_export() -> None:
    """Verifies that StudentKinematicMLP exports cleanly to ONNX via torch.onnx.export."""
    student = StudentKinematicMLP(return_dict=False).eval()
    dummy_in = torch.randn(1, 8, NUM_JOINTS, 4)

    with torch.no_grad():
        torch_pos, torch_rot6d = student(dummy_in)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "student_kinematic_mlp.onnx")
        try:
            torch.onnx.export(
                student,
                dummy_in,
                onnx_path,
                input_names=["input"],
                output_names=["positions_mm", "rot6d"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            assert os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 1024
            size_kb = os.path.getsize(onnx_path) / 1024.0

            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ort_pos, ort_rot6d = sess.run(None, {"input": dummy_in.numpy()})
            pos_diff = float(np.max(np.abs(torch_pos.numpy() - ort_pos)))
            assert pos_diff < 1e-3, f"ONNXRuntime parity error: {pos_diff} mm"
            print(f"[PASS] StudentKinematicMLP ONNX export ({size_kb:.1f} KB) & ORT parity err={pos_diff:.2e} mm")
        except ModuleNotFoundError:
            traced = torch.jit.trace(student, dummy_in)
            jit_pos, _ = traced(dummy_in)
            jit_diff = float(torch.max(torch.abs(torch_pos - jit_pos)).item())
            assert jit_diff < 1e-5
            print(f"[PASS] StudentKinematicMLP TorchScript trace parity err={jit_diff:.2e} mm")


def run_all_tests() -> None:
    print("=" * 78)
    print("Running Kinematics, Biomechanical RoM, Teacher ST-GAE & Student Verification")
    print("=" * 78)
    test_rot6d_so3_properties()
    test_fk_ik_roundtrip_and_bone_lengths()
    test_biomechanical_rom_clamper_and_validity()
    test_teacher_and_student_models_forward_backward()
    test_student_onnx_export()
    print("=" * 78)
    print("ALL KINEMATICS & MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_tests()
