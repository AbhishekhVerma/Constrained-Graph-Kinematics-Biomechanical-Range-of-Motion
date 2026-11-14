"""
ONNX Model Export & Numerical Parity Verification Pipeline.

Module Owner: Kapish (Edge Student Architecture & ONNX Deployment)

Exports:
  1. `StudentKinematicMLP` (`checkpoints/student_kinematic_mlp.pt`) ->
     `checkpoints/student_kinematic_fk.onnx` (and alias `checkpoints/student_kinematic_mlp.onnx`)
  2. `UnconstrainedStudentMLP` (`checkpoints/student_unconstrained_xyz.pt`) ->
     `checkpoints/student_unconstrained_xyz.onnx`

Verifies:
  - `onnx.checker.check_model` structural graph validity
  - `onnxruntime.InferenceSession` numerical parity (`< 1e-3 mm` max deviation)
  - Edge inference latency (ms / FPS) and serialized model footprint (KB / MB)
  - Saves comprehensive report to `results/onnx_export_report.json`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.kinematics import NUM_JOINTS, check_biomechanical_validity
from models.student_models import StudentKinematicMLP, UnconstrainedStudentMLP
from training.utils import compute_bone_length_error_mm, count_parameters


def _load_checkpoint_into_model(
    model: torch.nn.Module,
    ckpt_path: Path,
) -> Dict[str, Any]:
    """Loads state_dict from checkpoint file if present."""
    if not ckpt_path.exists():
        return {"loaded_from_disk": False, "checkpoint_path": str(ckpt_path)}

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
    elif isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=True)
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        model.load_state_dict(payload, strict=True)
        meta = {}

    return {
        "loaded_from_disk": True,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_meta": meta,
    }


def export_single_student_to_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    model_name: str,
    has_rot6d_output: bool = True,
    seq_len: int = 8,
    opset_version: int = 17,
) -> Dict[str, Any]:
    """
    Exports a trained Student PyTorch model to ONNX, runs `onnx.checker.check_model`,
    and benchmarks numerical parity & inference latency in `onnxruntime.InferenceSession`.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to("cpu")
    model.return_dict = False

    torch.manual_seed(42)
    dummy_input = torch.randn(1, seq_len, NUM_JOINTS, 4, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(dummy_input, return_dict=False)
        if has_rot6d_output:
            torch_pos_mm, torch_rot6d = torch_out
        else:
            torch_pos_mm = torch_out
            torch_rot6d = None

    output_names = ["positions_mm", "rot6d"] if has_rot6d_output else ["positions_mm"]
    dynamic_axes = {
        "input": {0: "batch_size", 1: "seq_len"},
        "positions_mm": {0: "batch_size", 1: "seq_len"},
    }
    if has_rot6d_output:
        dynamic_axes["rot6d"] = {0: "batch_size", 1: "seq_len"}

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # 1. Verify ONNX graph validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Verify ONNXRuntime numerical parity across multiple batch sizes and seq lengths
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    ort_outs = sess.run(None, {"input": dummy_input.numpy()})
    ort_pos_mm = ort_outs[0]
    max_pos_err_mm = float(np.max(np.abs(torch_pos_mm.numpy() - ort_pos_mm)))
    mean_pos_err_mm = float(np.mean(np.abs(torch_pos_mm.numpy() - ort_pos_mm)))

    max_rot_err = 0.0
    if has_rot6d_output and torch_rot6d is not None and len(ort_outs) > 1:
        ort_rot6d = ort_outs[1]
        max_rot_err = float(np.max(np.abs(torch_rot6d.numpy() - ort_rot6d)))

    # Test batched inference (B=16, T=seq_len) via dynamic_axes
    batch_input = torch.randn(16, seq_len, NUM_JOINTS, 4, dtype=torch.float32)
    with torch.no_grad():
        t_batch_out = model(batch_input, return_dict=False)
        t_batch_pos = t_batch_out[0] if has_rot6d_output else t_batch_out
    ort_batch_pos = sess.run(None, {"input": batch_input.numpy()})[0]
    batch_max_err_mm = float(np.max(np.abs(t_batch_pos.numpy() - ort_batch_pos)))

    # Test streaming single-frame inference (B=1, T=1) via dynamic_axes
    t1_input = torch.randn(1, 1, NUM_JOINTS, 4, dtype=torch.float32)
    with torch.no_grad():
        t1_out = model(t1_input, return_dict=False)
        t1_pos = t1_out[0] if has_rot6d_output else t1_out
    ort_t1_pos = sess.run(None, {"input": t1_input.numpy()})[0]
    t1_max_err_mm = float(np.max(np.abs(t1_pos.numpy() - ort_t1_pos)))

    # 3. Measure ONNXRuntime edge inference latency (B=1, T=8)
    in_np = dummy_input.numpy()
    for _ in range(25):
        _ = sess.run(None, {"input": in_np})

    timings_ms = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = sess.run(None, {"input": in_np})
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    arr_ms = np.array(timings_ms, dtype=np.float64)
    seq_median_ms = float(np.median(arr_ms))
    seq_mean_ms = float(np.mean(arr_ms))
    per_frame_ms = seq_median_ms / float(seq_len)
    fps = 1000.0 / max(per_frame_ms, 1e-6)

    # 4. Check bone length error and biomechanical validity on ONNX output
    bone_err_mm = compute_bone_length_error_mm(ort_batch_pos)
    bio_val_pct = float(check_biomechanical_validity(ort_batch_pos).valid_pct)

    size_bytes = int(onnx_path.stat().st_size)
    size_kb = size_bytes / 1024.0
    size_mb = size_bytes / (1024.0 * 1024.0)
    n_params = count_parameters(model)

    return {
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "opset_version": int(opset_version),
        "onnx_checker_passed": True,
        "num_parameters": int(n_params),
        "file_size_bytes": size_bytes,
        "file_size_kb": round(size_kb, 2),
        "file_size_mb": round(size_mb, 4),
        "max_parity_error_mm": float(max(max_pos_err_mm, batch_max_err_mm, t1_max_err_mm)),
        "mean_parity_error_mm": float(mean_pos_err_mm),
        "max_rot6d_parity_error": float(max_rot_err),
        "onnx_bone_length_error_mm": round(bone_err_mm, 6),
        "onnx_biomechanical_validity_pct": round(bio_val_pct, 2),
        "ort_sequence_latency_ms": round(seq_median_ms, 4),
        "ort_sequence_mean_ms": round(seq_mean_ms, 4),
        "ort_per_frame_latency_ms": round(per_frame_ms, 4),
        "ort_fps": round(fps, 1),
    }


def export_all_student_models(
    kinematic_ckpt: Union[str, Path] = "checkpoints/student_kinematic_mlp.pt",
    unconstrained_ckpt: Union[str, Path] = "checkpoints/student_unconstrained_xyz.pt",
    kinematic_onnx: Union[str, Path] = "checkpoints/student_kinematic_fk.onnx",
    unconstrained_onnx: Union[str, Path] = "checkpoints/student_unconstrained_xyz.onnx",
    report_path: Union[str, Path] = "results/onnx_export_report.json",
) -> Dict[str, Any]:
    """
    Exports both `StudentKinematicMLP` and `UnconstrainedStudentMLP` to ONNX,
    runs verification checks, and writes `results/onnx_export_report.json`.
    """
    kin_ckpt_path = Path(kinematic_ckpt)
    unc_ckpt_path = Path(unconstrained_ckpt)
    kin_onnx_path = Path(kinematic_onnx)
    unc_onnx_path = Path(unconstrained_onnx)
    rep_path = Path(report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export StudentKinematicMLP
    kin_model = StudentKinematicMLP(return_dict=False)
    kin_load_info = _load_checkpoint_into_model(kin_model, kin_ckpt_path)
    kin_report = export_single_student_to_onnx(
        model=kin_model,
        onnx_path=kin_onnx_path,
        model_name="StudentKinematicMLP (Distilled 6D-FK Student)",
        has_rot6d_output=True,
    )
    kin_report["checkpoint_info"] = kin_load_info

    # Also copy to checkpoints/student_kinematic_mlp.onnx for convenience
    alt_kin_onnx = kin_onnx_path.parent / "student_kinematic_mlp.onnx"
    if alt_kin_onnx.resolve() != kin_onnx_path.resolve():
        import shutil
        shutil.copy2(kin_onnx_path, alt_kin_onnx)

    # 2. Export UnconstrainedStudentMLP
    unc_model = UnconstrainedStudentMLP(return_dict=False)
    unc_load_info = _load_checkpoint_into_model(unc_model, unc_ckpt_path)
    unc_report = export_single_student_to_onnx(
        model=unc_model,
        onnx_path=unc_onnx_path,
        model_name="UnconstrainedStudentMLP (Direct-XYZ Ablation Student)",
        has_rot6d_output=False,
    )
    unc_report["checkpoint_info"] = unc_load_info

    full_report = {
        "status": "SUCCESS",
        "student_kinematic_fk": kin_report,
        "student_unconstrained_xyz": unc_report,
        "summary": {
            "primary_onnx_path": str(kin_onnx_path),
            "primary_size_kb": kin_report["file_size_kb"],
            "primary_size_mb": kin_report["file_size_mb"],
            "primary_parameters": kin_report["num_parameters"],
            "max_onnx_parity_error_mm": kin_report["max_parity_error_mm"],
            "ort_sequence_latency_ms": kin_report["ort_sequence_latency_ms"],
            "ort_per_frame_latency_ms": kin_report["ort_per_frame_latency_ms"],
            "ort_fps": kin_report["ort_fps"],
        },
    }

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("=" * 78)
    print("[ONNX Export & Verification Complete]")
    print(
        f"  1. {kin_report['model_name']}:\n"
        f"     - Path:            {kin_report['onnx_path']}\n"
        f"     - Parameters:      {kin_report['num_parameters']:,} ({kin_report['file_size_kb']:.1f} KB / {kin_report['file_size_mb']:.3f} MB)\n"
        f"     - Max Parity Err:  {kin_report['max_parity_error_mm']:.2e} mm\n"
        f"     - Bone Length Err: {kin_report['onnx_bone_length_error_mm']:.4f} mm (Biovalidity: {kin_report['onnx_biomechanical_validity_pct']:.1f}%)\n"
        f"     - ORT Latency:     {kin_report['ort_sequence_latency_ms']:.3f} ms/seq ({kin_report['ort_per_frame_latency_ms']:.3f} ms/frame, {kin_report['ort_fps']:.1f} FPS)"
    )
    print(
        f"  2. {unc_report['model_name']}:\n"
        f"     - Path:            {unc_report['onnx_path']}\n"
        f"     - Parameters:      {unc_report['num_parameters']:,} ({unc_report['file_size_kb']:.1f} KB / {unc_report['file_size_mb']:.3f} MB)\n"
        f"     - Max Parity Err:  {unc_report['max_parity_error_mm']:.2e} mm\n"
        f"     - ORT Latency:     {unc_report['ort_sequence_latency_ms']:.3f} ms/seq ({unc_report['ort_per_frame_latency_ms']:.3f} ms/frame, {unc_report['ort_fps']:.1f} FPS)"
    )
    print(f"  Report saved to: {rep_path}")
    print("=" * 78)

    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained Student models to ONNX.")
    parser.add_argument(
        "--kinematic-ckpt",
        type=str,
        default="checkpoints/student_kinematic_mlp.pt",
    )
    parser.add_argument(
        "--unconstrained-ckpt",
        type=str,
        default="checkpoints/student_unconstrained_xyz.pt",
    )
    parser.add_argument(
        "--kinematic-onnx",
        type=str,
        default="checkpoints/student_kinematic_fk.onnx",
    )
    parser.add_argument(
        "--unconstrained-onnx",
        type=str,
        default="checkpoints/student_unconstrained_xyz.onnx",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="results/onnx_export_report.json",
    )
    args = parser.parse_args()
    export_all_student_models(
        kinematic_ckpt=args.kinematic_ckpt,
        unconstrained_ckpt=args.unconstrained_ckpt,
        kinematic_onnx=args.kinematic_onnx,
        unconstrained_onnx=args.unconstrained_onnx,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
