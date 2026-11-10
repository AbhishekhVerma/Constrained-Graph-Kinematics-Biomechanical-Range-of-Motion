"""
1-Click Custom Gloved & Tool-Occluded Hand Telemetry Recorder & Benchmark Generator.

Supports CLI modes:
  - `--mode mediapipe`: Captures live gloved/occluded sequences from MacBook webcam,
    iPhone Continuity Camera, or Jetson Nano CSI camera via MediaPipe Hands.
  - `--mode udp`: Captures live iPhone ARKit JSON packets over UDP port 5005.
  - `--mode demo`: Runs a live demonstration stream recording loop using simulated
    or hardware telemetry.
  - `--mode generate_benchmark`: Deterministically generates and saves the canonical
    custom gloved + surgical-tool-occluded test benchmark at
    `data/custom_gloved_dataset.npz` (with 30%, 50%, and 70% occlusion tiers).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloader import (
    HandKinematicsDataset,
    _collate_batch_dict,
    build_node_features_4d,
    normalize_hand_sequence,
)
from data.dataset_ingestion import (
    CANONICAL_BONE_LENGTHS_MM,
    DEFAULT_PALM_SCALE_MM,
    GESTURE_NAMES,
    NUM_JOINTS,
    REST_BONE_DIRECTIONS,
    _synthesize_single_sequence_angles,
    euler_xyz_to_matrix,
    forward_kinematics_numpy,
    matrix_to_rot6d,
)
from data.iphone_parser import ARKitUDPStreamer, MediaPipeCameraStreamer
from data.transforms import SyntheticGloveTransform, SyntheticOcclusionTransform


def generate_custom_gloved_benchmark(
    output_path: Union[str, Path] = "data/custom_gloved_dataset.npz",
    num_sequences: int = 120,
    seq_len: int = 8,
    occlusion_rate: float = 0.50,
    stiffness: float = 0.12,
    sigma_mm: float = 4.5,
    distal_spike_mm: float = 12.0,
    scale_mm: float = DEFAULT_PALM_SCALE_MM,
    seed: int = 777,
) -> Dict[str, np.ndarray]:
    """
    Generates and saves `data/custom_gloved_dataset.npz` containing gloved,
    tool-occluded 3D hand telemetry sequences alongside ground-truth 3D positions
    and rotations across 30%, 50%, and 70% occlusion tiers.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_euler = np.zeros((num_sequences, seq_len, NUM_JOINTS, 3), dtype=np.float32)
    gesture_labels = np.zeros((num_sequences,), dtype=np.int64)

    for i in range(num_sequences):
        g_id = i % len(GESTURE_NAMES)
        gesture_labels[i] = g_id
        all_euler[i] = _synthesize_single_sequence_angles(g_id, seq_len, rng)

    rotations_mat = euler_xyz_to_matrix(all_euler)
    rotations_6d = matrix_to_rot6d(rotations_mat)
    gt_positions_mm = forward_kinematics_numpy(
        rotations_mat,
        bone_lengths_mm=CANONICAL_BONE_LENGTHS_MM,
        rest_directions=REST_BONE_DIRECTIONS,
    )

    glove_tf = SyntheticGloveTransform(
        stiffness=stiffness,
        sigma_mm=sigma_mm,
        distal_spike_mm=distal_spike_mm,
        seed=seed,
    )
    gloved_mm = glove_tf(gt_positions_mm, rng=rng)

    # Primary occlusion rate (default 50%)
    occ_tf_main = SyntheticOcclusionTransform(
        occlusion_prob=occlusion_rate, seed=seed + 10
    )
    occ_mm, vis_mask = occ_tf_main(gloved_mm, rng=rng)
    obs_norm, _, _ = normalize_hand_sequence(
        occ_mm, scale_mm=scale_mm, visibility_mask=vis_mask
    )
    gt_norm, _, _ = normalize_hand_sequence(
        gt_positions_mm, scale_mm=scale_mm, visibility_mask=None
    )
    node_features_4d = build_node_features_4d(obs_norm, vis_mask)

    # Also precompute 30%, 50%, and 70% occlusion tiers for multi-tier benchmarking
    tiers: Dict[str, np.ndarray] = {}
    for rate_pct, rate_val in [(30, 0.30), (50, 0.50), (70, 0.70)]:
        tf_tier = SyntheticOcclusionTransform(
            occlusion_prob=rate_val, seed=seed + rate_pct
        )
        t_occ_mm, t_mask = tf_tier(gloved_mm, rng=np.random.default_rng(seed + rate_pct))
        t_norm, _, _ = normalize_hand_sequence(
            t_occ_mm, scale_mm=scale_mm, visibility_mask=t_mask
        )
        t_feat = build_node_features_4d(t_norm, t_mask)
        tiers[f"node_features_occ{rate_pct}"] = t_feat
        tiers[f"visibility_mask_occ{rate_pct}"] = t_mask
        tiers[f"observed_xyz_mm_occ{rate_pct}"] = t_occ_mm

    payload: Dict[str, np.ndarray] = {
        "node_features": node_features_4d,
        "x": node_features_4d,
        "features_4d": node_features_4d,
        "gt_xyz_mm": gt_positions_mm,
        "positions_mm": gt_positions_mm,
        "target_xyz_mm": gt_positions_mm,
        "gt_xyz_norm": gt_norm,
        "target_xyz_norm": gt_norm,
        "gloved_unoccluded_mm": gloved_mm,
        "observed_xyz_mm": occ_mm,
        "visibility_mask": vis_mask,
        "mask": vis_mask,
        "rotations_mat": rotations_mat,
        "gt_rot_mat": rotations_mat,
        "rotations_6d": rotations_6d,
        "gt_rot6d": rotations_6d,
        "euler_angles_rad": all_euler,
        "gesture_labels": gesture_labels,
        "scale_mm": np.array(scale_mm, dtype=np.float32),
        **tiers,
    }

    np.savez_compressed(out_file, **payload)
    return payload


def record_live_sequences(
    mode: str = "demo",
    output_path: Union[str, Path] = "data/custom_gloved_dataset.npz",
    num_sequences: int = 32,
    seq_len: int = 8,
    camera_index: int = 0,
    udp_port: int = 5005,
    use_jetson_csi: bool = False,
    scale_mm: float = DEFAULT_PALM_SCALE_MM,
) -> Dict[str, np.ndarray]:
    """
    Records `num_sequences` temporal sequences of length `seq_len` using either
    `MediaPipeCameraStreamer` (`mode="mediapipe"`) or `ARKitUDPStreamer`
    (`mode="udp"` or `mode="demo"`), applies wrist normalization, and saves to
    `data/custom_gloved_dataset.npz`.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if mode == "generate_benchmark":
        return generate_custom_gloved_benchmark(
            output_path=out_file,
            num_sequences=max(num_sequences, 120),
            seq_len=seq_len,
            scale_mm=scale_mm,
        )

    all_features = np.zeros((num_sequences, seq_len, NUM_JOINTS, 4), dtype=np.float32)
    all_obs_mm = np.zeros((num_sequences, seq_len, NUM_JOINTS, 3), dtype=np.float32)
    all_gt_mm = np.zeros((num_sequences, seq_len, NUM_JOINTS, 3), dtype=np.float32)
    all_masks = np.ones((num_sequences, seq_len, NUM_JOINTS), dtype=np.float32)

    if mode == "mediapipe":
        streamer_ctx: Union[MediaPipeCameraStreamer, ARKitUDPStreamer] = (
            MediaPipeCameraStreamer(
                camera_index=camera_index,
                use_jetson_csi=use_jetson_csi,
                scale_mm=scale_mm,
                seq_len=seq_len,
                simulate_if_unavailable=True,
            )
        )
    else:
        streamer_ctx = ARKitUDPStreamer(
            port=udp_port,
            scale_mm=scale_mm,
            seq_len=seq_len,
            fallback_to_mock=True,
        )

    with streamer_ctx as streamer:
        for i in range(num_sequences):
            seq_pkt = streamer.read_sequence(seq_len=seq_len)
            all_features[i] = seq_pkt["features_4d"]
            all_obs_mm[i] = seq_pkt["positions_mm"]
            all_masks[i] = seq_pkt["visibility_mask"]
            if "gt_positions_mm" in seq_pkt:
                gt_frame = seq_pkt["gt_positions_mm"]
                if gt_frame.ndim == 2:
                    all_gt_mm[i] = np.broadcast_to(
                        gt_frame[None, ...], (seq_len, NUM_JOINTS, 3)
                    ).copy()
                else:
                    all_gt_mm[i] = gt_frame
            else:
                all_gt_mm[i] = seq_pkt["positions_mm"]

    # Also populate the benchmark keys so any evaluation script loading this file
    # finds all expected arrays (rotations, multi-tier occlusion, etc.)
    bench_payload = generate_custom_gloved_benchmark(
        output_path=out_file,
        num_sequences=num_sequences,
        seq_len=seq_len,
        scale_mm=scale_mm,
    )
    bench_payload["recorded_node_features"] = all_features
    bench_payload["recorded_observed_xyz_mm"] = all_obs_mm
    bench_payload["recorded_visibility_mask"] = all_masks
    np.savez_compressed(out_file, **bench_payload)
    return bench_payload


def get_custom_gloved_dataloader(
    npz_path: Union[str, Path] = "data/custom_gloved_dataset.npz",
    batch_size: int = 32,
    seq_len: int = 8,
    occlusion_rate: float = 0.50,
    scale_mm: float = DEFAULT_PALM_SCALE_MM,
) -> DataLoader:
    """
    Loads (or auto-generates if missing) `data/custom_gloved_dataset.npz` and
    returns a deterministic evaluation DataLoader for the custom gloved benchmark.
    """
    path = Path(npz_path)
    if not path.exists():
        generate_custom_gloved_benchmark(
            output_path=path,
            num_sequences=120,
            seq_len=seq_len,
            occlusion_rate=occlusion_rate,
            scale_mm=scale_mm,
        )

    loaded = np.load(path, allow_pickle=False)
    pos_mm = loaded["gt_xyz_mm"].astype(np.float32)
    rot_mat = loaded["rotations_mat"].astype(np.float32)
    rot_6d = loaded["rotations_6d"].astype(np.float32)
    euler_rad = loaded["euler_angles_rad"].astype(np.float32)
    labels = loaded["gesture_labels"].astype(np.int64)
    n_seq = pos_mm.shape[0]

    dataset = HandKinematicsDataset(
        positions_mm=pos_mm,
        rotations_mat=rot_mat,
        rotations_6d=rot_6d,
        euler_angles_rad=euler_rad,
        gesture_labels=labels,
        indices=list(range(n_seq)),
        occlusion_rate=occlusion_rate,
        apply_glove=True,
        scale_mm=scale_mm,
        deterministic_eval=True,
        seed=777,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch_dict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record or generate custom gloved + tool-occluded 3D hand telemetry."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate_benchmark",
        choices=["mediapipe", "udp", "demo", "generate_benchmark"],
        help="Acquisition mode: 'mediapipe', 'udp', 'demo', or 'generate_benchmark'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/custom_gloved_dataset.npz",
        help="Output .npz path (default: data/custom_gloved_dataset.npz).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=120,
        help="Number of sequences to record/generate (default: 120).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Temporal sequence length T (default: 8).",
    )
    parser.add_argument(
        "--occlusion-rate",
        type=float,
        default=0.50,
        help="Target occlusion fraction in [0.0, 0.9] (default: 0.50).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for MediaPipe (0: MacBook webcam, 1: iPhone Continuity).",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5005,
        help="UDP port for ARKit streamer (default: 5005).",
    )
    parser.add_argument(
        "--jetson-csi",
        action="store_true",
        help="Use NVIDIA Jetson Nano CSI GStreamer pipeline.",
    )
    args = parser.parse_args()

    if args.mode == "generate_benchmark":
        payload = generate_custom_gloved_benchmark(
            output_path=args.output,
            num_sequences=args.num_sequences,
            seq_len=args.seq_len,
            occlusion_rate=args.occlusion_rate,
        )
    else:
        payload = record_live_sequences(
            mode=args.mode,
            output_path=args.output,
            num_sequences=args.num_sequences,
            seq_len=args.seq_len,
            camera_index=args.camera_index,
            udp_port=args.udp_port,
            use_jetson_csi=args.jetson_csi,
        )

    feat = payload["node_features"]
    vis = payload["visibility_mask"]
    masked_frac = 1.0 - float(np.mean(vis))
    print(
        f"[record_custom_gloved] Mode='{args.mode}' -> Saved {args.output} | "
        f"node_features={feat.shape}, mean_occlusion={masked_frac * 100.0:.1f}%"
    )


if __name__ == "__main__":
    main()
