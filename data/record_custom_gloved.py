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
    ingest_or_generate_dataset,
    matrix_to_rot6d,
)
from data.iphone_parser import ARKitUDPStreamer, MediaPipeCameraStreamer
from data.transforms import SyntheticGloveTransform, SyntheticOcclusionTransform
from models.kinematics import (
    BiomechanicalRoMClamper,
    DifferentiableFK,
    positions_to_rot6d,
    rot6d_to_matrix,
)


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
    ds_hybrid = ingest_or_generate_dataset(
        data_dir=out_file.parent,
        num_sequences=max(num_sequences, 240),
        seq_len=seq_len,
        seed=seed,
        force_regenerate=False,
    )
    pick_idx = rng.choice(ds_hybrid["positions_mm"].shape[0], size=num_sequences, replace=False)
    all_euler = ds_hybrid["euler_angles_rad"][pick_idx].astype(np.float32)
    gesture_labels = ds_hybrid["gesture_labels"][pick_idx].astype(np.int64)
    rotations_mat = ds_hybrid["rotations_mat"][pick_idx].astype(np.float32)
    rotations_6d = ds_hybrid["rotations_6d"][pick_idx].astype(np.float32)
    gt_positions_mm = ds_hybrid["positions_mm"][pick_idx].astype(np.float32)

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
        "is_real_recording": np.array(False),
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
    simulate_if_unavailable: bool = False,
    show_preview: bool = True,
) -> Dict[str, Any]:
    """
    Records `num_sequences` temporal sequences of length `seq_len` using either
    `MediaPipeCameraStreamer` (`mode="mediapipe"`) or `ARKitUDPStreamer`
    (`mode="udp"` or `mode="demo"`), applies wrist normalization, harmonizes
    ground-truth reference trajectories onto exact canonical bone lengths via
    `positions_to_rot6d -> BiomechanicalRoMClamper -> DifferentiableFK`, and saves
    directly to `data/custom_gloved_dataset.npz`.
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
    all_raw_unmasked_mm = np.zeros((num_sequences, seq_len, NUM_JOINTS, 3), dtype=np.float32)
    all_masks = np.ones((num_sequences, seq_len, NUM_JOINTS), dtype=np.float32)

    if mode == "mediapipe":
        streamer_ctx: Union[MediaPipeCameraStreamer, ARKitUDPStreamer] = (
            MediaPipeCameraStreamer(
                camera_index=camera_index,
                use_jetson_csi=use_jetson_csi,
                scale_mm=scale_mm,
                seq_len=seq_len,
                simulate_if_unavailable=simulate_if_unavailable,
                simulated_occlusion_rate=0.0,  # Capture real camera observations without artificial random dropout
            )
        )
    else:
        streamer_ctx = ARKitUDPStreamer(
            port=udp_port,
            scale_mm=scale_mm,
            seq_len=seq_len,
            fallback_to_mock=simulate_if_unavailable,
        )

    any_real_hardware_frame = False
    gesture_prompts = [
        "1. Surgical Scalpel / Forceps Pinch",
        "2. Needle Driver Wrist & Finger Rotation",
        "3. Power Drill / Handle Grasp",
        "4. Sequential Finger Flexion Wave",
        "5. Precision Thumb-to-Tip Opposition",
        "6. Flat Palpation & Finger Spread",
    ]

    with streamer_ctx as streamer:
        # If in MediaPipe camera mode with preview enabled, let the user see their gloved hand first
        if mode == "mediapipe" and show_preview and not getattr(streamer, "_is_simulated", True):
            try:
                import cv2
                print("\n[record_custom_gloved] Camera open! Place your gloved hand in view. Recording starts in 3 seconds (or press SPACE to start immediately)...")
                warmup_start = torch.cuda.Event if False else None
                import time
                t_start = time.perf_counter()
                while (time.perf_counter() - t_start) < 3.0:
                    pkt_w = streamer.read_frame()
                    fbgr = pkt_w.get("frame_bgr")
                    if fbgr is not None:
                        disp = fbgr.copy()
                        rem = max(0.0, 3.0 - (time.perf_counter() - t_start))
                        cv2.rectangle(disp, (10, 10), (disp.shape[1] - 10, 72), (15, 23, 42), -1)
                        cv2.putText(
                            disp,
                            f"GLOVED HAND RECORDER - Starting in {rem:.1f}s (Keep Gloved Hand in Frame)",
                            (22, 46),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (52, 211, 153),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Gloved Hand Telemetry Recorder (Press Q to Abort)", disp)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord(" "):
                            break
            except Exception:
                pass

        seq_idx = 0
        while seq_idx < num_sequences:
            prompt_str = gesture_prompts[seq_idx % len(gesture_prompts)]
            frames_captured = 0
            while frames_captured < seq_len:
                f_pkt = streamer.read_frame()
                is_sim = bool(f_pkt.get("simulated", True))
                hand_det = bool(f_pkt.get("hand_detected", not is_sim))
                if not is_sim:
                    any_real_hardware_frame = True

                fbgr = f_pkt.get("frame_bgr")
                if mode == "mediapipe" and show_preview and fbgr is not None:
                    try:
                        import cv2
                        disp = fbgr.copy()
                        cv2.rectangle(disp, (10, 10), (disp.shape[1] - 10, 82), (15, 23, 42), -1)
                        cv2.rectangle(disp, (10, 10), (disp.shape[1] - 10, 82), (16, 185, 129) if hand_det else (0, 140, 255), 2)
                        status_txt = (
                            f"Recording Seq {seq_idx + 1}/{num_sequences} (Frame {frames_captured + 1}/{seq_len})"
                            if hand_det
                            else f"Paused Seq {seq_idx + 1}/{num_sequences} - Waiting for hand detection..."
                        )
                        cv2.putText(disp, status_txt, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(disp, f"Perform Gesture: {prompt_str}", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)
                        cv2.imshow("Gloved Hand Telemetry Recorder (Press Q to Abort)", disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            cv2.destroyAllWindows()
                            raise KeyboardInterrupt("User aborted recording via Q/ESC.")
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass

                # Only advance frame counter when a hand is actually detected (or if running simulated demo mode)
                if hand_det or is_sim:
                    frames_captured += 1

            seq_pkt = streamer.read_sequence(seq_len=seq_len)
            all_features[seq_idx] = seq_pkt["features_4d"]
            all_obs_mm[seq_idx] = seq_pkt["positions_mm"]
            all_masks[seq_idx] = seq_pkt["visibility_mask"]
            if "gt_positions_mm" in seq_pkt:
                gt_frame = seq_pkt["gt_positions_mm"]
                if gt_frame.ndim == 2:
                    all_raw_unmasked_mm[seq_idx] = np.broadcast_to(
                        gt_frame[None, ...], (seq_len, NUM_JOINTS, 3)
                    ).copy()
                else:
                    all_raw_unmasked_mm[seq_idx] = gt_frame
            else:
                all_raw_unmasked_mm[seq_idx] = seq_pkt["positions_mm"]
            print(f"[record_custom_gloved] Captured sequence {seq_idx + 1:02d}/{num_sequences:02d} ({prompt_str})")
            seq_idx += 1

    if mode == "mediapipe" and show_preview:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Harmonize reference ground-truth trajectories (`all_gt_mm`) via temporal smoothing +
    # analytical IK -> BiomechanicalRoMClamper -> DifferentiableFK so `gt_xyz_mm` has exact
    # canonical bone lengths (0.0000 mm error) and 100% biomechanical validity.
    with torch.no_grad():
        raw_t = torch.from_numpy(all_raw_unmasked_mm)
        if seq_len >= 3:
            # 3-tap temporal smoothing filter to remove single-frame camera depth jitter from reference GT
            smoothed = raw_t.clone()
            smoothed[:, 1:-1] = 0.25 * raw_t[:, :-2] + 0.50 * raw_t[:, 1:-1] + 0.25 * raw_t[:, 2:]
        else:
            smoothed = raw_t
        raw_rot6d = positions_to_rot6d(smoothed)
        clamper = BiomechanicalRoMClamper()
        fk_layer = DifferentiableFK()
        rot6d_torch, euler_torch = clamper(raw_rot6d, return_euler=True)
        rot_mat_torch = rot6d_to_matrix(rot6d_torch)
        harmonized_gt_torch = fk_layer(rot6d_torch)

        all_gt_mm = harmonized_gt_torch.numpy().astype(np.float32)
        rotations_6d = rot6d_torch.numpy().astype(np.float32)
        rotations_mat = rot_mat_torch.numpy().astype(np.float32)
        euler_angles_rad = euler_torch.numpy().astype(np.float32)

    if any_real_hardware_frame:
        # Real physical camera / UDP capture: keep the actual recorded camera observations
        # (`all_obs_mm`) as `gloved_unoccluded_mm` without adding fake synthetic noise!
        gloved_unoccluded_mm = all_obs_mm.copy()
    else:
        glove_tf = SyntheticGloveTransform(
            stiffness=0.12,
            sigma_mm=4.5,
            distal_spike_mm=12.0,
            temporal_corr=0.65,
            spike_prob=0.20,
            seed=777,
        )
        gloved_unoccluded_mm = glove_tf(all_gt_mm, rng=np.random.default_rng(777))

    # Precompute 30%, 50%, and 70% surgical tool occlusion evaluation tiers on top of the gloved trajectories
    tiers: Dict[str, np.ndarray] = {}
    for rate_pct, rate_val in [(30, 0.30), (50, 0.50), (70, 0.70)]:
        tf_tier = SyntheticOcclusionTransform(
            occlusion_prob=rate_val, seed=777 + rate_pct
        )
        t_occ_mm, t_mask = tf_tier(gloved_unoccluded_mm, rng=np.random.default_rng(777 + rate_pct))
        t_norm, _, _ = normalize_hand_sequence(
            t_occ_mm, scale_mm=scale_mm, visibility_mask=t_mask
        )
        t_feat = build_node_features_4d(t_norm, t_mask)
        tiers[f"node_features_occ{rate_pct}"] = t_feat
        tiers[f"visibility_mask_occ{rate_pct}"] = t_mask
        tiers[f"observed_xyz_mm_occ{rate_pct}"] = t_occ_mm

    gt_norm, _, _ = normalize_hand_sequence(
        all_gt_mm, scale_mm=scale_mm, visibility_mask=None
    )
    gesture_labels = np.arange(num_sequences, dtype=np.int64) % len(GESTURE_NAMES)

    payload: Dict[str, Any] = {
        "node_features": tiers["node_features_occ50"] if not any_real_hardware_frame else all_features,
        "x": tiers["node_features_occ50"] if not any_real_hardware_frame else all_features,
        "features_4d": tiers["node_features_occ50"] if not any_real_hardware_frame else all_features,
        "gt_xyz_mm": all_gt_mm,
        "positions_mm": all_gt_mm,
        "target_xyz_mm": all_gt_mm,
        "gt_xyz_norm": gt_norm,
        "target_xyz_norm": gt_norm,
        "gloved_unoccluded_mm": gloved_unoccluded_mm,
        "observed_xyz_mm": all_obs_mm,
        "visibility_mask": all_masks,
        "mask": all_masks,
        "rotations_mat": rotations_mat,
        "gt_rot_mat": rotations_mat,
        "rotations_6d": rotations_6d,
        "gt_rot6d": rotations_6d,
        "euler_angles_rad": euler_angles_rad,
        "gesture_labels": gesture_labels,
        "scale_mm": np.array(scale_mm, dtype=np.float32),
        "is_real_recording": np.array(bool(any_real_hardware_frame)),
        "recording_mode": np.array(mode if any_real_hardware_frame else f"{mode}_simulated"),
        **tiers,
    }
    np.savez_compressed(out_file, **payload)
    return payload


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
    parser.add_argument(
        "--strict-hardware",
        action="store_true",
        help="Fail if physical camera/UDP hardware stream is unavailable (do not simulate).",
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
            simulate_if_unavailable=not args.strict_hardware,
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
