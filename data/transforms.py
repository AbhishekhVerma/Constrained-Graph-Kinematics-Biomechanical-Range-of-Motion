"""
3D Graph-Space Occlusion & Zero-Shot Glove Domain Adaptation Transforms.

Implements:
1. `SyntheticOcclusionTransform`: 3D Volumetric Clipping (virtual surgical tool /
   cylindrical handle volume) combined with Kinematic Subtree Masking to achieve
   exact target joint occlusion rates (e.g., 30%, 50%, 70%) while keeping the
   wrist root (joint 0) visible as the kinematic anchor.
2. `SyntheticGloveTransform`: Zero-Shot Glove Domain Adaptation combining
   Kinematic Range Reduction (stiffness damping toward a neutral glove posture)
   with temporally correlated ARKit/MediaPipe specular sensor jitter (4.5 mm
   baseline, 12.0 mm distal fingertip spikes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data.dataset_ingestion import (
    CANONICAL_BONE_LENGTHS_MM,
    DISTAL_TIP_INDICES,
    FINGER_SUBTREES,
    NUM_JOINTS,
    PARENT_INDICES,
    REST_BONE_DIRECTIONS,
    REST_JOINT_POSITIONS_MM,
)


class SyntheticOcclusionTransform:
    """
    Operates in 3D coordinate space on hand sequences `(T, 21, 3)` or `(N, T, 21, 3)`.

    Combines:
      1. **3D Volumetric Clipping**: Spawns a virtual 3D cylinder/sphere near the
         fingertips/palm representing a surgical instrument or industrial handle
         and scores joints falling inside or behind the occluding volume.
      2. **Kinematic Subtree Masking**: Propagates occlusion contiguously toward
         distal joints (`MCP -> PIP -> DIP -> TIP`) of the affected digits so that
         the exact target percentage of joints (e.g., 30%, 50%, or 70%) have
         visibility flag `m = 0.0` and zeroed/corrupted coordinates, while wrist
         joint `0` always remains visible (`m = 1.0`) as the root anchor.
    """

    def __init__(
        self,
        occlusion_prob: float = 0.50,
        target_occlusion_rate: Optional[float] = None,
        volume_type: str = "cylinder",
        cylinder_radius_mm: float = 28.0,
        zero_occluded_coords: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        rate = (
            target_occlusion_rate
            if target_occlusion_rate is not None
            else occlusion_prob
        )
        self.occlusion_prob: float = float(np.clip(rate, 0.0, 0.95))
        self.target_occlusion_rate: float = self.occlusion_prob
        self.volume_type: str = volume_type
        self.cylinder_radius_mm: float = float(cylinder_radius_mm)
        self.zero_occluded_coords: bool = bool(zero_occluded_coords)
        self.seed: Optional[int] = seed
        self._rng = np.random.default_rng(seed)

    def _compute_num_masked_joints(self) -> int:
        """
        Computes exact number of joints to mask out of 21 so that the masked
        fraction matches `target_occlusion_rate` (clamped to [0, 20] so wrist 0
        is always unmasked).
        """
        if self.target_occlusion_rate <= 0.0:
            return 0
        k = int(np.rint(self.target_occlusion_rate * NUM_JOINTS))
        return int(np.clip(k, 1, NUM_JOINTS - 1))

    def _mask_single_sequence(
        self,
        seq_xyz: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies 3D Volumetric Clipping + Kinematic Subtree Masking to a single
        sequence `seq_xyz` of shape `(T, 21, 3)`.

        Returns:
            occluded_xyz: `(T, 21, 3)` float32
            visibility_mask: `(T, 21)` float32 with `1.0` = visible, `0.0` = occluded
        """
        t_len = seq_xyz.shape[0]
        visibility_mask = np.ones((t_len, NUM_JOINTS), dtype=np.float32)
        occluded_xyz = seq_xyz.copy().astype(np.float32)

        k_mask = self._compute_num_masked_joints()
        if k_mask == 0:
            return occluded_xyz, visibility_mask

        # 1. Spawn a virtual 3D surgical tool / handle volume near the fingertips/palm.
        # Compute centroid of distal tips & PIP joints at t=0 to anchor the tool volume.
        focus_joints = [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        hand_focus_center = np.mean(seq_xyz[:, focus_joints, :], axis=(0, 1))
        offset_start = rng.uniform(-22.0, 22.0, size=(3,)).astype(np.float32)
        offset_end = offset_start + rng.uniform(-15.0, 15.0, size=(3,)).astype(np.float32)

        # Tool axis (unit vector across palm/fingers representing scalpel/drill/probe)
        raw_axis = rng.normal(0.0, 1.0, size=(3,)).astype(np.float32)
        raw_axis[0] += 0.8  # Bias tool axis laterally across fingers
        tool_axis = raw_axis / max(float(np.linalg.norm(raw_axis)), 1e-6)

        # Finger chains ordered from distal TIP to proximal MCP so subtree masking
        # naturally masks contiguous distal segments (TIP -> DIP -> PIP -> MCP)
        finger_chains_distal_first: List[List[int]] = [
            [4, 3, 2, 1],     # Thumb
            [8, 7, 6, 5],     # Index
            [12, 11, 10, 9],  # Middle
            [16, 15, 14, 13], # Ring
            [20, 19, 18, 17], # Pinky
        ]

        for t in range(t_len):
            alpha = 0.0 if t_len <= 1 else float(t) / float(t_len - 1)
            vol_center = hand_focus_center + (1.0 - alpha) * offset_start + alpha * offset_end

            pts = seq_xyz[t]  # (21, 3)
            rel = pts - vol_center[None, :]  # (21, 3)

            if self.volume_type == "sphere":
                # Signed distance to virtual sphere + depth behind sphere (+Z occluded)
                dist_to_vol = np.linalg.norm(rel, axis=-1) - self.cylinder_radius_mm
            else:
                # 3D Cylinder clipping: perpendicular distance to tool axis
                proj_len = np.sum(rel * tool_axis[None, :], axis=-1, keepdims=True)
                perp_vec = rel - proj_len * tool_axis[None, :]
                perp_dist = np.linalg.norm(perp_vec, axis=-1)
                axial_dist = np.abs(proj_len[:, 0])
                # Also account for joints lying behind the tool along camera ray (+Z)
                depth_behind = np.maximum(0.0, pts[:, 2] - vol_center[2])
                dist_to_vol = (
                    perp_dist
                    + 0.25 * np.maximum(0.0, axial_dist - 45.0)
                    - 0.20 * depth_behind
                    - self.cylinder_radius_mm
                )

            # 2. Kinematic Subtree Scoring:
            # Rank fingers by how deeply their distal joints intersect the 3D tool volume,
            # and within each finger enforce contiguous distal-to-proximal subtree priority
            # so that whenever a proximal joint is masked, all its distal children are masked.
            candidate_scores = np.full((NUM_JOINTS,), fill_value=1e6, dtype=np.float32)
            for chain in finger_chains_distal_first:
                # Mean proximity of this finger's distal segment to the 3D occluding volume
                finger_score = float(np.mean(dist_to_vol[chain[:3]]))
                for rank_in_finger, j_idx in enumerate(chain):
                    # Distal joints get lower (higher-priority) score than their parent
                    # to guarantee contiguous subtree occlusion from TIP inward
                    candidate_scores[j_idx] = (
                        finger_score
                        + 0.35 * dist_to_vol[j_idx]
                        + 6.0 * float(rank_in_finger)
                    )

            # Wrist (joint 0) is strictly anchored as visible
            candidate_scores[0] = 1e9

            # Select exact top `k_mask` joints to mask
            masked_indices = np.argsort(candidate_scores)[:k_mask]
            visibility_mask[t, masked_indices] = 0.0

            if self.zero_occluded_coords:
                occluded_xyz[t, masked_indices, :] = 0.0

        return occluded_xyz, visibility_mask

    def __call__(
        self,
        positions_mm: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies synthetic 3D occlusion to `positions_mm`.

        Args:
            positions_mm: Shape `(T, 21, 3)`, `(N, T, 21, 3)`, or `(21, 3)` in mm.
            rng: Optional numpy Generator for reproducible masking.

        Returns:
            occluded_positions_mm: Same shape as `positions_mm`, with occluded joints zeroed.
            visibility_mask: Shape `(..., 21)` with `1.0` for visible and `0.0` for occluded.
        """
        active_rng = rng if rng is not None else self._rng
        arr = np.asarray(positions_mm, dtype=np.float32)

        if arr.ndim == 2 and arr.shape == (NUM_JOINTS, 3):
            occ, mask = self._mask_single_sequence(arr[None, ...], active_rng)
            return occ[0], mask[0]
        elif arr.ndim == 3 and arr.shape[1:] == (NUM_JOINTS, 3):
            return self._mask_single_sequence(arr, active_rng)
        elif arr.ndim == 4 and arr.shape[2:] == (NUM_JOINTS, 3):
            n_seq, t_len = arr.shape[0], arr.shape[1]
            occ_all = np.zeros_like(arr, dtype=np.float32)
            mask_all = np.ones((n_seq, t_len, NUM_JOINTS), dtype=np.float32)
            for i in range(n_seq):
                occ_all[i], mask_all[i] = self._mask_single_sequence(arr[i], active_rng)
            return occ_all, mask_all
        else:
            raise ValueError(
                f"Expected positions_mm of shape (T, 21, 3) or (N, T, 21, 3), got {arr.shape}"
            )


class SyntheticGloveTransform:
    """
    Zero-Shot Glove Domain Adaptation Transform in 3D Graph Space.

    Models the physical and optical domain shift induced by surgical latex/nitrile
    or industrial protective gloves via two mechanisms:
      1. **Kinematic Range Reduction (Stiffness)**: Dampens finger articulation/flexion
         toward the stiff neutral glove posture by `stiffness` (default `0.12`),
         preserving exact bone lengths along the kinematic tree.
      2. **Structured Sensor Jitter**: Injects temporally correlated AR(1) tracking
         noise (`sigma_mm=4.5` mm on visible joints, amplified to `distal_spike_mm=12.0` mm
         specular spikes on distal fingertips) modeling optical/LiDAR degradation
         on shiny gloved surfaces.
    """

    def __init__(
        self,
        stiffness: float = 0.12,
        sigma_mm: float = 4.5,
        distal_spike_mm: float = 12.0,
        temporal_corr: float = 0.65,
        spike_prob: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.stiffness: float = float(np.clip(stiffness, 0.0, 0.90))
        self.sigma_mm: float = float(max(0.0, sigma_mm))
        self.distal_spike_mm: float = float(max(0.0, distal_spike_mm))
        self.temporal_corr: float = float(np.clip(temporal_corr, 0.0, 0.98))
        self.spike_prob: float = float(np.clip(spike_prob, 0.0, 1.0))
        self.seed: Optional[int] = seed
        self._rng = np.random.default_rng(seed)

    def apply_kinematic_stiffness(self, positions_mm: np.ndarray) -> np.ndarray:
        """
        Dampens 3D articulation toward the stiff neutral glove posture along the
        kinematic chain while preserving canonical bone lengths.
        """
        arr = np.asarray(positions_mm, dtype=np.float32)
        if self.stiffness <= 0.0:
            return arr.copy()

        stiff_pos = np.zeros_like(arr, dtype=np.float32)
        stiff_pos[..., 0, :] = arr[..., 0, :]  # Keep wrist root

        # Neutral glove directions: slightly flexed semi-open equilibrium posture
        neutral_dirs = REST_BONE_DIRECTIONS.copy()

        for j in range(1, NUM_JOINTS):
            p = PARENT_INDICES[j]
            bone_vec = arr[..., j, :] - arr[..., p, :]
            bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
            safe_len = np.maximum(bone_len, 1e-6)
            curr_dir = bone_vec / safe_len

            # Blend current bone unit vector toward neutral glove bone direction
            target_dir = neutral_dirs[j]
            blended_dir = (1.0 - self.stiffness) * curr_dir + self.stiffness * target_dir
            blended_norm = np.maximum(
                np.linalg.norm(blended_dir, axis=-1, keepdims=True), 1e-6
            )
            unit_blended = blended_dir / blended_norm

            # Preserve exact bone length along kinematic tree
            use_len = np.where(bone_len > 1e-3, bone_len, CANONICAL_BONE_LENGTHS_MM[j])
            stiff_pos[..., j, :] = stiff_pos[..., p, :] + use_len * unit_blended

        return stiff_pos

    def _inject_sensor_jitter_single(
        self,
        seq_xyz: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Injects temporally correlated AR(1) tracking jitter + distal fingertip spikes
        into a single sequence `(T, 21, 3)`.
        """
        t_len = seq_xyz.shape[0]
        if self.sigma_mm <= 0.0 and self.distal_spike_mm <= 0.0:
            return seq_xyz.copy()

        # Per-joint baseline standard deviation (mm)
        joint_sigma = np.full((NUM_JOINTS, 1), self.sigma_mm, dtype=np.float32)
        joint_sigma[0, 0] = 0.0  # Keep root anchor noise-free (or wrist-centered)
        # Slightly higher noise on DIP joints
        for dip_j in [3, 7, 11, 15, 19]:
            joint_sigma[dip_j, 0] = self.sigma_mm * 1.25

        # Generate temporally correlated AR(1) Gaussian jitter across frames
        rho = self.temporal_corr
        innovation_scale = float(np.sqrt(max(1.0 - rho * rho, 1e-6)))
        ar_noise = np.zeros((t_len, NUM_JOINTS, 3), dtype=np.float32)
        prev = rng.normal(0.0, 1.0, size=(NUM_JOINTS, 3)).astype(np.float32)
        ar_noise[0] = prev * joint_sigma

        for t in range(1, t_len):
            z = rng.normal(0.0, 1.0, size=(NUM_JOINTS, 3)).astype(np.float32)
            prev = rho * prev + innovation_scale * z
            ar_noise[t] = prev * joint_sigma

        # Add structured specular tracking spikes on distal fingertips (12.0 mm scale)
        if self.distal_spike_mm > 0.0 and self.spike_prob > 0.0:
            for tip_j in DISTAL_TIP_INDICES:
                spike_mask = (
                    rng.uniform(0.0, 1.0, size=(t_len, 1)) < self.spike_prob
                ).astype(np.float32)
                spike_vec = rng.normal(
                    0.0, self.distal_spike_mm, size=(t_len, 3)
                ).astype(np.float32)
                ar_noise[:, tip_j, :] += spike_mask * spike_vec

        return (seq_xyz + ar_noise).astype(np.float32)

    def __call__(
        self,
        positions_mm: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Applies Kinematic Range Reduction (stiffness) + Structured Sensor Jitter.

        Args:
            positions_mm: Shape `(T, 21, 3)`, `(N, T, 21, 3)`, or `(21, 3)` in mm.
            rng: Optional numpy Generator.

        Returns:
            gloved_positions_mm: Same shape as `positions_mm` in millimeters.
        """
        active_rng = rng if rng is not None else self._rng
        stiff_xyz = self.apply_kinematic_stiffness(positions_mm)

        if stiff_xyz.ndim == 2 and stiff_xyz.shape == (NUM_JOINTS, 3):
            return self._inject_sensor_jitter_single(stiff_xyz[None, ...], active_rng)[0]
        elif stiff_xyz.ndim == 3 and stiff_xyz.shape[1:] == (NUM_JOINTS, 3):
            return self._inject_sensor_jitter_single(stiff_xyz, active_rng)
        elif stiff_xyz.ndim == 4 and stiff_xyz.shape[2:] == (NUM_JOINTS, 3):
            out = np.zeros_like(stiff_xyz, dtype=np.float32)
            for i in range(stiff_xyz.shape[0]):
                out[i] = self._inject_sensor_jitter_single(stiff_xyz[i], active_rng)
            return out
        else:
            raise ValueError(
                f"Expected positions_mm of shape (T, 21, 3) or (N, T, 21, 3), got {stiff_xyz.shape}"
            )
