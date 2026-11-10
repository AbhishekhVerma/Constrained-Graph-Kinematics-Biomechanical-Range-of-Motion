"""
Root Normalization, 4D Graph Node Feature Builder, Strict 80/20 Split Management,
and PyTorch DataLoaders for Biomechanical Hand Telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset_ingestion import (
    CANONICAL_BONE_LENGTHS_MM,
    DEFAULT_PALM_SCALE_MM,
    GESTURE_NAMES,
    HAND_EDGES,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENT_INDICES,
    REST_BONE_DIRECTIONS,
    get_hand_adjacency_matrix,
    get_hand_edge_index,
    ingest_or_generate_dataset,
)
from data.transforms import SyntheticGloveTransform, SyntheticOcclusionTransform


# ---------------------------------------------------------------------------
# 1. Root Normalization & 4D Node Feature Utilities
# ---------------------------------------------------------------------------


def normalize_hand_sequence(
    positions_mm: np.ndarray,
    scale_mm: float = DEFAULT_PALM_SCALE_MM,
    visibility_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Centers each hand frame at the wrist (`joint 0 = (0, 0, 0)`) and normalizes
    coordinates by `scale_mm` (default 100.0 mm) so neural network inputs lie
    roughly within `[-1.5, 1.5]`.

    If `visibility_mask` is provided, occluded joints (`m == 0`) are re-zeroed
    after wrist centering so they do not leak negative wrist offsets.

    Args:
        positions_mm: `(..., 21, 3)` array in millimeters.
        scale_mm: Reference palm scale in millimeters (default: 100.0).
        visibility_mask: Optional `(..., 21)` mask in `{0.0, 1.0}`.

    Returns:
        normalized_xyz: `(..., 21, 3)` float32 in normalized units.
        wrist_origin_mm: `(..., 1, 3)` float32 wrist position in mm.
        scale_mm: float reference scale used.
    """
    arr = np.asarray(positions_mm, dtype=np.float32)
    wrist_origin_mm = arr[..., 0:1, :].copy()
    centered_mm = arr - wrist_origin_mm
    norm_xyz = (centered_mm / float(scale_mm)).astype(np.float32)

    if visibility_mask is not None:
        mask_exp = np.asarray(visibility_mask, dtype=np.float32)[..., None]
        norm_xyz = norm_xyz * mask_exp

    return norm_xyz, wrist_origin_mm, float(scale_mm)


def denormalize_hand_sequence(
    normalized_xyz: Union[np.ndarray, torch.Tensor],
    scale_mm: Union[float, np.ndarray, torch.Tensor] = DEFAULT_PALM_SCALE_MM,
    wrist_origin_mm: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts normalized hand coordinates `(..., 21, 3)` directly back to
    millimeters for MPJPE calculation.
    """
    if isinstance(normalized_xyz, torch.Tensor):
        if isinstance(scale_mm, torch.Tensor):
            while scale_mm.ndim < normalized_xyz.ndim:
                scale_mm = scale_mm.unsqueeze(-1)
            pos_mm = normalized_xyz * scale_mm
        else:
            pos_mm = normalized_xyz * float(scale_mm)
        if wrist_origin_mm is not None:
            pos_mm = pos_mm + wrist_origin_mm
        return pos_mm

    pos_mm_np = np.asarray(normalized_xyz, dtype=np.float32) * float(
        np.mean(scale_mm) if isinstance(scale_mm, np.ndarray) else scale_mm
    )
    if wrist_origin_mm is not None:
        pos_mm_np = pos_mm_np + np.asarray(wrist_origin_mm, dtype=np.float32)
    return pos_mm_np


def build_node_features_4d(
    normalized_xyz: np.ndarray,
    visibility_mask: np.ndarray,
) -> np.ndarray:
    """
    Builds 4D Node Features `[x, y, z, m]` of shape `(..., 21, 4)` where
    `m in {0.0, 1.0}` is the joint visibility mask and occluded coordinates
    (`m == 0`) are strictly zeroed.
    """
    xyz = np.asarray(normalized_xyz, dtype=np.float32)
    mask = np.asarray(visibility_mask, dtype=np.float32)
    if mask.ndim == xyz.ndim - 1:
        mask_4d = mask[..., None]
    else:
        mask_4d = mask
    masked_xyz = xyz * mask_4d
    return np.concatenate([masked_xyz, mask_4d], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Strict 80/20 Train/Test Split Management (Zero Data Leakage)
# ---------------------------------------------------------------------------


def create_or_load_split_indices(
    num_sequences: int,
    split_path: Union[str, Path] = "data/splits/split_indices.json",
    train_ratio: float = 0.80,
    seed: int = 42,
    force_resplit: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Enforces a strict 80/20 Train/Test split and persists the exact indices to
    `data/splits/split_indices.json` so Teacher training and Student distillation
    share the identical split without any data leakage.
    """
    path = Path(split_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_resplit:
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if int(saved.get("num_sequences", -1)) == int(num_sequences):
                train_idx = [int(i) for i in saved["train_indices"]]
                test_idx = [int(i) for i in saved["test_indices"]]
                if set(train_idx).isdisjoint(set(test_idx)) and (
                    len(train_idx) + len(test_idx) == num_sequences
                ):
                    return train_idx, test_idx
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_sequences).tolist()
    n_train = int(round(train_ratio * num_sequences))
    train_indices = sorted(perm[:n_train])
    test_indices = sorted(perm[n_train:])

    assert set(train_indices).isdisjoint(set(test_indices)), (
        "Data leakage detected between train and test splits!"
    )

    payload = {
        "num_sequences": int(num_sequences),
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return train_indices, test_indices


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & Batch Container
# ---------------------------------------------------------------------------


class BatchDict(dict):
    """
    Dictionary subclass returned by DataLoader that supports `.to(device)`
    and provides consistent alias keys across training, distillation, and benchmark modules.
    """

    def to(self, device: Union[str, torch.device]) -> "BatchDict":
        moved = BatchDict()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return moved


class HandKinematicsDataset(Dataset):
    """
    PyTorch Dataset yielding 4D graph node features `[x, y, z, m]` of shape `(T, 21, 4)`
    along with ground-truth 3D joint positions in millimeters `(T, 21, 3)`, normalized
    ground-truth coordinates, and ground-truth SO(3) / 6D joint rotations.
    """

    def __init__(
        self,
        positions_mm: np.ndarray,
        rotations_mat: np.ndarray,
        rotations_6d: np.ndarray,
        euler_angles_rad: np.ndarray,
        gesture_labels: np.ndarray,
        indices: List[int],
        occlusion_rate: float = 0.50,
        apply_glove: bool = True,
        stiffness: float = 0.12,
        sigma_mm: float = 4.5,
        distal_spike_mm: float = 12.0,
        scale_mm: float = DEFAULT_PALM_SCALE_MM,
        deterministic_eval: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.indices = list(indices)
        self.positions_mm = positions_mm[self.indices].astype(np.float32)
        self.rotations_mat = rotations_mat[self.indices].astype(np.float32)
        self.rotations_6d = rotations_6d[self.indices].astype(np.float32)
        self.euler_angles_rad = euler_angles_rad[self.indices].astype(np.float32)
        self.gesture_labels = gesture_labels[self.indices].astype(np.int64)

        self.occlusion_rate = float(occlusion_rate)
        self.apply_glove = bool(apply_glove)
        self.scale_mm = float(scale_mm)
        self.deterministic_eval = bool(deterministic_eval)
        self.base_seed = int(seed)

        self.glove_transform = SyntheticGloveTransform(
            stiffness=stiffness,
            sigma_mm=sigma_mm,
            distal_spike_mm=distal_spike_mm,
            seed=seed,
        )
        self.occlusion_transform = SyntheticOcclusionTransform(
            occlusion_prob=occlusion_rate,
            target_occlusion_rate=occlusion_rate,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt_xyz_mm = self.positions_mm[idx]          # (T, 21, 3) in mm (wrist already at 0)
        gt_rot_mat = self.rotations_mat[idx]        # (T, 21, 3, 3)
        gt_rot6d = self.rotations_6d[idx]           # (T, 21, 6)
        gt_euler = self.euler_angles_rad[idx]       # (T, 21, 3)
        g_label = int(self.gesture_labels[idx])

        # Per-sample RNG for deterministic evaluation or stochastic training augmentation
        if self.deterministic_eval:
            sample_rng = np.random.default_rng(self.base_seed + 10007 * int(self.indices[idx]))
        else:
            sample_rng = None

        # 1. Optional Zero-Shot Glove Domain Adaptation (stiffness + sensor jitter)
        if self.apply_glove:
            observed_mm = self.glove_transform(gt_xyz_mm, rng=sample_rng)
        else:
            observed_mm = gt_xyz_mm.copy()

        # 2. 3D Volumetric Clipping + Kinematic Subtree Masking
        occluded_mm, visibility_mask = self.occlusion_transform(
            observed_mm, rng=sample_rng
        )

        # 3. Root Normalization (center at wrist joint 0 and divide by scale_mm)
        obs_norm, _, _ = normalize_hand_sequence(
            occluded_mm,
            scale_mm=self.scale_mm,
            visibility_mask=visibility_mask,
        )
        gt_norm, _, _ = normalize_hand_sequence(
            gt_xyz_mm,
            scale_mm=self.scale_mm,
            visibility_mask=None,
        )

        # 4. Build 4D Node Features [x, y, z, m] of shape (T, 21, 4)
        node_features_4d = build_node_features_4d(obs_norm, visibility_mask)

        x_tensor = torch.from_numpy(node_features_4d)         # (T, 21, 4)
        gt_mm_tensor = torch.from_numpy(gt_xyz_mm)            # (T, 21, 3)
        gt_norm_tensor = torch.from_numpy(gt_norm)            # (T, 21, 3)
        obs_mm_tensor = torch.from_numpy(occluded_mm)         # (T, 21, 3)
        mask_tensor = torch.from_numpy(visibility_mask)       # (T, 21)
        rot6d_tensor = torch.from_numpy(gt_rot6d)             # (T, 21, 6)
        rotmat_tensor = torch.from_numpy(gt_rot_mat)          # (T, 21, 3, 3)
        euler_tensor = torch.from_numpy(gt_euler)             # (T, 21, 3)
        scale_tensor = torch.tensor(self.scale_mm, dtype=torch.float32)
        label_tensor = torch.tensor(g_label, dtype=torch.long)

        return {
            # Primary keys
            "node_features": x_tensor,
            "x": x_tensor,
            "inputs": x_tensor,
            "features": x_tensor,
            "gt_xyz_mm": gt_mm_tensor,
            "target_xyz_mm": gt_mm_tensor,
            "positions_mm": gt_mm_tensor,
            "gt_positions_mm": gt_mm_tensor,
            "gt_xyz_norm": gt_norm_tensor,
            "target_xyz_norm": gt_norm_tensor,
            "target_xyz": gt_norm_tensor,
            "observed_xyz_mm": obs_mm_tensor,
            "visibility_mask": mask_tensor,
            "mask": mask_tensor,
            "gt_rot6d": rot6d_tensor,
            "target_rot6d": rot6d_tensor,
            "rotations_6d": rot6d_tensor,
            "gt_rot_mat": rotmat_tensor,
            "rotations_mat": rotmat_tensor,
            "euler_angles_rad": euler_tensor,
            "scale_mm": scale_tensor,
            "gesture_labels": label_tensor,
            "gesture_label": label_tensor,
        }


def _collate_batch_dict(batch_list: List[Dict[str, torch.Tensor]]) -> BatchDict:
    """Collates a list of sample dicts into a `BatchDict` of stacked tensors."""
    out = BatchDict()
    if not batch_list:
        return out
    keys = batch_list[0].keys()
    for k in keys:
        out[k] = torch.stack([sample[k] for sample in batch_list], dim=0)
    return out


# ---------------------------------------------------------------------------
# 4. Public Factory: `get_dataloaders`
# ---------------------------------------------------------------------------


def get_dataloaders(
    batch_size: int = 32,
    seq_len: int = 8,
    occlusion_rate: float = 0.50,
    apply_glove: bool = True,
    num_sequences: int = 1200,
    data_dir: Union[str, Path] = "data",
    scale_mm: float = DEFAULT_PALM_SCALE_MM,
    stiffness: float = 0.12,
    sigma_mm: float = 4.5,
    distal_spike_mm: float = 12.0,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs 80/20 Train and Test DataLoaders with Root Normalization,
    4D Node Features `[x, y, z, m]`, 3D Volumetric/Subtree Occlusion, and
    Zero-Shot Glove Domain Adaptation.

    Saves exact train/test split indices to `data/splits/split_indices.json`.

    Returns:
        train_loader: PyTorch DataLoader for training (80% split).
        test_loader: PyTorch DataLoader for evaluation (20% split).
        metadata: Dictionary of skeleton topology, scale, and split metadata.
    """
    base_dir = Path(data_dir)
    raw_ds = ingest_or_generate_dataset(
        data_dir=base_dir,
        num_sequences=num_sequences,
        seq_len=seq_len,
        seed=seed,
    )

    split_path = base_dir / "splits" / "split_indices.json"
    train_indices, test_indices = create_or_load_split_indices(
        num_sequences=num_sequences,
        split_path=split_path,
        train_ratio=0.80,
        seed=seed,
    )

    train_dataset = HandKinematicsDataset(
        positions_mm=raw_ds["positions_mm"],
        rotations_mat=raw_ds["rotations_mat"],
        rotations_6d=raw_ds["rotations_6d"],
        euler_angles_rad=raw_ds["euler_angles_rad"],
        gesture_labels=raw_ds["gesture_labels"],
        indices=train_indices,
        occlusion_rate=occlusion_rate,
        apply_glove=apply_glove,
        stiffness=stiffness,
        sigma_mm=sigma_mm,
        distal_spike_mm=distal_spike_mm,
        scale_mm=scale_mm,
        deterministic_eval=False,
        seed=seed,
    )

    test_dataset = HandKinematicsDataset(
        positions_mm=raw_ds["positions_mm"],
        rotations_mat=raw_ds["rotations_mat"],
        rotations_6d=raw_ds["rotations_6d"],
        euler_angles_rad=raw_ds["euler_angles_rad"],
        gesture_labels=raw_ds["gesture_labels"],
        indices=test_indices,
        occlusion_rate=occlusion_rate,
        apply_glove=apply_glove,
        stiffness=stiffness,
        sigma_mm=sigma_mm,
        distal_spike_mm=distal_spike_mm,
        scale_mm=scale_mm,
        deterministic_eval=True,
        seed=seed + 999,
    )

    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_batch_dict,
        generator=g_cpu,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_batch_dict,
        drop_last=False,
    )

    metadata: Dict[str, Any] = {
        "scale_mm": float(scale_mm),
        "num_joints": NUM_JOINTS,
        "joint_names": list(JOINT_NAMES),
        "parent_indices": list(PARENT_INDICES),
        "bone_lengths_mm": CANONICAL_BONE_LENGTHS_MM.copy(),
        "bone_lengths_norm": (CANONICAL_BONE_LENGTHS_MM / float(scale_mm)).astype(np.float32),
        "rest_directions": REST_BONE_DIRECTIONS.copy(),
        "hand_edges": list(HAND_EDGES),
        "edge_index": get_hand_edge_index(include_self_loops=True),
        "adjacency_matrix": get_hand_adjacency_matrix(include_self_loops=True, normalize=True),
        "seq_len": int(seq_len),
        "occlusion_rate": float(occlusion_rate),
        "apply_glove": bool(apply_glove),
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "split_file": str(split_path),
        "gesture_names": list(GESTURE_NAMES),
    }

    return train_loader, test_loader, metadata
