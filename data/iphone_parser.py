"""
Dual-Mode 3D Hand Keypoint Streamer for Live Hardware Telemetry.

Supports two real-time acquisition modes:
1. `ARKitUDPStreamer`: Listens on UDP port 5005 for JSON packets from iPhone
   ARKit (e.g., Zig Sim Pro / custom ARKit hand-tracking sender), retargets
   ARKit's 21 joint names to MANO 0..20 ordering, normalizes to the wrist, and
   sets `m = 0` for missing or low-confidence joints. Includes a local UDP mock
   sender `send_mock_arkit_udp_packet()` for self-contained testing.
2. `MediaPipeCameraStreamer`: Captures live video from MacBook Air webcam
   (`camera_index=0`), iPhone Continuity Camera (`camera_index=1`), or NVIDIA
   Jetson Nano with Raspberry Pi CSI Camera via `build_jetson_csi_pipeline()`,
   extracts 21 3D `multi_hand_world_landmarks` in millimeters, and converts
   them into 4D graph node features `[x, y, z, m]`. Gracefully falls back to a
   simulated biomechanical stream in headless environments.
"""

from __future__ import annotations

from collections import deque
import json
import socket
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from data.dataloader import build_node_features_4d, normalize_hand_sequence
from data.dataset_ingestion import (
    CANONICAL_BONE_LENGTHS_MM,
    DEFAULT_PALM_SCALE_MM,
    JOINT_NAMES,
    NUM_JOINTS,
    REST_BONE_DIRECTIONS,
     _synthesize_single_sequence_angles,
    euler_xyz_to_matrix,
    forward_kinematics_numpy,
)
from data.transforms import SyntheticGloveTransform, SyntheticOcclusionTransform


# ---------------------------------------------------------------------------
# 1. ARKit Joint Name Retargeting Map -> MANO 0..20
# ---------------------------------------------------------------------------

ARKIT_CANONICAL_JOINT_NAMES: List[str] = [
    "wrist",       # 0
    "thumbCMC",    # 1
    "thumbMP",     # 2
    "thumbIP",     # 3
    "thumbTip",    # 4
    "indexMCP",    # 5
    "indexPIP",    # 6
    "indexDIP",    # 7
    "indexTip",    # 8
    "middleMCP",   # 9
    "middlePIP",   # 10
    "middleDIP",   # 11
    "middleTip",   # 12
    "ringMCP",     # 13
    "ringPIP",     # 14
    "ringDIP",     # 15
    "ringTip",     # 16
    "littleMCP",   # 17
    "littlePIP",   # 18
    "littleDIP",   # 19
    "littleTip",   # 20
]

_ARKIT_ALIASES: Dict[int, List[str]] = {
    0: ["wrist", "vnhlkwwrist", "hand_joint_wrist", "root"],
    1: ["thumbcmc", "thumb_cmc", "vnhlkthumbcmc"],
    2: ["thumbmp", "thumb_mp", "thumbmcp", "thumb_mcp", "vnhlkthumbmp"],
    3: ["thumbip", "thumb_ip", "vnhlkthumbip"],
    4: ["thumbtip", "thumb_tip", "vnhlkthumbtip"],
    5: ["indexmcp", "index_mcp", "vnhlkmcpindex"],
    6: ["indexpip", "index_pip", "vnhlkpipindex"],
    7: ["indexdip", "index_dip", "vnhlkdipindex"],
    8: ["indextip", "index_tip", "vnhlktipindex"],
    9: ["middlemcp", "middle_mcp", "vnhlkmcpmiddle"],
    10: ["middlepip", "middle_pip", "vnhlkpipmiddle"],
    11: ["middledip", "middle_dip", "vnhlkdipmiddle"],
    12: ["middletip", "middle_tip", "vnhlktipmiddle"],
    13: ["ringmcp", "ring_mcp", "vnhlkmcpthird"],
    14: ["ringpip", "ring_pip", "vnhlkpipthird"],
    15: ["ringdip", "ring_dip", "vnhlkdipthird"],
    16: ["ringtip", "ring_tip", "vnhlktipthird"],
    17: ["littlemcp", "little_mcp", "pinkymcp", "pinky_mcp", "vnhlkmcpfourth"],
    18: ["littlepip", "little_pip", "pinkypip", "pinky_pip", "vnhlkpipfourth"],
    19: ["littledip", "little_dip", "pinkydip", "pinky_dip", "vnhlkdipfourth"],
    20: ["littletip", "little_tip", "pinkytip", "pinky_tip", "vnhlktipfourth"],
}

ARKIT_NAME_TO_MANO_IDX: Dict[str, int] = {}
for _idx, _aliases in _ARKIT_ALIASES.items():
    for _alias in _aliases:
        ARKIT_NAME_TO_MANO_IDX[_alias.lower()] = _idx
    ARKIT_NAME_TO_MANO_IDX[JOINT_NAMES[_idx].lower()] = _idx
    ARKIT_NAME_TO_MANO_IDX[str(_idx)] = _idx


# ---------------------------------------------------------------------------
# 2. Mock ARKit UDP Packet Sender & Frame Generator
# ---------------------------------------------------------------------------


def build_mock_arkit_packet_dict(
    step_idx: int = 0,
    occlusion_rate: float = 0.30,
    apply_glove: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Constructs a realistic ARKit hand telemetry JSON dictionary with 21 joints
    in meters (`x, y, z`) and per-joint tracking confidence in `[0.0, 1.0]`.
    """
    rng = np.random.default_rng(seed + step_idx)
    gesture_id = step_idx % 6
    angles = _synthesize_single_sequence_angles(gesture_id, seq_len=4, rng=rng)
    rot_mat = euler_xyz_to_matrix(angles)
    gt_pos_mm = forward_kinematics_numpy(
        rot_mat,
        bone_lengths_mm=CANONICAL_BONE_LENGTHS_MM,
        rest_directions=REST_BONE_DIRECTIONS,
    )[0]  # (21, 3)

    if apply_glove:
        glove_tf = SyntheticGloveTransform(
            stiffness=0.12, sigma_mm=4.5, distal_spike_mm=12.0, seed=seed + step_idx
        )
        obs_mm = glove_tf(gt_pos_mm, rng=rng)
    else:
        obs_mm = gt_pos_mm.copy()

    occ_tf = SyntheticOcclusionTransform(
        occlusion_prob=occlusion_rate, seed=seed + step_idx
    )
    occ_mm, vis_mask = occ_tf(obs_mm, rng=rng)

    # ARKit reports coordinates in meters; convert mm -> meters for realistic wire format
    joints_dict: Dict[str, Dict[str, float]] = {}
    for j_idx, arkit_name in enumerate(ARKIT_CANONICAL_JOINT_NAMES):
        is_vis = bool(vis_mask[j_idx] > 0.5)
        conf = float(rng.uniform(0.82, 0.99)) if is_vis else float(rng.uniform(0.05, 0.25))
        pos_m = (occ_mm[j_idx] / 1000.0).tolist() if is_vis else [0.0, 0.0, 0.0]
        joints_dict[arkit_name] = {
            "x": float(pos_m[0]),
            "y": float(pos_m[1]),
            "z": float(pos_m[2]),
            "confidence": conf,
            "tracked": is_vis,
        }

    return {
        "source": "iPhone_ARKit_HandSkeleton",
        "timestamp": time.time(),
        "step_idx": int(step_idx),
        "units": "meters",
        "chirality": "right",
        "joints": joints_dict,
        "gt_positions_mm": gt_pos_mm.tolist(),
    }


def send_mock_arkit_udp_packet(
    host: str = "127.0.0.1",
    port: int = 5005,
    step_idx: int = 0,
    occlusion_rate: float = 0.30,
    apply_glove: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sends a single mock ARKit JSON hand packet over UDP to `(host, port)` so
    `ARKitUDPStreamer` can be tested locally without physical iPhone hardware.
    """
    packet_dict = build_mock_arkit_packet_dict(
        step_idx=step_idx,
        occlusion_rate=occlusion_rate,
        apply_glove=apply_glove,
        seed=seed,
    )
    raw_bytes = json.dumps(packet_dict).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(raw_bytes, (host, port))
    finally:
        sock.close()
    return packet_dict


# ---------------------------------------------------------------------------
# 3. ARKitUDPStreamer
# ---------------------------------------------------------------------------


class ARKitUDPStreamer:
    """
    Real-Time iPhone ARKit Hand Keypoint UDP Streamer (Default Port: 5005).

    Listens for UDP JSON datagrams emitted by iPhone ARKit apps (e.g., Zig Sim Pro
    or custom `VNHumanHandPoseObservation` / `ARHandSkeleton` senders), maps
    ARKit's 21 joint names to our canonical MANO `0..20` index ordering, converts
    meters to millimeters, normalizes to the wrist (`joint 0`), and assigns
    visibility `m = 0.0` to missing or low-confidence joints.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        confidence_threshold: float = 0.50,
        scale_mm: float = DEFAULT_PALM_SCALE_MM,
        timeout_sec: float = 0.25,
        seq_len: int = 8,
        fallback_to_mock: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.confidence_threshold = float(confidence_threshold)
        self.scale_mm = float(scale_mm)
        self.timeout_sec = float(timeout_sec)
        self.seq_len = int(seq_len)
        self.fallback_to_mock = bool(fallback_to_mock)

        self._step_counter = 0
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._pos_mm_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._mask_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

        self._sock: Optional[socket.socket] = None
        self._bind_socket()

    def _bind_socket(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.timeout_sec)
            sock.bind((self.host, self.port))
            self._sock = sock
        except OSError:
            self._sock = None

    def parse_arkit_json_packet(
        self,
        payload: Union[str, bytes, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parses an ARKit JSON packet and retargets all 21 joints to MANO ordering.

        Returns a dictionary containing:
            - "features_4d": `(21, 4)` float32 `[x_norm, y_norm, z_norm, m]`
            - "positions_mm": `(21, 3)` float32 wrist-centered coordinates in mm
            - "visibility_mask": `(21,)` float32 in `{0.0, 1.0}`
            - "scale_mm": float
            - "gt_positions_mm": optional `(21, 3)` float32 if present in mock packet
        """
        if isinstance(payload, bytes):
            data = json.loads(payload.decode("utf-8"))
        elif isinstance(payload, str):
            data = json.loads(payload)
        elif isinstance(payload, dict):
            data = payload
        else:
            raise ValueError(f"Unsupported ARKit payload type: {type(payload)}")

        raw_positions = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        visibility_mask = np.zeros((NUM_JOINTS,), dtype=np.float32)

        units = str(data.get("units", "meters")).lower()
        unit_scale = 1.0 if units in ("mm", "millimeters") else 1000.0

        joints_container = data.get("joints", data.get("landmarks", data))
        if isinstance(joints_container, dict):
            for key_name, val in joints_container.items():
                clean_key = str(key_name).strip().lower()
                if clean_key not in ARKIT_NAME_TO_MANO_IDX:
                    continue
                mano_idx = ARKIT_NAME_TO_MANO_IDX[clean_key]

                if isinstance(val, dict):
                    x = float(val.get("x", 0.0))
                    y = float(val.get("y", 0.0))
                    z = float(val.get("z", 0.0))
                    conf = float(val.get("confidence", val.get("score", 1.0)))
                    tracked = bool(val.get("tracked", True))
                elif isinstance(val, (list, tuple)) and len(val) >= 3:
                    x, y, z = float(val[0]), float(val[1]), float(val[2])
                    conf = float(val[3]) if len(val) >= 4 else 1.0
                    tracked = True
                else:
                    continue

                pos_vec = np.array([x, y, z], dtype=np.float32)
                is_valid = (
                    tracked
                    and (conf >= self.confidence_threshold)
                    and np.all(np.isfinite(pos_vec))
                )
                if is_valid:
                    raw_positions[mano_idx] = pos_vec * unit_scale
                    visibility_mask[mano_idx] = 1.0

        elif isinstance(joints_container, list):
            for idx, item in enumerate(joints_container[:NUM_JOINTS]):
                if isinstance(item, dict):
                    j_name = str(item.get("name", idx)).lower()
                    mano_idx = ARKIT_NAME_TO_MANO_IDX.get(j_name, idx)
                    pos_vec = np.array(
                        [
                            float(item.get("x", 0.0)),
                            float(item.get("y", 0.0)),
                            float(item.get("z", 0.0)),
                        ],
                        dtype=np.float32,
                    )
                    conf = float(item.get("confidence", 1.0))
                    if conf >= self.confidence_threshold and np.all(np.isfinite(pos_vec)):
                        raw_positions[mano_idx] = pos_vec * unit_scale
                        visibility_mask[mano_idx] = 1.0
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    pos_vec = np.array(item[:3], dtype=np.float32)
                    conf = float(item[3]) if len(item) >= 4 else 1.0
                    if conf >= self.confidence_threshold and np.all(np.isfinite(pos_vec)):
                        raw_positions[idx] = pos_vec * unit_scale
                        visibility_mask[idx] = 1.0

        # Auto-detect if coordinates were already in mm (e.g. > 5.0 before scaling)
        if unit_scale == 1000.0 and np.max(np.abs(raw_positions)) > 5000.0:
            raw_positions /= 1000.0

        # Ensure wrist (joint 0) is anchored
        visibility_mask[0] = 1.0

        norm_xyz, _, _ = normalize_hand_sequence(
            raw_positions,
            scale_mm=self.scale_mm,
            visibility_mask=visibility_mask,
        )
        centered_mm = norm_xyz * self.scale_mm
        features_4d = build_node_features_4d(norm_xyz, visibility_mask)

        result: Dict[str, Any] = {
            "features_4d": features_4d,
            "positions_mm": centered_mm,
            "visibility_mask": visibility_mask,
            "scale_mm": self.scale_mm,
        }
        if "gt_positions_mm" in data:
            result["gt_positions_mm"] = np.asarray(
                data["gt_positions_mm"], dtype=np.float32
            )
        return result

    def read_frame(self) -> Dict[str, Any]:
        """
        Reads one UDP frame from port 5005, or synthesizes a mock ARKit packet if
        no hardware sender is active within `timeout_sec` and `fallback_to_mock=True`.
        """
        packet_bytes: Optional[bytes] = None
        if self._sock is not None:
            try:
                packet_bytes, _ = self._sock.recvfrom(65535)
            except (socket.timeout, BlockingIOError, OSError):
                packet_bytes = None

        if packet_bytes is not None:
            parsed = self.parse_arkit_json_packet(packet_bytes)
        elif self.fallback_to_mock:
            mock_dict = build_mock_arkit_packet_dict(step_idx=self._step_counter)
            parsed = self.parse_arkit_json_packet(mock_dict)
        else:
            raise TimeoutError(
                f"No ARKit UDP packet received on {self.host}:{self.port} within {self.timeout_sec}s"
            )

        self._step_counter += 1
        self._frame_buffer.append(parsed["features_4d"])
        self._pos_mm_buffer.append(parsed["positions_mm"])
        self._mask_buffer.append(parsed["visibility_mask"])
        return parsed

    def read_sequence(self, seq_len: Optional[int] = None) -> Dict[str, Any]:
        """
        Returns a temporal window of `T = seq_len` frames of shape `(T, 21, 4)`.
        """
        target_len = int(seq_len or self.seq_len)
        last_frame = self.read_frame()
        while len(self._frame_buffer) < target_len:
            last_frame = self.read_frame()

        frames_4d = np.stack(list(self._frame_buffer)[-target_len:], axis=0)
        pos_mm = np.stack(list(self._pos_mm_buffer)[-target_len:], axis=0)
        mask = np.stack(list(self._mask_buffer)[-target_len:], axis=0)
        out = {
            "node_features": frames_4d,
            "features_4d": frames_4d,
            "positions_mm": pos_mm,
            "visibility_mask": mask,
            "scale_mm": self.scale_mm,
        }
        if "gt_positions_mm" in last_frame:
            out["gt_positions_mm"] = last_frame["gt_positions_mm"]
        return out

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self) -> "ARKitUDPStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 4. Jetson Nano CSI GStreamer Builder & MediaPipeCameraStreamer
# ---------------------------------------------------------------------------


def build_jetson_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 480,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    Constructs the NVIDIA Jetson Nano GStreamer pipeline string for a
    Raspberry Pi CSI Camera (`nvarguscamerasrc`).
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1"
    )


class MediaPipeCameraStreamer:
    """
    Live 3D Hand Keypoint Streamer using OpenCV + `mediapipe.solutions.hands`.

    Supports:
      - MacBook Air built-in FaceTime HD webcam (`camera_index=0`)
      - iPhone Continuity Camera (`camera_index=1`)
      - NVIDIA Jetson Nano with Raspberry Pi CSI Camera (`use_jetson_csi=True`)
      - Automatic graceful fallback to a simulated biomechanical stream when
        running in headless environments without a physical camera.
    """

    def __init__(
        self,
        camera_index: int = 0,
        use_jetson_csi: bool = False,
        scale_mm: float = DEFAULT_PALM_SCALE_MM,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        seq_len: int = 8,
        simulate_if_unavailable: bool = True,
        simulated_occlusion_rate: float = 0.30,
    ) -> None:
        self.camera_index = int(camera_index)
        self.use_jetson_csi = bool(use_jetson_csi)
        self.scale_mm = float(scale_mm)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.seq_len = int(seq_len)
        self.simulate_if_unavailable = bool(simulate_if_unavailable)
        self.simulated_occlusion_rate = float(simulated_occlusion_rate)

        self._cap: Any = None
        self._mp_hands: Any = None
        self._is_simulated: bool = False
        self._step_counter: int = 0

        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._pos_mm_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._mask_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

        self._init_hardware_or_fallback()

    def _init_hardware_or_fallback(self) -> None:
        try:
            import cv2
            import mediapipe as mp

            mp_hands_mod = getattr(mp, "solutions", None)
            if mp_hands_mod is None or not hasattr(mp_hands_mod, "hands"):
                raise RuntimeError("mediapipe.solutions.hands unavailable")

            self._mp_hands = mp_hands_mod.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            cap = None
            if self.use_jetson_csi:
                gst_str = build_jetson_csi_pipeline(sensor_id=self.camera_index)
                cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index)

            if cap is not None and cap.isOpened():
                self._cap = cap
                self._is_simulated = False
            else:
                if cap is not None:
                    cap.release()
                self._cap = None
                self._is_simulated = True
        except Exception:
            self._cap = None
            self._mp_hands = None
            self._is_simulated = True

        if self._is_simulated and not self.simulate_if_unavailable:
            raise RuntimeError(
                f"Camera index {self.camera_index} (use_jetson_csi={self.use_jetson_csi}) "
                "or MediaPipe is unavailable and simulate_if_unavailable=False."
            )

    def read_frame(self) -> Dict[str, Any]:
        """
        Captures a single frame from the camera (or simulated stream) and extracts
        21 3D hand world landmarks in millimeters plus `[x, y, z, m]` features.
        """
        if not self._is_simulated and self._cap is not None and self._mp_hands is not None:
            import cv2

            ok, bgr_frame = self._cap.read()
            if ok and bgr_frame is not None:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                results = self._mp_hands.process(rgb_frame)
                if getattr(results, "multi_hand_world_landmarks", None):
                    hand_world = results.multi_hand_world_landmarks[0]
                    raw_mm = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
                    vis_mask = np.ones((NUM_JOINTS,), dtype=np.float32)

                    for j_idx in range(NUM_JOINTS):
                        lm = hand_world.landmark[j_idx]
                        # MediaPipe world landmarks are in meters; convert to mm
                        raw_mm[j_idx, 0] = float(lm.x) * 1000.0
                        raw_mm[j_idx, 1] = -float(lm.y) * 1000.0  # +Y distal
                        raw_mm[j_idx, 2] = -float(lm.z) * 1000.0
                        vis = float(getattr(lm, "visibility", 1.0) or 1.0)
                        if vis < 0.35:
                            vis_mask[j_idx] = 0.0

                    vis_mask[0] = 1.0
                    norm_xyz, _, _ = normalize_hand_sequence(
                        raw_mm, scale_mm=self.scale_mm, visibility_mask=vis_mask
                    )
                    centered_mm = norm_xyz * self.scale_mm
                    features_4d = build_node_features_4d(norm_xyz, vis_mask)

                    self._step_counter += 1
                    self._frame_buffer.append(features_4d)
                    self._pos_mm_buffer.append(centered_mm)
                    self._mask_buffer.append(vis_mask)
                    return {
                        "features_4d": features_4d,
                        "positions_mm": centered_mm,
                        "visibility_mask": vis_mask,
                        "scale_mm": self.scale_mm,
                        "simulated": False,
                    }

        # Fallback: generate simulated MediaPipe world landmarks frame
        mock_pkt = build_mock_arkit_packet_dict(
            step_idx=self._step_counter,
            occlusion_rate=self.simulated_occlusion_rate,
            apply_glove=True,
        )
        parser = ARKitUDPStreamer(
            port=0, fallback_to_mock=True, scale_mm=self.scale_mm
        )
        parser.close()
        parsed = parser.parse_arkit_json_packet(mock_pkt)
        parsed["simulated"] = True

        self._step_counter += 1
        self._frame_buffer.append(parsed["features_4d"])
        self._pos_mm_buffer.append(parsed["positions_mm"])
        self._mask_buffer.append(parsed["visibility_mask"])
        return parsed

    def read_sequence(self, seq_len: Optional[int] = None) -> Dict[str, Any]:
        """Returns a sliding window sequence `(T, 21, 4)` of 4D node features."""
        target_len = int(seq_len or self.seq_len)
        last_frame = self.read_frame()
        while len(self._frame_buffer) < target_len:
            last_frame = self.read_frame()

        frames_4d = np.stack(list(self._frame_buffer)[-target_len:], axis=0)
        pos_mm = np.stack(list(self._pos_mm_buffer)[-target_len:], axis=0)
        mask = np.stack(list(self._mask_buffer)[-target_len:], axis=0)
        out = {
            "node_features": frames_4d,
            "features_4d": frames_4d,
            "positions_mm": pos_mm,
            "visibility_mask": mask,
            "scale_mm": self.scale_mm,
            "simulated": bool(last_frame.get("simulated", True)),
        }
        if "gt_positions_mm" in last_frame:
            out["gt_positions_mm"] = last_frame["gt_positions_mm"]
        return out

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._mp_hands is not None:
            try:
                self._mp_hands.close()
            except Exception:
                pass
            self._mp_hands = None

    def __enter__(self) -> "MediaPipeCameraStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
