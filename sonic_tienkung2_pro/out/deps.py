"""所有 gear_sonic 依赖的内联版本，使 out/ 目录完全自包含。"""

# ============================================================
# 标准库与第三方导入
# ============================================================
import enum
import json
import os
import struct
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from scipy.spatial.transform import Rotation as sRot

# ============================================================
# JIT 工具（kornia_transform 和 torch_transform 共用）
# ============================================================
USE_JIT_TORCH_TRANSFORM = os.getenv("USE_JIT_TORCH_TRANSFORM", "1").lower() in ("1", "true", "yes")


def conditional_jit_script(func):
    if USE_JIT_TORCH_TRANSFORM:
        return torch.jit.script(func)
    return func


# ============================================================
# kornia_transform 函数（仅保留所需部分）
# ============================================================

class QuaternionCoeffOrder(enum.Enum):
    XYZW = "xyzw"
    WXYZ = "wxyz"


@conditional_jit_script
def _kt_torch_safe_atan2(y, x, eps: float = 1e-6):
    y = y.clone()
    if len(y.shape) == 0:
        if y.abs() < eps and x.abs() < eps:
            y += eps
    else:
        y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


def _kt_safe_zero_division(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    denominator = denominator.clone()
    if len(denominator.shape) == 0:
        if denominator.abs() < eps:
            denominator += eps
    else:
        denominator = torch.where(denominator.abs() < eps, denominator + eps, denominator)
    return numerator / denominator


@conditional_jit_script
def _compute_rotation_matrix(angle_axis, theta2, eps: float = 1e-6):
    k_one = 1.0
    theta = torch.sqrt(theta2.clamp_min(eps))
    wxyz = angle_axis / (theta + eps)
    wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(-1, 3, 3)


@conditional_jit_script
def _compute_rotation_matrix_taylor(angle_axis):
    rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
    k_one = torch.ones_like(rx)
    rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
    return rotation_matrix.view(-1, 3, 3)


@conditional_jit_script
def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))
    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input size must be a (*, 3) tensor. Got {}".format(angle_axis.shape))
    orig_shape = angle_axis.shape
    angle_axis = angle_axis.reshape(-1, 3)
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == torch.tensor(False)).type_as(theta2)  # noqa
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    rotation_matrix[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )
    rotation_matrix = rotation_matrix.view(orig_shape[:-1] + (3, 3))
    return rotation_matrix


def _kt_normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
) -> torch.Tensor:
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")
    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")
    quaternion_norm: torch.Tensor = _kt_normalize_quaternion(quaternion)
    if order == QuaternionCoeffOrder.XYZW:
        x, y, z, w = (
            quaternion_norm[..., 0], quaternion_norm[..., 1],
            quaternion_norm[..., 2], quaternion_norm[..., 3],
        )
    else:
        w, x, y, z = (
            quaternion_norm[..., 0], quaternion_norm[..., 1],
            quaternion_norm[..., 2], quaternion_norm[..., 3],
        )
    tx = 2.0 * x; ty = 2.0 * y; tz = 2.0 * z
    twx = tx * w; twy = ty * w; twz = tz * w
    txx = tx * x; txy = ty * x; txz = tz * x
    tyy = ty * y; tyz = tz * y; tzz = tz * z
    one = torch.tensor(1.0)
    matrix = torch.stack((
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy),
    ), dim=-1).view(quaternion.shape[:-1] + (3, 3))
    return matrix


@conditional_jit_script
def quaternion_to_angle_axis(
    quaternion: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")
    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])
    if order == QuaternionCoeffOrder.XYZW:
        q1 = quaternion[..., 0]; q2 = quaternion[..., 1]
        q3 = quaternion[..., 2]; cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]; q1 = quaternion[..., 1]
        q2 = quaternion[..., 2]; q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta = torch.sqrt((sin_squared_theta).clamp_min(eps))
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        _kt_torch_safe_atan2(-sin_theta, -cos_theta),
        _kt_torch_safe_atan2(sin_theta, cos_theta),
    )
    k_pos = _kt_safe_zero_division(two_theta, sin_theta, eps)
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


@conditional_jit_script
def angle_axis_to_quaternion(
    angle_axis: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {angle_axis.shape}")
    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")
    a0 = angle_axis[..., 0:1]; a1 = angle_axis[..., 1:2]; a2 = angle_axis[..., 2:3]
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2
    theta = torch.sqrt((theta_squared).clamp_min(eps))
    half_theta = theta * 0.5
    mask = theta_squared > 0.0
    ones = torch.ones_like(half_theta)
    k_neg = 0.5 * ones
    k_pos = _kt_safe_zero_division(torch.sin(half_theta), theta, eps)
    k = torch.where(mask, k_pos, k_neg)
    w = torch.where(mask, torch.cos(half_theta), ones)
    quaternion = torch.zeros(
        size=angle_axis.shape[:-1] + (4,), dtype=angle_axis.dtype, device=angle_axis.device,
    )
    if order == QuaternionCoeffOrder.XYZW:
        quaternion[..., 0:1] = a0 * k; quaternion[..., 1:2] = a1 * k
        quaternion[..., 2:3] = a2 * k; quaternion[..., 3:4] = w
    else:
        quaternion[..., 1:2] = a0 * k; quaternion[..., 2:3] = a1 * k
        quaternion[..., 3:4] = a2 * k; quaternion[..., 0:1] = w
    return quaternion


# ============================================================
# torch_transform 函数（仅保留所需部分）
# ============================================================

def _tt_normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@conditional_jit_script
def _tt_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


@conditional_jit_script
def _tt_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@conditional_jit_script
def quat_inv(a):
    return _tt_normalize(_tt_quat_conjugate(a))


@conditional_jit_script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


_human_joints_info = None


def compute_human_joints(
    body_pose,
    global_orient,
    human_joints_info_path="gear_sonic/data/human/human_joints_info.pkl",
    use_thumb_joints=True,
):
    global _human_joints_info
    if _human_joints_info is None:
        _human_joints_info = torch.load(human_joints_info_path)
    J = _human_joints_info["J"]
    parents_list = _human_joints_info["parents_list"]
    device = body_pose.device
    J = J.to(device)
    other_pose = torch.zeros(*body_pose.shape[:-1], 99, device=device)
    full_pose = torch.cat([global_orient, body_pose, other_pose], dim=-1)
    rot_mats = angle_axis_to_rotation_matrix(full_pose.reshape(*full_pose.shape[:-1], 55, 3))
    J = J.expand(*rot_mats.shape[:-3], -1, -1)
    rel_joints = J.clone()
    rel_joints[..., 1:, :] -= J[..., parents_list[1:], :]
    transforms_mat = F.pad(
        torch.cat([rot_mats, rel_joints[..., :, None]], dim=-1), [0, 0, 0, 1], value=0.0
    )
    transforms_mat[..., 3, 3] = 1.0
    transform_chain = [transforms_mat[..., 0, :, :]]
    for i in range(1, len(parents_list)):
        transform_chain.append(
            torch.matmul(transform_chain[parents_list[i]], transforms_mat[..., i, :, :])
        )
    joints = torch.stack(transform_chain, dim=-3)[..., :3, 3]
    output_joint_index = np.arange(22)
    if use_thumb_joints:
        output_joint_index = np.concatenate([output_joint_index, np.array([39, 54])])
    joints = joints[:, output_joint_index]
    return joints


# ============================================================
# rotation_conversion 函数
# ============================================================

def _rc_quaternion_multiply_np(a, b):
    aw, ax, ay, az = np.split(a, 4, axis=1)
    bw, bx, by, bz = np.split(b, 4, axis=1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate([ow, ox, oy, oz], axis=1)


def decompose_rotation_aa(rotation_aa, v2):
    angle = np.linalg.norm(rotation_aa, axis=1)[:, None]
    w = np.cos(angle / 2)
    v = np.sin(angle / 2) * rotation_aa / angle
    q = np.concatenate([w, v], axis=1)
    v_twist = np.dot(v, v2)[:, None] * v2
    q_twist = np.concatenate([w, v_twist], axis=1)
    q_twist = q_twist / np.linalg.norm(q_twist, axis=1)[:, None]
    q_twist_inv = q_twist * np.array([1, -1, -1, -1])
    q_swing = _rc_quaternion_multiply_np(q_twist_inv, q)
    return q_twist, q_swing


# ============================================================
# isaac_utils/rotations 函数（仅保留所需部分，内部函数加前缀避免冲突）
# ============================================================

@torch.jit.script
def _rot_quat_conjugate(a: torch.Tensor, w_last: bool) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def _rot_quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)
    return quat


@torch.jit.script
def remove_smpl_base_rot(quat, w_last: bool):
    base_rot = _rot_quat_conjugate(
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat), w_last=w_last
    )
    return _rot_quat_mul(quat, base_rot.repeat(quat.shape[0], 1), w_last=w_last)


@torch.jit.script
def smpl_root_ytoz_up(root_quat_y_up) -> torch.Tensor:
    base_rot = angle_axis_to_quaternion(
        torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
    )
    root_quat_z_up = _rot_quat_mul(
        base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up, w_last=False
    )
    return root_quat_z_up


# ============================================================
# ZMQPoller
# ============================================================

class ZMQPoller:
    """简单 ZMQ 订阅者，用于非阻塞读取最新消息。"""

    def __init__(self, host: str = "localhost", port: int = 5555, topic: str = ""):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic

    def __del__(self):
        self.close()

    def get_data(self):
        if self._socket.poll(timeout=0):
            data = self._socket.recv(zmq.NOBLOCK)
            if data is None:
                print("ZMQPoller: no data received")
                return None
            return data[len(self._topic):]
        print("ZMQPoller: no data available")
        return None

    def close(self):
        self._socket.close()
        self._context.term()


# ============================================================
# zmq_planner_sender 函数
# ============================================================

_HEADER_SIZE = 1280


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > _HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {_HEADER_SIZE}")
    return header_json.ljust(_HEADER_SIZE, b"\x00")


def build_command_message(
    start: bool, stop: bool, planner: bool, delta_heading=None
) -> bytes:
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("B", 1 if start else 0),
        struct.pack("B", 1 if stop else 0),
        struct.pack("B", 1 if planner else 0),
    ))
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    header = _build_header(fields, version=1, count=1)
    return b"command" + header + payload


def build_planner_message(
    mode: int, movement, facing, speed: float = -1.0, height: float = -1.0,
    upper_body_position=None, upper_body_velocity=None,
    left_hand_position=None, right_hand_position=None,
    vr_3pt_position=None, vr_3pt_orientation=None, vr_3pt_compliance=None,
) -> bytes:
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("<i", int(mode)),
        struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
        struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
        struct.pack("<f", float(speed)),
        struct.pack("<f", float(height)),
    ))
    for name, arr, field_name in [
        ("upper_body_position", upper_body_position, "upper_body_position"),
        ("upper_body_velocity", upper_body_velocity, "upper_body_velocity"),
        ("left_hand_position", left_hand_position, "left_hand_joints"),
        ("right_hand_position", right_hand_position, "right_hand_joints"),
        ("vr_3pt_position", vr_3pt_position, "vr_position"),
        ("vr_3pt_orientation", vr_3pt_orientation, "vr_orientation"),
        ("vr_3pt_compliance", vr_3pt_compliance, "vr_compliance"),
    ]:
        if arr is not None:
            fields.append({"name": field_name, "dtype": "f32", "shape": [len(arr)]})
            for value in arr:
                payload += struct.pack("<f", float(value))
    header = _build_header(fields, version=1, count=1)
    return b"planner" + header + payload


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    fields = []
    binary_data = []
    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float32:
                dtype_str = "f32"
            elif value.dtype == np.float64:
                dtype_str = "f64"
            elif value.dtype == np.int32:
                dtype_str = "i32"
            elif value.dtype == np.int64:
                dtype_str = "i64"
            elif value.dtype == bool:
                dtype_str = "bool"
            else:
                dtype_str = "f32"
                value = value.astype(np.float32)
            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            if value.dtype.byteorder == ">":
                value = value.astype(value.dtype.newbyteorder("<"))
            binary_data.append(value.tobytes())
    header_bytes = _build_header(fields, version=version, count=1)
    topic_bytes = topic.encode("utf-8")
    data_bytes = b"".join(binary_data)
    return topic_bytes + header_bytes + data_bytes


# ============================================================
# Solver 基类
# ============================================================

class Solver(ABC):
    def __init__(self):
        pass

    def register_robot(self, robot):
        pass

    def calibrate(self, data):
        pass

    @abstractmethod
    def __call__(self, target) -> Any:
        pass


# ============================================================
# G1GripperInverseKinematicsSolver
# ============================================================

class G1GripperInverseKinematicsSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        fingertips = finger_data["position"]
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))
        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        middle_pos = positions[4 + 10, :]
        ring_pos = positions[4 + 15, :]
        pinky_pos = positions[4 + 20, :]
        index_dist = np.linalg.norm(thumb_pos - index_pos)
        middle_dist = np.linalg.norm(thumb_pos - middle_pos)
        ring_dist = np.linalg.norm(thumb_pos - ring_pos)
        pinky_dist = np.linalg.norm(thumb_pos - pinky_pos)
        dist_threshold = 0.05
        index_grip = np.clip(1.0 - index_dist, 0.0, 1.0)
        middle_grip = np.clip(1.0 - middle_dist, 0.0, 1.0)
        ring_grip = np.clip(1.0 - ring_dist, 0.0, 1.0)
        pinky_grip = np.clip(1.0 - pinky_dist, 0.0, 1.0)

        def apply_dead_zone(grip, threshold):
            if grip < threshold:
                return 0.0
            return grip

        index_grip = apply_dead_zone(index_grip, dist_threshold)
        middle_grip = apply_dead_zone(middle_grip, dist_threshold)
        ring_grip = apply_dead_zone(ring_grip, dist_threshold)
        pinky_grip = apply_dead_zone(pinky_grip, dist_threshold)
        q_open = np.zeros(7)
        grips = [index_grip, middle_grip, ring_grip, pinky_grip]
        max_grip = max(grips)
        if max_grip == 0:
            q_desired = q_open
        elif index_grip == max_grip:
            q_closed = self._get_index_close_q_desired()
            q_desired = q_open + index_grip * (q_closed - q_open)
        elif middle_grip == max_grip:
            q_closed = self._get_middle_close_q_desired()
            q_desired = q_open + middle_grip * (q_closed - q_open)
        elif ring_grip == max_grip:
            q_closed = self._get_ring_close_q_desired()
            q_desired = q_open + ring_grip * (q_closed - q_open)
        else:
            q_closed = self._get_pinky_close_q_desired()
            q_desired = q_open + pinky_grip * (q_closed - q_open)
        return q_desired

    def _get_index_close_q_desired(self):
        q_desired = np.zeros(7)
        amp0 = 0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0
        amp = 0.7
        q_desired[1] += amp; q_desired[2] += amp
        q_desired[3] -= 1.5; q_desired[4] -= 1.5
        q_desired[5] -= 0.6; q_desired[6] -= 1.5
        return q_desired if self.side == "L" else -q_desired

    def _get_middle_close_q_desired(self):
        q_desired = np.zeros(7)
        amp = 0.7
        q_desired[1] += amp; q_desired[2] += amp
        q_desired[3] -= 1.0; q_desired[4] -= 1.5
        q_desired[5] -= 1.0; q_desired[6] -= 1.5
        return q_desired if self.side == "L" else -q_desired

    def _get_ring_close_q_desired(self):
        q_desired = np.zeros(7)
        amp0 = -0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0
        amp = 0.7
        q_desired[1] += amp; q_desired[2] += amp
        q_desired[3] -= 0.6; q_desired[4] -= 1.5
        q_desired[5] -= 1.5; q_desired[6] -= 1.5
        return q_desired if self.side == "L" else -q_desired

    def _get_pinky_close_q_desired(self):
        q_desired = np.zeros(7)
        return q_desired if self.side == "L" else -q_desired


# ============================================================
# RobotSupplementalInfo
# ============================================================

@dataclass
class RobotSupplementalInfo:
    name: str
    body_actuated_joints: List[str]
    left_hand_actuated_joints: List[str]
    right_hand_actuated_joints: List[str]
    joint_groups: Dict[str, Dict[str, List[str]]]
    root_frame_name: str
    hand_frame_names: Dict[str, str]
    joint_limits: Dict[str, List[float]]
    calibration_joint_q: Mapping[str, Union[float, Mapping[str, float]]]
    joint_name_mapping: Mapping[str, Union[str, Mapping[str, str]]]
    default_joint_q: Mapping[str, Union[float, Mapping[str, float]]]
    hand_rotation_correction: np.ndarray
    teleop_upper_body_motion_scale: float


# ============================================================
# G1SupplementalInfo, ElbowPose, WaistLocation
# ============================================================

class WaistLocation(Enum):
    LOWER_BODY = "lower_body"
    UPPER_BODY = "upper_body"
    LOWER_AND_UPPER_BODY = "lower_and_upper_body"


class ElbowPose(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class G1SupplementalInfo(RobotSupplementalInfo):
    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
    ):
        name = "G1_G1ThreeFinger"
        body_actuated_joints = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        left_hand_actuated_joints = [
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        ]
        right_hand_actuated_joints = [
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        ]
        joint_limits = {
            "left_hip_pitch_joint": [-2.5307, 2.8798], "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576], "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236], "left_ankle_roll_joint": [-0.2618, 0.2618],
            "right_hip_pitch_joint": [-2.5307, 2.8798], "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576], "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236], "right_ankle_roll_joint": [-0.2618, 0.2618],
            "waist_yaw_joint": [-2.618, 2.618], "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [0.19, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618], "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, -0.19],
            "right_shoulder_yaw_joint": [-2.618, 2.618], "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
            "left_hand_thumb_0_joint": [-1.04719755, 1.04719755],
            "left_hand_thumb_1_joint": [-0.72431163, 1.04719755],
            "left_hand_thumb_2_joint": [0, 1.74532925],
            "left_hand_index_0_joint": [-1.57079632, 0], "left_hand_index_1_joint": [-1.74532925, 0],
            "left_hand_middle_0_joint": [-1.57079632, 0], "left_hand_middle_1_joint": [-1.74532925, 0],
            "right_hand_thumb_0_joint": [-1.04719755, 1.04719755],
            "right_hand_thumb_1_joint": [-0.72431163, 1.04719755],
            "right_hand_thumb_2_joint": [0, 1.74532925],
            "right_hand_index_0_joint": [-1.57079632, 0], "right_hand_index_1_joint": [-1.74532925, 0],
            "right_hand_middle_0_joint": [-1.57079632, 0], "right_hand_middle_1_joint": [-1.74532925, 0],
        }
        joint_groups = {
            "waist": {"joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], "groups": []},
            "left_leg": {"joints": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"], "groups": []},
            "right_leg": {"joints": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"], "groups": []},
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            "left_arm": {"joints": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"], "groups": []},
            "right_arm": {"joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"], "groups": []},
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            "left_hand": {"joints": ["left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"], "groups": []},
            "right_hand": {"joints": ["right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"], "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }
        joint_name_mapping = {
            "waist_pitch": "waist_pitch_joint", "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            "shoulder_pitch": {"left": "left_shoulder_pitch_joint", "right": "right_shoulder_pitch_joint"},
            "shoulder_roll": {"left": "left_shoulder_roll_joint", "right": "right_shoulder_roll_joint"},
            "shoulder_yaw": {"left": "left_shoulder_yaw_joint", "right": "right_shoulder_yaw_joint"},
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
        }
        root_frame_name = "pelvis"
        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
        calibration_joint_q = {"elbow_pitch": {"left": 0.0, "right": 0.0}}
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
            }
        else:
            default_joint_q = {"shoulder_roll": {"left": 0.2, "right": -0.2}}
        teleop_upper_body_motion_scale = 1.0
        modified_joint_groups = joint_groups.copy()
        if waist_location == WaistLocation.UPPER_BODY:
            modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            modified_joint_groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}
        elif waist_location == WaistLocation.LOWER_AND_UPPER_BODY:
            modified_joint_groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}
        super().__init__(
            name=name, body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits, joint_groups=modified_joint_groups,
            root_frame_name=root_frame_name, hand_frame_names=hand_frame_names,
            calibration_joint_q=calibration_joint_q, joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction, default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )


# ============================================================
# RobotModel（需要 pinocchio）
# ============================================================

try:
    import pinocchio as pin
    _PINOCCHIO_AVAILABLE = True
except ImportError:
    pin = None
    _PINOCCHIO_AVAILABLE = False


class RobotModel:
    def __init__(
        self,
        urdf_path,
        asset_path,
        set_floating_base=False,
        supplemental_info: Optional[RobotSupplementalInfo] = None,
    ):
        if not _PINOCCHIO_AVAILABLE:
            raise ImportError("pinocchio is required for RobotModel")
        self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[asset_path],
            root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
        )
        self.is_floating_base_model = set_floating_base
        self.joint_to_dof_index = {}
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if set_floating_base
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q
        root_nq = 7 if set_floating_base else 0
        self.upper_joint_limits = self.pinocchio_wrapper.model.upperPositionLimit[root_nq:].copy()
        self.lower_joint_limits = self.pinocchio_wrapper.model.lowerPositionLimit[root_nq:].copy()
        self.supplemental_info = supplemental_info
        if self.supplemental_info is not None:
            self._body_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.body_actuated_joints
            ]
            self._left_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.left_hand_actuated_joints
            ]
            self._right_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.right_hand_actuated_joints
            ]
            self._hand_actuated_joint_indices = (
                self._left_hand_actuated_joint_indices + self._right_hand_actuated_joint_indices
            )
            self._joint_group_indices = {}
            for group_name, group_info in self.supplemental_info.joint_groups.items():
                indices = []
                indices.extend([self.dof_index(name) for name in group_info["joints"]])
                for subgroup_name in group_info["groups"]:
                    indices.extend(self.get_joint_group_indices(subgroup_name))
                self._joint_group_indices[group_name] = sorted(set(indices))
            if hasattr(self.supplemental_info, "joint_limits") and self.supplemental_info.joint_limits:
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]
        self.default_body_pose = self.q_zero
        if self.supplemental_info is not None:
            default_joint_q = self.supplemental_info.default_joint_q
            for joint, joint_values in default_joint_q.items():
                joint_mapping = self.supplemental_info.joint_name_mapping[joint]
                if isinstance(joint_mapping, str):
                    if joint_mapping in self.joint_to_dof_index:
                        joint_idx = self.dof_index(joint_mapping)
                        self.default_body_pose[joint_idx] = joint_values
                else:
                    for side, value in joint_values.items():
                        if side in joint_mapping and joint_mapping[side] in self.joint_to_dof_index:
                            joint_idx = self.dof_index(joint_mapping[side])
                            self.default_body_pose[joint_idx] = value
        self.initial_body_pose = self.default_body_pose.copy()

    @property
    def num_dofs(self) -> int:
        return self.pinocchio_wrapper.model.nq

    @property
    def q_zero(self) -> np.ndarray:
        return self.pinocchio_wrapper.q0.copy()

    @property
    def joint_names(self) -> List[str]:
        return list(self.joint_to_dof_index.keys())

    @property
    def num_joints(self) -> int:
        return len(self.joint_to_dof_index)

    def dof_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_to_dof_index:
            raise ValueError(f"Unknown joint name: '{joint_name}'.")
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided")
        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_joint_group_indices(self, group_names) -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided")
        if isinstance(group_names, str):
            group_names = {group_names}
        all_indices: Set[int] = set()
        for group_name in group_names:
            if group_name not in self._joint_group_indices:
                raise ValueError(f"Unknown joint group: {group_name}")
            all_indices.update(self._joint_group_indices[group_name])
        return sorted(all_indices)

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip=True) -> None:
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]}")
        if auto_clip:
            q = self.clip_configuration(q)
        pin.framesForwardKinematics(self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q)

    def clip_configuration(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        q_clipped = q.copy()
        root_nq = 7 if self.is_floating_base_model else 0
        q_clipped[root_nq:] = np.clip(
            q[root_nq:], self.lower_joint_limits + margin, self.upper_joint_limits - margin
        )
        return q_clipped

    def frame_placement(self, frame_name: str):
        model = self.pinocchio_wrapper.model
        data = self.pinocchio_wrapper.data
        frame_id = model.getFrameId(frame_name)
        if frame_id < 0 or frame_id >= len(model.frames):
            valid_frames = [f.name for f in model.frames]
            raise ValueError(f"Unknown frame '{frame_name}'. Valid frames: {valid_frames}")
        return data.oMf[frame_id].copy()

    def get_configuration_from_actuated_joints(
        self,
        body_actuated_joint_values: np.ndarray,
        hand_actuated_joint_values: Optional[np.ndarray] = None,
        left_hand_actuated_joint_values: Optional[np.ndarray] = None,
        right_hand_actuated_joint_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q = self.pinocchio_wrapper.q0.copy()
        q[self.get_body_actuated_joint_indices()] = body_actuated_joint_values
        if hand_actuated_joint_values is not None:
            q[self.get_hand_actuated_joint_indices("both")] = hand_actuated_joint_values
        else:
            if left_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("left")] = left_hand_actuated_joint_values
            if right_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("right")] = right_hand_actuated_joint_values
        return q

    def get_default_body_pose(self) -> np.ndarray:
        return self.default_body_pose.copy()

    def set_initial_body_pose(self, q: np.ndarray, q_idx=None) -> None:
        if q_idx is None:
            self.initial_body_pose = q
        else:
            self.initial_body_pose[q_idx] = q


# ============================================================
# instantiate_g1_robot_model
# ============================================================

def instantiate_g1_robot_model(
    waist_location: str = "lower_body",
    high_elbow_pose: bool = False,
) -> RobotModel:
    # G1 资源路径：gear_sonic/data/robot_model/model_data/g1
    model_data_dir = Path(__file__).resolve().parent.parent / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
    robot_model_config = {
        "asset_path": str(model_data_dir),
        "urdf_path": str(model_data_dir / "g1_29dof_with_hand.urdf"),
    }
    assert waist_location in ["lower_body", "upper_body", "lower_and_upper_body"]
    waist_location_enum = {
        "lower_body": WaistLocation.LOWER_BODY,
        "upper_body": WaistLocation.UPPER_BODY,
        "lower_and_upper_body": WaistLocation.LOWER_AND_UPPER_BODY,
    }[waist_location]
    elbow_pose_enum = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW
    robot_model_supplemental_info = G1SupplementalInfo(
        waist_location=waist_location_enum, elbow_pose=elbow_pose_enum
    )
    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model


# ============================================================
# VR3PtPoseVisualizer 和 get_g1_key_frame_poses
# （从 gear_sonic/utils/teleop/vis/vr3pt_pose_visualizer.py 内联，
#   将 gear_sonic.data.robot_model.instantiation.g1 替换为本文件中的 instantiate_g1_robot_model）
# ============================================================

import time as _time
from collections import deque as _deque

try:
    import pyvista as pv
    _PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    _PYVISTA_AVAILABLE = False

try:
    import vtk
    _VTK_AVAILABLE = True
except ImportError:
    vtk = None
    _VTK_AVAILABLE = False

try:
    import pinocchio as _pin_vis
    _PINOCCHIO_VIS_AVAILABLE = True
except ImportError:
    _pin_vis = None
    _PINOCCHIO_VIS_AVAILABLE = False

G1_LEFT_WRIST_FRAME = "left_wrist_yaw_link"
G1_RIGHT_WRIST_FRAME = "right_wrist_yaw_link"
G1_TORSO_FRAME = "torso_link"

G1_KEY_FRAME_OFFSETS = {
    "left_wrist": np.array([0.18, -0.025, 0.0]),
    "right_wrist": np.array([0.18, 0.025, 0.0]),
    "torso": np.array([0.0, 0.0, 0.35]),
}

G1_FRAME_MAPPING = {
    "left_wrist": G1_LEFT_WRIST_FRAME,
    "right_wrist": G1_RIGHT_WRIST_FRAME,
    "torso": G1_TORSO_FRAME,
}

# TienKung2 Pro 的 frame 映射（link 名称来自 tienkung2_pro.urdf）
TIENKUNG2_PRO_FRAME_MAPPING = {
    "left_wrist": "wrist_roll_l_link",
    "right_wrist": "wrist_roll_r_link",
    "torso": "body_yaw_link",
}

# TienKung2 Pro wrist frame 的局部偏移（沿 link 局部 X 轴方向）
TIENKUNG2_PRO_KEY_FRAME_OFFSETS = {
    "left_wrist": np.array([0.0, 0.0, 0.0]),
    "right_wrist": np.array([0.0, 0.0, 0.0]),
    "torso": np.array([0.0, 0.0, 0.0]),
}


def get_g1_key_frame_poses(
    robot_model,
    q: np.ndarray = None,
    root_position: np.ndarray = None,
    apply_offset: bool = True,
    frame_mapping: Dict[str, str] = None,
    key_frame_offsets: Dict[str, np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if q is None:
        q = robot_model.default_body_pose
    if root_position is None:
        root_position = np.array([0.0, 0.0, 0.0])
    if frame_mapping is None:
        frame_mapping = G1_FRAME_MAPPING
    if key_frame_offsets is None:
        key_frame_offsets = G1_KEY_FRAME_OFFSETS
    robot_model.cache_forward_kinematics(q, auto_clip=False)
    result = {}
    for key, frame_name in frame_mapping.items():
        try:
            placement = robot_model.frame_placement(frame_name)
        except ValueError as e:
            raise RuntimeError(
                f"Cannot find frame '{frame_name}' (key='{key}'). Original error: {e}"
            ) from e
        rotation_matrix = placement.rotation
        if apply_offset and key in key_frame_offsets:
            local_offset = key_frame_offsets[key]
            world_offset = rotation_matrix @ local_offset
            position = placement.translation + world_offset + root_position
        else:
            position = placement.translation + root_position
        rot = sRot.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        result[key] = {
            "position": position.copy(),
            "orientation_xyzw": quat_xyzw.copy(),
            "orientation_wxyz": quat_wxyz.copy(),
        }
    return result


class G1RobotVisualizer:
    ROBOT_COLOR = "#404040"
    ROBOT_OPACITY = 0.1
    LEFT_WRIST_FRAME = G1_LEFT_WRIST_FRAME
    RIGHT_WRIST_FRAME = G1_RIGHT_WRIST_FRAME
    TORSO_FRAME = G1_TORSO_FRAME
    KEY_FRAME_OFFSETS = G1_KEY_FRAME_OFFSETS
    KEY_POINT_COLORS = {"left_wrist": "lightgreen", "right_wrist": "lightblue", "torso": "yellow"}
    KEY_POINT_LABELS = {
        "left_wrist": "L-Wrist (with offset)",
        "right_wrist": "R-Wrist (with offset)",
        "torso": "Torso (with offset)",
    }
    WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    def __init__(self, robot_model=None):
        if not _PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")
        if not _PINOCCHIO_VIS_AVAILABLE:
            raise ImportError("Pinocchio is required. Install with: pip install pin")
        self.robot_model = robot_model if robot_model is not None else instantiate_g1_robot_model()
        gear_sonic_root = Path(__file__).resolve().parent.parent / "gear_sonic"
        self.mesh_dir = gear_sonic_root / "data" / "robot_model" / "model_data" / "g1" / "meshes"
        self._load_visual_geometries()
        self.mesh_actors: Dict[str, any] = {}
        self.key_point_actors: Dict[str, any] = {}
        self._initialized = False
        self._key_points_initialized = False
        self._waist_joint_indices: Optional[List[int]] = None
        try:
            self._waist_joint_indices = self.robot_model.get_joint_group_indices("waist")
        except (ValueError, AttributeError) as e:
            raise RuntimeError(f"Could not get waist joint indices: {e}") from e

    def compute_waist_joints_from_orientation(self, neck_quat_wxyz, scale_factor=1.0):
        if self._waist_joint_indices is None:
            return None
        quat_xyzw = np.array([neck_quat_wxyz[1], neck_quat_wxyz[2], neck_quat_wxyz[3], neck_quat_wxyz[0]])
        rot = sRot.from_quat(quat_xyzw)
        euler_zyx = rot.as_euler("ZYX", degrees=False)
        waist_yaw = euler_zyx[0] * scale_factor
        waist_roll = euler_zyx[2] * scale_factor
        waist_pitch = euler_zyx[1] * scale_factor
        return np.array([waist_yaw, waist_roll, waist_pitch])

    def apply_waist_joints_to_config(self, q, waist_joints):
        if self._waist_joint_indices is None or waist_joints is None:
            return q
        q_new = q.copy()
        for i, idx in enumerate(self._waist_joint_indices):
            if i < len(waist_joints):
                q_new[idx] = waist_joints[i]
        return q_new

    def _load_visual_geometries(self):
        self.visual_geometries: List[Dict] = []
        visual_model = self.robot_model.pinocchio_wrapper.visual_model
        model = self.robot_model.pinocchio_wrapper.model
        if len(visual_model.geometryObjects) == 0:
            raise RuntimeError("No visual geometries found in Pinocchio visual model.")
        for geom_id, geom in enumerate(visual_model.geometryObjects):
            mesh_path = geom.meshPath
            if not mesh_path:
                continue
            frame_id = geom.parentFrame
            frame_name = model.frames[frame_id].name if frame_id < len(model.frames) else None
            local_placement = geom.placement
            self.visual_geometries.append({
                "geom_id": geom_id, "mesh_path": str(mesh_path),
                "frame_id": frame_id, "frame_name": frame_name,
                "local_placement": local_placement, "mesh": None,
            })

    def _load_mesh(self, mesh_path: str):
        import os as _os
        original_path = mesh_path
        if not _os.path.exists(mesh_path):
            mesh_name = _os.path.basename(mesh_path)
            mesh_path = str(self.mesh_dir / mesh_name)
        if not _os.path.exists(mesh_path):
            raise FileNotFoundError(f"Robot mesh file not found: '{original_path}'")
        try:
            return pv.read(mesh_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load robot mesh file '{mesh_path}': {e}") from e

    def _get_geometry_world_transform(self, geom_info, q=None):
        if q is None:
            q = self.robot_model.q_zero
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        frame_placement = self.robot_model.pinocchio_wrapper.data.oMf[geom_info["frame_id"]]
        world_placement = frame_placement * geom_info["local_placement"]
        return world_placement.translation, world_placement.rotation

    def add_to_plotter(self, plotter, q=None, color=None, opacity=None, root_position=None):
        if q is None:
            q = self.robot_model.default_body_pose
        if color is None:
            color = self.ROBOT_COLOR
        if opacity is None:
            opacity = self.ROBOT_OPACITY
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])
        actors = {}
        for geom_info in self.visual_geometries:
            if geom_info["mesh"] is None:
                geom_info["mesh"] = self._load_mesh(geom_info["mesh_path"])
            position, rotation = self._get_geometry_world_transform(geom_info, q)
            position = position + root_position
            mesh = geom_info["mesh"].copy()
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position
            mesh.transform(transform)
            actor = plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, name=geom_info["frame_name"])
            actors[geom_info["frame_name"]] = {"actor": actor, "geom_info": geom_info}
        self.mesh_actors = actors
        self._initialized = True
        return actors

    def add_to_plotter_realtime(self, plotter, q=None, color=None, opacity=None, root_position=None):
        if not _VTK_AVAILABLE:
            raise ImportError("VTK is required for real-time mode")
        if q is None:
            q = self.robot_model.default_body_pose
        if color is None:
            color = self.ROBOT_COLOR
        if opacity is None:
            opacity = self.ROBOT_OPACITY
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])
        actors = {}
        for geom_info in self.visual_geometries:
            if geom_info["mesh"] is None:
                geom_info["mesh"] = self._load_mesh(geom_info["mesh_path"])
            mesh = geom_info["mesh"].copy()
            actor = plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, name=geom_info["frame_name"])
            actors[geom_info["frame_name"]] = {"actor": actor, "geom_info": geom_info}
        self.mesh_actors = actors
        self._root_position = root_position
        self._initialized = True
        self.update_pose(q, root_position)
        return actors

    def add_key_points_to_plotter(self, plotter, q=None, root_position=None, axis_length=0.08, ball_radius=0.02, show_axes=True):
        if q is None:
            q = self.robot_model.default_body_pose
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])
        poses = get_g1_key_frame_poses(self.robot_model, q=q, root_position=root_position)
        axis_colors = ["red", "green", "blue"]
        axis_dirs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        for key, pose in poses.items():
            position = pose["position"]
            quat_xyzw = pose["orientation_xyzw"]
            rot = sRot.from_quat(quat_xyzw)
            rot_matrix = rot.as_matrix()
            if show_axes:
                for i, (color, local_dir) in enumerate(zip(axis_colors, axis_dirs)):
                    world_dir = rot_matrix @ local_dir
                    arrow = pv.Arrow(start=position, direction=world_dir, scale=axis_length, tip_length=0.3, tip_radius=0.15, shaft_radius=0.05)
                    plotter.add_mesh(arrow, color=color, smooth_shading=True)
            ball = pv.Sphere(radius=ball_radius, center=position)
            plotter.add_mesh(ball, color=self.KEY_POINT_COLORS[key], smooth_shading=True, name=f"keypoint_{key}")
            plotter.add_point_labels([position + np.array([0, 0, ball_radius * 2])], [self.KEY_POINT_LABELS[key]], font_size=10, point_color=self.KEY_POINT_COLORS[key], text_color="white", always_visible=True, shape_opacity=0.7)
        return poses

    def add_key_points_realtime(self, plotter, q=None, root_position=None, axis_length=0.08, ball_radius=0.02):
        if not _VTK_AVAILABLE:
            raise ImportError("VTK is required for real-time mode")
        if q is None:
            q = self.robot_model.default_body_pose
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])
        self.key_point_actors = {}
        axis_colors = ["red", "green", "blue"]
        for key in ["left_wrist", "right_wrist", "torso"]:
            actors = {"arrows": [], "ball": None}
            for color in axis_colors:
                arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=axis_length, tip_length=0.3, tip_radius=0.15, shaft_radius=0.05)
                actor = plotter.add_mesh(arrow, color=color, smooth_shading=True)
                actors["arrows"].append(actor)
            ball = pv.Sphere(radius=ball_radius, center=(0, 0, 0))
            actors["ball"] = plotter.add_mesh(ball, color=self.KEY_POINT_COLORS[key], smooth_shading=True)
            self.key_point_actors[key] = actors
        self._key_points_initialized = True
        self._key_point_axis_length = axis_length
        poses = self.update_key_points(q, root_position)
        return poses

    def update_key_points(self, q, root_position=None):
        if not self._key_points_initialized or not _VTK_AVAILABLE:
            return {}
        if root_position is None:
            root_position = getattr(self, "_root_position", np.array([0.0, 0.0, 0.0]))
        poses = get_g1_key_frame_poses(self.robot_model, q=q, root_position=root_position)
        axis_dirs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        for key, pose in poses.items():
            if key not in self.key_point_actors:
                continue
            position = pose["position"]
            quat_xyzw = pose["orientation_xyzw"]
            rot = sRot.from_quat(quat_xyzw)
            rot_matrix = rot.as_matrix()
            actors = self.key_point_actors[key]
            for j, local_dir in enumerate(axis_dirs):
                world_dir = rot_matrix @ local_dir
                x_axis = np.array([1.0, 0.0, 0.0])
                v = np.cross(x_axis, world_dir)
                c = np.dot(x_axis, world_dir)
                if np.linalg.norm(v) > 1e-6:
                    s = np.linalg.norm(v)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    arrow_rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-9))
                elif c < 0:
                    arrow_rot = np.diag([-1.0, 1.0, -1.0])
                else:
                    arrow_rot = np.eye(3)
                transform = vtk.vtkTransform()
                mat = vtk.vtkMatrix4x4()
                for ri in range(3):
                    for ci in range(3):
                        mat.SetElement(ri, ci, arrow_rot[ri, ci])
                mat.SetElement(0, 3, position[0]); mat.SetElement(1, 3, position[1]); mat.SetElement(2, 3, position[2])
                transform.SetMatrix(mat)
                actors["arrows"][j].SetUserTransform(transform)
            ball_transform = vtk.vtkTransform()
            ball_transform.Translate(position[0], position[1], position[2])
            actors["ball"].SetUserTransform(ball_transform)
        return poses

    def update_pose(self, q, root_position=None):
        if not self._initialized or not _VTK_AVAILABLE:
            return None
        if root_position is None:
            root_position = getattr(self, "_root_position", np.array([0.0, 0.0, 0.0]))
        else:
            self._root_position = root_position
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        for name, actor_info in self.mesh_actors.items():
            geom_info = actor_info["geom_info"]
            actor = actor_info["actor"]
            frame_placement = self.robot_model.pinocchio_wrapper.data.oMf[geom_info["frame_id"]]
            world_placement = frame_placement * geom_info["local_placement"]
            position = world_placement.translation + root_position
            rotation = world_placement.rotation
            transform = vtk.vtkTransform()
            mat = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    mat.SetElement(i, j, rotation[i, j])
            mat.SetElement(0, 3, position[0]); mat.SetElement(1, 3, position[1]); mat.SetElement(2, 3, position[2])
            transform.SetMatrix(mat)
            actor.SetUserTransform(transform)
        if self._key_points_initialized:
            return self.update_key_points(q, root_position)
        return None


# ============================================================
# TienKung2ProSupplementalInfo
# ============================================================

@dataclass
class TienKung2ProSupplementalInfo(RobotSupplementalInfo):
    def __init__(self):
        name = "TienKung2Pro"
        body_actuated_joints = [
            "body_yaw_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
        ]
        left_hand_actuated_joints = []
        right_hand_actuated_joints = []
        joint_limits = {
            "body_yaw_joint": [-2.96706, 3.14159],
            "shoulder_pitch_l_joint": [-2.96706, 2.96706],
            "shoulder_roll_l_joint": [-1.0472, 1.8326],
            "shoulder_yaw_l_joint": [-2.96706, 2.96706],
            "elbow_pitch_l_joint": [-2.61799, 0.261799],
            "elbow_yaw_l_joint": [-2.96706, 2.96706],
            "wrist_pitch_l_joint": [-0.785398, 1.0472],
            "wrist_roll_l_joint": [-1.65806, 1.309],
            "shoulder_pitch_r_joint": [-2.96706, 2.96706],
            "shoulder_roll_r_joint": [-1.8326, 1.0472],
            "shoulder_yaw_r_joint": [-2.96706, 2.96706],
            "elbow_pitch_r_joint": [-2.61799, 0.261799],
            "elbow_yaw_r_joint": [-2.96706, 2.96706],
            "wrist_pitch_r_joint": [-0.785398, 1.0472],
            "wrist_roll_r_joint": [-1.309, 1.65806],
            "hip_roll_l_joint": [-0.785398, 0.785398],
            "hip_pitch_l_joint": [-2.79253, 2.0944],
            "hip_yaw_l_joint": [-1.0472, 1.0472],
            "knee_pitch_l_joint": [0.0, 2.3911],
            "ankle_pitch_l_joint": [-1.22173, 0.523599],
            "ankle_roll_l_joint": [-0.523599, 0.523599],
            "hip_roll_r_joint": [-0.785398, 0.785398],
            "hip_pitch_r_joint": [-2.79253, 2.0944],
            "hip_yaw_r_joint": [-1.0472, 1.0472],
            "knee_pitch_r_joint": [0.0, 2.3911],
            "ankle_pitch_r_joint": [-1.22173, 0.523599],
            "ankle_roll_r_joint": [-0.523599, 0.523599],
        }
        joint_groups = {
            "waist": {"joints": ["body_yaw_joint"], "groups": []},
            "left_leg": {"joints": ["hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint"], "groups": []},
            "right_leg": {"joints": ["hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint"], "groups": []},
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            "left_arm": {"joints": ["shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint"], "groups": []},
            "right_arm": {"joints": ["shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"], "groups": []},
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands"]},
        }
        joint_name_mapping = {
            "waist_yaw": "body_yaw_joint",
            "shoulder_pitch": {"left": "shoulder_pitch_l_joint", "right": "shoulder_pitch_r_joint"},
            "shoulder_roll": {"left": "shoulder_roll_l_joint", "right": "shoulder_roll_r_joint"},
            "shoulder_yaw": {"left": "shoulder_yaw_l_joint", "right": "shoulder_yaw_r_joint"},
            "elbow_pitch": {"left": "elbow_pitch_l_joint", "right": "elbow_pitch_r_joint"},
            "elbow_yaw": {"left": "elbow_yaw_l_joint", "right": "elbow_yaw_r_joint"},
            "wrist_pitch": {"left": "wrist_pitch_l_joint", "right": "wrist_pitch_r_joint"},
            "wrist_roll": {"left": "wrist_roll_l_joint", "right": "wrist_roll_r_joint"},
        }
        root_frame_name = "Base_link"
        hand_frame_names = {"left": "wrist_roll_l_link", "right": "wrist_roll_r_link"}
        calibration_joint_q = {"elbow_pitch": {"left": 0.0, "right": 0.0}}
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        default_joint_q = {
            "shoulder_roll": {"left": 0.1, "right": -0.1},
            "elbow_pitch": {"left": -0.3, "right": -0.3},
        }
        teleop_upper_body_motion_scale = 1.0
        super().__init__(
            name=name, body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits, joint_groups=joint_groups,
            root_frame_name=root_frame_name, hand_frame_names=hand_frame_names,
            calibration_joint_q=calibration_joint_q, joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction, default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )


# ============================================================
# instantiate_tienkung2_pro_robot_model
# 资源路径：out/assets/（相对于本文件所在目录）
# ============================================================

def instantiate_tienkung2_pro_robot_model() -> RobotModel:
    """实例化 TienKung2 Pro 机器人模型，资源从 out/assets/ 加载。"""
    assets_dir = Path(__file__).resolve().parent / "assets"
    urdf_path = assets_dir / "tienkung2_pro.urdf"
    # STL 网格文件直接在 assets/tienkung2_pro/ 目录下
    meshes_dir = assets_dir / "tienkung2_pro"
    supplemental_info = TienKung2ProSupplementalInfo()
    return RobotModel(
        str(urdf_path),
        str(meshes_dir),
        supplemental_info=supplemental_info,
    )
