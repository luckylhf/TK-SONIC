"""PICO 全身遥操作 MuJoCo 仿真 — 使用 smpl.onnx 模型。

数据流：
    pico_manager_thread_server.py (ZMQ topic="pose")
        ↓ smpl_joints (N,24,3)  body_quat_w (N,4)  joint_pos (N,29)
    SmplObsBuilder
        ├── smpl_joints_multi_future_local_nonflat  (10,72)
        ├── smpl_root_ori_b_multi_future            (10,6)
        └── joint_pos_multi_future_wrist_for_smpl   (10,6)
        + proprioception                             (870,)
        ↓
    smpl ONNX → action (27,) IsaacLab 顺序
        ↓
    MuJoCo PD 控制

用法（实时 PICO）：
    ./gear_sonic/shell/run_smpl_mujoco_onnx.sh
    或
    /isaac-sim/python.sh gear_sonic/scripts/run_smpl_mujoco_onnx.py \\
        --onnx <path/to/model_step_*_smpl.onnx> \\
        --zmq-host localhost --zmq-port 5555

用法（无 PICO，用 pkl 文件回放验证）：
    /isaac-sim/python.sh gear_sonic/scripts/run_smpl_mujoco_onnx.py \\
        --onnx <path/to/model_step_*_smpl.onnx> \\
        --replay-pkl data/motion_lib_bones_seed/tienkung2_pro_filtered/230112/alone_002__A116_M.pkl
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
import zmq
from scipy.spatial.transform import Rotation


# ── 常量（与 run_policy_mujoco_onnx.py 保持一致）─────────────────────────────
MJCF_PATH = "assets/tienkung2_pro.xml"
NUM_JOINTS = 27
QPOS_OFFSET = 7

DEFAULT_QPOS = np.array([
    0.0,
    0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
], dtype=np.float32)

KP = np.array([200, 60, 20, 10, 10, 10, 5, 5,
               60, 20, 10, 10, 10, 5, 5,
               700, 700, 500, 700, 30, 15,
               700, 700, 500, 700, 30, 15], dtype=np.float32)
KD = np.array([10, 3, 1.5, 1, 1, 1, 0.5, 0.5,
               3, 1.5, 1, 1, 1, 0.5, 0.5,
               20, 20, 15, 10, 1.25, 1.25,
               20, 20, 15, 10, 1.25, 1.25], dtype=np.float32)

# factor=0.25 + overrides（与训练一致）
ACTION_SCALE = np.array([
    0.1250,
    0.2188, 0.3938, 0.3150, 0.3150, 0.0, 0.0, 0.0,
    0.2188, 0.3938, 0.3150, 0.3150, 0.0, 0.0, 0.0,
    0.0839, 0.1179, 0.1175, 0.1179, 0.1833, 0.1833,
    0.0839, 0.1179, 0.1175, 0.1179, 0.1833, 0.1833,
], dtype=np.float32)

# MuJoCo DOF → IsaacLab 顺序映射
MUJOCO_TO_ISAACLAB = [
    0,  3,  7, 11, 15, 19, 23, 25,
    4,  8, 12, 16, 20, 24, 26,
    1,  5,  9, 13, 17, 21,
    2,  6, 10, 14, 18, 22,
]
ISAACLAB_TO_MUJOCO = [0] * 27
for _mj, _il in enumerate(MUJOCO_TO_ISAACLAB):
    ISAACLAB_TO_MUJOCO[_il] = _mj

# MuJoCo DOF → Actuator 顺序
DOF_TO_ACTUATOR = [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# smpl encoder 专用常量
NUM_SMPL_FUTURE = 10          # smpl_num_future_frames
NUM_SMPL_JOINTS = 24          # SMPL 关节数
# joint_pos_multi_future_wrist_for_smpl 的 IsaacLab 索引
# 与训练配置 joints_idx: [15, 16, 25, 26, 23, 24] 一致
# 15=elbow_pitch_l, 16=elbow_pitch_r, 25=wrist_roll_l, 26=wrist_roll_r, 23=wrist_pitch_l, 24=wrist_pitch_r
WRIST_ISAACLAB_IDX = [15, 16, 25, 26, 23, 24]

# SMPL base rotation（wxyz）：[0.5, 0.5, 0.5, 0.5] 的共轭 = [0.5, -0.5, -0.5, -0.5]
# pico_manager_thread_server 已经 remove_smpl_base_rot，body_quat_w 无需再处理
_SMPL_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)  # wxyz


# ── ZMQ 消息解包（与 run_data_exporter.py 的 unpack_pose_message 一致）────────
HEADER_SIZE = 1280

def _unpack_pose_message(data: bytes, topic: str = "pose") -> dict | None:
    topic_b = topic.encode()
    if not data.startswith(topic_b):
        return None
    off = len(topic_b)
    if len(data) < off + HEADER_SIZE:
        return None
    hdr_raw = data[off: off + HEADER_SIZE]
    null = hdr_raw.find(b"\x00")
    hdr = json.loads(hdr_raw[:null].decode() if null > 0 else hdr_raw.decode())
    dtype_map = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}
    result = {}
    cur = off + HEADER_SIZE
    for field in hdr.get("fields", []):
        dt = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        nb = int(np.prod(shape)) * np.dtype(dt).itemsize
        result[field["name"]] = np.frombuffer(data[cur: cur + nb], dtype=dt).reshape(shape).copy()
        cur += nb
    return result


# ── PICO ZMQ 接收器 ────────────────────────────────────────────────────────────
class PicoReceiver:
    """非阻塞 ZMQ 订阅器，接收 pico_manager_thread_server 发布的 pose 消息。"""

    def __init__(self, host: str, port: int, topic: str = "pose"):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sock.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞
        self._latest: dict | None = None

    def poll(self) -> dict | None:
        """尝试接收最新消息，无消息时返回 None。"""
        try:
            while True:
                raw = self.sock.recv(zmq.NOBLOCK)
                msg = _unpack_pose_message(raw)
                if msg is not None:
                    self._latest = msg
        except zmq.Again:
            pass
        return self._latest


# ── MuJoCo 辅助函数 ────────────────────────────────────────────────────────────
def _get_root_rot(mj_data, root_id) -> Rotation:
    q = mj_data.xquat[root_id]  # [w,x,y,z]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _gravity_local(mj_data, root_id) -> np.ndarray:
    return _get_root_rot(mj_data, root_id).inv().apply([0, 0, -1]).astype(np.float32)


# ── 四元数工具（numpy，wxyz 格式）─────────────────────────────────────────────
def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz 四元数求逆（单位四元数）。"""
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz 四元数乘法，支持 broadcast。"""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1).astype(np.float32)


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用 wxyz 四元数旋转向量 v，支持 batch。"""
    # 转为 xyzw 给 scipy
    q_xyzw = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)
    return Rotation.from_quat(q_xyzw.reshape(-1, 4)).apply(v.reshape(-1, 3)).reshape(v.shape).astype(np.float32)


def _quat_to_6d(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 6D 旋转（旋转矩阵前两列，行优先展平）。"""
    q_xyzw = np.concatenate([q_wxyz[..., 1:], q_wxyz[..., :1]], axis=-1)
    mat = Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_matrix().reshape(*q_wxyz.shape[:-1], 3, 3)
    return mat[..., :2].reshape(*q_wxyz.shape[:-1], 6).astype(np.float32)


# ── 观测构建器 ─────────────────────────────────────────────────────────────────
class SmplObsBuilder:
    """构建 smpl encoder 的完整 ONNX 输入。

    Tokenizer obs 布局（总计 267+870=1137D）：
        smpl_joints_multi_future_local_nonflat  (10,72) → 720D
        smpl_root_ori_b_multi_future            (10,6)  →  60D
        joint_pos_multi_future_wrist_for_smpl   (10,6)  →  60D
        proprioception                                  → 870D

    Proprioception 布局（PolicyCfg 字段顺序）：
        ang_vel(30) | jpos(270) | jvel(270) | action(270) | gravity(30)
    """

    PROP_HISTORY = 10

    def __init__(self, mj_model, root_id: int, expected_dim: int):
        self.root_id = root_id
        self.expected_dim = expected_dim

        self._ang_vel_hist  = np.zeros((self.PROP_HISTORY, 3),          dtype=np.float32)
        self._jpos_hist     = np.zeros((self.PROP_HISTORY, NUM_JOINTS),  dtype=np.float32)
        self._jvel_hist     = np.zeros((self.PROP_HISTORY, NUM_JOINTS),  dtype=np.float32)
        self._action_hist   = np.zeros((self.PROP_HISTORY, NUM_JOINTS),  dtype=np.float32)
        self._grav_hist     = np.zeros((self.PROP_HISTORY, 3),           dtype=np.float32)
        self.prev_action    = np.zeros(NUM_JOINTS,                       dtype=np.float32)

    def reset(self):
        for arr in (self._ang_vel_hist, self._jpos_hist, self._jvel_hist,
                    self._action_hist, self._grav_hist):
            arr[:] = 0
        self.prev_action[:] = 0

    def update_prop(self, mj_data):
        jpos_mj = mj_data.qpos[QPOS_OFFSET: QPOS_OFFSET + NUM_JOINTS].astype(np.float32)
        jvel_mj = mj_data.qvel[6: 6 + NUM_JOINTS].astype(np.float32)
        grav    = _gravity_local(mj_data, self.root_id)
        ang_vel_world = mj_data.qvel[3:6].astype(np.float32)
        ang_vel = _get_root_rot(mj_data, self.root_id).inv().apply(ang_vel_world).astype(np.float32)

        def _shift(buf, val):
            buf[:-1] = buf[1:]
            buf[-1]  = val

        _shift(self._ang_vel_hist,  ang_vel)
        _shift(self._jpos_hist,     (jpos_mj - DEFAULT_QPOS)[ISAACLAB_TO_MUJOCO])
        _shift(self._jvel_hist,     jvel_mj[ISAACLAB_TO_MUJOCO])
        _shift(self._action_hist,   self.prev_action.copy())
        _shift(self._grav_hist,     grav)

        self._cur_root_rot = _get_root_rot(mj_data, self.root_id)

    def _build_prop(self) -> np.ndarray:
        return np.concatenate([
            self._ang_vel_hist.flatten(),
            self._jpos_hist.flatten(),
            self._jvel_hist.flatten(),
            self._action_hist.flatten(),
            self._grav_hist.flatten(),
        ])

    def build(self, pico_msg: dict) -> np.ndarray:
        """构建完整 ONNX 输入向量。

        Args:
            pico_msg: unpack_pose_message 返回的字典，包含：
                smpl_joints  (N,24,3) root-local，已 remove_smpl_base_rot
                body_quat_w  (N,4)    wxyz，已 ytoz_up + remove_smpl_base_rot
                joint_pos    (N,29)   机器人关节角（IsaacLab 顺序）
        """
        smpl_joints = pico_msg["smpl_joints"].astype(np.float32)   # (N,24,3)
        body_quat_w = pico_msg["body_quat_w"].astype(np.float32)   # (N,4) wxyz
        joint_pos   = pico_msg["joint_pos"].astype(np.float32)     # (N,29)
        N = smpl_joints.shape[0]

        # ── 1. smpl_joints_multi_future_local_nonflat (10,72) ──────────────
        # smpl_joints 已经是 root-local（pico_manager 已处理），直接使用
        # 取前 10 帧（未来帧）；不足时用最后一帧末尾填充（与训练 clip 到末帧一致）
        if N >= NUM_SMPL_FUTURE:
            joints_10 = smpl_joints[:NUM_SMPL_FUTURE]               # (10,24,3)
            quats_10  = body_quat_w[:NUM_SMPL_FUTURE]               # (10,4)
            jpos_10   = joint_pos[:NUM_SMPL_FUTURE]                 # (10,29)
        else:
            pad = NUM_SMPL_FUTURE - N
            joints_10 = np.concatenate([
                smpl_joints, np.tile(smpl_joints[-1:], (pad, 1, 1))], axis=0)
            quats_10  = np.concatenate([
                body_quat_w, np.tile(body_quat_w[-1:], (pad, 1))], axis=0)
            jpos_10   = np.concatenate([
                joint_pos, np.tile(joint_pos[-1:], (pad, 1))], axis=0)

        smpl_joints_tok = joints_10.reshape(NUM_SMPL_FUTURE, -1)    # (10,72)

        # ── 2. smpl_root_ori_b_multi_future (10,6) ─────────────────────────
        # = quat_inv(robot_anchor) * smpl_root_quat，转 6D
        robot_quat_wxyz = np.array([
            self._cur_root_rot.as_quat()[3],   # w
            *self._cur_root_rot.as_quat()[:3]  # xyz
        ], dtype=np.float32)                                         # (4,) wxyz
        robot_quat_inv = _quat_inv_wxyz(robot_quat_wxyz)            # (4,)
        robot_quat_inv_rep = np.tile(robot_quat_inv, (NUM_SMPL_FUTURE, 1))  # (10,4)
        rel_quat = _quat_mul_wxyz(robot_quat_inv_rep, quats_10)     # (10,4)
        smpl_root_ori_tok = _quat_to_6d(rel_quat)                   # (10,6)

        # ── 3. joint_pos_multi_future_wrist_for_smpl (10,6) ────────────────
        # joint_pos (N,29) 是 IsaacLab 顺序
        # WRIST_ISAACLAB_IDX = [15,16,25,26,23,24]（与训练 joints_idx 一致）
        # 若 robot_pkl 未提供，joint_pos 全为零（降级模式）
        wrist_pos = np.zeros((NUM_SMPL_FUTURE, 6), dtype=np.float32)
        for i, idx in enumerate(WRIST_ISAACLAB_IDX):
            if idx < jpos_10.shape[1]:
                wrist_pos[:, i] = jpos_10[:, idx]

        # ── 拼接 tokenizer obs ─────────────────────────────────────────────
        tok = np.concatenate([
            smpl_joints_tok.flatten(),   # 720
            smpl_root_ori_tok.flatten(), #  60
            wrist_pos.flatten(),         #  60
        ])                               # 840

        prop = self._build_prop()        # 870
        full = np.concatenate([tok, prop])[None].astype(np.float32)

        assert full.shape[1] == self.expected_dim, (
            f"输入维度不匹配：{full.shape[1]} != {self.expected_dim}"
        )
        return full


def _load_robot_dof(robot_pkl_path: str, target_frames: int, target_fps: float) -> np.ndarray:
    """从 tienkung2_pro_filtered pkl 加载 DOF 数据并线性插值到目标帧数。

    Returns:
        joint_pos_29: (target_frames, 29) IsaacLab 顺序，后两列为零（无手部 DOF）
    """
    data = joblib.load(robot_pkl_path)
    m = list(data.values())[0] if not isinstance(data, dict) or "dof" not in data else data
    dof = m["dof"].astype(np.float32)  # (T_src, 27) IsaacLab 顺序
    src_fps = float(m.get("fps", 30))
    T_src = dof.shape[0]

    # 线性插值到 target_fps
    src_times = np.arange(T_src) / src_fps
    tgt_times = np.arange(target_frames) / target_fps
    # 截断到源数据时长
    tgt_times = np.clip(tgt_times, 0, src_times[-1])
    dof_resampled = np.zeros((target_frames, 27), dtype=np.float32)
    for j in range(27):
        dof_resampled[:, j] = np.interp(tgt_times, src_times, dof[:, j])

    joint_pos_29 = np.zeros((target_frames, 29), dtype=np.float32)
    joint_pos_29[:, :27] = dof_resampled
    print(f"  robot pkl: {T_src}帧@{src_fps}fps → {target_frames}帧@{target_fps}fps")
    return joint_pos_29


def _load_pkl_as_pico_stream(pkl_path: str, robot_pkl_path: str | None = None) -> tuple[list[dict], np.ndarray]:
    """从 pkl 动作文件生成模拟 PICO 消息序列（用于无 PICO 验证）。

    支持两种格式：
    1. smpl_filtered 格式：{'pose_aa':(T,72), 'transl':(T,3), 'smpl_joints':(T,24,3), 'fps':50}
       - smpl_joints 已是 Z-up 坐标系（compute_human_joints 用 ytoz_up 后的 global_orient 计算）
       - root quat 需要 ytoz_up + remove_smpl_base_rot
    2. tienkung2_pro_filtered 格式：{'dof':(T,27), 'root_rot':(T,4), 'smpl_joints':(T,24,3), ...}
       - smpl_joints 是 world 坐标系，需转 root-local

    robot_pkl_path: 可选，tienkung2_pro_filtered 格式的配对 pkl，用于提供真实 DOF 数据
                    （joint_pos_multi_future_wrist_for_smpl 需要）

    Returns:
        (msgs, first_root_quat_wxyz): 消息列表和第一帧处理后的 root 四元数 (wxyz)
    """
    data = joblib.load(pkl_path)
    # tienkung2_pro_filtered 是嵌套 dict，smpl_filtered 是直接 dict
    if isinstance(data, dict) and "pose_aa" in data:
        m = data  # smpl_filtered 格式
        smpl_format = "smpl_filtered"
    else:
        m = list(data.values())[0]  # tienkung2_pro_filtered 格式
        smpl_format = "robot_filtered"

    print(f"  pkl 格式：{smpl_format}，字段：{list(m.keys())}")
    T = m["smpl_joints"].shape[0]

    if smpl_format == "smpl_filtered":
        # 训练方式（smpl_joints_multi_future_local）：
        #   smpl_joints 已是 Z-up 坐标系（compute_human_joints 用 ytoz_up 后的 global_orient 计算）
        #   直接用 ytoz_up+remove_base_rot 后的 root_quat 逆旋转进行 local 化
        #   不做中心化，关节0不为0（保留 SMPL T-pose 骨盆偏移）
        joints_world = m["smpl_joints"].astype(np.float32)  # (T,24,3) world Z-up

        # root quat：pose_aa[:,:3] 轴角 → ytoz_up（绕 X 轴 90°）→ remove_smpl_base_rot → wxyz
        root_aa = m["pose_aa"][:, :3].astype(np.float32)
        root_quat_xyzw = Rotation.from_rotvec(root_aa.reshape(-1, 3)).as_quat().astype(np.float32)
        root_quat_wxyz = np.concatenate([root_quat_xyzw[:, 3:], root_quat_xyzw[:, :3]], axis=1)
        ytoz = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0], dtype=np.float32)  # 绕 X 轴 90°
        root_quat_wxyz = _quat_mul_wxyz(np.tile(ytoz, (T, 1)), root_quat_wxyz)
        base_conj = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)  # conj([0.5,0.5,0.5,0.5])
        root_quat_wxyz = _quat_mul_wxyz(root_quat_wxyz, np.tile(base_conj, (T, 1)))

        # local 化：quat_inv(root_quat_processed) * joints_world（与训练一致，不中心化）
        root_quat_inv = _quat_inv_wxyz(root_quat_wxyz)
        smpl_joints_local = _quat_apply_wxyz(
            np.tile(root_quat_inv[:, None, :], (1, NUM_SMPL_JOINTS, 1)),
            joints_world
        )  # (T,24,3)

        joint_pos_29 = np.zeros((T, 29), dtype=np.float32)

        # 若提供配对的 robot pkl，加载真实 DOF 数据（训练时 joint_pos_multi_future_wrist_for_smpl 来源）
        if robot_pkl_path is not None:
            joint_pos_29 = _load_robot_dof(robot_pkl_path, T, float(m.get("fps", 50)))
    else:
        # tienkung2_pro_filtered：smpl_joints 是 world，需转 root-local
        smpl_joints_world = m["smpl_joints"].astype(np.float32)
        root_rot_xyzw = m["root_rot"].astype(np.float32)
        root_quat_wxyz = np.concatenate([root_rot_xyzw[:, 3:], root_rot_xyzw[:, :3]], axis=1)
        root_quat_inv = _quat_inv_wxyz(root_quat_wxyz)
        smpl_joints_local = _quat_apply_wxyz(
            np.tile(root_quat_inv[:, None, :], (1, NUM_SMPL_JOINTS, 1)),
            smpl_joints_world
        )
        dof = m["dof"].astype(np.float32)  # (T,27) IsaacLab 顺序
        joint_pos_29 = np.zeros((T, 29), dtype=np.float32)
        joint_pos_29[:, :27] = dof  # dof 已是 IsaacLab 顺序，直接存入

    # 构建未来帧消息（与训练一致：当前帧 + 未来 N-1 帧，末尾用最后一帧填充）
    msgs = []
    for t in range(T):
        end = min(T, t + NUM_SMPL_FUTURE)
        n   = end - t
        pad = NUM_SMPL_FUTURE - n
        joints_w = smpl_joints_local[t:end]
        quats_w  = root_quat_wxyz[t:end]
        jpos_w   = joint_pos_29[t:end]
        if pad > 0:
            joints_w = np.concatenate([joints_w, np.tile(joints_w[-1:], (pad, 1, 1))], axis=0)
            quats_w  = np.concatenate([quats_w,  np.tile(quats_w[-1:],  (pad, 1))],    axis=0)
            jpos_w   = np.concatenate([jpos_w,   np.tile(jpos_w[-1:],   (pad, 1))],    axis=0)
        msgs.append({
            "smpl_joints": joints_w,
            "body_quat_w": quats_w,
            "joint_pos":   jpos_w,
        })
    return msgs, root_quat_wxyz[0].copy()


# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PICO 全身遥操作 MuJoCo 仿真（smpl encoder）")
    parser.add_argument("--onnx",       required=True, help="*_smpl.onnx 路径")
    parser.add_argument("--zmq-host",   default="localhost")
    parser.add_argument("--zmq-port",   type=int, default=5555)
    parser.add_argument("--replay-pkl", default=None,
                        help="无 PICO 时用 pkl 文件回放验证，例如 data/smpl_filtered/xxx.pkl")
    parser.add_argument("--robot-pkl", default=None,
                        help="配对的 tienkung2_pro_filtered pkl，提供真实 DOF 数据（joint_pos_wrist）")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--spawn-z",    type=float, default=1.0)
    parser.add_argument("--align-root", action="store_true",
                        help="将机器人 root 朝向初始化为 SMPL 首帧朝向（供模型质量不足时使用）")
    args = parser.parse_args()

    # ── 加载 ONNX ──────────────────────────────────────────────────────────
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name   = sess.get_inputs()[0].name
    expected_dim = sess.get_inputs()[0].shape[1]
    print(f"ONNX 输入维度：{expected_dim}")

    # ── 加载 MuJoCo ────────────────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data  = mujoco.MjData(mj_model)
    root_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Base_link")

    # ── 数据源：ZMQ 或 pkl 回放（在初始化机器人前加载）────────────────────
    replay_msgs: list[dict] | None = None
    pico: PicoReceiver | None = None
    first_root_quat_wxyz = None  # 仅 --align-root 时使用

    if args.replay_pkl:
        print(f"回放模式：从 {args.replay_pkl} 加载 SMPL 数据")
        replay_msgs, first_root_quat_wxyz = _load_pkl_as_pico_stream(args.replay_pkl, args.robot_pkl)
        print(f"  共 {len(replay_msgs)} 帧")
    else:
        print(f"ZMQ 模式：连接 {args.zmq_host}:{args.zmq_port}")
        pico = PicoReceiver(args.zmq_host, args.zmq_port)

    # ── 初始化机器人姿态 ───────────────────────────────────────────────────
    # 默认 identity 朝向，与训练一致（模型期望 robot root ≈ identity，
    # smpl_root_ori_b_multi_future 编码相对差异）。模型质量不足时可用
    # --align-root 将机器人对齐到 SMPL 首帧朝向。
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2]     = args.spawn_z
    mj_data.qpos[7:34]  = DEFAULT_QPOS
    if args.align_root and first_root_quat_wxyz is not None:
        mj_data.qpos[3] = float(first_root_quat_wxyz[0])  # w
        mj_data.qpos[4] = float(first_root_quat_wxyz[1])  # x
        mj_data.qpos[5] = float(first_root_quat_wxyz[2])  # y
        mj_data.qpos[6] = float(first_root_quat_wxyz[3])  # z
    mujoco.mj_forward(mj_model, mj_data)

    # ── Viewer ─────────────────────────────────────────────────────────────
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=False, show_right_ui=False)
        viewer.cam.azimuth  = 120
        viewer.cam.elevation = -20
        viewer.cam.distance  = 3.5
        viewer.cam.lookat    = np.array([0, 0, 1.0])

    policy_hz  = 50.0
    dt_policy  = 1.0 / policy_hz
    sim_dt     = mj_model.opt.timestep
    n_substeps = max(1, round(dt_policy / sim_dt))

    obs_builder = SmplObsBuilder(mj_model, root_id, expected_dim)

    print("等待数据……（按 Ctrl+C 退出）")
    step       = 0
    replay_idx = 0

    while True:
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.monotonic()

        # 获取当前帧数据
        if replay_msgs is not None:
            if replay_idx >= len(replay_msgs):
                print("回放结束")
                break
            pico_msg   = replay_msgs[replay_idx]
            replay_idx += 1
        else:
            pico_msg = pico.poll()

        obs_builder.update_prop(mj_data)

        if pico_msg is None:
            # 无 PICO 数据：保持站立（输出零动作）
            joint_targets = DEFAULT_QPOS.copy()
        else:
            onnx_input = obs_builder.build(pico_msg)
            action_il  = sess.run(None, {input_name: onnx_input})[0][0]  # IsaacLab 顺序
            action_mj  = action_il[MUJOCO_TO_ISAACLAB]                   # → MuJoCo 顺序
            joint_targets = np.clip(DEFAULT_QPOS + ACTION_SCALE * action_mj, -3.14, 3.14)
            obs_builder.prev_action[:] = action_il

        mj_data.ctrl[DOF_TO_ACTUATOR] = joint_targets
        for _ in range(n_substeps):
            mj_data.qfrc_applied[6: 6 + NUM_JOINTS] = -KD * mj_data.qvel[6: 6 + NUM_JOINTS]
            mujoco.mj_step(mj_model, mj_data)

        if viewer is not None:
            viewer.sync()

        if step < 50 and step % 10==0:
            x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
            xquat = mj_data.xquat[root_id]  # [w,x,y,z]
            root_rot = Rotation.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
            yaw = root_rot.as_euler("ZYX")[0]  # Z-up convention
            print(f"  [{step:2d}] x={x:7.3f}  y={y:7.3f}  z={z:.3f}  yaw={np.rad2deg(yaw):7.1f}°")
        elif step % 50 == 0:
            x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
            xquat = mj_data.xquat[root_id]  # [w,x,y,z]
            root_rot = Rotation.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
            yaw = root_rot.as_euler("ZYX")[0]  # Z-up convention
            src = "PICO" if pico_msg is not None else "等待"
            print(f"  步骤={step:5d}  x={x:.3f}  y={y:.3f}  z={z:.3f}  [{src}] yaw={np.rad2deg(yaw):7.1f}°")

        elapsed = time.monotonic() - step_start
        sleep   = dt_policy - elapsed
        if sleep > 0:
            time.sleep(sleep)
        step += 1

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
