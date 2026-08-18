"""Run tienkung2_pro policy on MuJoCo sim using ONNX model — no Isaac Lab needed.

Prerequisites:
    1. Export ONNX first (one-time, needs Isaac Lab):
       ./gear_sonic/shell/export_onnx.sh

    2. Then run this script:
       ./gear_sonic/shell/run_policy_mujoco_onnx.sh

The _g1.onnx model takes:
    input:  [tokenizer_obs_flat | proprioception]  shape (1, tok_dim + prop_dim)
    output: action  shape (1, 27)

All dims are read from model_config.yaml saved alongside the checkpoint.
"""

import argparse
import glob
import os
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation


# ── Constants ──────────────────────────────────────────────────────────────────
MJCF_PATH = "assets/tienkung2_pro.xml"
NUM_JOINTS = 27
QPOS_OFFSET = 7   # freejoint takes qpos[0:7]

DEFAULT_QPOS = np.array([
    0.0,
    0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
], dtype=np.float32)

# PD gains (from tienkung2_pro_sonic.yaml)
KP = np.array([200, 60, 20, 10, 10, 10, 5, 5,
               60, 20, 10, 10, 10, 5, 5,
               700, 700, 500, 700, 30, 15,
               700, 700, 500, 700, 30, 15], dtype=np.float32)
KD = np.array([10, 3, 1.5, 1, 1, 1, 0.5, 0.5,
               3, 1.5, 1, 1, 1, 0.5, 0.5,
               20, 20, 15, 10, 1.25, 1.25,
               20, 20, 15, 10, 1.25, 1.25], dtype=np.float32)

# Action scale from TIENKUNG2_PRO_ACTION_SCALE_TRACKING in tienkung2_pro.py
# Formula: factor * effort_limit / stiffness  (default factor=0.25, overrides for ankle/shoulder)
# Distal joints (elbow_yaw, wrist_pitch/roll) zeroed in TRACKING variant.
# MuJoCo DOF order: body_yaw, left_arm×7, right_arm×7, left_leg×6, right_leg×6
ACTION_SCALE = np.array([
    0.1250,                                                      # body_yaw:          0.25*100/200
    0.2188, 0.3938, 0.3150, 0.3150, 0.0, 0.0, 0.0,             # left arm:  shld_pit/rol/yaw, elb_pit (distal=0)
    0.2188, 0.3938, 0.3150, 0.3150, 0.0, 0.0, 0.0,             # right arm
    0.0839, 0.1179, 0.1175, 0.1179, 0.1833, 0.1833,             # left leg:  hip_rol/pit/yaw, knee, ank_pit/rol
    0.0839, 0.1179, 0.1175, 0.1179, 0.1833, 0.1833,             # right leg
], dtype=np.float32)

# DOF order conversion: MuJoCo → IsaacLab (from tienkung2_pro.py)
MUJOCO_TO_ISAACLAB = [
    0,  3,  7, 11, 15, 19, 23, 25,   # body_yaw, left arm
    4,  8, 12, 16, 20, 24, 26,        # right arm
    1,  5,  9, 13, 17, 21,            # left leg
    2,  6, 10, 14, 18, 22,            # right leg
]
ISAACLAB_TO_MUJOCO = [0]*27
for mj_i, il_i in enumerate(MUJOCO_TO_ISAACLAB):
    ISAACLAB_TO_MUJOCO[il_i] = mj_i

# MuJoCo ctrl is indexed by ACTUATOR, not joint DOF.
# Joint DOF order (qpos[7:34]): [body_yaw, L_arm×7, R_arm×7, L_leg×6, R_leg×6]
# Actuator order:             [body_yaw, L_leg×6, R_leg×6, L_arm×7, R_arm×7]
DOF_TO_ACTUATOR = [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ── Config loading ─────────────────────────────────────────────────────────────
def load_model_config(onnx_path: str) -> dict:
    """Load model_config.yaml from checkpoint directory."""
    onnx_path = Path(onnx_path)
    # exported/model_step_*.onnx → checkpoint_dir/model_config.yaml
    config_path = onnx_path.parent.parent / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.yaml not found at {config_path}")
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


# ── Motion library ─────────────────────────────────────────────────────────────
def load_motions(motion_path: str) -> list:
    if os.path.isfile(motion_path):
        paths = [motion_path]
    else:
        paths = sorted(glob.glob(os.path.join(motion_path, "**", "*.pkl"), recursive=True))
        paths = [p for p in paths if "metadata" not in p]
    motions = []
    for p in paths:
        data = joblib.load(p)
        for name, m in data.items():
            if "dof" in m:
                motions.append({
                    "name": name,
                    "dof": m["dof"].astype(np.float32),
                    "root_trans": m["root_trans_offset"].astype(np.float32),
                    "root_rot": m["root_rot"].astype(np.float32),  # [x,y,z,w]
                    "fps": float(m.get("fps", 30)),
                })
    return motions


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────
def get_root_rot(mj_data, root_id) -> Rotation:
    q = mj_data.xquat[root_id]  # [w,x,y,z]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def gravity_dir_local(mj_data, root_id) -> np.ndarray:
    rot_inv = get_root_rot(mj_data, root_id).inv()
    return rot_inv.apply([0, 0, -1]).astype(np.float32)


# ── Obs builder ────────────────────────────────────────────────────────────────
class ObsBuilder:
    """Builds the flat ONNX input from MuJoCo state + motion reference.

    Proprioception format (matches PolicyCfg class field order in observations.py):
        [ang_vel(t-9..t) | jpos(t-9..t) | jvel(t-9..t) | action(t-9..t) | gravity(t-9..t)]
    Each term is flattened (history, dim) row-major: oldest→newest.
    """

    def __init__(self, tok_dims: dict, required_tok: list, prop_dim: int,
                 mj_model, root_id):
        self.tok_dims = tok_dims
        self.required_tok = required_tok
        self.prop_dim = prop_dim
        self.mj_model = mj_model
        self.root_id = root_id

        self.required_tok_dim = sum(int(np.prod(tok_dims[n])) for n in required_tok)

        # Per-term history buffers (oldest at index 0, newest at index -1)
        self.prop_history = 10  # actor_prop_history_length
        self._grav_hist = np.zeros((self.prop_history, 3), dtype=np.float32)
        self._ang_vel_hist = np.zeros((self.prop_history, 3), dtype=np.float32)
        self._jpos_hist = np.zeros((self.prop_history, NUM_JOINTS), dtype=np.float32)
        self._jvel_hist = np.zeros((self.prop_history, NUM_JOINTS), dtype=np.float32)
        self._action_hist = np.zeros((self.prop_history, NUM_JOINTS), dtype=np.float32)
        self.prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    def reset(self):
        self._grav_hist[:] = 0
        self._ang_vel_hist[:] = 0
        self._jpos_hist[:] = 0
        self._jvel_hist[:] = 0
        self._action_hist[:] = 0
        self.prev_action[:] = 0

    def update_prop(self, mj_data):
        jpos_mj = mj_data.qpos[QPOS_OFFSET:QPOS_OFFSET + NUM_JOINTS].astype(np.float32)
        jvel_mj = mj_data.qvel[6:6 + NUM_JOINTS].astype(np.float32)
        grav = gravity_dir_local(mj_data, self.root_id)
        # 训练用 root_ang_vel_b（body frame），MuJoCo qvel[3:6] 是 world frame，需转换
        ang_vel_world = mj_data.qvel[3:6].astype(np.float32)
        ang_vel = get_root_rot(mj_data, self.root_id).inv().apply(ang_vel_world).astype(np.float32)

        # Shift history: oldest→newest, drop oldest, append current
        self._grav_hist[:-1] = self._grav_hist[1:]
        self._grav_hist[-1] = grav
        self._ang_vel_hist[:-1] = self._ang_vel_hist[1:]
        self._ang_vel_hist[-1] = ang_vel
        self._jpos_hist[:-1] = self._jpos_hist[1:]
        self._jpos_hist[-1] = (jpos_mj - DEFAULT_QPOS)[ISAACLAB_TO_MUJOCO]  # 与训练一致：减去 default_joint_pos
        self._jvel_hist[:-1] = self._jvel_hist[1:]
        self._jvel_hist[-1] = jvel_mj[ISAACLAB_TO_MUJOCO]
        self._action_hist[:-1] = self._action_hist[1:]
        self._action_hist[-1] = self.prev_action.copy()

        # Cache current root rotation for build() to compute relative orientation
        self.cur_root_rot = get_root_rot(mj_data, self.root_id)

    def build(self, motion, t) -> np.ndarray:
        """Build flat input: [required_tokenizer_obs | proprioception].
        t: current motion frame index (already time-mapped from policy step)
        """
        T = motion["dof"].shape[0]
        fps = motion["fps"]
        dt_future = 0.1  # 0.1s between future frames (10 frames = 1s lookahead)

        num_future = int(np.prod(self.tok_dims["command_multi_future_nonflat"][:1]))  # 10
        per_frame_dim = int(np.prod(self.tok_dims["command_multi_future_nonflat"][1:]))  # 54 = 27+27

        # command_multi_future_nonflat: matches training layout exactly.
        # Training: cat([all_jpos(270), all_jvel(270)]).reshape(10, 54)
        # Rows 0-4: position pairs [p0,p1], [p2,p3], ..., [p8,p9]
        # Rows 5-9: velocity pairs [v0,v1], [v2,v3], ..., [v8,v9]
        all_pos = np.zeros((num_future, NUM_JOINTS), dtype=np.float32)
        all_vel = np.zeros((num_future, NUM_JOINTS), dtype=np.float32)
        for fi in range(num_future):
            ref_t = min(t + int(fi * dt_future * fps), T - 1)
            ref_t_next = min(ref_t + 1, T - 1)
            # pkl dof is IsaacLab order -> convert to MuJoCo for model input
            all_pos[fi] = motion["dof"][ref_t][ISAACLAB_TO_MUJOCO]
            all_vel[fi] = (motion["dof"][ref_t_next] - motion["dof"][ref_t]) * fps
            all_vel[fi] = all_vel[fi][ISAACLAB_TO_MUJOCO]
        all_flat = np.concatenate([all_pos.flatten(), all_vel.flatten()])
        future_jpos_jvel = all_flat.reshape(num_future, per_frame_dim)

        # motion_anchor_ori_b_mf_nonflat: ref root orientation RELATIVE to current robot
        # shape (10, 6) = 6D rotation: first 2 columns of rotation matrix
        robot_rot_inv = self.cur_root_rot.inv()
        future_root_ori = np.zeros((num_future, 6), dtype=np.float32)
        for fi in range(num_future):
            ref_t = min(t + int(fi * dt_future * fps), T - 1)
            ref_rot = Rotation.from_quat(motion["root_rot"][ref_t])  # [x,y,z,w]
            rel_rot = robot_rot_inv * ref_rot
            mat = rel_rot.as_matrix()
            future_root_ori[fi] = mat[:, :2].reshape(-1)  # 行优先展平，与训练一致：[R00,R01,R10,R11,R20,R21]

        # Build only the required tokenizer obs in order
        tok_parts = []
        for name in self.required_tok:
            if name == "command_multi_future_nonflat":
                tok_parts.append(future_jpos_jvel.flatten())
            elif name == "motion_anchor_ori_b_mf_nonflat":
                tok_parts.append(future_root_ori.flatten())
            else:
                dim = int(np.prod(self.tok_dims[name]))
                tok_parts.append(np.zeros(dim, dtype=np.float32))

        tok_flat = np.concatenate(tok_parts)

        # Proprioception: PolicyCfg 类字段顺序 ang_vel→jpos→jvel→action→gravity
        prop_parts = [
            self._ang_vel_hist.flatten(),
            self._jpos_hist.flatten(),
            self._jvel_hist.flatten(),
            self._action_hist.flatten(),
            self._grav_hist.flatten(),
        ]
        prop_flat = np.concatenate(prop_parts)

        return np.concatenate([tok_flat, prop_flat])[None].astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to *_g1.onnx")
    parser.add_argument("--motion-file", required=True,
                        help="Path to a single .pkl motion file or directory")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # Load model config
    cfg = load_model_config(args.onnx)
    env_cfg = cfg["env_config"]
    tok_dims = env_cfg["obs"]["group_obs_dims"]["tokenizer"]
    prop_dim = env_cfg["robot"]["algo_obs_dim_dict"]["actor_obs"]
    g1_inputs = cfg["algo_config"]["actor"]["backbone"]["encoders"]["g1"]["inputs"]

    # _g1.onnx only needs the encoder's tokenizer inputs (decoder uses token+prop, no extra tok obs)
    special_keys = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    dec_inputs = cfg["algo_config"]["actor"]["backbone"]["decoders"]["g1_dyn"].get("inputs", [])
    required_tok = list(dict.fromkeys(
        [f for f in g1_inputs if f not in special_keys] +
        [f for f in dec_inputs if f not in special_keys]
    ))

    print(f"Required tokenizer obs: {required_tok}")
    print(f"Proprioception dim: {prop_dim}")

    # Load ONNX
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    expected_dim = sess.get_inputs()[0].shape[1]
    print(f"\nONNX input dim: {expected_dim}")

    # Load motions
    motions = load_motions(args.motion_file)
    if not motions:
        raise FileNotFoundError(f"No motions in {args.motion_file}")
    print(f"Loaded {len(motions)} motions")

    # MuJoCo
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Base_link")

    obs_builder = ObsBuilder(tok_dims, required_tok, prop_dim,
                             mj_model, root_id)

    assert obs_builder.required_tok_dim + prop_dim == expected_dim, \
        f"Dim mismatch: {obs_builder.required_tok_dim} + {prop_dim} != {expected_dim}"
    print(f"Input dim verified: {expected_dim} ✓")

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=False, show_right_ui=False)
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.5
        viewer.cam.lookat = np.array([0, 0, 1.0])

    motion_idx = 0
    while True:
        if viewer is not None and not viewer.is_running():
            break

        motion = motions[motion_idx % len(motions)]
        T = motion["dof"].shape[0]
        motion_fps = motion["fps"]
        policy_hz = 50.0
        dt_policy = 1.0 / policy_hz
        sim_dt = mj_model.opt.timestep
        n_substeps = max(1, round(dt_policy / sim_dt))
        total_time = T / motion_fps
        n_motion_steps = int(total_time * policy_hz)
        n_stand_steps = int(3.0 * policy_hz)  # 1s standing at start and end

        # Reset sim to DEFAULT_QPOS (crouching) — same as training init_state
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[0:3] = motion["root_trans"][0]
        mj_data.qpos[2] = 1.0  # Isaac Lab init_state spawn height (root_trans z 是动捕偏移，不能直接用)
        mj_data.qpos[3] = motion["root_rot"][0, 3]
        mj_data.qpos[4:7] = motion["root_rot"][0, :3]
        mj_data.qpos[7:34] = DEFAULT_QPOS  # 与 Isaac Lab init_state 一致
        mujoco.mj_forward(mj_model, mj_data)
        obs_builder.reset()

        print(f"Playing '{motion['name']}' — stand(3s) → motion({total_time:.1f}s) → stand(3s)")

        global_step = 0  # 全局步数计数器，用于前 10 帧调试输出

        def policy_step(ref_frame_idx):
            """Run one policy step with given motion reference frame."""
            nonlocal global_step
            obs_builder.update_prop(mj_data)
            onnx_input = obs_builder.build(motion, ref_frame_idx)
            action_il = sess.run(None, {input_name: onnx_input})[0][0]  # IsaacLab order
            action_mj = action_il[MUJOCO_TO_ISAACLAB]                   # → MuJoCo order
            # Training: target = default_joint_pos + action_scale * action
            # DEFAULT_QPOS and ACTION_SCALE are both in MuJoCo DOF order
            joint_targets = DEFAULT_QPOS + ACTION_SCALE * action_mj
            joint_targets = np.clip(joint_targets, -3.14, 3.14)
            mj_data.ctrl[DOF_TO_ACTUATOR] = joint_targets
            for _ in range(n_substeps):
                # Add velocity damping: XML position actuators only have kp, no kv.
                # qfrc_applied provides -kd * qvel for proper PD control.
                mj_data.qfrc_applied[6:6 + NUM_JOINTS] = (
                    -KD * mj_data.qvel[6:6 + NUM_JOINTS])
                mujoco.mj_step(mj_model, mj_data)
            obs_builder.prev_action[:] = action_il  # store IsaacLab order for prop history
            if viewer is not None:
                viewer.sync()

            # 前 10 帧打印 x,y,z,yaw
            if step < 50 and step % 10==0:
                x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                xquat = mj_data.xquat[root_id]
                root_rot = Rotation.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
                yaw = root_rot.as_euler("ZYX")[0]
                print(f"  [{global_step:2d}] x={x:7.3f}  y={y:7.3f}  z={z:.3f}  yaw={np.rad2deg(yaw):7.1f}°")

            global_step += 1

        def wait(dt_policy):
            elapsed = time.monotonic() - step_start
            sleep = dt_policy - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # Phase 1: 1s — policy runs with frame 0 as reference (standing)
        for step in range(n_stand_steps):
            if viewer is not None and not viewer.is_running():
                break
            step_start = time.monotonic()
            policy_step(0)
            wait(dt_policy)

        # Phase 2: motion — policy runs with advancing reference frames
        print_interval = int(0.5 * policy_hz)  # print every 0.5s
        for step in range(n_motion_steps):
            if viewer is not None and not viewer.is_running():
                break
            step_start = time.monotonic()
            t_sec = step / policy_hz * args.speed
            t = min(int(t_sec * motion_fps), T - 1)
            policy_step(t)
            if step % print_interval == 0:
                x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                xquat = mj_data.xquat[root_id]
                root_rot = Rotation.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
                yaw = root_rot.as_euler("ZYX")[0]
                print(f"  t={t_sec:.1f}s  x={x:.3f}  y={y:.3f}  z={z:.3f}  yaw={np.rad2deg(yaw):.1f}°")
            wait(dt_policy)

        # Phase 3: 1s — policy runs with last frame as reference (return to stand)
        for step in range(n_stand_steps):
            if viewer is not None and not viewer.is_running():
                break
            step_start = time.monotonic()
            policy_step(0)
            wait(dt_policy)

        break  # play once only

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
