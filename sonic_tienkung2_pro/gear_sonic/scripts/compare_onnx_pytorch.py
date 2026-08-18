"""对比 ONNX 模型输出与 PyTorch 模型输出是否一致。

用法：
    conda activate groot
    python gear_sonic/scripts/compare_onnx_pytorch.py

该脚本：
1. 从 checkpoint 加载 SMPL 编码器 + FSQ + g1_dyn 解码器权重
2. 用 onnxruntime 加载 ONNX 模型
3. 用与 MuJoCo 脚本完全相同的方式构建输入
4. 逐级对比输出（编码器 → FSQ → 解码器 → 最终动作）
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from vector_quantize_pytorch import FSQ

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO = Path("/home/managers/rl/GR00T-WholeBodyControl")
RUN_DIR = REPO / "logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes/sonic_tienkung2_pro_test-20260529_023808"
CKPT_PATH = RUN_DIR / "model_step_001000.pt"
SMPL_ONNX = RUN_DIR / "exported/model_step_001000_smpl.onnx"
SMPL_PKL = REPO / "data/smpl_filtered/body_stretch_2_001__A035.pkl"
ROBOT_PKL = REPO / "data/motion_lib_bones_seed/tienkung2_pro_filtered/220721/body_stretch_2_001__A035.pkl"

# ── 常量 ──────────────────────────────────────────────────────────────────────
NUM_JOINTS = 27
NUM_SMPL_FUTURE = 10
NUM_TOKENS = 2  # max_num_tokens
TOKEN_DIM = 32  # num_fsq_levels
DEFAULT_QPOS = np.array([
    0.0,
    0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
    0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
], dtype=np.float32)

WRIST_ISAACLAB_IDX = [15, 16, 25, 26, 23, 24]


# ── 四元数工具 ─────────────────────────────────────────────────────────────────
def quat_inv_wxyz(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_to_6d(q_wxyz):
    q_xyzw = np.concatenate([q_wxyz[..., 1:], q_wxyz[..., :1]], axis=-1)
    mat = Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_matrix().reshape(*q_wxyz.shape[:-1], 3, 3)
    return mat[..., :2].reshape(*q_wxyz.shape[:-1], 6).astype(np.float32)


# ── 输入数据加载 ──────────────────────────────────────────────────────────────
def load_smpl_inputs(smpl_pkl_path, robot_pkl_path):
    """用与 run_smpl_mujoco_onnx.py 完全相同的方式构建输入。"""
    import joblib

    smpl_data = joblib.load(smpl_pkl_path)
    m = smpl_data
    T = m["smpl_joints"].shape[0]

    joints_world = m["smpl_joints"].astype(np.float32)

    root_aa = m["pose_aa"][:, :3].astype(np.float32)
    root_quat_xyzw = Rotation.from_rotvec(root_aa.reshape(-1, 3)).as_quat().astype(np.float32)
    root_quat_wxyz = np.concatenate([root_quat_xyzw[:, 3:], root_quat_xyzw[:, :3]], axis=1)
    ytoz = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0], dtype=np.float32)
    root_quat_wxyz = quat_mul_wxyz(np.tile(ytoz, (T, 1)), root_quat_wxyz)
    base_conj = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    root_quat_wxyz = quat_mul_wxyz(root_quat_wxyz, np.tile(base_conj, (T, 1)))

    root_quat_inv = quat_inv_wxyz(root_quat_wxyz)
    joints_local = np.zeros_like(joints_world)
    for i in range(T):
        ri = Rotation.from_quat(np.concatenate([root_quat_inv[i, 1:], root_quat_inv[i, :1]]))
        joints_local[i] = ri.apply(joints_world[i])

    if T >= NUM_SMPL_FUTURE:
        joints_10 = joints_local[:NUM_SMPL_FUTURE]
        quats_10 = root_quat_wxyz[:NUM_SMPL_FUTURE]
    else:
        pad = NUM_SMPL_FUTURE - T
        joints_10 = np.concatenate([joints_local, np.tile(joints_local[-1:], (pad, 1, 1))], axis=0)
        quats_10 = np.concatenate([root_quat_wxyz, np.tile(root_quat_wxyz[-1:], (pad, 1))], axis=0)

    smpl_joints_tok = joints_10.reshape(NUM_SMPL_FUTURE, -1)
    smpl_root_ori_tok = quat_to_6d(quats_10)

    wrist_pos = np.zeros((NUM_SMPL_FUTURE, 6), dtype=np.float32)
    if robot_pkl_path is not None and Path(robot_pkl_path).exists():
        robot_data = joblib.load(robot_pkl_path)
        rm = list(robot_data.values())[0] if not ("dof" in robot_data) else robot_data
        dof = rm["dof"].astype(np.float32)
        src_fps = float(rm.get("fps", 30))
        T_src = dof.shape[0]
        src_times = np.arange(T_src) / src_fps
        tgt_times = np.arange(NUM_SMPL_FUTURE) / 50.0
        tgt_times = np.clip(tgt_times, 0, src_times[-1])
        dof_interp = np.zeros((NUM_SMPL_FUTURE, 27), dtype=np.float32)
        for j in range(27):
            dof_interp[:, j] = np.interp(tgt_times, src_times, dof[:, j])
        jpos_29 = np.zeros((NUM_SMPL_FUTURE, 29), dtype=np.float32)
        jpos_29[:, :27] = dof_interp
        for i, idx in enumerate(WRIST_ISAACLAB_IDX):
            wrist_pos[:, i] = jpos_29[:, idx]

    tok = np.concatenate([
        smpl_joints_tok.flatten(),   # 720
        smpl_root_ori_tok.flatten(), #  60
        wrist_pos.flatten(),         #  60
    ])

    prop = np.zeros(870, dtype=np.float32)
    full_input = np.concatenate([tok, prop])[None].astype(np.float32)
    print(f"输入形状: {full_input.shape} (tokenizer={tok.shape[0]}, prop={prop.shape[0]})")
    return full_input


# ── PyTorch 模型重建 ──────────────────────────────────────────────────────────
def build_pytorch_model(ckpt_path):
    """从 checkpoint 重建 SMPL 编码器 + FSQ + g1_dyn 解码器。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["policy_state_dict"]

    # SMPL 编码器: 840 → 2048 → 1024 → 512 → 512 → 64
    smpl_encoder = nn.Sequential(
        nn.Linear(840, 2048), nn.SiLU(),
        nn.Linear(2048, 1024), nn.SiLU(),
        nn.Linear(1024, 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 64),
    )
    prefix = "actor_module.encoders.smpl.module."
    smpl_encoder.load_state_dict({
        "0.weight": sd[prefix + "0.weight"], "0.bias": sd[prefix + "0.bias"],
        "2.weight": sd[prefix + "2.weight"], "2.bias": sd[prefix + "2.bias"],
        "4.weight": sd[prefix + "4.weight"], "4.bias": sd[prefix + "4.bias"],
        "6.weight": sd[prefix + "6.weight"], "6.bias": sd[prefix + "6.bias"],
        "8.weight": sd[prefix + "8.weight"], "8.bias": sd[prefix + "8.bias"],
    })

    # FSQ: 32 个维度，每个维度 32 个量化级别
    # 编码器输出 (B, 64) → reshape 为 (B, 2, 32) → FSQ
    fsq = FSQ(levels=[32] * TOKEN_DIM)  # [32, ..., 32] (32 次)

    # g1_dyn 解码器: 934 → 2048 → 2048 → 1024 → 1024 → 512 → 512 → 27
    g1_dyn_decoder = nn.Sequential(
        nn.Linear(934, 2048), nn.SiLU(),
        nn.Linear(2048, 2048), nn.SiLU(),
        nn.Linear(2048, 1024), nn.SiLU(),
        nn.Linear(1024, 1024), nn.SiLU(),
        nn.Linear(1024, 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 27),
    )
    prefix = "actor_module.decoders.g1_dyn.module."
    g1_dyn_decoder.load_state_dict({
        "0.weight":  sd[prefix + "0.weight"],  "0.bias":  sd[prefix + "0.bias"],
        "2.weight":  sd[prefix + "2.weight"],  "2.bias":  sd[prefix + "2.bias"],
        "4.weight":  sd[prefix + "4.weight"],  "4.bias":  sd[prefix + "4.bias"],
        "6.weight":  sd[prefix + "6.weight"],  "6.bias":  sd[prefix + "6.bias"],
        "8.weight":  sd[prefix + "8.weight"],  "8.bias":  sd[prefix + "8.bias"],
        "10.weight": sd[prefix + "10.weight"], "10.bias": sd[prefix + "10.bias"],
        "12.weight": sd[prefix + "12.weight"], "12.bias": sd[prefix + "12.bias"],
    })

    smpl_encoder.eval()
    fsq.eval()
    g1_dyn_decoder.eval()
    return smpl_encoder, fsq, g1_dyn_decoder


def rearrange_tokenizer_input(tok_flat):
    """将特征分组顺序重排为逐帧交织顺序，与 ONNX 内部 Slice/Reshape/Concat 一致。

    ONNX 将 flat 840D 切分为 [0:720]=joints, [720:780]=root_ori, [780:840]=wrist,
    各自 reshape 为 (10, X) 后在最后一维 concat 成 (10, 84)，再 flatten 为 840D。

    效果：特征分组 [f0j..f9j | f0o..f9o | f0w..f9w] → 逐帧交织 [f0_all | f1_all | ... | f9_all]。
    """
    tok = tok_flat.view(-1)  # (840,)
    joints = tok[:720].view(10, 72)
    ori = tok[720:780].view(10, 6)
    wrist = tok[780:840].view(10, 6)
    interleaved = torch.cat([joints, ori, wrist], dim=-1)  # (10, 84)
    return interleaved.flatten().unsqueeze(0)  # (1, 840)


def pytorch_forward(encoder, fsq, decoder, tokenizer_input, proprioception):
    """PyTorch 前向传播：编码器 → reshape → FSQ → reshape → 解码器。"""
    with torch.no_grad():
        # 重排输入以匹配 ONNX 内部 Slice/Reshape/Concat 操作
        tokenizer_rearranged = rearrange_tokenizer_input(tokenizer_input)

        # 编码
        encoded = encoder(tokenizer_rearranged)  # (1, 64)
        print(f"  [PyTorch] 编码器输出: shape={encoded.shape}, "
              f"range=[{encoded.min():.4f}, {encoded.max():.4f}], "
              f"mean={encoded.mean():.4f}, std={encoded.std():.4f}")

        # Reshape: (1, 64) → (1, 2, 32)
        encoded_reshaped = encoded.view(1, NUM_TOKENS, TOKEN_DIM)

        # FSQ 量化: (1, 2, 32) → (1, 2, 32)
        quantized_reshaped, _ = fsq(encoded_reshaped)
        print(f"  [PyTorch] FSQ 输出: shape={quantized_reshaped.shape}, "
              f"range=[{quantized_reshaped.min():.4f}, {quantized_reshaped.max():.4f}], "
              f"mean={quantized_reshaped.mean():.4f}, std={quantized_reshaped.std():.4f}")

        # 验证量化前后的差异
        q_diff = (quantized_reshaped - encoded_reshaped).abs()
        print(f"  [PyTorch] 量化误差: max={q_diff.max():.6f}, mean={q_diff.mean():.6f}")

        # Reshape 回 (1, 64)
        quantized_flat = quantized_reshaped.reshape(1, -1)

        # 解码: concat(token_flattened(64), proprioception(870)) → (1, 934) → (1, 27)
        decoder_input = torch.cat([quantized_flat, proprioception], dim=-1)
        action = decoder(decoder_input)
        print(f"  [PyTorch] 解码器输出 (action): shape={action.shape}, "
              f"range=[{action.min():.4f}, {action.max():.4f}], "
              f"mean={action.mean():.4f}, std={action.std():.4f}")

    return encoded, quantized_flat, action


# ── ONNX 中间层提取 ──────────────────────────────────────────────────────────
def run_onnx_with_intermediates(sess, full_input):
    """运行 ONNX 并提取中间层输出。"""
    # 获取所有输出名称
    output_names = [o.name for o in sess.get_outputs()]
    # 添加中间层作为额外输出
    # 关键中间节点：
    # - /smpl/module/module.8/Gemm_output_0: 编码器输出 (1, 64)
    # - /quantizer/Reshape_1_output_0: FSQ 输出 (1, 2, 32)

    # 先获取基础输出
    results = sess.run(None, {"obs_dict": full_input})
    onnx_action = results[0]

    # 用额外输出再跑一次获取中间结果
    extra_outputs = [
        "/smpl/module/module.8/Gemm_output_0",      # encoder output (1, 64)
        "/smpl/Reshape_1_output_0",                   # after reshape to (1, 2, 32)
        "/quantizer/Reshape_1_output_0",              # FSQ output (1, 2, 32)
    ]
    try:
        all_results = sess.run(extra_outputs + ["action"], {"obs_dict": full_input})
        return all_results[:-1], all_results[-1]  # intermediates, action
    except Exception:
        return None, onnx_action


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("1. 加载输入数据")
    full_input = load_smpl_inputs(SMPL_PKL, ROBOT_PKL)
    tok_input = full_input[:, :840]
    prop_input = full_input[:, 840:]

    print(f"   tokenizer 范围: [{tok_input.min():.4f}, {tok_input.max():.4f}]")
    print(f"   prop 范围: [{prop_input.min():.4f}, {prop_input.max():.4f}]")

    print()
    print("=" * 70)
    print("2. PyTorch 模型前向传播")
    encoder, fsq, decoder = build_pytorch_model(CKPT_PATH)
    tok_tensor = torch.from_numpy(tok_input)
    prop_tensor = torch.from_numpy(prop_input)
    pt_encoded, pt_quantized, pt_action = pytorch_forward(
        encoder, fsq, decoder, tok_tensor, prop_tensor
    )

    print()
    print("=" * 70)
    print("3. ONNX 模型前向传播")
    sess = ort.InferenceSession(str(SMPL_ONNX))
    intermediates, onnx_action = run_onnx_with_intermediates(sess, full_input)

    if intermediates is not None:
        for i, name in enumerate(["/smpl/module/module.8/Gemm_output_0",
                                   "/smpl/Reshape_1_output_0",
                                   "/quantizer/Reshape_1_output_0"]):
            val = intermediates[i]
            print(f"  [ONNX] {name.split('/')[-1]}: shape={val.shape}, "
                  f"range=[{val.min():.4f}, {val.max():.4f}], "
                  f"mean={val.mean():.4f}, std={val.std():.4f}")

    print(f"  [ONNX] action: shape={onnx_action.shape}, "
          f"range=[{onnx_action.min():.4f}, {onnx_action.max():.4f}], "
          f"mean={onnx_action.mean():.4f}, std={onnx_action.std():.4f}")

    print()
    print("=" * 70)
    print("4. 对比结果")
    pt_action_np = pt_action.numpy()

    diff = np.abs(pt_action_np - onnx_action)
    l2 = np.sqrt(np.sum((pt_action_np - onnx_action) ** 2))
    cos_sim = np.dot(pt_action_np.flatten(), onnx_action.flatten()) / (
        np.linalg.norm(pt_action_np) * np.linalg.norm(onnx_action) + 1e-8
    )

    print(f"  L2 距离:        {l2:.6f}")
    print(f"  Cosine 相似度:  {cos_sim:.8f}")
    print(f"  最大绝对差:      {diff.max():.6f} (索引 {diff.argmax()})")
    print(f"  平均绝对差:      {diff.mean():.6f}")

    # 逐关节比较
    print()
    print("  逐关节差异 (IsaacLab 顺序):")
    joint_names = [
        "L_hip_roll", "L_hip_yaw", "L_hip_pitch", "L_knee", "L_ankle_roll", "L_ankle_pitch",
        "R_hip_roll", "R_hip_yaw", "R_hip_pitch", "R_knee", "R_ankle_roll", "R_ankle_pitch",
        "waist_yaw", "waist_pitch", "waist_roll",
        "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
        "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
        "L_wrist_roll", "L_wrist_pitch", "R_wrist_roll", "R_wrist_pitch",
    ]
    for j in range(NUM_JOINTS):
        match = "✓" if diff[0, j] < 1e-4 else "✗"
        print(f"    [{j:2d}] {joint_names[j]:20s}:  pt={pt_action_np[0,j]:8.4f}  "
              f"onnx={onnx_action[0,j]:8.4f}  diff={diff[0,j]:8.6f}  {match}")

    # 中间层对比
    if intermediates is not None:
        print()
        print("  中间层对比:")
        # 编码器输出
        onnx_enc = intermediates[0]  # (1, 64)
        pt_enc = pt_encoded.numpy()
        enc_diff = np.abs(onnx_enc - pt_enc)
        print(f"    编码器输出: L2={np.sqrt(np.sum((onnx_enc-pt_enc)**2)):.6f}, "
              f"max_diff={enc_diff.max():.6f}, mean_diff={enc_diff.mean():.6f}")

        # FSQ 输出
        onnx_fsq = intermediates[2]  # (1, 2, 32)
        pt_fsq = pt_quantized.numpy().reshape(1, 2, 32)
        fsq_diff = np.abs(onnx_fsq - pt_fsq)
        print(f"    FSQ 输出:    L2={np.sqrt(np.sum((onnx_fsq-pt_fsq)**2)):.6f}, "
              f"max_diff={fsq_diff.max():.6f}, mean_diff={fsq_diff.mean():.6f}")

    if l2 < 1e-4:
        print("\n✓ ONNX 与 PyTorch 输出一致！性能差距是模型本身造成的。")
    else:
        print(f"\n✗ ONNX 与 PyTorch 输出不一致！L2={l2:.6f}，存在导出 bug。")

    return pt_action_np, onnx_action


if __name__ == "__main__":
    main()
