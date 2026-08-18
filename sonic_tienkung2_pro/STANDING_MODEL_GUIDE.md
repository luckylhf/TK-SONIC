# 站立模型训练快速开始指南

## 已生成的文件

### 1. 动作文件
```
data/motion_lib_tienkung2_pro_standing/
├── standing_base.pkl          # 基础站立姿态
├── standing_var_01.pkl        # 变体 1（±5° 关节扰动）
├── standing_var_02.pkl        # 变体 2
├── standing_var_03.pkl        # 变体 3
└── standing_var_04.pkl        # 变体 4
```

**特点：**
- 每个动作 10 秒（500 帧 @ 50 FPS）
- 基于 TienKung2 Pro 初始站立姿态
- 包含 5 个变体，具有轻微的关节角度扰动（±5°）
- 文件大小：2-3 KB（高度压缩）

### 2. 训练配置
```
gear_sonic/config/exp/manager/universal_token/all_modes/sonic_tienkung2_pro_standing.yaml
```

**关键配置：**
- 使用 `data/motion_lib_tienkung2_pro_standing` 作为动作库
- 强调平衡和直立姿态的奖励函数
- 禁用数据增强（`freeze_frame_aug: false`）
- 其他配置与完整模型相同

### 3. 生成脚本
```
gear_sonic/scripts/generate_standing_motion.py
```

可用于生成其他参数的站立动作文件。

---

## 训练命令

### 基础训练（2 GPU）
```bash
accelerate launch --num_processes=2 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_tienkung2_pro_standing \
    num_envs=8192 \
    headless=True
```

### 多 GPU 训练（8 GPU）
```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_tienkung2_pro_standing \
    num_envs=4096 \
    headless=True
```

### 自定义参数
```bash
accelerate launch --num_processes=2 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_tienkung2_pro_standing \
    num_envs=8192 \
    headless=True \
    ++algo.config.num_learning_epochs=10 \
    ++algo.config.actor_learning_rate=3e-5
```

---

## 预期结果

### 训练曲线
- **收敛速度：** 快速（相比完整模型）
- **奖励：** 应在 100-200 步内快速上升
- **稳定性：** 在 1000-5000 步后应该稳定

### TensorBoard 监控
```bash
tensorboard --logdir logs_rl/TRL_Tienkung2Pro_Standing
```

关键指标：
- `train/reward` — 总奖励（应快速上升）
- `train/upright_penalty` — 直立惩罚（应接近 0）
- `train/tracking_anchor_pos` — 位置跟踪（应接近 1）
- `train/tracking_anchor_ori` — 方向跟踪（应接近 1）

---

## 动作文件格式说明

### 生成的动作文件结构
```python
{
    "root_trans_offset": (T, 3),      # 根节点位置 [x, y, z]
    "pose_aa": (T, 28, 3),            # 轴角表示 (28 个体 = 1 根 + 27 关节)
    "dof": (T, 27),                   # 关节角度 (27 DOF)
    "root_rot": (T, 4),               # 根节点四元数 [x, y, z, w]
    "smpl_joints": (T, 24, 3),        # SMPL 关节（未使用，全 0）
    "fps": 50                         # 帧率
}
```

### 初始站立姿态（27 DOF）
```
body_yaw:        0.0°
shoulder_pitch:  0.0° (L), 0.0° (R)
shoulder_roll:   5.7° (L), -5.7° (R)
shoulder_yaw:    0.0°
elbow_pitch:    -17.2° (L), -17.2° (R)
elbow_yaw:       0.0°
wrist_pitch:     0.0°
wrist_roll:      0.0°
hip_roll:        0.0°
hip_pitch:      -28.6°
hip_yaw:         0.0°
knee_pitch:      57.3°
ankle_pitch:    -28.6°
ankle_roll:      0.0°
```

---

## 自定义动作文件

### 修改站立姿态
编辑 `gear_sonic/scripts/generate_standing_motion.py` 中的 `initial_dof_pos` 数组。

### 生成不同参数的动作
```bash
python gear_sonic/scripts/generate_standing_motion.py \
    --output data/motion_lib_tienkung2_pro_standing_custom \
    --duration 20 \
    --fps 50 \
    --num_variations 10
```

参数说明：
- `--duration`: 每个动作的时长（秒）
- `--fps`: 帧率
- `--num_variations`: 生成的变体数量

---

## 故障排除

### 问题：训练立即失败
**原因：** 动作库路径错误或文件损坏
**解决：** 检查 `data/motion_lib_tienkung2_pro_standing/` 目录是否存在且包含 `.pkl` 文件

### 问题：奖励不上升
**原因：** 初始姿态不稳定或奖励权重不合适
**解决：** 
1. 检查 TensorBoard 中的 `upright_penalty` 是否过高
2. 调整 `sonic_tienkung2_pro_standing.yaml` 中的奖励权重

### 问题：机器人摔倒
**原因：** 扰动过大或初始姿态不平衡
**解决：** 
1. 减少 `generate_standing_motion.py` 中的扰动幅度（当前 ±5°）
2. 增加 `upright_penalty` 权重

---

## 下一步

### 1. 验证站立模型
训练 5000-10000 步后，评估模型在不同扰动下的稳定性。

### 2. 迁移到完整 MoE 模型
使用训练好的站立头初始化完整 MoE 模型的静止头（可选）。

### 3. 生成其他速度类别的动作
- 低速行走：0.1-0.5 m/s
- 中速行走：0.5-1.5 m/s
- 高速跑步：≥1.5 m/s

---

## 参考

- 动作库格式：`gear_sonic/utils/motion_lib/motion_lib_base.py`
- 转换脚本：`gear_sonic/data_process/convert_gmr_to_motion_lib_tienkung2_pro.py`
- 机器人配置：`gear_sonic/envs/manager_env/robots/tienkung2_pro.py`
