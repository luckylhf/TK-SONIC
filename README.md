# Gear Sonic — Whole Body Control

NVIDIA Gear Lab 的全身控制框架，用于 TienKung2 Pro 机器人的强化学习训练与策略部署。

> **本包分发说明**
>
> 本包按 **“代码全量交付、数据/权重/闭源 SDK 按需重建”** 模式分发：动作数据集
> （Bones-SEED，约 30 GB）、官方权重 `sonic_release/last.pt`、部署用 ONNX 模型
> （`out/exported/`）与 Manus / PICO 遥操作 SDK **均不随包提供**，需按下述指引
> 自行获取并重建后，才能得到与分发时等价的完整训练包。
>
> - **完整复原步骤**（数据下载、GMR 重定向、过滤与格式转换、metadata 重建、
>   基线校验）：见 **[`sonic_tienkung2_pro/RESTORE.md`](sonic_tienkung2_pro/RESTORE.md)**。
> - **快速入口**：`cd sonic_tienkung2_pro` 后按该文档第 0 步准备环境即可。
> - **复原校验**：`python tools/restore_motion_data.py --verify`（对照
>   `data_manifest.json` 基线，覆盖 80k+ 动作数据、权重、ONNX 与样例输出）。
>
> 注意：`sonic_tienkung2_pro/README.md` 为 NVIDIA 原版仓库 README（含同样的分发
> 说明与 RESTORE.md 链接），本文件是对应 TienKung2 Pro 的本地化使用指南。

## 目录结构

```
gear_sonic/
├── train_agent_trl.py     # 训练入口
├── eval_agent_trl.py      # 评估 & ONNX 导出入口
├── config/                # Hydra 实验配置 (algo, env, exp)
├── trl/                   # RL 训练核心 (loss, modules, trainer)
├── envs/                  # 环境封装 (manager_env, wrapper)
├── scripts/               # 推理/可视化/数据处理脚本
├── shell/                 # 常用操作脚本 (训练/评估/导出/部署)
├── data_process/          # 数据格式转换与过滤
├── isaac_utils/           # Isaac Sim 工具
├── camera/                # 相机服务
└── utils/                 # 通用工具
```

## 安装

```bash
cd sonic_tienkung2_pro
# 训练环境
pip install -e "gear_sonic[training]"

# MuJoCo 仿真 (无需 Isaac Lab)
pip install -e "gear_sonic[sim]"

# 遥操作 / 数据采集 / 推理
pip install -e "[teleop,data_collection,inference]"
```

Isaac Lab 需单独安装，详见 `docs/source/getting_started/installation_training.md`。

## 实验配置

训练入口配置位于 `config/exp/manager/universal_token/all_modes/sonic_tienkung2_pro.yaml`，基于 `sonic_h2.yaml` 改编，适配 TienKung2 Pro（28 刚体、27 自由度）。

**三编码器架构**：策略同时接收三种输入——

| 编码器 | 输入 | 用途 |
|--------|------|------|
| `g1` | 未来运动帧 + 锚点朝向 | 运动跟踪（模仿参考动作） |
| `teleop` | 下半身运动 + VR 三点目标 | 遥操作实时控制 |
| `smpl` | SMPL 关节 + 腕部姿态 | 人体骨骼重定向 |

解码器 `g1_kin` 负责重建运动参考帧与锚点朝向，辅助损失 `g1_recon_and_all_latent` 正则化三种编码器的潜在空间。

**关键配置**：
- 算法：PPO + 模仿学习 + 辅助损失，逐步衰减动作噪声 (0.5 → 0.1)
- 并行环境：8192，单回合 5 秒
- 奖励：5 点局部足部加速度跟踪 + 抗抖动 + 非预期接触惩罚
- 终止条件：自适应严格朝向 + 足部局部位置 + 末端超出阈值
- 课程学习：3 阶段渐进 (`tienkung2_pro_3stage.yaml`)
- 模型每 200 step 保存一次

**与 G1 的差异**：身体命名用 `_l/_r` 后缀代替 `left_/right_` 前缀，`body_yaw_link` 代替 `torso_link`，无 `wrist_yaw`（以 `wrist_roll` 替代）。

## Shell 脚本

位于 `shell/`，一键执行常见操作。

### 训练

```bash
./shell/pro.sh                        # 启动训练 (64 env, 有头)
```

脚本内含数据预处理命令（格式转换、骨骼过滤），取消注释即可使用。
如需从 checkpoint 恢复，取消脚本底部 `+checkpoint=...` 行的注释。

### 评估

```bash
./shell/eval.sh [checkpoint] [output_dir]    # 批量评估 (256 env, 双卡)
./shell/save_recoder.sh [checkpoint] [dir]   # 渲染评估视频
```

### ONNX 导出与部署

```bash
./shell/export_onnx.sh [checkpoint]                    # 导出 ONNX (需 Isaac Lab)
./shell/run_policy_mujoco_onnx.sh [onnx] [motion.pkl]  # 纯 MuJoCo 推理 (无需 Isaac Lab)
./shell/run_smpl_mujoco_onnx.sh [onnx] [zmq_port]      # SMPL 全身遥操作仿真
```

### 辅助工具

```bash
./shell/log.sh          # 每 3s 刷新训练日志
./shell/play_motion.sh  # MuJoCo 播放动作文件
```

## 典型工作流

```
数据预处理 → pro.sh 训练 → eval.sh 评估 → export_onnx.sh 导出 → run_policy_mujoco_onnx.sh 部署
```
