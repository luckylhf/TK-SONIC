# 数据与资产复原指南（Restore Guide）

本训练包按 **“代码全量交付、数据/权重按需重建”** 的模式分发：
仓库内只保留无争议的开源代码与转换工具，凡涉及第三方数据集、官方权重和闭源
SDK 的大体量文件均不随包分发。按下述步骤操作，即可把本包还原为与分发时
**等价的完整训练包**（文件数、数据格式、动作集合一致；唯一例外是
`out/exported/` 与 `logs_rl/` 两个训练导出物，见第 7 步与第 9 步说明，
二者不影响运行）。

> 注意：所有步骤在 **Linux + Python 3.10** 环境下执行（SONIC 训练/数据管线均面向 Linux）。
> 包根目录的 `data_manifest.json` 记录了分发时的数据基线，用于复原后校验。

---

## 0. 环境准备

```bash
# 1) 解压本包到任意位置，进入包根目录（含 README.md / RESTORE.md / tools/ 的那一层）
cd /path/to/sonic_tienkung2_pro

# 2) 创建 conda 环境（SONIC 训练环境）
conda create -n sonic python=3.10 -y
conda activate sonic
pip install -r requirements.txt

# 3) 登录 Hugging Face（下载数据与权重必需）
pip install huggingface_hub
huggingface-cli login
#    需先在 https://huggingface.co 注册，并访问 nvidia/GEAR-SONIC
#    与 bones-studio/seed 页面同意数据集条款（gated repository）
export HF_TOKEN=hf_xxxx   # 或写入环境变量
```

---

## 1. 训练权重 `sonic_release/`（~470 MB，可选保留在包内）

官方发布权重 `last.pt`（NVIDIA Open Model License）。若分发时已删除：

```bash
python download_from_hf.py --training --no-smpl --token $HF_TOKEN
# 下载到 sonic_release/last.pt + sonic_release/config.yaml
```

---

## 2. Bones-SEED 源数据（SMPL 格式，~30 GB）—— 运动库重建的输入

```bash
python download_from_hf.py --training --token $HF_TOKEN
```

- 下载 `bones_seed_smpl.tar.part_*`（约 30 GB，7 个分片）并自动解压到
  `data/smpl_filtered/`。
- **来源与许可**：该 SMPL 版数据由 NVIDIA 托管在 `nvidia/GEAR-SONIC`，
  原始数据集为 `bones-studio/seed`（Bones-SEED，142K+ 动作）。Bones-SEED 采用
  **自定义许可证（gated，需同意条款后才能下载）**。许可证限制原始数据的
  再分发，并对使用范围、Results 和署名另有条件；它并非笼统禁止所有衍生结果。
  使用前必须阅读下载时适用的完整条款：
  https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md

  随模型或软件分发时应保留以下署名：Training data includes Motion Data by
  Bones Studio with a link to https://bones.studio/. Use of the underlying dataset
  is subject to the BONES Motion Capture Dataset License Agreement.

> 仅训练 TienKung2 Pro 运控策略时，本步数据即 retarget 的输入；若只想跑通
> 数据管线，可用 `python download_from_hf.py --sample`（1 条行走序列，约 4 MB）
> 快速验证。

---

## 2A. SMPL/SMPL-X 运行辅助资产（受限，不随 v1.3 分发）

v1.3 不包含以下模型辅助张量和分割数据：

- `coco_aug_dict.pth`
- `seg_part_info.npy`
- `smpl_3dpw14_J_regressor_sparse.pt`
- `smpl_coco17_J_regressor.pt`
- `smpl_neutral_J_regressor.pt`
- `smpl_vert_segmentation.json`
- `smplx2smpl_sparse.pt`
- `smplx_verts437.pt`

这些文件的来源和许可可能不完全相同，本包不代为下载或接受任何
SMPL/SMPL-X 条款。使用者必须自行核定每个文件的来源、获得所需授权，
然后放入：

```text
gear_sonic/trl/utils/smplx/body_model/
```

不提供自动下载脚本，以免绕过原始站点的授权门禁。相关代码在文件
缺失时会明确报错并指向本节。`python tools/restore_motion_data.py --verify`
会将它们单独标为“干净发布版预期缺失”，不影响原有基线校验结果。

---

## 3. GMR 重定向工具（Bones-SEED → TienKung2 Pro 骨架）

数据目录 `data/tienkung2_pro_filtered/` 是由 **GMR（General Motion
Retargeting，`github.com/Roboparty/GMR`，开源）** 将 SMPL 数据重定向到
TienKung2 Pro 机器人骨架后，再经本包 `gear_sonic/data_process/` 转换得到。

```bash
git clone https://github.com/Roboparty/GMR
cd GMR && pip install -e .
```

为 TienKung2 Pro 准备 retarget 配置（一次性工作）：

1. 参照 `GMR/ik_configs/smplx_to_g1.json` 编写 `smplx_to_tienkung2_pro.json`，
   关节映射以包内
   `gear_sonic/data/assets/robot_description/mjcf/tienkung2_pro.xml`
   （或 `out/assets/tienkung2_pro.xml`）的关节定义为准（27 DOF）。
2. 将 TienKung2 Pro 的网格/STL 资产拷入 `GMR/assets/` 对应目录。
3. 用
   `python -c "from general_motion_retargeting.kinematics import KinematicsModel; KinematicsModel('assets/tienkung2_pro/mjcf/tienkung.xml')"`
   验证配置可加载。

---

## 4. Retarget（生成 GMR 运动库）

```bash
python GMR/scripts/smplx_to_robot_dataset.py \
  --src_folder data/smpl_filtered \
  --tgt_folder data/bones_gmr/0903_all \
  --robot tienkung2_pro \
  --config GMR/ik_configs/smplx_to_tienkung2_pro.json
```

输出为每个动作一个 pkl（字段：`fps, root_pos, root_rot, dof_pos, local_body_pos,
link_body_list`），目录结构与 `data/smpl_filtered` 一致（session 日期目录）。

---

## 5. 过滤 + 格式转换（重建 `data/tienkung2_pro_filtered/`）

```bash
# 5.1 过滤（剔除 bed/bike/chair 等不适用动作，按 session 组织）
python gear_sonic/data_process/filter_and_copy_bones_data.py \
  --source data/bones_gmr/0903_all \
  --dest data/single_pkls

# 5.2 转换为训练用 motion-lib 格式（joblib 压缩，30 DOF → 27 DOF）
python gear_sonic/data_process/convert_gmr_to_motion_lib_tienkung2_pro.py \
  --input data/single_pkls \
  --output data/tienkung2_pro_filtered \
  --num_workers 16
```

---

## 6. 生成 metadata + 校验（一键）

```bash
# 生成 metadata.pkl（动作名 → {fps, length}，共 82112 条）并对照基线校验
python tools/restore_motion_data.py --make-metadata --verify
```

校验通过的标准（对照 `data_manifest.json`）：

| 目标路径 | 基线文件数 | 基线大小 |
|---|---|---|
| `data/tienkung2_pro_filtered/` | 82113 | ~4.8 GB |
| `sonic_release/` | 2 | ~469 MB |
| `out/exported/` | 5 | ~261 MB（训练导出物，见第 7 步）|
| `out/pkl/` + `out/smpl_pkl/` | 4 | <1 MB（可选，见第 9 步）|

---

## 7. 部署用 ONNX 模型（约 865 MB）

官方 ONNX 模型受 NVIDIA Open Model License 约束。**注意区分两组文件**：

- **G1 演示策略（随包提供，不需要 RESTORE）**：
  `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/` 下的
  `GR00T-WholeBodyControl-Balance.onnx` 和
  `GR00T-WholeBodyControl-Walk.onnx`。它们被 G1 示例配置直接引用，
  保留可避免用户运行示例时因下载地址、版本或网络问题报错。
  该组权重不是 TienKung2 Pro 策略；若只发布 TienKung2 Pro 最小包，
  可同时删除 G1 示例配置与这两个文件。许可全文和必要署名已置于
  `LICENSE` 和 `NOTICE`。

- **部署组（HF 直接提供，下载即用）**：
  `model_encoder.onnx`、`model_decoder.onnx`、`planner_sonic.onnx`：

  ```bash
  python download_from_hf.py --no-planner   # 下载到 gear_sonic_deploy/policy/release/ 与 planner/
  ```

- **训练导出组 `out/exported/`（5 个 `model_step_001000_*.onnx`，约 261 MB）**：
  这是训练过程中阶段性导出的缓存，**HF 仓库不提供**（HF 的 `sonic_v1_1/` 仅含
  顶层 `model_encoder.onnx`/`model_decoder.onnx`），包内代码也不引用它；只有
  `tools/restore_motion_data.py --verify` 会对照基线检查。若确需重建，须在完成
  训练后从 checkpoint 自行导出；日常部署与运控不受影响，`--verify` 对该项报
  DIFF 属预期，可忽略。

---

## 8. 闭源 SDK（需向厂商申请，无法自动下载）

| 组件 | 位置 | 厂商 |
|---|---|---|
| Manus VR 手部追踪 SDK（`libManusSDK.so` 等） | `decoupled_wbc/control/teleop/device/SDKClient_Linux/` | Manus VR（闭源，需授权） |
| PICO XR 机器人服务 SDK（`*.deb`） | `decoupled_wbc/control/teleop/device/pico/` | 字节跳动 / PICO（闭源，需授权） |
| `libPXREARobotSDK.so` | `external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/` | PICO（闭源二进制；Python 包装为 MIT） |

请向各厂商开发者渠道申请 SDK 后放入对应路径（保持相对结构即可）。

---

## 9. 其他可选项

- **MotionBricks 预训练权重**（如需）：`nvidia/MotionBricks`（HF），下载后放入
  `motionbricks/`。
- **AMASS 数据（`out/pkl/`、`out/smpl_pkl/`，已删除）**：如确需该部分数据，
  到 https://amass.is.tue.mpg.de 注册下载（学术非商业许可、禁止再分发），再经
  `out/convert_gmr_to_motion_lib_tienkung2_pro.py` 转换。分发时该部分量极小，
  建议直接省略。

---

## 校验与故障排查

- **校验基线**：`python tools/restore_motion_data.py --verify` 会输出每个目标
  目录的文件数/大小与 `data_manifest.json` 的差异；仅做数量级核对，不做逐文件
  哈希（80k+ 文件哈希耗时）。
- **metadata 数量不符**：多为过滤关键词或 retarget 时缺失个别 session，重跑
  第 4、5 步并对比 `data/bones_gmr/0903_all` 与 `data/smpl_filtered` 的 session
  列表。
- **HF 下载失败**：脚本已内置 `HF_ENDPOINT=https://hf-mirror.com`；如网络受限
  可手动设置该环境变量。gated 仓库需在网页端先同意条款。
- **GMR 安装失败**：GMR 依赖 `numpy、scipy、trimesh、open3d` 等，按 GMR 仓库
  README 安装；配置 JSON 的关键是 `joint_match` 与机器人 27 个自由度一一对应。
