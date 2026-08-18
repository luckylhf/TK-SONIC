#!/bin/bash
# 使用 ONNX 模型在 MuJoCo 中运行策略（无需 Isaac Lab）
#
# 前提：先运行 export_onnx.sh 导出 ONNX 模型
#
# 用法：
#   ./gear_sonic/shell/run_policy_mujoco_onnx.sh [onnx_path] [motion_dir]
RUN="sonic_tienkung2_pro_test-20260528_053101"
ONNX="${1:-logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes/sonic_tienkung2_pro_test-20260529_023808/exported/model_step_001000_g1.onnx}"
MOTION="${2:-data/motion_lib_bones_seed/tienkung2_pro_filtered/220721/body_stretch_2_001__A035.pkl}"
 
# /confusion_002__A438.pkl
# model_step_*_g1.onnx

echo "ONNX:   ${ONNX}"
echo "Motion: ${MOTION}"

# 不覆盖 DISPLAY，使用容器继承的值（X11 转发或 Xvfb）
COOKIE=$(xauth list | grep "unix:10 " | awk '{print $3}')
export XAUTHORITY=/tmp/.Xauthority-local
xauth add localhost:10 MIT-MAGIC-COOKIE-1 $COOKIE
xauth list

echo "==============="
python gear_sonic/scripts/run_policy_mujoco_onnx.py \
    --onnx "${ONNX}" \
    --motion-file "${MOTION}" \
    "$@"

# logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes/sonic_tienkung2_pro_test-20260521_222900/exported/model_step_000250_decoder.onnx