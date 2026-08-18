#!/bin/bash
# PICO 全身遥操作 MuJoCo 仿真（smpl encoder）
# 前提：先运行 pico_manager_thread_server.py
#
# 用法：
#   ./gear_sonic/shell/run_smpl_mujoco_onnx.sh [onnx_path] [zmq_port]
ONNX="${1:-exported/model_step_001000_smpl.onnx}"

PORT="${2:-5555}"

SMPL_PKL="smpl_pkl/walk_sideway_045_loop_001__A023_M.pkl"
SMPL_PKL="smpl_pkl/body_stretch_2_001__A035.pkl"

echo "ONNX:     ${ONNX}"
echo "ZMQ 端口: ${PORT}"
echo "SMPL pkl: ${SMPL_PKL}"
echo "Robot pkl:${ROBOT_PKL}"

ROBOT_PKL="${ROBOT_PKL:-}"

echo "==============="
# 构建命令参数；用 --align-root 可将机器人初始朝向对齐到 SMPL 首帧
set -- \
    --onnx "${ONNX}" \
    --replay-pkl "${SMPL_PKL}" \
    --headless
python run_smpl_mujoco_onnx.py "$@"
