#!/bin/bash
# Step 1: Export ONNX + model_config.yaml from checkpoint
# Run this ONCE with Isaac Lab env, then use run_policy_mujoco_onnx.sh for inference.
#
# Usage:
#   ./gear_sonic/shell/export_onnx.sh [checkpoint_path]

CHECKPOINT_BASE="logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes"
CHECKPOINT="${1:-${CHECKPOINT_BASE}/sonic_tienkung2_pro_test-20260529_023808/last.pt}"

echo "Exporting ONNX from: ${CHECKPOINT}"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/nccl/lib/
python gear_sonic/eval_agent_trl.py \
    checkpoint="${CHECKPOINT}" \
    num_envs=1 \
    headless=True \
    export_onnx_only=True

echo ""
echo "Done. ONNX files saved to: $(dirname ${CHECKPOINT})/exported/"
echo "model_config.yaml saved to: $(dirname ${CHECKPOINT})/model_config.yaml"
