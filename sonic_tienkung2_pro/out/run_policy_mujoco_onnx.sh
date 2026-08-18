
ONNX="${1:-exported/model_step_001000_pro.onnx}"
MOTION="${2:-pkl/walk_sideway_045_loop_001__A023_M.pkl}"
 
echo "ONNX:   ${ONNX}"
echo "Motion: ${MOTION}"

echo "==============="
python run_policy_mujoco_onnx.py \
    --onnx "${ONNX}" \
    --motion-file "${MOTION}" \
    "$@"
