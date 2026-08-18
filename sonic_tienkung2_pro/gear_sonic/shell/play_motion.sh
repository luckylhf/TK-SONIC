#!/bin/bash
# 播放 TienKung2 Pro 动作文件
# 用法:
#   ./gear_sonic/shell/play_motion.sh                          # 播放默认动作库（随机顺序，循环）
#   ./gear_sonic/shell/play_motion.sh <motion.pkl>             # 播放指定文件
#   ./gear_sonic/shell/play_motion.sh <dir> --speed 0.5        # 慢速播放目录下所有动作

# MOTION_FILE="${2:-data/tienkung2_pro_filtered/230112/alone_002__A116_M.pkl}"
MOTION_FILE="${2:-data/tienkung2_pro_filtered/230112/idle_loop_002__A054.pkl}"

shift 2>/dev/null  # 移除第一个参数，剩余参数透传

python gear_sonic/scripts/play_motion.py \
    --motion-file "${MOTION_FILE}" \
    --loop \
    --shuffle \
    "$@"
