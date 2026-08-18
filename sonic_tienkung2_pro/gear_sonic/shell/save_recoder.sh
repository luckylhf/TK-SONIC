CHECKPOINT_BASE="logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes"
CHECKPOINT="${1:-${CHECKPOINT_BASE}/sonic_tienkung2_pro_test-20260528_053101/last.pt}"
OUTPUT_DIR="${2:-output/eval_$(date +%Y%m%d_%H%M%S)}"

# --- Render videos ---

python gear_sonic/eval_agent_trl.py \
    +checkpoint=$CHECKPOINT \
    ++headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=16 \
    ++manager_env.config.render_results=True \
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=64" \
    "++manager_env.config.save_rendering_dir=output/renders" \
    ++manager_env.config.env_spacing=10.0 \
    "++manager_env.commands.motion.motion_lib_cfg.exclude_patterns=[jump.*,danc.*,flip.*,high_jump.*,turn_jump.*,reach_jump.*,acro.*,cartwheel.*,somersault.*,jog.*,run.*,sprint.*,ab_bicycle.*,balled_up.*,kneeling_start.*,sitting.*,lying.*,crawl.*,fall.*,stumble.*,drunk.*,injured.*,inj_.*_walk.*,inj_.*_jog.*,inj_.*_run.*,inj_.*_jump.*,turn_jog.*,.*_jog_.*,.*_danc.*,.*_jump_.*,Jump.*,ROM.*,.*__A18\[4-8\].*,.*__A23\[4-8\].*,.*__A25\[5-7\].*,.*__A54\[2-5\].*]" \
    "~manager_env/recorders=empty" "+manager_env/recorders=render"
