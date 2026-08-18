# 格式转换
#  python gear_sonic/data_process/convert_gmr_to_motion_lib_tienkung2_pro.py \
#       --input /home/managers/rl/seed/soma_uniform/tienkung2_pro \
#       --output data/tienkung2_pro \
#       --num_workers 60

#数据过滤
# python gear_sonic/data_process/filter_and_copy_bones_data.py \
#     --source data/tienkung2_pro --dest data/tienkung2_pro_filtered


accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_tienkung2_pro \
    num_envs=64 headless=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/tienkung2_pro_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
    # +checkpoint=logs_rl/TRL_Tienkung2Pro_Track/manager/universal_token/all_modes/sonic_tienkung2_pro_test-20260529_023808/last.pt

# data/smpl_filtered 数据
# 参考  docs/source/getting_started/installation_training.md