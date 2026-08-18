# Downstream modifications

TienKung2 Pro adaptation and compliance modifications are Copyright 2026
luckylhf.

This package modifies NVIDIA GR00T-WholeBodyControl / GEAR-SONIC to add
TienKung2 Pro robot descriptions, joint ordering, environment configuration,
motion conversion, training, simulation, deployment, and documentation. Files
whose names contain `tienkung2_pro`, together with the TienKung branches in the
following shared files, carry those changes:

- `gear_sonic/envs/manager_env/mdp/rewards.py`
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
- `gear_sonic/envs/manager_env/robots/__init__.py`
- `gear_sonic/trl/utils/order_converter.py`
- `gear_sonic/utils/mujoco_sim/base_sim.py`
- `gear_sonic/utils/mujoco_sim/configs.py`

SONIC v1.3 also changes licensing metadata, removes restricted SMPL/SMPL-X
auxiliary assets from the distribution, adds explicit missing-asset diagnostics,
and stops the UltraLeap installer from auto-accepting vendor terms. These
compliance changes were made by luckylhf on 2026-08-18. The source package did
not contain a complete historical change log, so earlier exact modification
dates are unavailable.

The 2026-08-18 compliance pass also:

- replaced the abbreviated NVIDIA Open Model License with the complete official
  October 24, 2025 agreement;
- declared the mixed Apache-2.0/AGPL-3.0-or-later Python package metadata;
- corrected RoboCasa v0.2 code licensing to MIT, recorded RoboCasa's CC BY 4.0
  upstream asset terms, and identified the bundled gear_factory and locomanip
  assets as RoboCasa upstream assets;
- recorded Cyclone DDS and Cyclone DDS C++ version 0.10.2 source locations;
- added the BONES-SEED-required model/software attribution for MotionBricks
  normalization statistics;
- removed duplicate PICO setup videos and generated build/cache artifacts; and
- redirected downstream support and security reports away from NVIDIA contacts.

The exact NVIDIA GR00T-WholeBodyControl and Open-X-Humanoid/TienKung_URDF source
commits cannot be recovered from this history-free source package and are not
guessed here. Future imports must record their commit hashes at acquisition time.
