# TienKung2 Pro adaptation; Copyright 2026 luckylhf.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets/robot_description/urdf/tienkung2_pro"

# All bodies in IsaacLab traversal order (merge_fixed_joints=True, 28 bodies).
# Isaac Lab uses breadth-first traversal with children sorted alphabetically
# at each depth level.  Head links (fixed joints) are merged away.
#
# Pelvis children (alpha): body_yaw, hip_roll_l, hip_roll_r
#   Level 1 children (alpha): hip_pitch_l, hip_pitch_r, shoulder_pitch_l, shoulder_pitch_r
#     Level 2 children (alpha): hip_yaw_l, hip_yaw_r, shoulder_roll_l, shoulder_roll_r
#       Level 3 children (alpha): knee_pitch_l, knee_pitch_r, shoulder_yaw_l, shoulder_yaw_r
#         Level 4 children (alpha): ankle_pitch_l, ankle_pitch_r, elbow_pitch_l, elbow_pitch_r
#           Level 5 children (alpha): ankle_roll_l, ankle_roll_r, elbow_yaw_l, elbow_yaw_r
#             Level 6 children (alpha): wrist_pitch_l, wrist_pitch_r
#               Level 7 children (alpha): wrist_roll_l, wrist_roll_r
TIENKUNG2_PRO_ISAACLAB_JOINTS = [
    "pelvis",
    "body_yaw_link",
    "hip_roll_l_link",
    "hip_roll_r_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "hip_pitch_l_link",
    "hip_pitch_r_link",
    "shoulder_roll_l_link",
    "shoulder_roll_r_link",
    "hip_yaw_l_link",
    "hip_yaw_r_link",
    "shoulder_yaw_l_link",
    "shoulder_yaw_r_link",
    "knee_pitch_l_link",
    "knee_pitch_r_link",
    "elbow_pitch_l_link",
    "elbow_pitch_r_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "elbow_yaw_l_link",
    "elbow_yaw_r_link",
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "wrist_pitch_l_link",
    "wrist_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
]

# MuJoCo DOF order (XML depth-first, 27 joints):
#  0:    body_yaw
#  1-7:  left arm
#  8-14: right arm
#  15-20: left leg
#  21-26: right leg
#
# IsaacLab DOF order (breadth-first, children alpha-sorted per level):
#  0: body_yaw    1: hip_roll_l      2: hip_roll_r
#  3: shld_pit_l  4: shld_pit_r      5: hip_pit_l   6: hip_pit_r
#  7: shld_rol_l  8: shld_rol_r      9: hip_yaw_l  10: hip_yaw_r
# 11: shld_yaw_l 12: shld_yaw_r     13: knee_pit_l 14: knee_pit_r
# 15: elb_pit_l  16: elb_pit_r      17: ank_pit_l  18: ank_pit_r
# 19: elb_yaw_l  20: elb_yaw_r      21: ank_rol_l  22: ank_rol_r
# 23: wri_pit_l  24: wri_pit_r      25: wri_rol_l  26: wri_rol_r

TIENKUNG2_PRO_ISAACLAB_TO_MUJOCO_DOF = [
    0,                          #  0: body_yaw
    15,                         #  1: hip_roll_l
    21,                         #  2: hip_roll_r
    1,                          #  3: shoulder_pitch_l
    8,                          #  4: shoulder_pitch_r
    16,                         #  5: hip_pitch_l
    22,                         #  6: hip_pitch_r
    2,                          #  7: shoulder_roll_l
    9,                          #  8: shoulder_roll_r
    17,                         #  9: hip_yaw_l
    23,                         # 10: hip_yaw_r
    3,                          # 11: shoulder_yaw_l
    10,                         # 12: shoulder_yaw_r
    18,                         # 13: knee_pitch_l
    24,                         # 14: knee_pitch_r
    4,                          # 15: elbow_pitch_l
    11,                         # 16: elbow_pitch_r
    19,                         # 17: ankle_pitch_l
    25,                         # 18: ankle_pitch_r
    5,                          # 19: elbow_yaw_l
    12,                         # 20: elbow_yaw_r
    20,                         # 21: ankle_roll_l
    26,                         # 22: ankle_roll_r
    6,                          # 23: wrist_pitch_l
    13,                         # 24: wrist_pitch_r
    7,                          # 25: wrist_roll_l
    14,                         # 26: wrist_roll_r
]

TIENKUNG2_PRO_MUJOCO_TO_ISAACLAB_DOF = [
    0,                          #  0: body_yaw
    3,                          #  1: shoulder_pitch_l
    7,                          #  2: shoulder_roll_l
    11,                         #  3: shoulder_yaw_l
    15,                         #  4: elbow_pitch_l
    19,                         #  5: elbow_yaw_l
    23,                         #  6: wrist_pitch_l
    25,                         #  7: wrist_roll_l
    4,                          #  8: shoulder_pitch_r
    8,                          #  9: shoulder_roll_r
    12,                         # 10: shoulder_yaw_r
    16,                         # 11: elbow_pitch_r
    20,                         # 12: elbow_yaw_r
    24,                         # 13: wrist_pitch_r
    26,                         # 14: wrist_roll_r
    1,                          # 15: hip_roll_l
    5,                          # 16: hip_pitch_l
    9,                          # 17: hip_yaw_l
    13,                         # 18: knee_pitch_l
    17,                         # 19: ankle_pitch_l
    21,                         # 20: ankle_roll_l
    2,                          # 21: hip_roll_r
    6,                          # 22: hip_pitch_r
    10,                         # 23: hip_yaw_r
    14,                         # 24: knee_pitch_r
    18,                         # 25: ankle_pitch_r
    22,                         # 26: ankle_roll_r
]

# MuJoCo body order (28 bodies, XML depth-first):
#  0: pelvis    1: body_yaw    2-8: left_arm    9-15: right_arm
#  16-21: left_leg   22-27: right_leg
#
# IsaacLab body order (breadth-first with alpha-sorted children — same as joints)

TIENKUNG2_PRO_ISAACLAB_TO_MUJOCO_BODY = [
    0,                          #  0: pelvis
    1,                          #  1: body_yaw_link
    16,                         #  2: hip_roll_l_link
    22,                         #  3: hip_roll_r_link
    2,                          #  4: shoulder_pitch_l_link
    9,                          #  5: shoulder_pitch_r_link
    17,                         #  6: hip_pitch_l_link
    23,                         #  7: hip_pitch_r_link
    3,                          #  8: shoulder_roll_l_link
    10,                         #  9: shoulder_roll_r_link
    18,                         # 10: hip_yaw_l_link
    24,                         # 11: hip_yaw_r_link
    4,                          # 12: shoulder_yaw_l_link
    11,                         # 13: shoulder_yaw_r_link
    19,                         # 14: knee_pitch_l_link
    25,                         # 15: knee_pitch_r_link
    5,                          # 16: elbow_pitch_l_link
    12,                         # 17: elbow_pitch_r_link
    20,                         # 18: ankle_pitch_l_link
    26,                         # 19: ankle_pitch_r_link
    6,                          # 20: elbow_yaw_l_link
    13,                         # 21: elbow_yaw_r_link
    21,                         # 22: ankle_roll_l_link
    27,                         # 23: ankle_roll_r_link
    7,                          # 24: wrist_pitch_l_link
    14,                         # 25: wrist_pitch_r_link
    8,                          # 26: wrist_roll_l_link
    15,                         # 27: wrist_roll_r_link
]

TIENKUNG2_PRO_MUJOCO_TO_ISAACLAB_BODY = [
    0,                          #  0: pelvis
    1,                          #  1: body_yaw_link
    4,                          #  2: shoulder_pitch_l_link
    8,                          #  3: shoulder_roll_l_link
    12,                         #  4: shoulder_yaw_l_link
    16,                         #  5: elbow_pitch_l_link
    20,                         #  6: elbow_yaw_l_link
    24,                         #  7: wrist_pitch_l_link
    26,                         #  8: wrist_roll_l_link
    5,                          #  9: shoulder_pitch_r_link
    9,                          # 10: shoulder_roll_r_link
    13,                         # 11: shoulder_yaw_r_link
    17,                         # 12: elbow_pitch_r_link
    21,                         # 13: elbow_yaw_r_link
    25,                         # 14: wrist_pitch_r_link
    27,                         # 15: wrist_roll_r_link
    2,                          # 16: hip_roll_l_link
    6,                          # 17: hip_pitch_l_link
    10,                         # 18: hip_yaw_l_link
    14,                         # 19: knee_pitch_l_link
    18,                         # 20: ankle_pitch_l_link
    22,                         # 21: ankle_roll_l_link
    3,                          # 22: hip_roll_r_link
    7,                          # 23: hip_pitch_r_link
    11,                         # 24: hip_yaw_r_link
    15,                         # 25: knee_pitch_r_link
    19,                         # 26: ankle_pitch_r_link
    23,                         # 27: ankle_roll_r_link
]

# Leg joints in IsaacLab DOF order (used for lower-body observations)
# hip_roll_l/r, hip_pitch_l/r, hip_yaw_l/r, knee_pitch_l/r, ankle_pitch_l/r, ankle_roll_l/r
TIENKUNG2_PRO_LOWER_JOINT_ISAACLAB_INDICES = [
    1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22,
]

TIENKUNG2_PRO_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": TIENKUNG2_PRO_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": TIENKUNG2_PRO_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": TIENKUNG2_PRO_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": TIENKUNG2_PRO_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": TIENKUNG2_PRO_MUJOCO_TO_ISAACLAB_BODY,
    "lower_joint_isaaclab_indices": TIENKUNG2_PRO_LOWER_JOINT_ISAACLAB_INDICES,
}

TIENKUNG2_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_DIR}/urdf/tienkung2_pro.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.00),
        joint_pos={
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.5,
            "hip_yaw_l_joint": 0.0,
            "knee_pitch_l_joint": 1.0,
            "ankle_pitch_l_joint": -0.5,
            "ankle_roll_l_joint": 0.0,
            "hip_roll_r_joint": 0.0,
            "hip_pitch_r_joint": -0.5,
            "hip_yaw_r_joint": 0.0,
            "knee_pitch_r_joint": 1.0,
            "ankle_pitch_r_joint": -0.5,
            "ankle_roll_r_joint": 0.0,
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": -0.3,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.3,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
            "body_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 235.0,
                "hip_pitch_.*_joint": 330.0,
                "hip_yaw_.*_joint": 235.0,
                "knee_pitch_.*_joint": 330.0,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 16.755,
                "hip_pitch_.*_joint": 15.707,
                "hip_yaw_.*_joint": 16.755,
                "knee_pitch_.*_joint": 15.707,
            },
            stiffness={
                "hip_roll_.*_joint": 700.0,
                "hip_pitch_.*_joint": 700.0,
                "hip_yaw_.*_joint": 500.0,
                "knee_pitch_.*_joint": 700.0,
            },
            damping={
                "hip_roll_.*_joint": 20.0,
                "hip_pitch_.*_joint": 20.0,
                "hip_yaw_.*_joint": 15.0,
                "knee_pitch_.*_joint": 10.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 55.0,
                "ankle_roll_.*_joint": 55.0,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 14.137,
                "ankle_roll_.*_joint": 14.137,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30.0,
                "ankle_roll_.*_joint": 15.0,
            },
            damping={
                "ankle_pitch_.*_joint": 1.25,
                "ankle_roll_.*_joint": 1.25,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
                "shoulder_roll_.*_joint": 52.5,
                "shoulder_yaw_.*_joint": 52.5,
                "elbow_pitch_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
                "shoulder_roll_.*_joint": 14.1,
                "shoulder_yaw_.*_joint": 14.1,
                "elbow_pitch_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60.0,
                "shoulder_roll_.*_joint": 20.0,
                "shoulder_yaw_.*_joint": 10.0,
                "elbow_pitch_.*_joint": 10.0,
            },
            damping={
                "shoulder_pitch_.*_joint": 3.0,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1.0,
                "elbow_pitch_.*_joint": 1.0,
            },
        ),
        "arms_distal": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_yaw_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
            ],
            effort_limit_sim={
                "elbow_yaw_.*_joint": 24.0,
                "wrist_pitch_.*_joint": 6.3,
                "wrist_roll_.*_joint": 6.3,
            },
            velocity_limit_sim={
                "elbow_yaw_.*_joint": 10.472,
                "wrist_pitch_.*_joint": 7.540,
                "wrist_roll_.*_joint": 7.540,
            },
            stiffness={
                "elbow_yaw_.*_joint": 10.0,
                "wrist_pitch_.*_joint": 5.0,
                "wrist_roll_.*_joint": 5.0,
            },
            damping={
                "elbow_yaw_.*_joint": 1.0,
                "wrist_pitch_.*_joint": 0.5,
                "wrist_roll_.*_joint": 0.5,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["body_yaw_joint"],
            effort_limit_sim={"body_yaw_joint": 100.0},
            velocity_limit_sim={"body_yaw_joint": 5.0},
            stiffness={"body_yaw_joint": 200.0},
            damping={"body_yaw_joint": 10.0},
        ),
    },
)

# Per-joint factor overrides for joints where the default 0.25 formula gives
# unreasonably large scales due to low stiffness relative to effort limit.
# ankle_roll: 0.25 * 55/15 = 0.917 rad → capped via 0.05 factor → 0.183 rad
# ankle_pitch: 0.25 * 55/30 = 0.458 rad → capped via 0.10 factor → 0.183 rad
# shoulder_roll/yaw, elbow_pitch: similarly over-scaled, reduced to ~0.3 rad
_ACTION_SCALE_FACTOR_OVERRIDES = {
    "ankle_pitch_.*_joint": 0.10,
    "ankle_roll_.*_joint":  0.05,
    "shoulder_roll_.*_joint": 0.15,
    "shoulder_yaw_.*_joint":  0.06,
    "elbow_pitch_.*_joint":   0.06,
}

TIENKUNG2_PRO_ACTION_SCALE = {}
for _actuator in TIENKUNG2_PRO_CFG.actuators.values():
    _effort = _actuator.effort_limit_sim
    _stiff = _actuator.stiffness
    _names = _actuator.joint_names_expr
    if not isinstance(_effort, dict):
        _effort = {n: _effort for n in _names}
    if not isinstance(_stiff, dict):
        _stiff = {n: _stiff for n in _names}
    for _n in _names:
        if _n in _effort and _n in _stiff and _stiff[_n]:
            import re as _re
            factor = next(
                (v for pat, v in _ACTION_SCALE_FACTOR_OVERRIDES.items() if _re.fullmatch(pat, _n)),
                0.25,
            )
            TIENKUNG2_PRO_ACTION_SCALE[_n] = factor * _effort[_n] / _stiff[_n]

# Tracking variant: zero out distal wrist/elbow_yaw joints so PD tracks reference directly.
# These joints have high finite-difference velocity noise (mae~8 rad/s) — RL residual only
# adds oscillation. action_scale=0 means the PD controller follows the reference pose exactly.
TIENKUNG2_PRO_ACTION_SCALE_TRACKING = {**TIENKUNG2_PRO_ACTION_SCALE}
for _k in list(TIENKUNG2_PRO_ACTION_SCALE_TRACKING):
    if any(_k.startswith(p) for p in ("elbow_yaw_", "wrist_pitch_", "wrist_roll_")):
        TIENKUNG2_PRO_ACTION_SCALE_TRACKING[_k] = 0.0
