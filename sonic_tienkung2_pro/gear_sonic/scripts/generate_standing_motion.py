#!/usr/bin/env python3
"""Generate a standing motion file for TienKung2 Pro robot.

This script creates a simple motion file containing only the robot's initial standing pose.
The motion is repeated for a configurable duration, allowing the policy to learn to maintain
balance under perturbations during training.

Output format matches motion_lib format:
    root_trans_offset (T,3), pose_aa (T,28,3), dof (T,27), root_rot (T,4 xyzw), smpl_joints, fps

Usage:
    python gear_sonic/scripts/generate_standing_motion.py \
        --output data/motion_lib_tienkung2_pro_standing \
        --duration 10 \
        --fps 50
"""

import argparse
import os
import pickle
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import transform


def create_standing_motion(duration_seconds=10, fps=50):
    """Create a standing motion file.

    Args:
        duration_seconds: Duration of the motion in seconds
        fps: Frames per second

    Returns:
        dict with motion_lib format
    """
    num_frames = int(duration_seconds * fps)

    # TienKung2 Pro initial pose (from robots/tienkung2_pro.py)
    # 27 DOF in MuJoCo order
    # 原始姿态 - 已验证稳定
    initial_dof_pos = np.array([
        0.0,      # 0:  body_yaw
        0.0,      # 1:  shoulder_pitch_l
        0.1,      # 2:  shoulder_roll_l
        0.0,      # 3:  shoulder_yaw_l
        -0.3,     # 4:  elbow_pitch_l
        0.0,      # 5:  elbow_yaw_l
        0.0,      # 6:  wrist_pitch_l
        0.0,      # 7:  wrist_roll_l
        0.0,      # 8:  shoulder_pitch_r
        -0.1,     # 9:  shoulder_roll_r
        0.0,      # 10: shoulder_yaw_r
        -0.3,     # 11: elbow_pitch_r
        0.0,      # 12: elbow_yaw_r
        0.0,      # 13: wrist_pitch_r
        0.0,      # 14: wrist_roll_r
        0.0,      # 15: hip_roll_l
        -0.5,     # 16: hip_pitch_l
        0.0,      # 17: hip_yaw_l
        1.0,      # 18: knee_pitch_l
        -0.5,     # 19: ankle_pitch_l
        0.0,      # 20: ankle_roll_l
        0.0,      # 21: hip_roll_r
        -0.5,     # 22: hip_pitch_r
        0.0,      # 23: hip_yaw_r
        1.0,      # 24: knee_pitch_r
        -0.5,     # 25: ankle_pitch_r
        0.0,      # 26: ankle_roll_r
    ], dtype=np.float32)

    # Repeat initial pose for all frames
    dof_pos = np.tile(initial_dof_pos[None, :], (num_frames, 1))  # (T, 27)

    # Root position: standing at origin, slight height variation for realism
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_pos[:, 2] = 1.0  # z-height (standing)

    # Root orientation: identity (no rotation)
    root_rot = np.tile([0, 0, 0, 1], (num_frames, 1)).astype(np.float32)  # xyzw

    # Convert DOF to axis-angle representation
    # Joint rotation axes in MuJoCo DOF order
    DOF_AXIS = np.array([
        [0, 0, 1],  # 0:  body_yaw_joint
        [0, 1, 0],  # 1:  shoulder_pitch_l_joint
        [1, 0, 0],  # 2:  shoulder_roll_l_joint
        [0, 0, 1],  # 3:  shoulder_yaw_l_joint
        [0, 1, 0],  # 4:  elbow_pitch_l_joint
        [0, 0, 1],  # 5:  elbow_yaw_l_joint
        [0, 1, 0],  # 6:  wrist_pitch_l_joint
        [1, 0, 0],  # 7:  wrist_roll_l_joint
        [0, 1, 0],  # 8:  shoulder_pitch_r_joint
        [1, 0, 0],  # 9:  shoulder_roll_r_joint
        [0, 0, 1],  # 10: shoulder_yaw_r_joint
        [0, 1, 0],  # 11: elbow_pitch_r_joint
        [0, 0, 1],  # 12: elbow_yaw_r_joint
        [0, 1, 0],  # 13: wrist_pitch_r_joint
        [1, 0, 0],  # 14: wrist_roll_r_joint
        [1, 0, 0],  # 15: hip_roll_l_joint
        [0, 1, 0],  # 16: hip_pitch_l_joint
        [0, 0, 1],  # 17: hip_yaw_l_joint
        [0, 1, 0],  # 18: knee_pitch_l_joint
        [0, 1, 0],  # 19: ankle_pitch_l_joint
        [1, 0, 0],  # 20: ankle_roll_l_joint
        [1, 0, 0],  # 21: hip_roll_r_joint
        [0, 1, 0],  # 22: hip_pitch_r_joint
        [0, 0, 1],  # 23: hip_yaw_r_joint
        [0, 1, 0],  # 24: knee_pitch_r_joint
        [0, 1, 0],  # 25: ankle_pitch_r_joint
        [1, 0, 0],  # 26: ankle_roll_r_joint
    ], dtype=np.float32)

    # pose_aa: axis-angle per body (28 bodies: root + 27 actuated)
    pose_aa = np.zeros((num_frames, 28, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa[:, 1:28, :] = DOF_AXIS[None, :, :] * dof_pos[:, :, None]

    # SMPL joints (not used for standing, but required by format)
    smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof_pos,
        "root_rot": root_rot,
        "smpl_joints": smpl_joints,
        "fps": fps,
    }


def save_motion_file(motion_data, output_path, motion_name, compress=True):
    """Save motion data to file.

    Args:
        motion_data: dict with motion_lib format
        output_path: path to save file
        motion_name: name of the motion (used as key in the saved dict)
        compress: whether to compress with zlib
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Wrap motion_data in a dict with motion_name as key
    # This matches the format expected by motion_lib_base.py
    wrapped_data = {motion_name: motion_data}

    serialized = pickle.dumps(wrapped_data)
    if compress:
        serialized = zlib.compress(serialized)

    with open(output_path, 'wb') as f:
        f.write(serialized)

    print(f"Saved motion file: {output_path}")
    print(f"  Motion name: {motion_name}")
    print(f"  Frames: {motion_data['root_trans_offset'].shape[0]}")
    print(f"  FPS: {motion_data['fps']}")
    print(f"  Duration: {motion_data['root_trans_offset'].shape[0] / motion_data['fps']:.1f}s")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a standing motion file for TienKung2 Pro robot"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/motion_lib_tienkung2_pro_standing",
        help="Output directory for motion files",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10,
        help="Duration of each motion in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Frames per second",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=5,
        help="Number of motion variations to generate (with slight pose variations)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_variations} standing motion files...")
    print(f"  Duration: {args.duration}s per motion")
    print(f"  FPS: {args.fps}")
    print(f"  Output: {output_dir}")

    # Generate base standing motion
    base_motion = create_standing_motion(duration_seconds=args.duration, fps=args.fps)

    # Save base motion
    base_path = output_dir / "standing_base.pkl"
    save_motion_file(base_motion, str(base_path), "standing_base", compress=True)

    # Generate variations with slight pose perturbations
    for i in range(1, args.num_variations):
        motion_data = create_standing_motion(duration_seconds=args.duration, fps=args.fps)

        # Add small random perturbations to joint angles (±5 degrees)
        perturbation = np.random.randn(27) * 0.087  # 0.087 rad ≈ 5 degrees
        motion_data["dof"] = motion_data["dof"] + perturbation[None, :]

        # Recompute pose_aa with perturbed DOF
        DOF_AXIS = np.array([
            [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
            [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
        ], dtype=np.float32)
        motion_data["pose_aa"][:, 1:28, :] = DOF_AXIS[None, :, :] * motion_data["dof"][:, :, None]

        var_path = output_dir / f"standing_var_{i:02d}.pkl"
        save_motion_file(motion_data, str(var_path), f"standing_var_{i:02d}", compress=True)

    print(f"\nGenerated {args.num_variations} motion files in {output_dir}")
    print("Ready for training with: +exp=manager/universal_token/all_modes/sonic_tienkung2_pro_standing")


if __name__ == "__main__":
    main()
