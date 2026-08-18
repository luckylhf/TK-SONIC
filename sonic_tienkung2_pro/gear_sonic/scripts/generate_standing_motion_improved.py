#!/usr/bin/env python3
"""Generate improved standing motion file for TienKung2 Pro robot.

This script creates a standing motion file with variations to improve adaptive sampling.
Instead of a completely static pose, it includes small perturbations and variations
to make the motion more realistic and reduce adaptive sampling failures.

Output format matches motion_lib format:
    root_trans_offset (T,3), pose_aa (T,28,3), dof (T,27), root_rot (T,4 xyzw), smpl_joints, fps

Usage:
    python gear_sonic/scripts/generate_standing_motion_improved.py \
        --output data/motion_lib_tienkung2_pro_standing \
        --duration 10 \
        --fps 50 \
        --num_variations 10
"""

import argparse
import os
import pickle
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import transform


def create_standing_motion_improved(duration_seconds=10, fps=50, num_variations=10):
    """Create an improved standing motion file with variations.

    Args:
        duration_seconds: Duration of each motion variation in seconds
        fps: Frames per second
        num_variations: Number of motion variations to generate

    Returns:
        dict mapping motion names to motion_lib format dicts
    """
    num_frames = int(duration_seconds * fps)

    # TienKung2 Pro initial pose (from robots/tienkung2_pro.py)
    # 27 DOF in MuJoCo order
    base_dof_pos = np.array([
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

    motions = {}

    for var_idx in range(num_variations):
        # Create variation with small perturbations
        # This helps adaptive sampling by providing diverse standing poses
        variation_scale = 0.1 * (var_idx / max(1, num_variations - 1))  # 0 to 0.1

        # Add small random perturbations to the base pose
        dof_perturbation = np.random.randn(27) * variation_scale * 0.05
        dof_pos_var = base_dof_pos + dof_perturbation

        # Repeat for all frames with slight oscillation
        dof_pos = np.tile(dof_pos_var[None, :], (num_frames, 1))

        # Add very small oscillation (< 0.01 rad) to make motion less static
        # This helps adaptive sampling recognize it as a valid motion
        t = np.linspace(0, 2 * np.pi, num_frames)
        oscillation = 0.005 * np.sin(t)[:, None]  # Small oscillation
        dof_pos = dof_pos + oscillation

        # Root position: standing at origin
        root_pos = np.zeros((num_frames, 3), dtype=np.float32)
        root_pos[:, 2] = 1.0  # z-height (standing)

        # Add small horizontal drift to make motion more realistic
        root_pos[:, 0] = 0.01 * np.sin(t)  # Small x drift
        root_pos[:, 1] = 0.01 * np.cos(t)  # Small y drift

        # Root orientation: identity (no rotation)
        root_rot = np.tile([0, 0, 0, 1], (num_frames, 1)).astype(np.float32)  # xyzw

        # Convert DOF to axis-angle representation
        pose_aa = np.zeros((num_frames, 28, 3), dtype=np.float32)
        pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
        pose_aa[:, 1:28, :] = DOF_AXIS[None, :, :] * dof_pos[:, :, None]

        # SMPL joints (not used for standing, but required by format)
        smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)

        motion_data = {
            "root_trans_offset": root_pos,
            "pose_aa": pose_aa,
            "dof": dof_pos,
            "root_rot": root_rot,
            "smpl_joints": smpl_joints,
            "fps": fps,
        }

        motion_name = f"standing_var_{var_idx:03d}"
        motions[motion_name] = motion_data

    return motions


def save_motion_file(motions_dict, output_path, compress=True):
    """Save motion data to file.

    Args:
        motions_dict: dict mapping motion names to motion_lib format dicts
        output_path: path to save file
        compress: whether to compress with zlib
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serialized = pickle.dumps(motions_dict)
    if compress:
        serialized = zlib.compress(serialized)

    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"Saved {len(motions_dict)} motion variations to {output_path}")
    print(f"File size: {len(serialized) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate improved standing motion file")
    parser.add_argument(
        "--output",
        type=str,
        default="data/motion_lib_tienkung2_pro_standing",
        help="Output file path",
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
        default=10,
        help="Number of motion variations to generate",
    )

    args = parser.parse_args()

    print(f"Generating {args.num_variations} standing motion variations...")
    print(f"  Duration: {args.duration}s @ {args.fps} fps = {int(args.duration * args.fps)} frames")

    motions = create_standing_motion_improved(
        duration_seconds=args.duration,
        fps=args.fps,
        num_variations=args.num_variations,
    )

    save_motion_file(motions, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
