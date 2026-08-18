#!/usr/bin/env python3
# TienKung2 Pro adaptation; Copyright 2026 luckylhf.
# ruff: noqa: T201, DOC
"""Convert GMR retargeted PKL files (Tienkung2 Pro) to motion_lib format for SONIC training.

GMR smplx_to_robot_dataset.py outputs per-motion PKL files with:
    fps, root_pos (T,3), root_rot (T,4 xyzw), dof_pos (T,27), local_body_pos, link_body_list

This script converts those to the motion_lib format expected by SONIC:
    root_trans_offset (T,3), pose_aa (T,28,3), dof (T,27), root_rot (T,4 xyzw), smpl_joints, fps

Tienkung2 Pro has 27 DOF and 28 bodies (pelvis + 27 actuated links).
MuJoCo DOF order (from tienkung.xml, depth-first):
  0:    body_yaw_joint         (axis Z)
  1-7:  left arm               (shoulder_pitch Y, shoulder_roll X, shoulder_yaw Z,
                                 elbow_pitch Y, elbow_yaw Z, wrist_pitch Y, wrist_roll X)
  8-14: right arm              (same pattern)
  15-20: left leg              (hip_roll X, hip_pitch Y, hip_yaw Z, knee_pitch Y,
                                 ankle_pitch Y, ankle_roll X)
  21-26: right leg             (same pattern)

Usage:
    # Single GMR output folder → individual PKLs
    python gear_sonic/data_process/convert_gmr_to_motion_lib_tienkung2_pro.py \\
        --input /path/to/gmr_output/AMASS_tienkung2_pro \\
        --output data/motion_lib_tienkung2_pro/amass \\
        --num_workers 16

    # Single PKL file
    python gear_sonic/data_process/convert_gmr_to_motion_lib_tienkung2_pro.py \\
        --input /path/to/motion.pkl \\
        --output data/motion_lib_tienkung2_pro/test.pkl
"""

import argparse
import multiprocessing
import os
import pickle
import sys

import joblib
import numpy as np
from scipy.spatial import transform

# ---------------------------------------------------------------------------
# Tienkung2 Pro constants
# ---------------------------------------------------------------------------

NUM_DOF = 27
NUM_BODIES = 28  # Base_link (pelvis) + 27 actuated links

# GMR PKL 输出 30 DOF（含 head_yaw/pitch/roll），训练管线只用 27 DOF。
# 从 30 DOF 中去掉 index 1,2,3（head 关节）得到 27 DOF。
DOF30_TO_DOF27 = [0] + list(range(4, 30))  # body_yaw + left_arm + right_arm + left_leg + right_leg

# Joint rotation axes in MuJoCo DOF order (from tienkung.xml / GMR KinematicsModel).
# Verified by running: KinematicsModel('assets/tienkung2_pro/mjcf/tienkung.xml')
DOF_AXIS = np.array(
    [
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
    ],
    dtype=np.float32,
)


def convert_gmr_pkl(gmr_data: dict, fps: int) -> dict:
    """Convert a single GMR output dict to motion_lib format.

    Args:
        gmr_data: dict with keys fps, root_pos (T,3), root_rot (T,4 xyzw),
                  dof_pos (T,27), local_body_pos, link_body_list
        fps: target fps (use gmr_data['fps'] if None)

    Returns:
        motion_lib entry dict
    """
    root_pos = np.array(gmr_data["root_pos"], dtype=np.float32)   # (T, 3) meters
    root_rot = np.array(gmr_data["root_rot"], dtype=np.float32)   # (T, 4) xyzw
    dof_pos_raw = np.array(gmr_data["dof_pos"], dtype=np.float32)
    if dof_pos_raw.shape[1] == 30:
        dof_pos = dof_pos_raw[:, DOF30_TO_DOF27]
    elif dof_pos_raw.shape[1] == 27:
        dof_pos = dof_pos_raw
    else:
        raise ValueError(f"Unexpected dof_pos dimension: {dof_pos_raw.shape[1]}, expected 27 or 30")
    src_fps = int(gmr_data["fps"])

    T = root_pos.shape[0]

    # Downsample if needed
    if fps and fps != src_fps:
        jump = max(1, int(src_fps / fps))
        root_pos = root_pos[::jump]
        root_rot = root_rot[::jump]
        dof_pos = dof_pos[::jump]
        T = root_pos.shape[0]
        out_fps = fps
    else:
        out_fps = src_fps

    # pose_aa: axis-angle per body
    # body 0 = root (pelvis), bodies 1-27 = actuated joints
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa[:, 1:NUM_BODIES, :] = DOF_AXIS[None, :, :] * dof_pos[:, :, None]

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof_pos,
        "root_rot": root_rot,          # xyzw (scipy convention)
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": out_fps,
    }


def _process_file(args_tuple):
    """Worker: convert one GMR PKL → one motion_lib PKL."""
    src_path, tgt_path, fps = args_tuple
    try:
        with open(src_path, "rb") as f:
            gmr_data = pickle.load(f)

        motion_name = os.path.splitext(os.path.basename(src_path))[0]
        entry = convert_gmr_pkl(gmr_data, fps)

        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        joblib.dump({motion_name: entry}, tgt_path, compress=True)
        return src_path, True, None
    except Exception as e:  # noqa: BLE001
        return src_path, False, str(e)


def collect_pkl_files(src_root: str, tgt_root: str, fps: int):
    """Walk src_root and build (src, tgt, fps) tuples for all .pkl files."""
    tasks = []
    for dirpath, _, filenames in os.walk(src_root):
        for fname in sorted(filenames):
            if not fname.endswith(".pkl"):
                continue
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, src_root)
            tgt = os.path.join(tgt_root, rel)
            tasks.append((src, tgt, fps))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR Tienkung2 Pro PKL → motion_lib PKL for SONIC training"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="GMR output folder (recursive) or single .pkl file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (individual PKLs) or single .pkl file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30). GMR outputs at 30fps by default.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Parallel workers (default: 8)",
    )
    args = parser.parse_args()

    print(f"Tienkung2 Pro: {NUM_DOF} DOFs, {NUM_BODIES} bodies")

    # Single file mode
    if args.input.endswith(".pkl") and os.path.isfile(args.input):
        with open(args.input, "rb") as f:
            gmr_data = pickle.load(f)
        motion_name = os.path.splitext(os.path.basename(args.input))[0]
        entry = convert_gmr_pkl(gmr_data, args.fps)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        joblib.dump({motion_name: entry}, args.output, compress=True)
        print(f"Saved: {args.output}  ({entry['dof'].shape[0]} frames @ {entry['fps']} fps)")
        return

    # Batch folder mode
    if not os.path.isdir(args.input):
        print(f"ERROR: {args.input} is not a directory or .pkl file")
        sys.exit(1)

    tasks = collect_pkl_files(args.input, args.output, args.fps)
    if not tasks:
        print(f"ERROR: No .pkl files found in {args.input}")
        sys.exit(1)

    # Skip already-converted files
    pending = [(s, t, f) for s, t, f in tasks if not os.path.exists(t)]
    print(f"Found {len(tasks)} PKLs, {len(pending)} pending (skipping {len(tasks)-len(pending)} existing)")

    if not pending:
        print("All files already converted.")
        return

    print(f"Converting with {args.num_workers} workers → {args.output}")
    converted = 0
    failed = 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for src, ok, err in pool.imap_unordered(_process_file, pending):
            if ok:
                converted += 1
            else:
                failed += 1
                print(f"  FAILED: {src}: {err}")
            if (converted + failed) % 500 == 0:
                print(f"  Progress: {converted+failed}/{len(pending)} ({failed} failed)")

    print(f"\nDone: {converted} converted, {failed} failed out of {len(pending)} total")


if __name__ == "__main__":
    main()
