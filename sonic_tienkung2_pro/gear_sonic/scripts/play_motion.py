"""Play back a recorded motion file on the TienKung2 Pro MuJoCo sim.

Loads a .pkl motion file and replays it frame-by-frame by directly setting
qpos in MuJoCo, bypassing the policy and unitree_sdk2py entirely.

Usage:
    python gear_sonic/scripts/play_motion.py --motion-file <path.pkl> [options]

Examples:
    # Play a specific file
    python gear_sonic/scripts/play_motion.py \
        --motion-file data/motion_lib_bones_seed/tienkung2_pro_filtered/210531/jump_and_land_heavy_001__A001.pkl

    # Play all motions in a directory, looping
    python gear_sonic/scripts/play_motion.py \
        --motion-file data/motion_lib_bones_seed/tienkung2_pro_filtered \
        --loop
"""

import argparse
import glob
import os
import random
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np


MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/tienkung2_pro.xml"

# MuJoCo DOF order for tienkung2_pro (27 joints):
#   0:    body_yaw
#   1-7:  left arm  (shoulder_pitch/roll/yaw, elbow_pitch/yaw, wrist_pitch/roll)
#   8-14: right arm (same)
#   15-20: left leg  (hip_roll/pitch/yaw, knee_pitch, ankle_pitch/roll)
#   21-26: right leg (same)


def load_motion_files(motion_file: str) -> list[dict]:
    """Load one or all .pkl motion files from a path."""
    if os.path.isfile(motion_file):
        paths = [motion_file]
    else:
        paths = sorted(glob.glob(os.path.join(motion_file, "**", "*.pkl"), recursive=True))
        paths = [p for p in paths if not p.endswith("metadata.pkl")]
        if not paths:
            raise FileNotFoundError(f"No .pkl files found under {motion_file}")

    motions = []
    for path in paths:
        data = joblib.load(path)
        for name, motion in data.items():
            if "dof" not in motion:
                continue
            motions.append({
                "name": name,
                "path": path,
                "dof": motion["dof"].astype(np.float64),           # (T, 27)
                "root_trans": motion["root_trans_offset"].astype(np.float64),  # (T, 3)
                "root_rot": motion["root_rot"].astype(np.float64),  # (T, 4) [x,y,z,w]
                "fps": float(motion.get("fps", 30)),
            })
    print(f"Loaded {len(motions)} motion(s) from {motion_file}")
    return motions


def play_motion(mj_model, mj_data, viewer, motion: dict, speed: float = 1.0):
    """Replay a single motion by directly setting qpos each frame."""
    dof = motion["dof"]          # (T, 27)
    root_trans = motion["root_trans"]  # (T, 3)
    root_rot = motion["root_rot"]      # (T, 4) [x,y,z,w] → MuJoCo wants [w,x,y,z]
    T = dof.shape[0]
    dt = 1.0 / (motion["fps"] * speed)

    print(f"  Playing '{motion['name']}' — {T} frames @ {motion['fps']} fps ({T/motion['fps']:.1f}s)")

    for t in range(T):
        if viewer is not None and not viewer.is_running():
            return False  # viewer closed

        step_start = time.monotonic()

        # qpos layout: [x, y, z, qw, qx, qy, qz, joint_0, ..., joint_26]
        mj_data.qpos[0:3] = root_trans[t]
        # Convert [x,y,z,w] → MuJoCo [w,x,y,z]
        mj_data.qpos[3] = root_rot[t, 3]
        mj_data.qpos[4:7] = root_rot[t, 0:3]
        mj_data.qpos[7:34] = dof[t]

        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        if viewer is not None:
            viewer.sync()

        elapsed = time.monotonic() - step_start
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

    return True  # completed normally


def main():
    parser = argparse.ArgumentParser(description="Play back motion files on TienKung2 Pro sim")
    parser.add_argument(
        "--motion-file",
        default="data/motion_lib_bones_seed/tienkung2_pro_filtered",
        help="Path to a .pkl file or directory of .pkl files",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop through all motions indefinitely",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle motion order",
    )
    parser.add_argument(
        "--motion-index", type=int, default=None,
        help="Play only the motion at this index (0-based)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer (for testing)",
    )
    args = parser.parse_args()

    # Load MuJoCo model
    xml_path = str(Path(MJCF_PATH).resolve())
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Load motions
    motions = load_motion_files(args.motion_file)
    if not motions:
        print("No motions found, exiting.")
        return

    if args.motion_index is not None:
        motions = [motions[args.motion_index]]

    if args.shuffle:
        random.shuffle(motions)

    print(f"Total motions: {len(motions)}")

    if args.headless:
        # Headless: just run through motions without viewer
        for motion in motions:
            play_motion(mj_model, mj_data, None, motion, args.speed)
        return

    # Launch passive viewer
    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat = np.array([0, 0, 1.0])

        while viewer.is_running():
            playlist = motions if not args.shuffle else random.sample(motions, len(motions))
            for motion in playlist:
                if not viewer.is_running():
                    break
                ok = play_motion(mj_model, mj_data, viewer, motion, args.speed)
                if not ok:
                    break
                # Pause 0.5s between motions
                time.sleep(0.5)

            if not args.loop:
                break

    print("Done.")


if __name__ == "__main__":
    main()
