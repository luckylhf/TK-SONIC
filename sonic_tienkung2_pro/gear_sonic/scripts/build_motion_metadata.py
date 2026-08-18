#!/usr/bin/env python3
"""Generate metadata.pkl for a motion library directory.

Scans all .pkl files, reads fps and frame count, saves a single
metadata.pkl in the root directory. This allows init_adaptive_sampling()
to skip opening individual pkl files during startup.

Usage:
    python gear_sonic/scripts/build_motion_metadata.py \
        data/motion_lib_bones_seed/tienkung2_pro_filtered
"""

import argparse
import glob
import os
import joblib
from pathlib import Path
from rich.progress import track


def build_metadata(motion_dir: str) -> None:
    motion_dir = Path(motion_dir)
    assert motion_dir.is_dir(), f"Not a directory: {motion_dir}"

    pkl_files = [
        f for f in glob.glob(str(motion_dir / "**" / "*.pkl"), recursive=True)
        if not f.endswith("metadata.pkl")
    ]
    print(f"Found {len(pkl_files)} motion files in {motion_dir}")

    metadata = {}
    errors = 0
    for f in track(pkl_files, description="Reading motion files..."):
        key = os.path.splitext(os.path.basename(f))[0]
        try:
            data = joblib.load(f)
            motion_key = list(data.keys())[0]
            motion = data[motion_key]
            fps = float(motion.get("fps", 30.0))
            length = int(motion["root_trans_offset"].shape[0])
            metadata[key] = {"fps": fps, "length": length}
        except Exception as e:
            print(f"  Warning: skipping {key}: {e}")
            errors += 1

    out_path = motion_dir / "metadata.pkl"
    joblib.dump(metadata, out_path)
    print(f"\nSaved {len(metadata)} entries to {out_path}")
    if errors:
        print(f"  ({errors} files skipped due to errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_dir", help="Path to motion library directory")
    args = parser.parse_args()
    build_metadata(args.motion_dir)
