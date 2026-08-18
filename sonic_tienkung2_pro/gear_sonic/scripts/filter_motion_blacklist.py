#!/usr/bin/env python3
"""
Filter motion lib directory, keeping only files matching patterns from
tienkung2_pro_4stage.yaml stages "纯站立" and "站立行走".

Usage:
    python gear_sonic/scripts/filter_motion_blacklist.py \
        --source data/motion_lib_bones_seed/tienkung2_pro_filtered \
        --dest data/motion_lib_bones_seed/tienkung2_pro_filtered_1
"""

import argparse
import pickle
import re
import shutil
from pathlib import Path


# Combined patterns from stages "纯站立" + "站立行走" in tienkung2_pro_4stage.yaml.
# These are regex fullmatch patterns — "." is a wildcard, ".*" matches the rest.
KEEP_PATTERNS = [
    r"neutral_idle.*",
    r"idle_loop.*",
    r"inj_.*_idle.*",
    r"stand.*",
    r"walk_ff.*",
    r"walk_sideway.*",
    r"slow_walk.*",
    r"neutral_walk.*",
    r"arc_walk.*",
    r"change_idle.*",
]

KEEP_REGEX = re.compile("^(" + "|".join(KEEP_PATTERNS) + ")")


def should_keep(filename: str) -> bool:
    """Check if a filename stem matches keep patterns (fullmatch)."""
    return bool(KEEP_REGEX.fullmatch(filename))


def main():
    parser = argparse.ArgumentParser(description="Filter motion lib by blacklist patterns")
    parser.add_argument("--source", required=True, help="Source directory path")
    parser.add_argument("--dest", required=True, help="Destination directory path")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        print(f"Error: source directory not found: {src}")
        return

    # --- Copy matching .pkl files from subdirectories ---
    motion_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    total_files = 0
    kept_files = 0

    for motion_dir in motion_dirs:
        pkl_files = sorted(motion_dir.glob("*.pkl"))
        kept = [f for f in pkl_files if should_keep(f.stem)]

        if kept:
            dest_dir = dst / motion_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in kept:
                shutil.copy2(f, dest_dir / f.name)
            kept_files += len(kept)
            print(f"{motion_dir.name}: kept {len(kept)}/{len(pkl_files)}")
        else:
            print(f"{motion_dir.name}: skipped (0/{len(pkl_files)} kept)")

        total_files += len(pkl_files)

    # --- Filter metadata.pkl, keeping only matched motions ---
    metadata_src = src / "metadata.pkl"
    if metadata_src.exists():
        with open(metadata_src, "rb") as f:
            all_metadata = pickle.load(f)

        filtered_metadata = {k: v for k, v in all_metadata.items() if should_keep(k)}

        with open(dst / "metadata.pkl", "wb") as f:
            pickle.dump(filtered_metadata, f)

        print(f"metadata.pkl: kept {len(filtered_metadata)}/{len(all_metadata)} motions")

    print(f"\nDone. Kept {kept_files}/{total_files} motion files in {dst}")


if __name__ == "__main__":
    main()
