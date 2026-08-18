#!/usr/bin/env python3
"""Generate a data baseline manifest for the SONIC training bundle.

Records the exact file inventory (per top-level data asset) that was present
when the bundle was distributed, WITHOUT hashing every file (too slow for
80k+ files).  It records:

  - per directory: file count, total size, per-subdir breakdown
  - a sampled list of file names (sorted first/last N per dir) for spot checks
  - per-file sha256 for the first file of each session dir (lightweight)

After the user rebuilds the dataset with tools/restore_motion_data.py, re-run
this script and diff the JSON to verify the rebuild is complete.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level asset dirs to baseline (paths relative to repo root)
DEFAULT_TARGETS = [
    "data/tienkung2_pro_filtered",
    "sonic_release",
    "out/exported",
    "out/pkl",
    "out/smpl_pkl",
    "motionbricks/out",
]

SAMPLE_PER_DIR = 8


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan_dir(root: Path, target: Path) -> dict:
    """Return {count, size_bytes, subdirs: {rel: {count, size_bytes}}, samples: [...], hashes: [...]}."""
    count = 0
    size = 0
    subdirs = {}
    samples = []
    hashes = []
    for dirpath, dirnames, filenames in os.walk(target):
        rel_dir = os.path.relpath(dirpath, target)
        dcount = 0
        dsize = 0
        names = sorted(filenames)
        for fname in names:
            fp = Path(dirpath) / fname
            try:
                fsize = fp.stat().st_size
            except OSError:
                continue
            count += 1
            size += fsize
            dcount += 1
            dsize += fsize
        if names:
            subdirs[rel_dir] = {"count": dcount, "size_bytes": dsize}
            # sample: first/last few file names per dir (no hashing all)
            for fname in names[:SAMPLE_PER_DIR] + names[-SAMPLE_PER_DIR:]:
                samples.append(os.path.join(rel_dir, fname))
            # hash one file per non-root dir (lightweight integrity anchor)
            if rel_dir != ".":
                first = Path(dirpath) / names[0]
                try:
                    hashes.append({"path": os.path.join(rel_dir, names[0]),
                                   "sha256": sha256_file(first)})
                except OSError:
                    pass
    return {"count": count, "size_bytes": size, "subdirs": subdirs,
            "samples": sorted(set(samples)), "hashes": hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / "data_manifest.json"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                        help="Paths (relative to repo root) to baseline")
    args = parser.parse_args()

    manifest = {"repo_root": str(REPO_ROOT), "targets": {}}
    for t in args.targets:
        target = REPO_ROOT / t
        if not target.exists():
            print(f"[skip] {t}: not found")
            continue
        info = scan_dir(REPO_ROOT, target)
        manifest["targets"][t] = info
        print(f"[ok] {t}: {info['count']} files, {info['size_bytes']/1e6:.1f} MB")

    out = Path(args.output)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nManifest written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
