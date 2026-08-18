#!/usr/bin/env python3
"""One-shot restore helper for the SONIC training bundle.

Rebuilds the removed training assets so a user can reconstruct an equivalent of
the originally distributed bundle.  Works together with:

  - RESTORE.md        : step-by-step restore guide
  - data_manifest.json: baseline inventory recorded at distribution time

Features:
  --make-metadata   scan data/tienkung2_pro_filtered/ and rebuild metadata.pkl
  --verify          compare current inventory against data_manifest.json
  --download-cmd    print the exact download command for a given asset

Run inside the repo root on Linux with the SONIC conda environment active
(needs numpy + joblib, both are part of the SONIC training stack).
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "tienkung2_pro_filtered"
METADATA = DATA_DIR / "metadata.pkl"
MANIFEST = REPO_ROOT / "data_manifest.json"
SMPL_AUX_DIR = REPO_ROOT / "gear_sonic" / "trl" / "utils" / "smplx" / "body_model"
SMPL_AUX_FILES = (
    "coco_aug_dict.pth",
    "seg_part_info.npy",
    "smpl_3dpw14_J_regressor_sparse.pt",
    "smpl_coco17_J_regressor.pt",
    "smpl_neutral_J_regressor.pt",
    "smpl_vert_segmentation.json",
    "smplx2smpl_sparse.pt",
    "smplx_verts437.pt",
)

# order matters: first key found wins
LENGTH_KEYS = ("root_trans_offset", "dof_pos", "root_pos", "local_body_pos")
FPS_DEFAULT = 30.0


def _extract_meta(path: Path):
    """Load one motion-lib pkl and return (motion_name, fps, length)."""
    if joblib is None:
        return None
    try:
        data = joblib.load(str(path))
        if not isinstance(data, dict):
            return None
        name = next(iter(data))
        entry = data[name]
        fps = entry.get("fps", FPS_DEFAULT) if isinstance(entry, dict) else FPS_DEFAULT
        length = 0
        if isinstance(entry, dict):
            for k in LENGTH_KEYS:
                v = entry.get(k)
                if v is not None:
                    try:
                        length = int(v.shape[0])
                    except (AttributeError, IndexError, TypeError):
                        length = 0
                    if length:
                        break
        if not length and hasattr(entry, "__len__"):
            try:
                length = len(entry)
            except TypeError:
                length = 0
        return name, float(fps), int(length)
    except Exception as exc:  # noqa: BLE001 - per-file resilience
        print(f"    [warn] failed to read {path.name}: {exc}", flush=True)
        return None


def make_metadata(workers: int = 8) -> int:
    """Rebuild data/tienkung2_pro_filtered/metadata.pkl (pickle dict)."""
    if joblib is None:
        print("joblib not installed; run inside the SONIC conda env first.")
        return 1
    if not DATA_DIR.exists():
        print(f"[error] {DATA_DIR} not found. Run the GMR retarget + convert "
              f"steps in RESTORE.md first.")
        return 1

    pkls = sorted(
        p for p in DATA_DIR.rglob("*.pkl") if p.name != METADATA.name
    )
    print(f"Scanning {len(pkls)} motion pkl files (workers={workers}) ...")
    results = []
    with mp.Pool(min(workers, mp.cpu_count())) as pool:
        for res in pool.imap_unordered(_extract_meta, pkls, chunksize=64):
            results.append(res)

    meta = {}
    skipped = 0
    for res in results:
        if res is None:
            skipped += 1
            continue
        name, fps, length = res
        meta[name] = {"fps": fps, "length": length}

    with open(METADATA, "wb") as f:
        pickle.dump(meta, f, protocol=4)

    print(f"metadata.pkl written: {len(meta)} motions "
          f"({skipped} skipped/corrupt)")
    return 0


def _scan_inventory(root: Path):
    """Return (count, size_bytes) for a target dir."""
    count = 0
    size = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            try:
                fsize = (Path(dirpath) / fname).stat().st_size
            except OSError:
                continue
            count += 1
            size += fsize
    return count, size


def verify() -> int:
    """Compare current inventory with the baseline manifest."""
    if not MANIFEST.exists():
        print(f"[error] baseline {MANIFEST} missing.")
        return 1
    baseline = json.loads(MANIFEST.read_text(encoding="utf-8"))["targets"]

    print(f"{'target':<38}{'files':>10}{'size':>12}   check")
    print("-" * 76)
    ok = True
    for target, info in baseline.items():
        p = REPO_ROOT / target
        if p.exists():
            count, size = _scan_inventory(p)
        else:
            count, size = 0, 0
        b_count = info["count"]
        b_size = info["size_bytes"]
        # allow small slack: metadata.pkl differs, sizes ~equal within 2%
        size_ok = abs(size - b_size) <= max(b_size * 0.02, 1_000_000)
        count_ok = count == b_count
        status = "OK" if (count_ok and size_ok) else "DIFF"
        if status == "DIFF":
            ok = False
        print(f"{target:<38}{count:>10}{size/1e6:>9.1f}MB   {status}"
              f"   (baseline {b_count} files / {b_size/1e6:.1f}MB)")
    print("-" * 76)
    print("All targets OK." if ok else "Some targets differ — see RESTORE.md §6/§7/§9.")
    missing = [name for name in SMPL_AUX_FILES if not (SMPL_AUX_DIR / name).is_file()]
    print("\nRestricted SMPL/SMPL-X auxiliary assets (not part of the baseline result):")
    if missing:
        print(f"  MISSING AS EXPECTED IN CLEAN RELEASE: {len(missing)}/{len(SMPL_AUX_FILES)}")
        print("  Supply them yourself under the applicable licenses; see RESTORE.md §2A.")
    else:
        print(f"  USER-SUPPLIED: {len(SMPL_AUX_FILES)}/{len(SMPL_AUX_FILES)}")
    return 0 if ok else 2


def print_download_cmd(asset: str) -> int:
    cmds = {
        "checkpoint": "python download_from_hf.py --training --no-smpl --token $HF_TOKEN",
        "smpl":       "python download_from_hf.py --training --token $HF_TOKEN",
        "sample":     "python download_from_hf.py --sample --token $HF_TOKEN",
        "onnx":       "python download_from_hf.py --no-planner",
        "smpl-aux":   "See RESTORE.md §2A; no automatic download is provided.",
    }
    if asset not in cmds:
        print(f"unknown asset '{asset}'. choose from: {', '.join(cmds)}")
        return 1
    print(cmds[asset])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-metadata", action="store_true",
                        help="rebuild data/tienkung2_pro_filtered/metadata.pkl")
    parser.add_argument("--verify", action="store_true",
                        help="check current inventory against data_manifest.json")
    parser.add_argument("--download-cmd", metavar="ASSET",
                        help="print download command: checkpoint|smpl|smpl-aux|sample|onnx")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel workers for metadata scan (default 8)")
    args = parser.parse_args()

    if not (args.make_metadata or args.verify or args.download_cmd):
        parser.print_help()
        return 0

    if args.download_cmd:
        return print_download_cmd(args.download_cmd)
    if args.make_metadata:
        rc = make_metadata(args.workers)
        if rc:
            return rc
    if args.verify:
        return verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
