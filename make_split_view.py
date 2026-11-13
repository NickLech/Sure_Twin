"""
Builds datasets/MILAN_val (or MILAN_train): same as MILAN, but the chosen split is
exposed as 'test', so the EXISTING checkpoints can be evaluated on it.
train_* files are linked unchanged, so the z-score statistics stay identical.
No data is duplicated (symlinks; falls back to copies if symlinks are not allowed).

Usage:
    python make_split_view.py --split val
    python make_split_view.py --split train
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/MILAN")
ap.add_argument("--split", choices=["val", "train"], required=True)
args = ap.parse_args()

src = Path(args.data).resolve()
dst = src.parent / f"{src.name}_{args.split}"
dst.mkdir(exist_ok=True)


def link(target, name):
    p = dst / name
    if p.is_symlink() or p.exists():
        p.unlink()
    try:
        os.symlink(target, p)
    except OSError:
        shutil.copy(target, p)


for name in ("train_data", "train_timestamps", "val_data", "val_timestamps"):
    link(src / f"{name}.npy", f"{name}.npy")
link(src / f"{args.split}_data.npy", "test_data.npy")
link(src / f"{args.split}_timestamps.npy", "test_timestamps.npy")

meta = json.load(open(src / "meta.json"))
meta["name"] = dst.name
with open(dst / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

import numpy as np
n = np.load(dst / "test_data.npy", mmap_mode="r").shape[0]
print(f"created {dst}")
print(f"'test' of this view = original {args.split} split: {n} steps = {n / 144:.1f} days")
