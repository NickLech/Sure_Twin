"""
Check whether zeros in MILAN come from data gaps (filled by the converter)
or from genuine no-traffic cells.

Usage:
    python check_zeros.py
    python check_zeros.py --csv milan_raw.csv     # also check gaps in the raw file
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/MILAN")
ap.add_argument("--csv", default=None)
args = ap.parse_args()


def longest_zero_run(col):
    """Length of the longest run of consecutive zeros in a 1-D array."""
    best = cur = 0
    for z in (col == 0):
        cur = cur + 1 if z else 0
        best = max(best, cur)
    return best


def runs(mask):
    """(start, end) of consecutive True values in a boolean array."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2] - 1))


for split in ["train", "val", "test"]:
    x = np.load(Path(args.data) / f"{split}_data.npy")[..., 0]  # [T, N]
    T, N = x.shape
    zero = x == 0
    print(f"\n=== {split}: T={T}, N={N} ===")
    print(f"zeros overall: {zero.mean():.2%}")

    # 1) timestamps where most cells are zero at once -> data gap, not real traffic
    frac_t = zero.mean(1)
    gaps = runs(frac_t > 0.5)
    print(f"timesteps with >50% cells at zero: {int((frac_t > 0.5).sum())}  blocks: {[(int(a), int(b)) for a, b in gaps[:10]]}")

    # 2) cells that are mostly zero -> dead or sparse cells
    frac_n = zero.mean(0)
    print(f"cells with >90% zeros: {int((frac_n > 0.9).sum())} | all-zero cells: {int((frac_n == 1).sum())}")

    # 3) long zero runs in single cells (excluding gap timesteps)
    valid = frac_t <= 0.5
    lr = np.array([longest_zero_run(x[valid, n]) for n in range(N)])
    print(f"cells with a zero run >= 36 steps (6h): {int((lr >= 36).sum())} (max run {lr.max()} steps)")

if args.csv:
    import pandas as pd
    df = pd.read_csv(args.csv, parse_dates=["time"])
    n_t, n_c = df["time"].nunique(), df["cell_id"].nunique()
    step = df["time"].drop_duplicates().sort_values().diff().dropna().mode().iloc[0]
    expected_t = (df["time"].max() - df["time"].min()) / step + 1
    print(f"\n=== raw csv ===\ntimestamps present: {n_t} / expected {int(expected_t)}")
    print(f"rows: {len(df)} / full grid {n_t * n_c} -> missing (time, cell) pairs: {n_t * n_c - len(df)}")
