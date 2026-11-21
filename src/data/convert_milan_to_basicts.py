"""
Converte milan_raw.csv (nella root del progetto; colonne time, cell_id, y)
nel formato di ETTh1, scrivendo i file in datasets/MILAN/:

datasets/MILAN/
    meta.json
    train_data.npy        [T_train, N, 1]
    train_timestamps.npy  [T_train, 4]
    val_data.npy / val_timestamps.npy
    test_data.npy / test_timestamps.npy

Uso:
    python convert_milan_to_basicts.py
    python convert_milan_to_basicts.py --csv datasets/MILAN/milan_raw.csv --ref datasets/ETTh1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--csv", default="milan_raw.csv")
p.add_argument("--out", default="datasets/MILAN")
p.add_argument("--ref", default="datasets/ETTh1", help="dataset di riferimento (per confronto)")
p.add_argument("--ratio", nargs=3, type=float, default=[0.6, 0.2, 0.2])
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# 1) long -> wide [T, N]
df = pd.read_csv(args.csv, parse_dates=["time"])
wide = df.pivot_table(index="time", columns="cell_id", values="y", aggfunc="mean").sort_index()
wide = wide[sorted(wide.columns)]

# 2) griglia temporale regolare
step = wide.index.to_series().diff().dropna().mode().iloc[0]
full_idx = pd.date_range(wide.index[0], wide.index[-1], freq=step)
print(f"passo: {step} | timestamp mancanti riempiti: {len(full_idx) - len(wide)}")
wide = wide.reindex(full_idx)
print(f"NaN: {int(wide.isna().sum().sum())} (riempiti con 0)")

data = wide.fillna(0.0).to_numpy(dtype=np.float32)  # [T, N] come ETTh1 (T, num_vars)
T, N = data.shape

# 3) timestamp: time-of-day, day-of-week, day-of-month, day-of-year (tutti normalizzati in [0,1))
idx = wide.index
ts = np.stack([
    (idx.hour * 3600 + idx.minute * 60 + idx.second) / 86400.0,
    idx.dayofweek / 7.0,
    (idx.day - 1) / 31.0,
    (idx.dayofyear - 1) / 366.0,
], axis=1).astype(np.float32)  # [T, 4]

# 4) split cronologico
r_train, r_val, _ = args.ratio
n_train, n_val = int(T * r_train), int(T * r_val)
bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, T)}
for split, (a, b) in bounds.items():
    np.save(out / f"{split}_data.npy", data[a:b])
    np.save(out / f"{split}_timestamps.npy", ts[a:b])
    print(f"{split}: data {data[a:b].shape}, timestamps {ts[a:b].shape}")

# 5) meta.json (generico: confrontalo con quello di ETTh1)
meta = {
    "name": "MILAN",
    "domain": "mobile network traffic",
    "frequency (minutes)": step.total_seconds() / 60,
    "shape": [T, N],
    "timestamps_shape": [T, 4],
    "timestamps_description": ["time of day", "day of week", "day of month", "day of year"],
    "num_time_steps": T,
    "num_vars": N,
    "has_graph": False,
    "regular_settings": {
        "train_val_test_ratio": args.ratio,
        "norm_each_channel": True,
        "rescale": False,
        "metrics": ["MAE", "MSE"],
        "null_val": 0.0,
    },
}
with open(out / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# 6) confronto con il dataset di riferimento
ref = Path(args.ref)
if (ref / "meta.json").exists():
    print("\n--- meta.json di riferimento ---")
    print(json.dumps(json.load(open(ref / "meta.json")), indent=2))
    for name in ["train_data", "train_timestamps"]:
        f = ref / f"{name}.npy"
        if f.exists():
            a = np.load(f, mmap_mode="r")
            print(f"{ref.name}/{name}.npy: shape={a.shape} dtype={a.dtype}")
            if "timestamps" in name:
                print("prime righe:\n", np.asarray(a[:3]))
