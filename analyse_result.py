"""
Post-training analysis for STID on MILAN.

Usage:
    python analyse_result.py
    python analyse_result.py --logs "out_1539399_*.log"
    python analyse_result.py --ckpts "checkpoints/STID_MILAN/seed*/*"
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--ckpts",
    default="checkpoints/STID_MILAN/seed*/*",
    help="Glob matching the directories that contain test_results/",
)
ap.add_argument("--logs", default="out_*.log")
ap.add_argument("--data", default="datasets/MILAN")
ap.add_argument("--input_len", type=int, default=144)
ap.add_argument("--output_len", type=int, default=12)
ap.add_argument("--out", default="analysis.png")
args = ap.parse_args()

IN, OUT = args.input_len, args.output_len
data = Path(args.data)


# ---------------------------------------------------------------------
# Ground truth in original units
# ---------------------------------------------------------------------
def load_series(path):
    a = np.load(path)
    if a.ndim == 3:      # [T, N, C] -> prima feature
        a = a[..., 0]
    elif a.ndim != 2:
        raise ValueError(f"Unexpected shape {a.shape} in {path}")
    return a            # [T, N]

train = load_series(data / "train_data.npy")
test = load_series(data / "test_data.npy")
test_ts = np.load(data / "test_timestamps.npy")

# Per-node z-score statistics, assuming norm_each_channel=True.
mean = train.mean(axis=0)
std = train.std(axis=0) + 1e-8

N = test.shape[1]
S = test.shape[0] - IN - OUT + 1

if S <= 0:
    raise ValueError(
        f"Test series is too short: T={test.shape[0]}, "
        f"input_len={IN}, output_len={OUT}"
    )


def load_prediction_file(path):
    """
    Load either a standard .npy file or a raw float32 memmap
    (BasicTS may save raw data with a .npy extension).

    The saved values are assumed to be normalized and are converted
    back to original units using the training-set mean and std.
    """
    path = Path(path)

    with path.open("rb") as f:
        has_npy_header = f.read(6) == b"\x93NUMPY"

    if has_npy_header:
        a = np.load(path, mmap_mode="r")
    else:
        a = np.memmap(path, dtype=np.float32, mode="r")

    a = np.asarray(a, dtype=np.float32)

    expected_size = S * OUT * N
    if a.size != expected_size:
        raise ValueError(
            f"Unexpected size in {path}: got {a.size}, "
            f"expected {expected_size} (= {S} x {OUT} x {N}). "
            "Check the prediction file format and input/output lengths."
        )

    a = a.reshape(S, OUT, N)
    return a * std[None, None, :] + mean[None, None, :]


# ---------------------------------------------------------------------
# Find checkpoint directories
# ---------------------------------------------------------------------
ckpt_dirs = sorted(
    d for d in glob.glob(args.ckpts)
    if (Path(d) / "test_results" / "prediction.npy").is_file()
)

if not ckpt_dirs:
    raise FileNotFoundError(
        f"No checkpoint directories containing test_results/prediction.npy "
        f"were found with pattern: {args.ckpts}"
    )

print(f"Found {len(ckpt_dirs)} checkpoint(s):")
for d in ckpt_dirs:
    print(f"  {d}")

targets = load_prediction_file(
    Path(ckpt_dirs[0]) / "test_results" / "targets.npy"
)

# ---------------------------------------------------------------------
# Sanity check 1: targets versus raw test windows
# ---------------------------------------------------------------------
win = sliding_window_view(test, IN + OUT, axis=0)  # [S, N, IN+OUT]
true = win[:, :, IN:].transpose(0, 2, 1)            # [S, OUT, N]

target_error = np.abs(targets - true).max()
print(
    f"[check] max |targets - raw test| = {target_error:.6g} "
    "(should be approximately 0)"
)

# ---------------------------------------------------------------------
# Predictions: individual seeds and ensemble
# ---------------------------------------------------------------------
preds = np.stack([
    load_prediction_file(Path(d) / "test_results" / "prediction.npy")
    for d in ckpt_dirs
])  # [M, S, OUT, N]

ens_mean = preds.mean(axis=0)
ens_std = preds.std(axis=0)

# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------
last = np.repeat(
    win[:, :, IN - 1][:, None, :],
    OUT,
    axis=1,
)
seasonal = win[:, :, :OUT].transpose(0, 2, 1)


def metrics(prediction):
    error = prediction - true
    denominator = np.abs(true).sum()
    wape = np.abs(error).sum() / denominator if denominator > 0 else np.nan

    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "WAPE": wape,
    }


rows = {
    "last value": last,
    "seasonal naive (24h)": seasonal,
}

for d, prediction in zip(ckpt_dirs, preds):
    seed_name = Path(d).parent.name
    rows[seed_name] = prediction

if len(preds) > 1:
    rows["ensemble mean"] = ens_mean

print(
    f"\n{'model':24s}{'MAE':>10s}{'RMSE':>10s}{'WAPE':>10s}   "
    "(original units)"
)

for name, prediction in rows.items():
    result = metrics(prediction)
    print(
        f"{name:24s}"
        f"{result['MAE']:10.3f}"
        f"{result['RMSE']:10.3f}"
        f"{result['WAPE']:10.3f}"
    )

# ---------------------------------------------------------------------
# Sanity check 2: ensemble spread versus ensemble error
# ---------------------------------------------------------------------
if len(preds) > 1:
    err = np.abs(ens_mean - true)
    corr = np.corrcoef(
        ens_std.ravel()[::50],
        err.ravel()[::50],
    )[0, 1]

    print(f"\n[ensemble] corr(std, |err|) = {corr:.3f} (>0 = useful)")

    coverage = (
        np.abs(true - ens_mean) <= 1.645 * ens_std
    ).mean()

    print(
        f"[ensemble] coverage of mean +/- 1.645 std = {coverage:.3f} "
        "(nominal 0.90; low is expected with few members)"
    )

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
main = ens_mean if len(preds) > 1 else preds[0]

fig, ax = plt.subplots(2, 3, figsize=(18, 9))

# (a) Learning curves from logs: overfitting check
log_files = sorted(glob.glob(args.logs))

if not log_files:
    print(f"[warning] No log files matched: {args.logs}")

for log_file in log_files:
    txt = Path(log_file).read_text(errors="replace")

    tr = [
        float(x)
        for x in re.findall(r"train/loss: ([\d.eE+-]+)", txt)
    ]
    va = [
        float(x)
        for x in re.findall(r"val/MAE: ([\d.eE+-]+)", txt)
    ]

    if tr:
        ax[0, 0].plot(tr, "--", alpha=0.6)
    if va:
        ax[0, 0].plot(va, "-", label=Path(log_file).stem)

ax[0, 0].set(
    title="train loss (--) vs val MAE (-), normalized",
    xlabel="epoch",
)
if log_files:
    ax[0, 0].legend(fontsize=6)

# (b) MAE per horizon versus baselines and ensemble
for name, prediction in rows.items():
    if name.startswith("seed"):
        continue

    ax[0, 1].plot(
        range(1, OUT + 1),
        np.abs(prediction - true).mean(axis=(0, 2)),
        marker="o",
        label=name,
    )

ax[0, 1].set(
    title="MAE per horizon",
    xlabel="horizon (x10 min)",
)
ax[0, 1].legend(fontsize=7)

# (c) Prediction versus truth scatter, horizon 1
sample_size = min(20000, S * N)
idx = np.random.default_rng(0).choice(
    S * N,
    sample_size,
    replace=False,
)

t1 = true[:, 0].ravel()[idx]
p1 = main[:, 0].ravel()[idx]

ax[0, 2].scatter(t1, p1, s=2, alpha=0.3)
lim = [min(0, t1.min(), p1.min()), max(t1.max(), p1.max())]
ax[0, 2].plot(lim, lim, "r-")
ax[0, 2].set(
    title="pred vs true (h=1)",
    xlabel="true",
    ylabel="pred",
)

# (d) Time series for busiest, median, and quietest nodes, horizon 1
order = np.argsort(true[:, 0].mean(axis=0))

for node, label in [
    (order[-1], "busiest"),
    (order[N // 2], "median"),
    (order[0], "quietest"),
]:
    ax[1, 0].plot(
        true[:432, 0, node],
        label=f"true {label}",
    )
    ax[1, 0].plot(
        main[:432, 0, node],
        "--",
        label=f"pred {label}",
    )

ax[1, 0].set(
    title="h=1, first 3 test days",
    xlabel="step (10 min)",
)
ax[1, 0].legend(fontsize=6)

# (e) Error by time of day, horizon 1
tod = (
    test_ts[IN:IN + S, 0] * 144
).round().astype(int) % 144

mae_t = np.abs(main[:, 0] - true[:, 0]).mean(axis=1)

mae_by_slot = [
    mae_t[tod == hour].mean() if np.any(tod == hour) else np.nan
    for hour in range(144)
]

ax[1, 1].plot(mae_by_slot)
ax[1, 1].set(
    title="MAE by time of day (h=1)",
    xlabel="slot (x10 min)",
)

# (f) Signed residuals, horizon 1
res = (main[:, 0] - true[:, 0]).ravel()[idx]

ax[1, 2].hist(res, bins=100)
ax[1, 2].set(
    title=f"residuals h=1 (mean={res.mean():.2f})",
)

plt.tight_layout()
plt.savefig(args.out, dpi=130)
print(f"\nSaved {args.out}")
