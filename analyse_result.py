"""
Post-training analysis for STID on MILAN.

Usage:
    python analyse_result.py --logs "out_1539399_*.log"
    python analyse_result.py --ckpts "checkpoints/STID_MILAN/seed*/*"
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STEPS_PER_DAY = 144  # 10-minute slots


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*",
                    help="glob of run directories containing test_results/")
    ap.add_argument("--logs", default="out_*.log")
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--out", default="analysis.png")
    return ap.parse_args()


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_series(path):
    """Load [T, N] or [T, N, C] and return [T, N] (first feature)."""
    a = np.load(path)
    return a[..., 0] if a.ndim == 3 else a


def find_run_dirs(pattern):
    dirs = sorted(d for d in glob.glob(pattern)
                  if (Path(d) / "test_results" / "prediction.npy").is_file())
    if not dirs:
        raise FileNotFoundError(f"No run with test_results/prediction.npy matches: {pattern}")
    seeds = [Path(d).parent.name for d in dirs]
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:  # duplicates would be counted twice in the ensemble
        raise ValueError(f"More than one run directory for: {dup}. Keep one per seed.")
    return dirs


def load_results(path, shape, mean, std):
    """
    Load a BasicTS results file and convert it back to original units.
    BasicTS may write a raw float32 memmap even with a .npy extension.
    """
    with open(path, "rb") as f:
        has_header = f.read(6) == b"\x93NUMPY"
    a = np.load(path, mmap_mode="r") if has_header else np.memmap(path, dtype=np.float32, mode="r")
    a = np.asarray(a, dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise ValueError(f"{path}: got {a.size} values, expected {shape}")
    return a.reshape(shape) * std + mean


# ----------------------------------------------------------------------
# Last-batch workaround
# ----------------------------------------------------------------------
def fix_misplaced_last_batch(targets, preds, true, tol=1e-2):
    """
    Some BasicTS versions write the last (partial) test batch at the wrong
    offset: it overwrites the first windows and leaves garbage at the end.
    Bad windows then form two equal blocks (head = overwritten, tail = garbage).
    The tail's real content sits in the head block: move it back and drop
    the lost head windows. Returns the arrays restricted to valid windows
    and the boolean mask of kept windows.
    """
    keep = np.ones(true.shape[0], dtype=bool)
    bad = np.flatnonzero(np.abs(targets - true).max(axis=(1, 2)) > tol)
    if bad.size == 0:
        return targets, preds, true, keep

    head, tail = bad[: bad.size // 2], bad[bad.size // 2:]
    if bad.size % 2 or not np.allclose(targets[head], true[tail], atol=tol):
        raise RuntimeError(
            f"{bad.size} windows differ from the raw test data and do not match the "
            "last-batch pattern. Check that mean/std match the scaler used by BasicTS."
        )

    print(f"[fix] last batch misplaced: moving windows {head[0]}-{head[-1]} to "
          f"{tail[0]}-{tail[-1]}, dropping {head.size} lost windows")
    preds[:, tail] = preds[:, head]
    targets[tail] = targets[head]
    keep[head] = False
    return targets[keep], preds[:, keep], true[keep], keep


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def metrics(pred, true):
    err = pred - true
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "WAPE": np.abs(err).sum() / max(np.abs(true).sum(), 1e-12),
    }


def cross_check_with_basicts(run_dirs, preds, true, std):
    """BasicTS computes test_metrics.json during evaluation, unaffected by the saving issue."""
    print("\n[check] normalized MAE: recomputed vs BasicTS test_metrics.json")
    for d, p in zip(run_dirs, preds):
        f = Path(d) / "test_metrics.json"
        if not f.is_file():
            print(f"  {Path(d).parent.name}: test_metrics.json not found")
            continue
        ref = json.load(open(f))["overall"]["MAE"]
        mine = np.abs((p - true) / std).mean()
        print(f"  {Path(d).parent.name}: {mine:.4f} vs {ref:.4f} (should agree within ~1%)")


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_learning_curves(ax, pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[warning] no log files match: {pattern}")
    for f in files:
        txt = Path(f).read_text(errors="replace")
        train = [float(x) for x in re.findall(r"train/loss: ([\d.eE+-]+)", txt)]
        val = [float(x) for x in re.findall(r"val/MAE: ([\d.eE+-]+)", txt)]
        if train:
            ax.plot(train, "--", alpha=0.6)
        if val:
            ax.plot(val, label=Path(f).stem)
    ax.set(title="train loss (--) vs val MAE (-), normalized", xlabel="epoch")
    if files:
        ax.legend(fontsize=6)


def main():
    args = parse_args()
    IN, OUT = args.input_len, args.output_len
    if IN < STEPS_PER_DAY:
        raise ValueError("input_len must cover at least one day for the seasonal baseline")
    data = Path(args.data)

    # --- ground truth (original units) ---
    train = load_series(data / "train_data.npy")
    test = load_series(data / "test_data.npy")
    test_ts = np.load(data / "test_timestamps.npy")
    mean, std = train.mean(0), train.std(0) + 1e-8  # per-node z-score (norm_each_channel=True)
    N = test.shape[1]
    S_all = test.shape[0] - IN - OUT + 1
    if S_all <= 0:
        raise ValueError("Test series shorter than input_len + output_len")

    win = sliding_window_view(test, IN + OUT, axis=0)  # [S, N, IN+OUT]
    true = win[:, :, IN:].transpose(0, 2, 1)           # [S, OUT, N]

    # --- saved results ---
    run_dirs = find_run_dirs(args.ckpts)
    print(f"Found {len(run_dirs)} run(s): " + ", ".join(Path(d).parent.name for d in run_dirs))
    shape = (S_all, OUT, N)
    targets = load_results(Path(run_dirs[0]) / "test_results" / "targets.npy", shape, mean, std)
    preds = np.stack([load_results(Path(d) / "test_results" / "prediction.npy", shape, mean, std)
                      for d in run_dirs])  # [M, S, OUT, N]

    print(f"[check] max |targets - raw test| = {np.abs(targets - true).max():.4g} (should be ~0)")
    targets, preds, true, keep = fix_misplaced_last_batch(targets, preds, true)
    S = true.shape[0]
    print(f"[check] windows evaluated: {S} of {S_all}; "
          f"max |targets - raw test| = {np.abs(targets - true).max():.4g}")

    # --- baselines (same windows as the models) ---
    last = np.repeat(win[:, :, IN - 1][keep][:, None, :], OUT, axis=1)           # last observed value
    s0 = IN - STEPS_PER_DAY
    seasonal = win[:, :, s0:s0 + OUT][keep].transpose(0, 2, 1)                   # same slot, 24h earlier

    ens_mean, ens_std = preds.mean(0), preds.std(0)
    rows = {"last value": last, "seasonal naive (24h)": seasonal}
    rows.update({Path(d).parent.name: p for d, p in zip(run_dirs, preds)})
    if len(preds) > 1:
        rows["ensemble mean"] = ens_mean

    print(f"\n{'model':24s}{'MAE':>10s}{'RMSE':>10s}{'WAPE':>10s}   (original units)")
    for name, p in rows.items():
        m = metrics(p, true)
        print(f"{name:24s}{m['MAE']:10.3f}{m['RMSE']:10.3f}{m['WAPE']:10.3f}")

    cross_check_with_basicts(run_dirs, preds, true, std)

    # --- ensemble spread vs error (preview of the uncertainty module) ---
    if len(preds) > 1:
        err = np.abs(ens_mean - true)
        corr = np.corrcoef(ens_std.ravel()[::50], err.ravel()[::50])[0, 1]
        cov = (err <= 1.645 * ens_std).mean()
        print(f"\n[ensemble] corr(std, |err|) = {corr:.3f} (>0 = spread is informative)")
        print(f"[ensemble] coverage of mean +/- 1.645 std = {cov:.3f} (nominal 0.90)")

    # --- figure ---
    best = ens_mean if len(preds) > 1 else preds[0]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))

    plot_learning_curves(ax[0, 0], args.logs)

    for name, p in rows.items():
        if not name.startswith("seed"):
            ax[0, 1].plot(range(1, OUT + 1), np.abs(p - true).mean((0, 2)), marker="o", label=name)
    ax[0, 1].set(title="MAE per horizon", xlabel="horizon (x10 min)")
    ax[0, 1].legend(fontsize=7)

    idx = np.random.default_rng(0).choice(S * N, min(20000, S * N), replace=False)
    t1, p1 = true[:, 0].ravel()[idx], best[:, 0].ravel()[idx]
    lim = [min(0, t1.min(), p1.min()), max(t1.max(), p1.max())]
    ax[0, 2].scatter(t1, p1, s=2, alpha=0.3)
    ax[0, 2].plot(lim, lim, "r-")
    ax[0, 2].set(title="pred vs true (h=1)", xlabel="true", ylabel="pred")

    order = np.argsort(true[:, 0].mean(0))
    for node, label in [(order[-1], "busiest"), (order[N // 2], "median"), (order[0], "quietest")]:
        ax[1, 0].plot(true[:432, 0, node], label=f"true {label}")
        ax[1, 0].plot(best[:432, 0, node], "--", label=f"pred {label}")
    ax[1, 0].set(title="h=1, first 432 evaluated windows", xlabel="step (10 min)")
    ax[1, 0].legend(fontsize=6)

    slot = ((test_ts[IN:IN + S_all, 0] * STEPS_PER_DAY).round().astype(int) % STEPS_PER_DAY)[keep]
    mae_step = np.abs(best[:, 0] - true[:, 0]).mean(1)
    ax[1, 1].plot([mae_step[slot == s].mean() if (slot == s).any() else np.nan
                   for s in range(STEPS_PER_DAY)])
    ax[1, 1].set(title="MAE by time of day (h=1)", xlabel="slot (x10 min)")

    res = (best[:, 0] - true[:, 0]).ravel()[idx]
    ax[1, 2].hist(res, bins=100)
    ax[1, 2].set(title=f"residuals h=1 (mean={res.mean():.2f})")

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
