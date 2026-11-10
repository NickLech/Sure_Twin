"""
Signed error (pred - true) over the whole evaluated test period.

  top    : signed error over time, all evaluated windows (about 5 days)
           red = over-estimate (pred > true), blue = under-estimate (pred < true)
  bottom : the same error folded on the 24 h clock and averaged over the days
           left  = mean over ALL days
           right = mean over weekdays only (Saturday/Sunday excluded)

Default series: mean error over all cells. Use --cell N for a single cell.
Units are the original traffic units.

Usage:
    python plot_signed_error.py
    python plot_signed_error.py --horizon 6
    python plot_signed_error.py --cell 1800
    python plot_signed_error.py --seed 0      # one seed instead of the ensemble mean
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from analyse_result import (STEP_H, STEPS_PER_DAY, WEEKDAYS, clock_axis, find_run_dirs,
                            fix_misplaced_last_batch, load_results, load_series)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*")
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1, help="1..output_len (x10 min ahead)")
    ap.add_argument("--cell", type=int, default=None,
                    help="single column index; default: mean error over all cells")
    ap.add_argument("--seed", type=int, default=None, help="use this seed only (default: ensemble mean)")
    ap.add_argument("--out", default="signed_error.png")
    return ap.parse_args()


def to_full(v, keep):
    """Put dropped windows back as NaN, so lines break instead of joining across the gap."""
    full = np.full(keep.size, np.nan, dtype=np.float32)
    full[keep] = v
    return full


def hhmm(slot):
    m = int(slot) * 10
    return f"{m // 60:02d}:{m % 60:02d}"


def signed_fill(ax, x, y):
    """Line + red/blue fill depending on the sign of y (NaN-safe)."""
    fin = np.isfinite(y)
    y0 = np.where(fin, y, 0.0)
    ax.plot(x, y, color="0.25", lw=1.0)
    ax.fill_between(x, 0, y0, where=fin & (y0 > 0), color="tab:red", alpha=0.35, lw=0, interpolate=True)
    ax.fill_between(x, 0, y0, where=fin & (y0 < 0), color="tab:blue", alpha=0.35, lw=0, interpolate=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)


def main():
    args = parse_args()
    IN, OUT, h = args.input_len, args.output_len, args.horizon - 1
    if not 0 <= h < OUT:
        raise ValueError(f"--horizon must be in 1..{OUT}")
    data = Path(args.data)

    train = load_series(data / "train_data.npy")
    test = load_series(data / "test_data.npy")
    test_ts = np.load(data / "test_timestamps.npy")
    mean, std = train.mean(0), train.std(0) + 1e-8
    N = test.shape[1]
    S_all = test.shape[0] - IN - OUT + 1

    win = sliding_window_view(test, IN + OUT, axis=0)
    true = win[:, :, IN:].transpose(0, 2, 1)  # [S, OUT, N]
    run_dirs = find_run_dirs(args.ckpts)
    shape = (S_all, OUT, N)
    targets = load_results(Path(run_dirs[0]) / "test_results" / "targets.npy", shape, mean, std)
    preds = np.stack([load_results(Path(d) / "test_results" / "prediction.npy", shape, mean, std)
                      for d in run_dirs])  # [M, S, OUT, N]
    targets, preds, true, keep = fix_misplaced_last_batch(targets, preds, true)

    # estimate at one horizon: ensemble mean or a single seed -> [S_kept, N]
    if args.seed is None:
        est, who = preds[:, :, h, :].mean(0), "ensemble mean"
    else:
        names = [Path(d).parent.name for d in run_dirs]
        if f"seed{args.seed}" not in names:
            raise ValueError(f"seed{args.seed} not found among {names}")
        est, who = preds[names.index(f"seed{args.seed}")][:, h, :], f"seed{args.seed}"
    err = est - true[:, h, :]                           # positive = over-estimate

    # --- the single series to plot ---
    if args.cell is None:
        vec, label = err.mean(1), "all cells (mean per cell)"
    else:
        if not 0 <= args.cell < N:
            raise ValueError(f"--cell must be in 0..{N - 1}")
        vec, label = err[:, args.cell], f"cell col {args.cell}"
    full = to_full(vec, keep)                           # [S_all], NaN where dropped

    # --- time axes ---
    x = np.arange(S_all) * STEP_H                       # hours since the first target
    tod0 = round(test_ts[IN + h, 0] * 24 * 6) / 6
    dow0 = int(round(test_ts[IN + h, 1] * 7)) % 7
    step_idx = IN + h + np.arange(S_all)                # test step each window predicts
    day = step_idx // STEPS_PER_DAY
    slot = step_idx % STEPS_PER_DAY
    days = np.unique(day)
    dow = np.round(test_ts[days * STEPS_PER_DAY, 1] * 7).astype(int) % 7
    is_wd = dow < 5                                     # Sat/Sun excluded

    # E[d, s] = error of day d at time-of-day slot s
    E = np.full((len(days), STEPS_PER_DAY), np.nan, dtype=np.float32)
    E[day - days[0], slot] = full
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(E, axis=0)                # [STEPS_PER_DAY]
        mean_wd = np.nanmean(E[is_wd], axis=0)

    # --- figure ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax_t = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1], sharey=ax_a)

    bias, mae = np.nanmean(full), np.nanmean(np.abs(full))
    signed_fill(ax_t, x, full)
    ax_t.set_ylabel("error = pred - true (traffic)")
    ax_t.set_title(f"{label} - signed error over time ({who}, horizon {args.horizon * 10} min)\n"
                   f"bias {bias:+.2f}  |  MAE {mae:.2f}  (red = over, blue = under)",
                   fontsize=10, pad=16)
    clock_axis(ax_t, x, tod0, dow0)

    xs = np.arange(STEPS_PER_DAY) * STEP_H
    wd_names = f"{WEEKDAYS[dow[is_wd][0]]}-{WEEKDAYS[dow[is_wd][-1]]}"
    all_names = f"{WEEKDAYS[dow[0]]}-{WEEKDAYS[dow[-1]]}"
    for ax, m, title in [(ax_a, mean_all, f"mean of {len(days)} days ({all_names})"),
                         (ax_w, mean_wd, f"mean of {int(is_wd.sum())} weekdays ({wd_names})")]:
        xs = np.arange(STEPS_PER_DAY) * STEP_H
        wd_names = f"{WEEKDAYS[dow[is_wd][0]]}-{WEEKDAYS[dow[is_wd][-1]]}"
        all_names = f"{WEEKDAYS[dow[0]]}-{WEEKDAYS[dow[-1]]}"
        n_days, n_wd = len(days), int(is_wd.sum())

        # left: mean over all days
        signed_fill(ax_a, xs, mean_all)
        ax_a.set_title(f"mean of {n_days} days ({all_names})", fontsize=10)

        # right: mean over weekdays, with the all-days mean overlaid for comparison
        signed_fill(ax_w, xs, mean_wd)
        ax_w.lines[0].set_label(f"mean of {n_wd} weekdays ({wd_names})")
        ax_w.plot(xs, mean_all, color="tab:orange", lw=1.6, ls="--",
                  label=f"mean of {n_days} days ({all_names})")
        ax_w.set_title(f"weekdays ({wd_names}) vs all days ({all_names})", fontsize=10)
        ax_w.legend(fontsize=8, loc="best")

        for ax in (ax_a, ax_w):
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 3))
            ax.set_xticklabels([f"{t:02d}:00" for t in range(0, 25, 3)], fontsize=8)
            ax.set_xlabel("time of day")
        ax_a.set_ylabel("mean error = pred - true (traffic)")

    for name, m in [("all days", mean_all), ("weekdays", mean_wd)]:
        w = int(np.nanargmax(np.abs(m)))
        print(f"{name:10s} largest mean error {m[w]:+.2f} at {hhmm(w)}")
    print(f"{label}: bias {bias:+.2f}, MAE {mae:.2f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
