"""
Signed error (pred - true) instead of MAE, over the whole evaluated test period.

  row 1: error over time, all evaluated windows (about 5 days, clock labels, weekday at midnight)
         red  = over-estimate (pred > true), blue = under-estimate (pred < true)
  row 2: the same error folded on the 24 h clock and averaged over the days: ONE mean line
         (with --show_days also one thin line per day and the Tue-Fri-only mean, to see whether a
         single day dominates the average; Saturday behaves differently from the weekdays)

One column per series: all cells (mean error per cell), then busiest / median / quietest cell
(same selection as plot_timeseries.py). Units are the original traffic units.

Usage:
    python plot_signed_error.py
    python plot_signed_error.py --horizon 6
    python plot_signed_error.py --cells 12 345 1800
    python plot_signed_error.py --seed 0          # one seed instead of the ensemble mean
    python plot_signed_error.py --no_all          # only the single cells
    python plot_signed_error.py --show_days       # row 2 also shows each day and the Tue-Fri mean

Loading and last-batch fix are reused from analyse_result.py (same folder).
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
    ap.add_argument("--cells", type=int, nargs="*", default=[],
                    help="column indices; default: busiest, median, quietest")
    ap.add_argument("--no_all", action="store_true", help="skip the 'all cells' column")
    ap.add_argument("--seed", type=int, default=None, help="use this seed only (default: ensemble mean)")
    ap.add_argument("--show_days", action="store_true",
                    help="row 2: also draw one line per day and the weekday-only mean")
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
    true_h = true[:, h, :]
    err = est - true_h                                  # positive = over-estimate

    # --- columns: (label, error vector [S_kept], MAE shown in the title) ---
    cols = []
    if not args.no_all:
        cols.append(("all cells (mean per cell)", err.mean(1), np.abs(err).mean()))
    if args.cells:
        nodes, labels = args.cells, [f"cell col {n}" for n in args.cells]
        if max(nodes) >= N or min(nodes) < 0:
            raise ValueError(f"--cells must be in 0..{N - 1}")
    else:
        order = np.argsort(true_h.mean(0))
        nodes = [int(order[-1]), int(order[N // 2]), int(order[0])]
        labels = ["busiest", "median", "quietest"]
    for n, lab in zip(nodes, labels):
        cols.append((f"{lab} (col {n})", err[:, n], np.abs(err[:, n]).mean()))

    # --- time axes ---
    x = np.arange(S_all) * STEP_H                       # hours since the first target
    tod0 = round(test_ts[IN + h, 0] * 24 * 6) / 6
    dow0 = int(round(test_ts[IN + h, 1] * 7)) % 7
    step_idx = IN + h + np.arange(S_all)                # test step each window predicts
    day = step_idx // STEPS_PER_DAY
    slot = step_idx % STEPS_PER_DAY
    days = np.unique(day)
    d_row = day - days[0]
    dow = np.round(test_ts[days * STEPS_PER_DAY, 1] * 7).astype(int) % 7
    dom = np.round(test_ts[days * STEPS_PER_DAY, 2] * 31).astype(int) + 1
    is_wd = dow < 5

    fig, axes = plt.subplots(2, len(cols), figsize=(8 * len(cols), 8), squeeze=False)
    print(f"\nsigned error = pred - true ({who}, horizon {args.horizon} = {args.horizon * 10} min ahead), "
          f"{len(days)} days ({S_all * STEP_H / 24:.1f} days of windows)")
    print(f"{'series':28s}{'bias':>9s}{'MAE':>9s}   bias by day: "
          + " ".join(f"{WEEKDAYS[d]}{m}".rjust(8) for d, m in zip(dow, dom)))

    for c, (label, vec, mae) in enumerate(cols):
        full = to_full(vec, keep)
        E = np.full((len(days), STEPS_PER_DAY), np.nan, dtype=np.float32)
        E[d_row, slot] = full                           # [day, time-of-day slot]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            day_bias = np.nanmean(E, axis=1)
            mean_all = np.nanmean(E, axis=0)
            mean_wd = np.nanmean(E[is_wd], axis=0) if is_wd.any() else None
        bias = np.nanmean(full)
        worst = int(np.nanargmax(np.abs(mean_all)))
        print(f"{label:28s}{bias:+9.2f}{mae:9.2f}   "
              + " ".join(f"{b:+8.1f}" for b in day_bias)
              + f"   | largest mean error {mean_all[worst]:+.1f} at {hhmm(worst)}")

        # ---- row 1: error over time ----
        ax = axes[0, c]
        fin = np.isfinite(full)
        y = np.where(fin, full, 0.0)
        ax.plot(x, full, color="0.25", lw=0.7)
        ax.fill_between(x, 0, y, where=fin & (y > 0), color="tab:red", alpha=0.35, lw=0, interpolate=True)
        ax.fill_between(x, 0, y, where=fin & (y < 0), color="tab:blue", alpha=0.35, lw=0, interpolate=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("error = pred - true (traffic)")
        ax.grid(alpha=0.3)
        ax.set_title(f"{label}\nbias {bias:+.2f}  |  MAE {mae:.2f}  (red = over, blue = under)",
                     fontsize=10, pad=16)
        clock_axis(ax, x, tod0, dow0)

        # ---- row 2: mean over the days, by time of day ----
        ax = axes[1, c]
        xs = np.arange(STEPS_PER_DAY) * STEP_H
        if args.show_days:
            for k in range(len(days)):
                ax.plot(xs, E[k], lw=0.9, alpha=0.55,
                        color="tab:red" if dow[k] >= 5 else plt.cm.Blues(0.45 + 0.12 * k),
                        label=f"{WEEKDAYS[dow[k]]} {dom[k]}")
            if mean_wd is not None and (~is_wd).any():
                ax.plot(xs, mean_wd, color="0.35", lw=1.4, ls="--", label=f"mean {int(is_wd.sum())} weekdays")
        ax.plot(xs, mean_all, color="k", lw=2.0, label=f"mean of {len(days)} days")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{t:02d}:00" for t in range(0, 25, 3)], fontsize=8)
        ax.set_xlabel("time of day")
        ax.set_ylabel("mean error = pred - true (traffic)")
        ax.set_title(f"mean of the {len(days)} days ({WEEKDAYS[dow[0]]}-{WEEKDAYS[dow[-1]]}), by time of day",
                     fontsize=10)
        ax.grid(alpha=0.3)
        if args.show_days:
            ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
