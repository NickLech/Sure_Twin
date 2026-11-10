"""
True vs predicted traffic for a few cells: one panel per cell, each with its own y-axis,
so cells with very different traffic stay readable (analysis.png squashes them together).

Usage:
    python plot_timeseries.py
    python plot_timeseries.py --horizon 6 --start 0 --length 144
    python plot_timeseries.py --cells 12 345 1800 --band 1.645

Loading and last-batch fix are reused from analyse_result.py (same folder).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from analyse_result import (STEPS_PER_DAY, find_run_dirs, fix_misplaced_last_batch,
                            load_results, load_series)

STEP_H = 24 / STEPS_PER_DAY  # hours per step
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*")
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1, help="1..output_len (x10 min ahead)")
    ap.add_argument("--start", type=int, default=0, help="first window to plot")
    ap.add_argument("--length", type=int, default=432, help="number of windows to plot")
    ap.add_argument("--cells", type=int, nargs="*", default=[],
                    help="column indices; default: busiest, median, quietest")
    ap.add_argument("--band", type=float, default=0.0,
                    help="if > 0 draw ensemble mean +/- band * std (NOT calibrated yet)")
    ap.add_argument("--out", default="timeseries.png")
    return ap.parse_args()


def to_full(x, keep):
    """Put dropped windows back as NaN, so the line breaks instead of joining across the gap."""
    full = np.full((keep.size,) + x.shape[1:], np.nan, dtype=np.float32)
    full[keep] = x
    return full


def clock_axis(ax, x, tod0, dow0):
    """Tick labels as clock time, dashed line + weekday at every midnight."""
    first = np.ceil(tod0 / 6) * 6 - tod0
    pos = np.arange(first, x[-1] + 1e-9, 6)
    clock = tod0 + pos
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{c % 24:02.0f}:00" for c in clock], fontsize=8)
    for p, c in zip(pos, clock):
        if np.isclose(c % 24, 0):
            ax.axvline(p, color="gray", ls=":", lw=1)
            ax.text(p, 1.0, WEEKDAYS[int(dow0 + np.floor(c / 24 + 1e-6)) % 7],
                    transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=8)


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

    # one horizon only: [S, N]
    true_h = true[:, h, :]
    ph = preds[:, :, h, :]
    mean_h, std_h = ph.mean(0), ph.std(0)

    # cells to plot
    if args.cells:
        nodes, labels = args.cells, [f"cell col {n}" for n in args.cells]
        if max(nodes) >= N or min(nodes) < 0:
            raise ValueError(f"--cells must be in 0..{N - 1}")
    else:
        order = np.argsort(true_h.mean(0))
        nodes = [int(order[-1]), int(order[N // 2]), int(order[0])]
        labels = ["busiest", "median", "quietest"]
    cell_mae = np.abs(mean_h[:, nodes] - true_h[:, nodes]).mean(0)  # over all evaluated windows
    cell_avg = true_h[:, nodes].mean(0)

    tf = to_full(true_h[:, nodes], keep)
    pf = to_full(mean_h[:, nodes], keep)
    sf = to_full(std_h[:, nodes], keep)

    a, b = args.start, min(args.start + args.length, S_all)
    if a >= S_all:
        raise ValueError(f"--start must be < {S_all}")
    x = np.arange(b - a) * STEP_H  # hours since the first plotted target
    tod0 = round(test_ts[IN + a + h, 0] * 24 * 6) / 6
    dow0 = int(round(test_ts[IN + a + h, 1] * 7)) % 7

    fig, axes = plt.subplots(len(nodes), 1, figsize=(16, 3.4 * len(nodes)), sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        ax.plot(x, tf[a:b, j], color="tab:blue", lw=1.3, label="true")
        ax.plot(x, pf[a:b, j], color="tab:orange", lw=1.3, ls="--", label="pred (ensemble mean)")
        if args.band > 0:
            lo = np.maximum(pf[a:b, j] - args.band * sf[a:b, j], 0)
            hi = pf[a:b, j] + args.band * sf[a:b, j]
            ax.fill_between(x, lo, hi, color="tab:orange", alpha=0.25,
                            label=f"mean ± {args.band:g}·std (uncalibrated)")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("traffic")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{labels[j]} (col {nodes[j]}): mean traffic {cell_avg[j]:.0f}, "
                     f"MAE {cell_mae[j]:.1f} (h={args.horizon}, all evaluated windows)", fontsize=10)
    clock_axis(axes[-1, 0], x, tod0, dow0)
    axes[-1, 0].set_xlabel("time (clock) - gaps = windows dropped by the last-batch fix")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"cells plotted (column index): {dict(zip(labels, nodes))}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
