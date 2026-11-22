"""
Activity concentration across cells: how many cells carry most of the activity?

Cells are sorted by mean activity (descending); the plot shows the cumulative
share of activity against the cumulative share of cells (Pareto / Lorenz-like
curve). Prints the number of cells needed to reach 50/80/90% of the activity,
the mean-activity level of the last cell in that group, the Gini coefficient
and the point where the slope of the curve is 1 (cell activity = mean activity
per cell), a data-driven threshold between "busy" and "quiet" cells.

Usage:
    python src/plotting/plot_concentration.py
    python src/plotting/plot_concentration.py --splits train val test
    python src/plotting/plot_concentration.py --exclude 858 808
    python src/plotting/plot_concentration.py --out results/figures/data/concentration.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))  # analyse_result.py
from analyse_result import kept_columns, load_series


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"],
                    help="splits to concatenate in time (default: test)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="cell_ids to drop (e.g. 858 808: cells with missing rows)")
    ap.add_argument("--csv", default="milan_raw.csv",
                    help="raw csv, only used with --exclude if datasets/MILAN/cell_ids.npy is missing")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.5, 0.8, 0.9],
                    help="activity shares to report")
    ap.add_argument("--out", default="results/figures/data/concentration.png")
    return ap.parse_args()


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def main():
    args = parse_args()
    data = Path(args.data)
    series = np.concatenate([load_series(data / f"{s}_data.npy") for s in args.splits], axis=0)
    N = series.shape[1]

    keep = kept_columns(args, N)
    m = series[:, keep].mean(0)  # mean activity per cell
    n = m.size

    order = np.argsort(m)[::-1]
    ms = m[order]
    cum = np.cumsum(ms) / ms.sum()
    frac = np.arange(1, n + 1) / n

    print(f"splits: {args.splits} | cells: {n} (excluded {len(args.exclude)})")
    marks = []
    for p in args.levels:
        k = int(np.searchsorted(cum, p)) + 1
        print(f"{k} cells ({k / n:.1%}) carry {p:.0%} of the activity; "
              f"mean activity of the k-th cell: {ms[k - 1]:.0f} a.u.")
        marks.append((k / n, p, k))
    G = gini(m)
    print(f"Gini = {G:.3f}")

    # slope = 1: slope of the curve at cell k is m_k / mean(m), so slope = 1
    # where a cell's activity equals the mean activity per cell
    mu = m.mean()
    k_star = int(np.sum(ms > mu))
    x_s, y_s = k_star / n, cum[k_star - 1]
    print(f"slope = 1: {k_star} cells ({x_s:.1%}) above the mean activity ({mu:.0f} a.u.) "
          f"carry {y_s:.1%} of the activity")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.r_[0, frac], np.r_[0, cum], color="tab:blue", lw=2, label="cells sorted by mean activity")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform activity")
    for x, y, k in marks:
        ax.plot([x, x], [0, y], color="tab:red", lw=0.8, ls=":")
        ax.plot([0, x], [y, y], color="tab:red", lw=0.8, ls=":")
        ax.plot(x, y, "o", color="tab:red")
        ax.annotate(f"{x:.0%} of cells\n= {y:.0%} of activity", (x, y),
                    textcoords="offset points", xytext=(8, -26), fontsize=8)
    ax.plot(x_s, y_s, "s", color="tab:green", ms=8, label="slope = 1 (cell activity = mean)")
    ax.annotate(f"{x_s:.0%} of cells\n= {y_s:.0%} of activity", (x_s, y_s),
                textcoords="offset points", xytext=(-90, 10), fontsize=8,
                color="tab:green", arrowprops=dict(arrowstyle="-", color="tab:green", lw=0.8))
    ax.set_xlabel("cumulative share of cells (busiest first)")
    ax.set_ylabel("cumulative share of activity")
    ax.set_title(f"Activity concentration (Gini = {G:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()