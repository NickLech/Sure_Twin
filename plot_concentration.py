"""
Traffic concentration across cells: how many cells carry most of the traffic?

Cells are sorted by mean traffic (descending); the plot shows the cumulative
share of traffic against the cumulative share of cells (Pareto / Lorenz-like
curve). Prints the number of cells needed to reach 50/80/90% of the traffic,
the mean-traffic level of the last cell in that group, the Gini coefficient
and the point where the slope of the curve is 1 (cell traffic = mean traffic
per cell), a data-driven threshold between "busy" and "quiet" cells.

Usage:
    python plot_concentration.py
    python plot_concentration.py --splits train val test
    python plot_concentration.py --exclude 858 808
    python plot_concentration.py --out results/concentration.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # reuse the project loader if available
    from analyse_result import load_series
except Exception:  # fallback: plain numpy, squeeze trailing singleton dims
    def load_series(path):
        x = np.load(path)
        return x.reshape(x.shape[0], -1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"],
                    help="splits to concatenate in time (default: test)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="column indices to drop (e.g. 858 808: cells with missing rows)")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.5, 0.8, 0.9],
                    help="traffic shares to report")
    ap.add_argument("--out", default="results/concentration.png")
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

    keep = np.setdiff1d(np.arange(N), args.exclude)
    m = series[:, keep].mean(0)  # mean traffic per cell
    n = m.size

    order = np.argsort(m)[::-1]
    ms = m[order]
    cum = np.cumsum(ms) / ms.sum()
    frac = np.arange(1, n + 1) / n

    print(f"splits: {args.splits} | cells: {n} (excluded {len(args.exclude)})")
    marks = []
    for p in args.levels:
        k = int(np.searchsorted(cum, p)) + 1
        print(f"{k} cells ({k / n:.1%}) carry {p:.0%} of the traffic; "
              f"mean traffic of the k-th cell: {ms[k - 1]:.0f}")
        marks.append((k / n, p, k))
    G = gini(m)
    print(f"Gini = {G:.3f}")

    # slope = 1: slope of the curve at cell k is m_k / mean(m), so slope = 1
    # where a cell's traffic equals the mean traffic per cell
    mu = m.mean()
    k_star = int(np.sum(ms > mu))
    x_s, y_s = k_star / n, cum[k_star - 1]
    print(f"slope = 1: {k_star} cells ({x_s:.1%}) above the mean traffic ({mu:.0f}) "
          f"carry {y_s:.1%} of the traffic")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.r_[0, frac], np.r_[0, cum], color="tab:blue", lw=2, label="cells sorted by mean traffic")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform traffic")
    for x, y, k in marks:
        ax.plot([x, x], [0, y], color="tab:red", lw=0.8, ls=":")
        ax.plot([0, x], [y, y], color="tab:red", lw=0.8, ls=":")
        ax.plot(x, y, "o", color="tab:red")
        ax.annotate(f"{x:.0%} of cells\n= {y:.0%} of traffic", (x, y),
                    textcoords="offset points", xytext=(8, -26), fontsize=8)
    ax.plot(x_s, y_s, "s", color="tab:green", ms=8, label="slope = 1 (cell traffic = mean)")
    ax.annotate(f"{x_s:.0%} of cells\n= {y_s:.0%} of traffic", (x_s, y_s),
                textcoords="offset points", xytext=(-90, 10), fontsize=8,
                color="tab:green", arrowprops=dict(arrowstyle="-", color="tab:green", lw=0.8))
    ax.set_xlabel("cumulative share of cells (busiest first)")
    ax.set_ylabel("cumulative share of traffic")
    ax.set_title(f"Traffic concentration (Gini = {G:.2f})")
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
