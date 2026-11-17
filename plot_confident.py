"""
True vs prediction vs conformal upper bound ("conformal by level"), same style as timeseries.png.

  blue   = real traffic
  orange = simple prediction (ensemble mean), dashed
  red    = prediction + conformal margin by level (upper bound)

Two files are saved:
  confident_pred.png    busiest / median / quietest cell (one panel each, own y-axis)
  busiest_confident.png the --n_busy busiest cells (default 5), busiest first

Calibration is done on the val split, the plots show the test split (as in conformal_eval.py).

Self-contained: needs only analyse_result.py and conformal_eval.py (already in the repo).

Usage:
    python plot_confident.py --horizon 0 --eps 0.05 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python plot_confident.py --horizon 2 --eps 0.10 --start 0 --length 432     # first 3 days only

--horizon is 0-based like in conformal_eval.py (0 = 10 min ahead, 2 = 30 min).
Everything is in ORIGINAL traffic units.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_result import STEP_H, STEPS_PER_DAY, clock_axis
from conformal_eval import N_BINS, conf_q, load_split, subset

C_TRUE, C_PRED, C_UP = "tab:blue", "tab:orange", "tab:red"


def level_bins(cal, ev):
    """Predicted-level deciles, defined on the calibration set."""
    edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    return (np.searchsorted(edges, cal["pred"], side="right"),
            np.searchsorted(edges, ev["pred"], side="right"))


def margins(res_c, lb_c, eps):
    """Per-decile conformal margin and the single global margin."""
    qb = np.array([conf_q(res_c[lb_c == b], eps) for b in range(N_BINS)])
    return qb, conf_q(res_c, eps)


def to_full(v, idx, n):
    """Place windows on a regular time grid, NaN where windows were dropped (line breaks at gaps)."""
    full = np.full(n, np.nan)
    full[idx] = v
    return full


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*", help="runs of the EVALUATION split (test)")
    ap.add_argument("--data", default="datasets/MILAN", help="dataset of the EVALUATION split (test)")
    ap.add_argument("--cal_ckpts", default=None, help="runs of the CALIBRATION split (e.g. val view)")
    ap.add_argument("--cal_data", default=None, help="dataset view of the calibration split")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=0, help="0 = first step = 10 min ahead")
    ap.add_argument("--cal_frac", type=float, default=0.5, help="only used without --cal_data")
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--csv", default="milan_raw.csv")
    ap.add_argument("--eps", type=float, default=0.05, help="target violation rate (0.05 = 95%% coverage)")
    ap.add_argument("--n_busy", type=int, default=5, help="cells in busiest_confident.png")
    ap.add_argument("--start", type=int, default=0, help="first time step to plot")
    ap.add_argument("--length", type=int, default=None, help="number of time steps (default: all)")
    ap.add_argument("--out_pred", default="confident_pred.png")
    ap.add_argument("--out_busy", default="busiest_confident.png")
    return ap.parse_args()


def conformal_upper(cal, ev, eps):
    """Upper bound on the evaluation windows: prediction + margin of its predicted-level decile."""
    lb_c, lb_e = level_bins(cal, ev)
    qb, _ = margins(cal["truth"] - cal["pred"], lb_c, eps)
    return ev["pred"] + qb[lb_e]


def draw(ev, up, cells, labels, h_min, eps, start, length, path):
    ok = ev["truth"] <= up
    u = ev["day"] * STEPS_PER_DAY + ev["slot"]           # absolute time step of each window
    idx = u - u.min()
    n = int(idx.max()) + 1
    a = start
    if a >= n:
        raise ValueError(f"--start must be < {n}")
    b = n if length is None else min(a + length, n)
    x = np.arange(a, b) * STEP_H
    tod0 = (int(ev["slot"][0]) + a) * STEP_H % 24
    dow0 = int(ev["dow"][0]) + (int(ev["slot"][0]) + a) // STEPS_PER_DAY

    fig, axes = plt.subplots(len(cells), 1, figsize=(24 if (b - a) > 432 else 16, 3.6 * len(cells)),
                             sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        c = cells[j]
        t = to_full(ev["truth"][:, c], idx, n)[a:b]
        p = to_full(ev["pred"][:, c], idx, n)[a:b]
        U = to_full(up[:, c], idx, n)[a:b]
        ax.plot(x, t, color=C_TRUE, lw=1.3, label="true")
        ax.plot(x, p, color=C_PRED, lw=1.3, ls="--", label="pred (ensemble mean)")
        ax.plot(x, U, color=C_UP, lw=1.2, label=f"pred + conformal margin ({1 - eps:.0%} target)")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("traffic")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        mae = np.abs(ev["truth"][:, c] - ev["pred"][:, c]).mean()
        ax.set_title(f"{labels[j]} (col {c}): mean traffic {ev['truth'][:, c].mean():.0f}, MAE {mae:.1f}, "
                     f"coverage {ok[:, c].mean():.3f}, mean margin {(up[:, c] - ev['pred'][:, c]).mean():.1f} "
                     f"(h={h_min} min, all evaluated windows)", fontsize=10, pad=14)
        clock_axis(ax, x, tod0, dow0)
    axes[-1, 0].set_xlabel("time (clock) - gaps = windows dropped by the last-batch fix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"cells plotted: {dict(zip(labels, cells))}")
    print(f"Saved {path}")


def make_figures(cal, ev, args):
    h_min = (args.horizon + 1) * 10
    up = conformal_upper(cal, ev, args.eps)
    print(f"overall test coverage {(ev['truth'] <= up).mean():.3f} (target {1 - args.eps:.0%})")

    order = np.argsort(ev["truth"].mean(0))              # ascending mean traffic
    N = order.size
    cells = [int(order[-1]), int(order[N // 2]), int(order[0])]
    draw(ev, up, cells, ["busiest", "median", "quietest"], h_min, args.eps, args.start, args.length, args.out_pred)

    k = min(args.n_busy, N)
    busy = [int(c) for c in order[::-1][:k]]
    draw(ev, up, busy, [f"#{i + 1} busiest" for i in range(k)], h_min, args.eps, args.start, args.length,
         args.out_busy)


def main():
    args = parse_args()
    if not 0 < args.eps < 1:
        raise SystemExit("--eps must be in (0, 1)")
    ev = load_split(args.data, args.ckpts, args, "evaluation")
    if args.cal_data:
        if not args.cal_ckpts:
            raise SystemExit("--cal_data needs --cal_ckpts")
        cal = load_split(args.cal_data, args.cal_ckpts, args, "calibration")
        if cal["members"] != ev["members"]:
            raise SystemExit("different number of members between calibration and evaluation")
    else:
        cut = int(ev["pred"].shape[0] * args.cal_frac)
        cal, ev = subset(ev, slice(0, cut)), subset(ev, slice(cut, ev["pred"].shape[0]))
        print(f"FALLBACK: calibration = first {args.cal_frac:.0%} of test, evaluation = the rest")
    make_figures(cal, ev, args)


if __name__ == "__main__":
    main()
