"""
The N busiest cells (default 5), one panel each, with the conformal-by-level upper bound drawn at
several coverage levels (default 70 / 75 / 80 / 85 / 90 / 95 %). Saves ONE png (default results/figures/conformal/busiest_levels.png).

  blue   = real activity
  orange = simple prediction (ensemble mean), dashed
  red    = prediction + conformal margin by level, one line per coverage level
           (darker red = higher coverage; the legend also gives the coverage reached by THAT cell)

Calibration on the val split, plot on the test split (same setup as conformal_eval.py).

Usage:
    python src/plotting/busiest_levels.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python src/plotting/busiest_levels.py --horizon 2 --start 0 --length 432        # 30 min ahead, first 3 days
    python src/plotting/busiest_levels.py --levels 0.95 0.90 0.80 --n_busy 3

--horizon is 0-based (0 = 10 min ahead, 2 = 30 min). Original scale [a.u.].
Needs only analyse_result.py and conformal_eval.py (in src/analysis/).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))  # analyse_result.py, conformal_eval.py
from analyse_result import STEP_H, STEPS_PER_DAY, clock_axis
from conformal_eval import N_BINS, conf_q, load_split, subset

C_TRUE, C_PRED = "tab:blue", "tab:orange"


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
    ap.add_argument("--levels", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                    help="target coverages to draw")
    ap.add_argument("--n_busy", type=int, default=5, help="number of busiest cells")
    ap.add_argument("--start", type=int, default=0, help="first time step to plot")
    ap.add_argument("--length", type=int, default=None, help="number of time steps (default: all)")
    ap.add_argument("--out", default="results/figures/conformal/busiest_levels.png")
    return ap.parse_args()


def to_full(v, idx, n):
    """Place windows on a regular time grid, NaN where windows were dropped (line breaks at gaps)."""
    full = np.full(n, np.nan)
    full[idx] = v
    return full


def bounds_for_cells(cal, ev, cells, levels):
    """Conformal-by-level upper bound at each level, only for the chosen cells -> {level: [S, k]}."""
    edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    lb_c = np.searchsorted(edges, cal["pred"], side="right")
    lb_e = np.searchsorted(edges, ev["pred"][:, cells], side="right")
    res_c = cal["truth"] - cal["pred"]
    out = {}
    for lev in levels:
        qb = np.array([conf_q(res_c[lb_c == b], 1 - lev) for b in range(N_BINS)])
        out[lev] = ev["pred"][:, cells] + qb[lb_e]
    return out


def plot(cal, ev, args):
    levels = sorted(args.levels)
    h_min = (args.horizon + 1) * 10
    order = np.argsort(ev["truth"].mean(0))[::-1]        # busiest first
    cells = [int(c) for c in order[:min(args.n_busy, order.size)]]
    bounds = bounds_for_cells(cal, ev, cells, levels)

    u = ev["day"] * STEPS_PER_DAY + ev["slot"]           # absolute time step of each window
    idx = u - u.min()
    n = int(idx.max()) + 1
    a = args.start
    if a >= n:
        raise ValueError(f"--start must be < {n}")
    b = n if args.length is None else min(a + args.length, n)
    x = np.arange(a, b) * STEP_H
    tod0 = (int(ev["slot"][0]) + a) * STEP_H % 24
    dow0 = int(ev["dow"][0]) + (int(ev["slot"][0]) + a) // STEPS_PER_DAY

    cmap = plt.get_cmap("Reds")
    shade = {lev: cmap(0.30 + 0.65 * k / max(len(levels) - 1, 1)) for k, lev in enumerate(levels)}

    fig, axes = plt.subplots(len(cells), 1, figsize=(24 if (b - a) > 432 else 16, 3.8 * len(cells)),
                             sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        c = cells[j]
        t = to_full(ev["truth"][:, c], idx, n)[a:b]
        p = to_full(ev["pred"][:, c], idx, n)[a:b]
        ax.fill_between(x, p, to_full(bounds[levels[-1]][:, j], idx, n)[a:b], color=shade[levels[-1]],
                        alpha=0.10, lw=0)
        ax.plot(x, t, color=C_TRUE, lw=1.3, label="true", zorder=3)
        ax.plot(x, p, color=C_PRED, lw=1.3, ls="--", label="pred (ensemble mean)", zorder=4)
        for lev in levels:
            U = to_full(bounds[lev][:, j], idx, n)[a:b]
            cov = (ev["truth"][:, c] <= bounds[lev][:, j]).mean()
            ax.plot(x, U, color=shade[lev], lw=1.0, label=f"{lev:.0%} (cell coverage {cov:.3f})", zorder=2)
        mae = np.abs(ev["truth"][:, c] - ev["pred"][:, c]).mean()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("activity [a.u.]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.set_title(f"#{j + 1} busiest (col {c}): mean activity {ev['truth'][:, c].mean():.0f}, MAE {mae:.1f} "
                     f"(h={h_min} min, all evaluated windows; values in a.u.)", fontsize=10, pad=14)
        clock_axis(ax, x, tod0, dow0)
    axes[-1, 0].set_xlabel("time (clock) - gaps = windows dropped by the last-batch fix")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)

    print("overall test coverage (all cells):")
    for lev in levels:
        edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
        lb_c = np.searchsorted(edges, cal["pred"], side="right")
        lb_e = np.searchsorted(edges, ev["pred"], side="right")
        qb = np.array([conf_q((cal["truth"] - cal["pred"])[lb_c == k], 1 - lev) for k in range(N_BINS)])
        print(f"  target {lev:.0%}: {(ev['truth'] <= ev['pred'] + qb[lb_e]).mean():.3f}")
    print(f"cells plotted (column index, busiest first): {cells}")
    print(f"Saved {args.out}")


def main():
    args = parse_args()
    if not all(0.5 < c < 1 for c in args.levels):
        raise SystemExit("--levels must be coverages in (0.5, 1), e.g. 0.70 0.95")
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
    plot(cal, ev, args)


if __name__ == "__main__":
    main()