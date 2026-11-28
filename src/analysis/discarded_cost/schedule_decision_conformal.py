"""
Conformal margin calibrated on the DECISION, not on y.

The sleep rule is: sleep iff upper_bound < theta. Only cells whose prediction is close to theta need a
margin: a cell predicted far below theta sleeps anyway, one predicted far above never does. So instead of
binning the residuals by predicted level (deciles, independent of theta) we bin them by the signed
distance to the threshold

        d = theta - prediction            (d > 0: the point forecast says "sleep")

and calibrate one margin per bin and per theta (Mondrian conformal on d):

        margin[b] = conformal quantile (1 - eps) of the calibration residuals (truth - prediction) with d in bin b
        asleep    = prediction + margin[bin(d)] < theta

Bins: n_bins // 2 on each side of 0, with edges at quantiles of |d| on the calibration split (so the bins
nearest to d = 0 are the narrowest). Negative margins are clipped at 0, so the bound stays an upper bound
(a cell with prediction >= theta never sleeps).

Compared on the test set, with the same minimum dwell time n (see schedule_hysteresis.py) for every approach:
    STID point               forecast alone
    STID + level margin      decile-of-level margin (the previous proposal)
    STID + decision margin   the new one
    last value point         reference
    oracle                   y < theta (instant; not an upper bound when kappa > 0)

    net sleep % = sleep_% - 100 * kappa * wakeups_per_cell_day / STEPS_PER_DAY
    x = unmet_%  (or viol_all_% with --x viol_all)

Saved in --out_dir (default results/costs):
    decision_conformal_h<min>.png   rows = dwell n, columns = theta, one kappa (--kappa)
    decision_conformal.csv          every (horizon, kappa, dwell, theta, approach, eps)
and printed: net sleep at eps_ref, and the mean gain of the decision margin over the level margin at equal
unmet traffic (over the x range where both curves exist).

Usage (from the repository root; other options as in schedule_costs.py):
    python src/analysis/schedule_decision_conformal.py --horizons 1 3 --kappas 3 6 12 --kappa 6 \
        --dwell 1 3 6 --n_bins 10 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
Needs schedule_hysteresis.py, schedule_costs.py, schedule_conformal.py, costs.py (same folder).
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import costs
import schedule_conformal as sc
import schedule_costs as scost
from analyse_result import STEPS_PER_DAY
from conformal_eval import conf_q
from analysis.discarded_cost.schedule_hysteresis import ORACLE_LABEL, hysteresis

DEC = "STID + decision margin"
KEEP = {scost.point("STID"), scost.level("STID"), scost.point("last value"), scost.ORACLE}


def own_args():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--kappas", type=float, nargs="*", default=[0, 3, 6, 12])
    ap.add_argument("--kappa", type=float, default=6, help="kappa of the figure (cell-steps per wake-up)")
    ap.add_argument("--dwell", type=int, nargs="*", default=[1, 3, 6])
    ap.add_argument("--n_bins", type=int, default=10, help="distance-to-theta bins (half on each side of 0)")
    ap.add_argument("--x", choices=["unmet", "viol_all"], default="unmet")
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return own


def decision_margin(cal, ev, theta, eps, n_bins):
    """Margin of every evaluation window/cell from its distance to theta. Returns (margin, (edges, table))."""
    d_c, d_e = theta - cal["pred"], theta - ev["pred"]
    res_c = cal["truth"] - cal["pred"]
    k = max(n_bins // 2, 1)
    inner = np.quantile(np.abs(d_c), np.linspace(0, 1, k + 1)[1:-1])
    inner = np.unique(inner[inner > 0])
    edges = np.concatenate([-inner[::-1], [0.0], inner])
    b_c = np.searchsorted(edges, d_c, side="right")
    b_e = np.searchsorted(edges, d_e, side="right")
    table = np.full(len(edges) + 1, max(conf_q(res_c, eps), 0.0))   # fallback for an empty bin
    for b in range(len(edges) + 1):
        r = res_c[b_c == b]
        if r.size:
            table[b] = max(conf_q(r, eps), 0.0)
    return table[b_e], (edges, table)


def run(cal, ev, args, own):
    truth = ev["truth"]
    steps = ev["day"] * STEPS_PER_DAY + ev["slot"]
    thetas = {q: np.quantile(cal["truth"], q) for q in args.theta_q}
    base = [(n, e, est) for n, e, est in scost.estimates(cal, ev, args) if n in KEEP]
    rows, tables = [], {}
    for q, theta in thetas.items():
        ests = list(base)
        for eps in args.eps_grid:
            m, tab = decision_margin(cal, ev, theta, eps, own.n_bins)
            ests.append((DEC, eps, ev["pred"] + m))
            tables[(q, eps)] = tab
        for name, eps, est in ests:
            for n in own.dwell:
                asleep = hysteresis(est, theta, steps, n)
                c = costs.sleep_cost(truth, asleep, theta)
                wk = costs.wakeups_per_cell_day(asleep, steps)
                for k in own.kappas:
                    rows.append({"theta_q": q, "theta": theta, "approach": name, "eps": eps, "dwell": n,
                                 "kappa": k, **c, "wakeups_day": wk,
                                 "viol_all_%": c["violation_%"] * c["sleep_%"] / 100,
                                 "net_sleep_%": c["sleep_%"] - 100 * k * wk / STEPS_PER_DAY})
    return rows, tables


def curve(sub, approach, xkey):
    pts = sorted((r for r in sub if r["approach"] == approach), key=lambda r: r[xkey])
    return np.array([r[xkey] for r in pts]), np.array([r["net_sleep_%"] for r in pts])


def mean_gain(sub, xkey):
    """Mean of (decision - level) net sleep at equal x, over the x range covered by both curves."""
    x1, y1 = curve(sub, scost.level("STID"), xkey)
    x2, y2 = curve(sub, DEC, xkey)
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return float("nan")
    g = np.linspace(lo, hi, 25)
    return float(np.mean(np.interp(g, x2, y2) - np.interp(g, x1, y1)))


def summary(rows, tables, args, own, xkey):
    for q in args.theta_q:
        edges, tab = tables[(q, args.eps_ref)]
        print(f"\ntheta quantile {q:g}: decision margin table at eps = {args.eps_ref:g} "
              f"(bin edges of d = theta - pred, margin [a.u.])")
        print("  edges :", " ".join(f"{e:8.1f}" for e in edges))
        print("  margin:", " ".join(f"{m:8.1f}" for m in tab))
    for k in own.kappas:
        for n in own.dwell:
            print(f"\nkappa = {k:g}, dwell = {n}, eps = {args.eps_ref:g}")
            for q in args.theta_q:
                sub = [r for r in rows if (r["kappa"], r["dwell"], r["theta_q"]) == (k, n, q)]
                print(f"  theta quantile {q:g}   (mean gain decision - level at equal {xkey}: "
                      f"{mean_gain(sub, xkey):+.2f} net-sleep points)")
                for r in sub:
                    if r["eps"] in (None, args.eps_ref):
                        print(f"    {r['approach']:26s} net {r['net_sleep_%']:6.2f} sleep {r['sleep_%']:6.2f} "
                              f"unmet {r['unmet_%']:5.2f} wake/day {r['wakeups_day']:5.2f}")


def figure(rows, args, own, xkey, xlabel, path):
    nd, nt = len(own.dwell), len(args.theta_q)
    fig, axes = plt.subplots(nd, nt, figsize=(5.5 * nt, 4 * nd), squeeze=False)
    for i, n in enumerate(own.dwell):
        for j, q in enumerate(args.theta_q):
            ax = axes[i, j]
            sub = [r for r in rows if r["kappa"] == own.kappa and r["dwell"] == n and r["theta_q"] == q]
            ax.plot(*curve(sub, scost.level("STID"), xkey), "o-", color="tab:orange", label=scost.level("STID"))
            ax.plot(*curve(sub, DEC, xkey), "o-", color="tab:red", label=DEC)
            ax.plot(*curve(sub, scost.point("STID"), xkey), "*", color="tab:orange", ms=12, label=scost.point("STID"))
            ax.plot(*curve(sub, scost.point("last value"), xkey), "*", color="tab:green", ms=12,
                    label=scost.point("last value"))
            ax.axhline(curve(sub, scost.ORACLE, xkey)[1][0], color="gray", ls=":", label=ORACLE_LABEL)
            ax.set_title(f"theta = {sub[0]['theta']:.0f} a.u. (q {q:g}), dwell n = {n}, kappa = {own.kappa:g}",
                         fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("net sleep [% of cell-steps]")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    sc.prepare(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    own = own_args()
    args = scost.parse_all()
    if own.kappa not in own.kappas:
        own.kappas.append(own.kappa)
    xkey, xlabel = (("unmet_%", "unmet activity [% of total y]") if own.x == "unmet"
                    else ("viol_all_%", "violations [% of ALL cell-steps]"))
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members")
        rows, tables = run(cal, ev, args, own)
        summary(rows, tables, args, own, xkey)
        figure(rows, args, own, xkey, xlabel, os.path.join(args.out_dir, f"decision_conformal_h{h * 10}.png"))
        out += [{"horizon_min": h * 10, **r} for r in rows]
    path = os.path.join(args.out_dir, "decision_conformal.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
