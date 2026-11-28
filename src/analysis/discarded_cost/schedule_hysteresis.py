"""
Fair comparison with a minimum dwell time (hysteresis) applied to EVERY approach.

A cell goes to sleep only after its estimate has been below theta for n consecutive steps
(n = 1 is the plain rule of costs.py) and wakes as soon as the estimate reaches theta. The same rule
is applied to STID, last value and seasonal naive, so a gap in wake-ups cannot come from last value
simply having no smoothing.

    net sleep % = sleep_% - 100 * kappa * wakeups_per_cell_day / STEPS_PER_DAY
    x = unmet_%  (or viol_all_% with --x viol_all)

The "oracle" (y < theta, instantaneous) is NOT an upper bound for kappa > 0: it follows the noise and
switches a lot. It is drawn as a reference only and labelled so.

Saved in --out_dir (default results/costs):
    hysteresis_h<min>.png   rows = dwell n, columns = theta, one kappa (--kappa)
    hysteresis.csv          every (horizon, kappa, dwell, theta, approach, eps)

Usage (from the repository root; other options as in schedule_costs.py):
    python src/analysis/schedule_hysteresis.py --horizons 1 3 --kappas 3 6 12 --kappa 6 --dwell 1 3 6 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
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
from conformal_eval import N_BINS

ORACLE_LABEL = "oracle y<theta (instant; not an upper bound if kappa>0)"


def own_args():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--kappas", type=float, nargs="*", default=[0, 3, 6, 12], help="values in the CSV / table")
    ap.add_argument("--kappa", type=float, default=6, help="kappa of the figure (cell-steps per wake-up)")
    ap.add_argument("--dwell", type=int, nargs="*", default=[1, 3, 6],
                    help="min consecutive steps below theta before sleeping (1 = no hysteresis)")
    ap.add_argument("--x", choices=["unmet", "viol_all"], default="unmet")
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return own


def hysteresis(est, theta, steps, n):
    """asleep[t, c] = est below theta for the last n consecutive steps (gaps in `steps` reset the count)."""
    below = est < theta
    asleep = np.zeros_like(below)
    run = np.zeros(below.shape[1], dtype=int)
    for t in range(len(below)):
        if t > 0 and steps[t] - steps[t - 1] != 1:
            run[:] = 0
        run = np.where(below[t], run + 1, 0)
        asleep[t] = run >= n
    return asleep


def run(cal, ev, args, own):
    truth = ev["truth"]
    steps = ev["day"] * STEPS_PER_DAY + ev["slot"]
    thetas = {q: np.quantile(cal["truth"], q) for q in args.theta_q}
    rows = []
    for name, eps, est in scost.estimates(cal, ev, args):
        if name == scost.CONST:
            continue
        for q, theta in thetas.items():
            for n in own.dwell:
                asleep = hysteresis(est, theta, steps, n)
                c = costs.sleep_cost(truth, asleep, theta)
                wk = costs.wakeups_per_cell_day(asleep, steps)
                for k in own.kappas:
                    rows.append({"theta_q": q, "theta": theta, "approach": name, "eps": eps, "dwell": n,
                                 "kappa": k, **c, "wakeups_day": wk,
                                 "viol_all_%": c["violation_%"] * c["sleep_%"] / 100,
                                 "net_sleep_%": c["sleep_%"] - 100 * k * wk / STEPS_PER_DAY})
    return rows


def summary(rows, args, own):
    for k in own.kappas:
        for n in own.dwell:
            print(f"\nkappa = {k:g}, dwell = {n}, eps = {args.eps_ref:g}")
            for q in args.theta_q:
                print(f"  theta quantile {q:g}")
                for r in rows:
                    if (r["kappa"], r["dwell"], r["theta_q"]) == (k, n, q) and r["eps"] in (None, args.eps_ref):
                        print(f"    {r['approach']:28s} net {r['net_sleep_%']:6.2f} sleep {r['sleep_%']:6.2f} "
                              f"unmet {r['unmet_%']:5.2f} wake/day {r['wakeups_day']:5.2f}")


def curve(sub, approach, xkey):
    pts = sorted((r for r in sub if r["approach"] == approach), key=lambda r: r[xkey])
    return [r[xkey] for r in pts], [r["net_sleep_%"] for r in pts]


def figure(rows, args, own, xkey, xlabel, path):
    nd, nt = len(own.dwell), len(args.theta_q)
    fig, axes = plt.subplots(nd, nt, figsize=(5.5 * nt, 4 * nd), squeeze=False)
    for i, n in enumerate(own.dwell):
        for j, q in enumerate(args.theta_q):
            ax = axes[i, j]
            sub = [r for r in rows if r["kappa"] == own.kappa and r["dwell"] == n and r["theta_q"] == q]
            for name in scost.PREDICTORS:
                col = scost.COLORS[name]
                ax.plot(*curve(sub, scost.level(name), xkey), "o-", color=col, label=scost.level(name))
                ax.plot(*curve(sub, scost.point(name), xkey), "*", color=col, ms=12, label=scost.point(name))
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
        rows = run(cal, ev, args, own)
        summary(rows, args, own)
        figure(rows, args, own, xkey, xlabel, os.path.join(args.out_dir, f"hysteresis_h{h * 10}.png"))
        out += [{"horizon_min": h * 10, **r} for r in rows]
    path = os.path.join(args.out_dir, "hysteresis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
