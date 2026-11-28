"""
Net energy saving vs QoS cost, with a variable wake-up cost kappa.

Companion of schedule_costs.py (same estimates, same sweeps over eps and theta). Instead of
"violations among sleeping cells" (whose denominator changes with the method) it plots:

    x = unmet_%     sum(max(y - theta, 0)) over sleeping cell-steps / sum(y)   (--x unmet, default)
        or viol_all_%  sleeping cell-steps with y >= theta / ALL cell-steps     (--x viol_all)
    y = net sleep % = sleep_% - 100 * kappa * wakeups_per_cell_day / STEPS_PER_DAY

kappa is the cost of one wake-up expressed in cell-steps of sleep (1 step = 10 min). kappa = 0 is
the plain sleep %; kappa = 6 means a wake-up wastes one hour of sleep. Rows = kappa, columns = theta.

Saved in --out_dir (default results/costs):
    net_savings_h<min>.png    one figure per horizon
    net_savings.csv           one row per (horizon, kappa, theta, approach, eps)

Usage (from the repository root; other options as in schedule_costs.py):
    python src/analysis/schedule_net_savings.py --horizons 1 2 3 --kappas 0 3 6 12 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import schedule_conformal as sc
import schedule_costs as scost
from analyse_result import STEPS_PER_DAY


def own_args():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--kappas", type=float, nargs="*", default=[0, 3, 6, 12],
                    help="wake-up cost in cell-steps of sleep (1 step = 10 min)")
    ap.add_argument("--x", choices=["unmet", "viol_all"], default="unmet")
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest          # schedule_costs.parse_all reads the rest
    return own


def add_metrics(r, kappa):
    r = dict(r)
    r["viol_all_%"] = r["violation_%"] * r["sleep_%"] / 100
    r["kappa"] = kappa
    r["net_sleep_%"] = r["sleep_%"] - 100 * kappa * r["wakeups_day"] / STEPS_PER_DAY
    return r


def curve(sub, approach, xkey):
    pts = sorted((r for r in sub if r["approach"] == approach), key=lambda r: r[xkey])
    return [r[xkey] for r in pts], [r["net_sleep_%"] for r in pts]


def figure(rows, args, kappas, xkey, xlabel, path):
    thetas = args.theta_q
    fig, axes = plt.subplots(len(kappas), len(thetas), figsize=(5.5 * len(thetas), 4 * len(kappas)),
                             squeeze=False)
    for i, k in enumerate(kappas):
        for j, q in enumerate(thetas):
            ax = axes[i, j]
            sub = [r for r in rows if r["theta_q"] == q and r["kappa"] == k]
            for name in scost.PREDICTORS:
                col = scost.COLORS[name]
                ax.plot(*curve(sub, scost.level(name), xkey), "o-", color=col, label=scost.level(name))
                ax.plot(*curve(sub, scost.point(name), xkey), "*", color=col, ms=12, label=scost.point(name))
            ax.plot(*curve(sub, scost.CONST, xkey), "o--", color="tab:blue", label=scost.CONST)
            ax.axhline(curve(sub, scost.ORACLE, xkey)[1][0], color="gray", ls=":", label=scost.ORACLE)
            ax.set_title(f"theta = {sub[0]['theta']:.0f} a.u. (q {q:g}), kappa = {k:g} steps", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("net sleep [% of cell-steps]")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    sc.prepare(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    own = own_args()
    args = scost.parse_all()
    xkey, xlabel = (("unmet_%", "unmet activity [% of total y]") if own.x == "unmet"
                    else ("viol_all_%", "violations [% of ALL cell-steps]"))
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members")
        base, _ = scost.run(cal, ev, args)
        rows = [add_metrics(r, k) for k in own.kappas for r in base]
        figure(rows, args, own.kappas, xkey, xlabel, os.path.join(args.out_dir, f"net_savings_h{h * 10}.png"))
        for k in own.kappas:
            print(f"\nkappa = {k:g} steps, eps = {args.eps_ref:g}")
            for q in args.theta_q:
                print(f"  theta quantile {q:g}")
                for r in rows:
                    if r["kappa"] == k and r["theta_q"] == q and r["eps"] in (None, args.eps_ref):
                        print(f"    {r['approach']:28s} net {r['net_sleep_%']:6.2f}  "
                              f"sleep {r['sleep_%']:6.2f}  unmet {r['unmet_%']:5.2f}  "
                              f"wake/day {r['wakeups_day']:5.2f}")
        out += [{"horizon_min": h * 10, **r} for r in rows]
    path = os.path.join(args.out_dir, "net_savings.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
