"""
Does the conformal upper bound let more cells sleep for the same number of violations, and does a
more accurate forecast sleep more?

A cell sleeps when its estimate is below theta (see costs.py). Estimates compared on the test set:
    <predictor> point            forecast alone, no margin
    <predictor> + level margin   forecast + conformal margin of the predicted-level decile (eps)
    STID + constant margin       forecast + ONE conformal margin for all cells (ablation)
    oracle                       the true y: upper limit of what can be saved
with predictor = STID (ensemble mean), last value, seasonal naive (same slot 24 h earlier). Each
predictor is calibrated on its own errors.

theta is a quantile of y on the calibration split (--theta_q). For each theta, eps is swept, so each
margin approach draws a curve (violations -> cells asleep): the higher curve sleeps more for the same
violations. All cells are evaluated; at --eps_ref the results are also split by the traffic level of
the cell (decile of its mean y), to show where the saving comes from.

Saved in --out_dir (default results/costs):
    sleep_costs.csv            one row per (horizon, theta, approach, eps)
    sleep_costs_by_level.csv   the same at eps_ref, per cell-traffic decile (0 = quietest)
    sleep_frontier_h<min>.png  violations vs cells asleep, one panel per theta

Usage (from the repository root; data/checkpoint options as in schedule_conformal.py):
    python src/analysis/schedule_costs.py --horizons 1 2 3 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import costs
import schedule_conformal as sc
from analyse_result import STEPS_PER_DAY
from conformal_eval import N_BINS, conf_q

PREDICTORS = {"STID": "pred", "last value": "last", "seasonal naive": "seasonal"}   # name -> key in split
COLORS = {"STID": "tab:orange", "last value": "tab:green", "seasonal naive": "tab:purple"}
CONST, ORACLE = "STID + constant margin", "oracle"


def point(name):
    return f"{name} point"


def level(name):
    return f"{name} + level margin"


def parse_all():
    """Own options + every option of schedule_conformal.py."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="results/costs")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1], help="1-based (1 = 10 min)")
    ap.add_argument("--theta_q", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                    help="sleep thresholds as quantiles of y on the calibration split")
    ap.add_argument("--eps_grid", type=float, nargs="*", default=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    ap.add_argument("--eps_ref", type=float, default=0.10, help="eps of the summary table (must be in --eps_grid)")
    own, rest = ap.parse_known_args()
    args = sc.parse_args(rest)
    vars(args).update(vars(own))
    if args.eps_ref not in args.eps_grid:
        raise SystemExit("--eps_ref must be one of --eps_grid")
    return args


def estimates(cal, ev, args):
    """(approach, eps, estimate [windows, cells]); eps is None for point forecasts and oracle."""
    yield ORACLE, None, ev["truth"]
    for name, key in PREDICTORS.items():
        c = sc.Calib({**cal, "pred": cal[key]}, {**ev, "pred": ev[key]}, args)
        yield point(name), None, ev[key]
        for eps in args.eps_grid:
            yield level(name), eps, ev[key] + c.table(eps)[c.lb_e]
            if name == "STID":
                yield CONST, eps, ev[key] + conf_q(c.res_c, eps)


def run(cal, ev, args):
    """Costs of every (theta, approach, eps), overall and per cell-traffic decile at eps_ref."""
    truth = ev["truth"]
    steps = ev["day"] * STEPS_PER_DAY + ev["slot"]
    thetas = {q: np.quantile(cal["truth"], q) for q in args.theta_q}
    cell_mean = cal["truth"].mean(0)
    decile = np.searchsorted(np.quantile(cell_mean, np.linspace(0, 1, N_BINS + 1)[1:-1]), cell_mean, side="right")

    rows, by_level = [], []
    for name, eps, est in estimates(cal, ev, args):
        for q, theta in thetas.items():
            asleep = costs.sleep_decision(est, theta)
            key = {"theta_q": q, "theta": theta, "approach": name, "eps": eps}
            rows.append({**key, **costs.sleep_cost(truth, asleep, theta),
                         "wakeups_day": costs.wakeups_per_cell_day(asleep, steps)})
            if eps in (None, args.eps_ref):
                for g in range(N_BINS):
                    cols = decile == g
                    by_level.append({**key, "decile": g,
                                     **costs.sleep_cost(truth[:, cols], asleep[:, cols], theta)})
    return rows, by_level


def print_summary(rows, args):
    for q in args.theta_q:
        print(f"\ntheta = quantile {q:g} of y   (eps = {args.eps_ref:g} for the margin approaches)")
        print(f"{'approach':30s} {'sleep%':>7s} {'violation%':>11s} {'unmet%':>7s} {'wakeups/day':>12s}")
        for r in rows:
            if r["theta_q"] == q and r["eps"] in (None, args.eps_ref):
                print(f"{r['approach']:30s} {r['sleep_%']:7.2f} {r['violation_%']:11.2f} "
                      f"{r['unmet_%']:7.2f} {r['wakeups_day']:12.2f}")


def curve(sub, approach):
    """(violation %, sleep %) of an approach, sorted by violation."""
    pts = sorted((r for r in sub if r["approach"] == approach), key=lambda r: r["violation_%"])
    return [r["violation_%"] for r in pts], [r["sleep_%"] for r in pts]


def figure_frontier(rows, args, path):
    fig, axes = plt.subplots(1, len(args.theta_q), figsize=(5.5 * len(args.theta_q), 4.5), squeeze=False)
    for ax, q in zip(axes[0], args.theta_q):
        sub = [r for r in rows if r["theta_q"] == q]
        for name in PREDICTORS:
            ax.plot(*curve(sub, level(name)), "o-", color=COLORS[name], label=level(name))
            ax.plot(*curve(sub, point(name)), "*", color=COLORS[name], ms=12, label=point(name))
        ax.plot(*curve(sub, CONST), "o--", color="tab:blue", label=CONST)
        ax.axhline(curve(sub, ORACLE)[1][0], color="gray", ls=":", label=ORACLE)
        ax.set(title=f"theta = {sub[0]['theta']:.0f} a.u. (quantile {q:g})",
               xlabel="violations among sleeping cells [%]", ylabel="cells asleep [%]")
    axes[0, 0].legend(fontsize=7)
    sc.prepare(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def main():
    args = parse_all()
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows, all_levels = [], []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | "
              f"evaluation windows {ev['pred'].shape[0]}")
        rows, by_level = run(cal, ev, args)
        print_summary(rows, args)
        figure_frontier(rows, args, os.path.join(args.out_dir, f"sleep_frontier_h{h * 10}.png"))
        all_rows += [{"horizon_min": h * 10, **r} for r in rows]
        all_levels += [{"horizon_min": h * 10, **r} for r in by_level]
    save_csv(all_rows, os.path.join(args.out_dir, "sleep_costs.csv"))
    save_csv(all_levels, os.path.join(args.out_dir, "sleep_costs_by_level.csv"))


if __name__ == "__main__":
    main()