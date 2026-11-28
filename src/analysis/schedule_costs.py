"""
Does the conformal upper bound let more cells sleep for the same number of violations?

A cell sleeps when its estimate is below theta (see costs.py). Estimates compared on the test set:
    STID point           ensemble mean, no margin
    constant margin      prediction + ONE conformal margin for all cells (ablation)
    level margin         prediction + conformal margin of the predicted-level decile (proposal)
    oracle               the true y: upper limit of what can be saved

theta is a quantile of y on the calibration split (--theta_q). For each theta, eps is swept, so the
two margin approaches draw a curve (violations -> cells asleep): the higher curve sleeps more for
the same violations. All cells are evaluated; at --eps_ref the results are also split by the
traffic level of the cell (decile of its mean y), to show where the saving comes from.

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

POINT, CONST, LEVEL, ORACLE = "STID point", "constant margin", "level margin (proposal)", "oracle"


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


def estimates(c, ev, eps_grid):
    """(approach, eps, estimate [windows, cells]); eps is None for point and oracle."""
    yield POINT, None, ev["pred"]
    yield ORACLE, None, ev["truth"]
    for eps in eps_grid:
        yield CONST, eps, ev["pred"] + conf_q(c.res_c, eps)
        yield LEVEL, eps, ev["pred"] + c.table(eps)[c.lb_e]


def run(cal, ev, args):
    """Costs of every (theta, approach, eps), overall and per cell-traffic decile at eps_ref."""
    c = sc.Calib(cal, ev, args)
    truth = ev["truth"]
    steps = ev["day"] * STEPS_PER_DAY + ev["slot"]
    thetas = {q: np.quantile(cal["truth"], q) for q in args.theta_q}
    cell_mean = cal["truth"].mean(0)
    level = np.searchsorted(np.quantile(cell_mean, np.linspace(0, 1, N_BINS + 1)[1:-1]), cell_mean, side="right")

    rows, by_level = [], []
    for name, eps, est in estimates(c, ev, args.eps_grid):
        for q, theta in thetas.items():
            asleep = costs.sleep_decision(est, theta)
            key = {"theta_q": q, "theta": theta, "approach": name, "eps": eps}
            rows.append({**key, **costs.sleep_cost(truth, asleep, theta),
                         "wakeups_day": costs.wakeups_per_cell_day(asleep, steps)})
            if eps in (None, args.eps_ref):
                for g in range(N_BINS):
                    cols = level == g
                    by_level.append({**key, "decile": g,
                                     **costs.sleep_cost(truth[:, cols], asleep[:, cols], theta)})
    return rows, by_level


def print_summary(rows, args):
    for q in args.theta_q:
        print(f"\ntheta = quantile {q:g} of y   (eps = {args.eps_ref:g} for the margin approaches)")
        print(f"{'approach':26s} {'sleep%':>7s} {'violation%':>11s} {'unmet%':>7s} {'wakeups/day':>12s}")
        for r in rows:
            if r["theta_q"] == q and r["eps"] in (None, args.eps_ref):
                print(f"{r['approach']:26s} {r['sleep_%']:7.2f} {r['violation_%']:11.2f} "
                      f"{r['unmet_%']:7.2f} {r['wakeups_day']:12.2f}")


def figure_frontier(rows, args, path):
    fig, axes = plt.subplots(1, len(args.theta_q), figsize=(5 * len(args.theta_q), 4), squeeze=False)
    for ax, q in zip(axes[0], args.theta_q):
        sub = [r for r in rows if r["theta_q"] == q]
        for name, style in ((CONST, "o--"), (LEVEL, "o-")):
            pts = sorted((r for r in sub if r["approach"] == name), key=lambda r: r["violation_%"])
            ax.plot([r["violation_%"] for r in pts], [r["sleep_%"] for r in pts], style, label=name)
        point = next(r for r in sub if r["approach"] == POINT)
        oracle = next(r for r in sub if r["approach"] == ORACLE)
        ax.plot(point["violation_%"], point["sleep_%"], "k*", ms=12, label=POINT)
        ax.axhline(oracle["sleep_%"], color="gray", ls=":", label=ORACLE)
        ax.set(title=f"theta = {point['theta']:.0f} a.u. (quantile {q:g})",
               xlabel="violations among sleeping cells [%]", ylabel="cells asleep [%]")
    axes[0, 0].legend()
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
