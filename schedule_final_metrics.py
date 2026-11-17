"""
Final metrics (companion of schedule_conformal.py): only the two metrics that matter.

    unmet_%      sum(max(truth - est, 0)) / sum(truth)   traffic NOT covered (under-provisioning)
    overprov_%   sum(max(est - truth, 0)) / sum(truth)   capacity reserved but unused (wasted energy)

Approaches, on the test set and in ORIGINAL units:
    STID point prediction (no confidence)   ensemble mean, the "before"
    constant margin per period              pred + ONE margin per period (no deciles): ablation
    fixed 90 % / fixed 75 %                 conformal by level, same eps all day
    scheduled, calibrated per period        the proposal

Reported for all hours / window hours / other hours.

Saved (prefix = --out_prefix, default "schedule_final_metrics"):
    schedule_final_metrics.png / .csv / .json

Usage (same arguments as schedule_conformal.py, plus --out_prefix):
    python schedule_final_metrics.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
"""
import argparse
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schedule_conformal as sc
from conformal_eval import conf_q, load_split, subset

K_POINT = "STID point prediction (no confidence)"
K_CONST = "constant margin per period (no deciles)"
PERIODS = ("all hours", "window", "other hours")
METRICS = [("unmet_%", "unmet traffic, % of total (lower = safer)"),
           ("overprov_%", "over-provisioning, % of total (lower = less waste)")]
SHORT = {K_POINT: "STID point\n(no confidence)", K_CONST: "const. margin\n(no deciles)",
         sc.K_SCHED: "scheduled\n(proposal)"}


def parse_all():
    """Own option (--out_prefix) + every option of schedule_conformal.py."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out_prefix", default="schedule_final_metrics")
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = sc.parse_args()
    args.out_prefix = extra.out_prefix
    return args


def load(args):
    """Same loading logic as schedule_conformal.main()."""
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
    return cal, ev


def constant_margin_bound(c, ev, args):
    """Ablation: one margin per period (window / other hours), no level deciles."""
    q_day = conf_q(c.res_c[c.day_c[:, 0]].ravel(), args.day_eps)
    q_night = conf_q(c.res_c[~c.day_c[:, 0]].ravel(), args.night_eps)
    return ev["pred"] + np.where(c.day_e, q_day, q_night)


def compute(est, truth):
    d = est - truth
    return {"unmet_%": float(100 * np.maximum(-d, 0).sum() / truth.sum()),
            "overprov_%": float(100 * np.maximum(d, 0).sum() / truth.sum())}


def evaluate(estimates, ev, c):
    day = c.day_e[:, 0]
    masks = {"all hours": np.ones_like(day, dtype=bool), "window": day, "other hours": ~day}
    return {n: {p: compute(est[m], ev["truth"][m]) for p, m in masks.items()} for n, est in estimates.items()}


def print_tables(res, args):
    print(f"\n{'approach':40s} | " + " | ".join(
        f"{(f'{args.day_start:g}-{args.day_end:g} h' if p == 'window' else p):^21s}" for p in PERIODS))
    print(f"{'':40s} | " + " | ".join(f"{'unmet%':>9s} {'overprov%':>11s}" for _ in PERIODS))
    for n, per in res.items():
        tag = "  <- proposal" if n == sc.K_SCHED else ""
        print(f"{n:40s} | " + " | ".join(
            f"{per[p]['unmet_%']:9.2f} {per[p]['overprov_%']:11.2f}" for p in PERIODS) + tag)


def save_tables(res, args):
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "period", "unmet_%", "overprov_%"])
        for n, per in res.items():
            for p, m in per.items():
                w.writerow([n, p, m["unmet_%"], m["overprov_%"]])
    meta = {"horizon_min": (args.horizon + 1) * 10, "window_hours": [args.day_start, args.day_end],
            "eps_window": args.day_eps, "eps_outside": args.night_eps}
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump({"meta": meta, "results": res}, f, indent=2)
    print(f"Saved {args.out_prefix}.csv, {args.out_prefix}.json")


def figure(res, args):
    names = list(res.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"][:len(names)]
    xt = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]
    x = np.arange(len(PERIODS))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (key, title) in zip(axes, METRICS):
        for i, n in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * w, [res[n][p][key] for p in PERIODS], w,
                   color=colors[i], label=SHORT.get(n, n.replace(" (always)", "")))
        ax.set_xticks(x)
        ax.set_xticklabels(xt)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Unmet traffic vs over-provisioning - horizon {(args.horizon + 1) * 10} min - test set",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(f"{args.out_prefix}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {args.out_prefix}.png")


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    cal, ev = load(args)
    print(f"\n{ev['members']} members | horizon {(args.horizon + 1) * 10} min | ORIGINAL units | "
          f"calibration windows {cal['pred'].shape[0]}, evaluation windows {ev['pred'].shape[0]}")

    c = sc.Calib(cal, ev, args)
    bounds, _, keys = sc.build_bounds(c, ev, args)
    estimates = {K_POINT: ev["pred"], K_CONST: constant_margin_bound(c, ev, args),
                 keys[0]: bounds[keys[0]], keys[1]: bounds[keys[1]], sc.K_SCHED: bounds[sc.K_SCHED]}

    res = evaluate(estimates, ev, c)
    print_tables(res, args)
    save_tables(res, args)
    figure(res, args)


if __name__ == "__main__":
    main()
