"""
Final metrics for the scheduled conformal bound (companion of schedule_conformal.py).

Compares, on the test set and in ORIGINAL units, against the TRUE traffic:
    point prediction (ensemble mean)      the "before"
    fixed 90 % bound                      always 90 %
    fixed 75 % bound                      always 75 %
    scheduled, calibrated per period      the proposal
    scheduled, calibrated on all day      same switch, margin from whole-day residuals

Metrics (all hours / window hours / other hours):
    MAE, RMSE, WAPE   closeness to the true traffic (|estimate - truth|)
    bias_%            mean (estimate - truth) / mean truth
    coverage          share of points with truth <= estimate (target: 90 % in window, 75 % outside)
    unmet_%           sum(max(truth - estimate, 0)) / sum(truth): traffic NOT covered (under-provisioning)
    overprov_%        sum(max(estimate - truth, 0)) / sum(truth): capacity reserved but unused (wasted energy)
    pinball           pinball loss at the target quantile (0.90 in window, 0.75 outside), lower = better

Saved (prefix = --out_prefix, default "schedule_final_metrics"):
    schedule_final_metrics.png    bar charts: coverage, unmet traffic, over-provisioning, pinball loss
    schedule_final_metrics.csv    all numbers, one row per (approach, period)
    schedule_final_metrics.json   same, nested

Usage (same arguments as schedule_conformal.py):
    python schedule_final_metrics.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python schedule_final_metrics.py --out_prefix results/final --day_eps 0.10 --night_eps 0.25
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
from conformal_eval import load_split, subset

K_POINT = "point prediction (before)"
PERIODS = ("all hours", "window", "other hours")
METRIC_KEYS = ["MAE", "RMSE", "WAPE_%", "bias_%", "coverage", "unmet_%", "overprov_%", "pinball"]


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


def compute(est, truth, tau):
    """Metrics of one estimate (point prediction or bound) against the truth. tau broadcasts on est."""
    d = est - truth
    q = truth - est
    return {
        "MAE": float(np.abs(d).mean()),
        "RMSE": float(np.sqrt((d ** 2).mean())),
        "WAPE_%": float(100 * np.abs(d).sum() / truth.sum()),
        "bias_%": float(100 * d.mean() / truth.mean()),
        "coverage": float((truth <= est).mean()),
        "unmet_%": float(100 * np.maximum(q, 0).sum() / truth.sum()),
        "overprov_%": float(100 * np.maximum(-q, 0).sum() / truth.sum()),
        "pinball": float(np.maximum(tau * q, (tau - 1) * q).mean()),
    }


def evaluate(estimates, ev, c, args):
    """{approach: {period: metrics}} on all hours, window hours and other hours."""
    day = c.day_e[:, 0]
    tau = np.where(day, 1 - args.day_eps, 1 - args.night_eps)[:, None] * np.ones_like(ev["truth"])
    masks = {"all hours": np.ones_like(day, dtype=bool), "window": day, "other hours": ~day}
    out = {}
    for name, est in estimates.items():
        out[name] = {p: compute(est[m], ev["truth"][m], tau[m]) for p, m in masks.items()}
    return out


def print_table(res, args):
    target = {"all hours": "-", "window": f"{1 - args.day_eps:.0%}", "other hours": f"{1 - args.night_eps:.0%}"}
    for p in PERIODS:
        print(f"\n=== {p} (coverage target: {target[p]}) ===")
        print(f"{'approach':36s} " + " ".join(f"{k:>10s}" for k in METRIC_KEYS))
        for name, per in res.items():
            m = per[p]
            print(f"{name:36s} " + " ".join(f"{m[k]:10.3f}" for k in METRIC_KEYS))


def print_summary(res, keys_sched):
    pt, sch = res[K_POINT], res[sc.K_SCHED]
    print("\n--- proposal vs point prediction ---")
    for p in PERIODS:
        a, b = pt[p], sch[p]
        print(f"{p:12s} coverage {a['coverage']:.3f} -> {b['coverage']:.3f} | "
              f"unmet traffic {a['unmet_%']:.2f}% -> {b['unmet_%']:.2f}% | "
              f"over-provisioning {a['overprov_%']:.2f}% -> {b['overprov_%']:.2f}% | "
              f"pinball {a['pinball']:.3f} -> {b['pinball']:.3f}")
    hi = res[keys_sched[0]]
    print("\n--- proposal vs fixed high-coverage bound ---")
    for p in PERIODS:
        a, b = hi[p], sch[p]
        print(f"{p:12s} over-provisioning {a['overprov_%']:.2f}% -> {b['overprov_%']:.2f}% "
              f"({b['overprov_%'] - a['overprov_%']:+.2f} pp) | coverage {a['coverage']:.3f} -> {b['coverage']:.3f}")
    print("\nNote: the bound is above the prediction by construction, so MAE/WAPE of the bound are NOT expected\n"
          "to beat the point prediction. The gain is in uncovered traffic (unmet_%) and in pinball loss at the\n"
          "target quantile; versus the fixed 90 % bound the gain is lower over-provisioning at the same coverage\n"
          "inside the window.")


def save_tables(res, args):
    rows = []
    for name, per in res.items():
        for p, m in per.items():
            rows.append({"approach": name, "period": p, **m})
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["approach", "period"] + METRIC_KEYS)
        w.writeheader()
        w.writerows(rows)
    meta = {"horizon_min": (args.horizon + 1) * 10, "window_hours": [args.day_start, args.day_end],
            "eps_window": args.day_eps, "eps_outside": args.night_eps}
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump({"meta": meta, "results": res}, f, indent=2)
    print(f"Saved {args.out_prefix}.csv, {args.out_prefix}.json")


def figure(res, args):
    short = {K_POINT: "point (before)"}
    names = list(res.keys())
    for n in names[1:]:
        short[n] = n.replace(", calibrated", "\ncalibrated").replace(" (always)", "")
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:red", "tab:gray"][:len(names)]
    panels = [("coverage", "coverage (black = target)"),
              ("unmet_%", "unmet traffic, % of total (lower = safer)"),
              ("overprov_%", "over-provisioning, % of total (lower = more energy saved)"),
              ("pinball", "pinball loss at target quantile (lower = better)")]
    target = {"all hours": np.nan, "window": 1 - args.day_eps, "other hours": 1 - args.night_eps}
    x = np.arange(len(PERIODS))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, (key, title) in zip(axes, panels):
        for i, n in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * w, [res[n][p][key] for p in PERIODS], w,
                   color=colors[i], label=short[n])
        if key == "coverage":
            for xi, p in zip(x, PERIODS):
                if not np.isnan(target[p]):
                    ax.hlines(target[p], xi - 0.4, xi + 0.4, color="k", ls="--", lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Final metrics vs true traffic - horizon {(args.horizon + 1) * 10} min - test set", fontsize=12)
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
    estimates = {K_POINT: ev["pred"], **bounds}

    res = evaluate(estimates, ev, c, args)
    print_table(res, args)
    print_summary(res, keys)
    save_tables(res, args)
    figure(res, args)


if __name__ == "__main__":
    main()
