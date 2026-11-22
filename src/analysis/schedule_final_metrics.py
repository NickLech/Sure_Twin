"""
Final metrics (companion of schedule_conformal.py): unmet activity and over-provisioning,
repeated for several horizons and several high-coverage windows.

    unmet_%      sum(max(truth - est, 0)) / sum(truth)   activity NOT covered (under-provisioning)
    overprov_%   sum(max(est - truth, 0)) / sum(truth)   capacity reserved but unused (wasted energy)

Approaches, on the test set and in ORIGINAL scale [a.u.]:
    STID point prediction (no confidence)   ensemble mean, the "before"
    constant margin per period              pred + ONE margin per period (no deciles): ablation
    fixed 90 % / fixed 75 %                 conformal by level, same eps all day
    scheduled, calibrated per period        the proposal

Every (horizon, window) combination gives one figure, with the metrics for
all hours / window hours / other hours. The model outputs are loaded once per horizon.

Saved in --out_dir (default "final_metrics"):
    schedule_final_metrics_h<min>_w<start>-<end>.png   one per combination (3 horizons x 3 windows = 9)
    schedule_final_metrics_all.csv / .json             every number of every combination

Usage (from the repository root; data/checkpoint options as in schedule_conformal.py):
    python src/analysis/schedule_final_metrics.py \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python src/analysis/schedule_final_metrics.py --horizons 0 --windows 9-18   # a single combination

--horizons are 0-based like --horizon (0 = 10 min, 1 = 20 min, 2 = 30 min). The per-run options of
schedule_conformal.py (--horizon, --day_start, --day_end, --out*) are accepted but overridden here.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schedule_conformal as sc
from conformal_eval import conf_q

K_POINT = "STID point prediction (no confidence)"
K_CONST = "constant margin per period (no deciles)"
PERIODS = ("all hours", "window", "other hours")
METRICS = [("unmet_%", "unmet activity, % of total (lower = safer)"),
           ("overprov_%", "over-provisioning, % of total (lower = less waste)")]
SHORT = {K_POINT: "STID point\n(no confidence)", K_CONST: "const. margin\n(no deciles)",
         sc.K_SCHED: "scheduled\n(proposal)"}


def parse_window(s):
    """'9-18' -> (9.0, 18.0)."""
    a, b = s.split("-")
    return float(a), float(b)


def parse_all():
    """Own options + every option of schedule_conformal.py."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="final_metrics")
    ap.add_argument("--horizons", type=int, nargs="*", default=[0, 1, 2],
                    help="0-based horizons to run (0 = 10 min, 1 = 20 min, 2 = 30 min)")
    ap.add_argument("--windows", nargs="*", default=["8-21", "9-18", "9-19"],
                    help="high-coverage windows as START-END hours")
    extra, rest = ap.parse_known_args()
    args = sc.parse_args(rest)
    args.out_dir, args.horizons = extra.out_dir, extra.horizons
    args.windows = [parse_window(w) for w in extra.windows]
    return args


def constant_margin_bound(c, ev, args):
    """Ablation: one margin per period (window / other hours), no level deciles."""
    q_day = conf_q(c.res_c[c.day_c[:, 0]].ravel(), args.day_eps)
    q_night = conf_q(c.res_c[~c.day_c[:, 0]].ravel(), args.night_eps)
    return ev["pred"] + np.where(c.day_e, q_day, q_night)


def compute(est, truth):
    """Unmet and over-provisioned activity as % of the total true activity."""
    d = est - truth
    return {"unmet_%": float(100 * np.maximum(-d, 0).sum() / truth.sum()),
            "overprov_%": float(100 * np.maximum(d, 0).sum() / truth.sum())}


def evaluate(estimates, ev, c):
    """{approach: {period: metrics}} for all hours / window hours / other hours."""
    day = c.day_e[:, 0]
    masks = {"all hours": np.ones_like(day, dtype=bool), "window": day, "other hours": ~day}
    return {n: {p: compute(est[m], ev["truth"][m]) for p, m in masks.items()} for n, est in estimates.items()}


def print_tables(res, args):
    """Print one table: unmet % and over-provisioning % per approach and period."""
    print(f"\n{'approach':40s} | " + " | ".join(
        f"{(f'{args.day_start:g}-{args.day_end:g} h' if p == 'window' else p):^21s}" for p in PERIODS))
    print(f"{'':40s} | " + " | ".join(f"{'unmet%':>9s} {'overprov%':>11s}" for _ in PERIODS))
    for n, per in res.items():
        tag = "  <- proposal" if n == sc.K_SCHED else ""
        print(f"{n:40s} | " + " | ".join(
            f"{per[p]['unmet_%']:9.2f} {per[p]['overprov_%']:11.2f}" for p in PERIODS) + tag)


def figure(res, args, path):
    """Two bar charts (unmet %, over-provisioning %) per period and approach."""
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
        ax.set_ylabel("% of total true activity")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Unmet activity vs over-provisioning - horizon {(args.horizon + 1) * 10} min - "
                 f"window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other hours "
                 f"({1 - args.night_eps:.0%}) - test set", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def run_one(cal, ev, args):
    """Metrics of every approach for the horizon and window currently set in args."""
    c = sc.Calib(cal, ev, args)
    bounds, _, keys = sc.build_bounds(c, ev, args)
    estimates = {K_POINT: ev["pred"], K_CONST: constant_margin_bound(c, ev, args),
                 keys[0]: bounds[keys[0]], keys[1]: bounds[keys[1]], sc.K_SCHED: bounds[sc.K_SCHED]}
    return evaluate(estimates, ev, c)


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    os.makedirs(args.out_dir, exist_ok=True)
    all_res = []   # one {"horizon_min", "window", "results"} entry per combination

    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {(h + 1) * 10} min | {ev['members']} members | "
              f"calibration windows {cal['pred'].shape[0]}, evaluation windows {ev['pred'].shape[0]}")
        for start, end in args.windows:
            args.day_start, args.day_end = start, end
            print(f"\n--- horizon {(h + 1) * 10} min, window {start:g}-{end:g} h ---")
            try:
                res = run_one(cal, ev, args)
            except SystemExit as e:
                print(f"SKIPPED: {e}")
                continue
            print_tables(res, args)
            figure(res, args, os.path.join(
                args.out_dir, f"schedule_final_metrics_h{(h + 1) * 10}_w{start:g}-{end:g}.png"))
            all_res.append({"horizon_min": (h + 1) * 10, "window": f"{start:g}-{end:g}", "results": res})

    csv_path = os.path.join(args.out_dir, "schedule_final_metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_min", "window", "approach", "period", "unmet_%", "overprov_%"])
        for r in all_res:
            for n, per in r["results"].items():
                for p, m in per.items():
                    w.writerow([r["horizon_min"], r["window"], n, p, m["unmet_%"], m["overprov_%"]])
    json_path = os.path.join(args.out_dir, "schedule_final_metrics_all.json")
    with open(json_path, "w") as f:
        json.dump({"eps_window": args.day_eps, "eps_outside": args.night_eps, "runs": all_res}, f, indent=2)
    print(f"\nSaved {csv_path}, {json_path} ({len(all_res)} combinations)")


if __name__ == "__main__":
    main()