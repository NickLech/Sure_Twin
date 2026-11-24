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

Weighted cost (third bar chart):  J = lam * unmet + (1 - lam) * overprov,  lam in [0, 1]
    lam = 1 -> only QoS (unmet), lam = 0 -> only energy (over-provisioning).
    --lam        lam inside the window       (default 1 - day_eps,   e.g. 0.90)
    --lam_night  lam outside the window      (default 1 - night_eps, e.g. 0.75)
    Default rationale: a conformal bound with coverage 1 - eps is the (1 - eps)-quantile of the truth,
    which minimises lam * unmet + (1 - lam) * overprov with lam = 1 - eps. Same lam everywhere:
    --lam 0.8 --lam_night 0.8. "all hours" combines the two periods with their share s of the true
    activity: J_all = s * J_window + (1 - s) * J_other (exact).

Every (horizon, window) combination gives three figures, with the metrics for
all hours / window hours / other hours. The model outputs are loaded once per horizon.
    - bars:   three bar charts side by side (unmet %, over-provisioning %, weighted cost J)
    - stacked: ONE bar chart, bar = unmet (solid, bottom) + over-provisioning (hatched, top);
              the total height is the normalised absolute error sum|est - truth| / sum(truth)
    - merged: ONE scatter per period, x = over-provisioning, y = unmet, one marker per approach
              (bottom-left is better; the dashed line is the Pareto front of the approaches)

Saved in --out_dir (default "final_metrics"):
    schedule_final_metrics_h<min>_w<start>-<end>.png          bars, one per combination (3 horizons x 3 windows = 9)
    schedule_final_metrics_h<min>_w<start>-<end>_stacked.png  stacked bars, one per combination
    schedule_final_metrics_h<min>_w<start>-<end>_merged.png   merged scatter, one per combination
    schedule_final_metrics_all.csv / .json                    every number of every combination

Usage (from the repository root; data/checkpoint options as in schedule_conformal.py):
    python src/analysis/schedule_final_metrics.py \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python src/analysis/schedule_final_metrics.py --horizons 1 --windows 9-18   # a single combination
    python src/analysis/schedule_final_metrics.py --horizons 1 --windows 9-19 --lam 0.8 --lam_night 0.8

--horizons are 1-based like --horizon (1 = 10 min, 2 = 20 min, 3 = 30 min). The per-run options of
schedule_conformal.py (--horizon, --day_start, --day_end, --out*) are accepted but overridden here.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
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
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 2, 3],
                    help="1-based horizons to run (1 = 10 min, 2 = 20 min, 3 = 30 min)")
    ap.add_argument("--windows", nargs="*", default=["8-21", "9-18", "9-19"],
                    help="high-coverage windows as START-END hours")
    ap.add_argument("--lam", type=float, default=None,
                    help="weight of unmet (QoS) in J = lam*unmet + (1-lam)*overprov inside the window, in [0, 1] "
                         "(1 = only QoS, 0 = only energy); default 1 - day_eps")
    ap.add_argument("--lam_night", type=float, default=None,
                    help="same, outside the window; default 1 - night_eps")
    extra, rest = ap.parse_known_args()
    args = sc.parse_args(rest)
    args.out_dir, args.horizons = extra.out_dir, extra.horizons
    args.windows = [parse_window(w) for w in extra.windows]
    args.lam = (1 - args.day_eps) if extra.lam is None else extra.lam
    args.lam_night = (1 - args.night_eps) if extra.lam_night is None else extra.lam_night
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
    """{approach: {period: metrics}} for all hours / window hours / other hours.
    Every period dict also carries "share": that period's fraction of the total true activity
    (1 for all hours), used to combine the per-period lambdas in the 'all hours' cost."""
    day = c.day_e[:, 0]
    masks = {"all hours": np.ones_like(day, dtype=bool), "window": day, "other hours": ~day}
    truth = ev["truth"]
    share = {p: float(truth[m].sum() / truth.sum()) for p, m in masks.items()}
    out = {}
    for n, est in estimates.items():
        out[n] = {}
        for p, m in masks.items():
            out[n][p] = compute(est[m], truth[m])
            out[n][p]["share"] = share[p]
    return out


def weighted_cost(res, args):
    """{approach: {period: J}}, J = lam * unmet + (1 - lam) * overprov, with lam = args.lam in the window
    and args.lam_night outside it. 'all hours' = s * J_window + (1 - s) * J_other (s = window share of the
    true activity): exact, because unmet / overprov of 'all hours' are the share-weighted sums of the
    two periods."""
    lam = {"window": args.lam, "other hours": args.lam_night}
    out = {}
    for n, per in res.items():
        j = {p: lam[p] * per[p]["unmet_%"] + (1 - lam[p]) * per[p]["overprov_%"] for p in lam}
        s = per["window"]["share"]
        j["all hours"] = s * j["window"] + (1 - s) * j["other hours"]
        out[n] = j
    return out


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
    """Three bar charts per period and approach: unmet %, over-provisioning %, and their weighted
    average J = lam * unmet + (1 - lam) * overprov (lam = 1: only QoS, lam = 0: only energy)."""
    names = list(res.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"][:len(names)]
    xt = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]
    x = np.arange(len(PERIODS))
    w = 0.8 / len(names)
    cost = weighted_cost(res, args)
    panels = [(lambda n, p, k=k: res[n][p][k], t) for k, t in METRICS]
    panels.append((lambda n, p: cost[n][p],
                   f"weighted cost J = $\\lambda\\cdot$unmet + (1-$\\lambda$)$\\cdot$overprov  (lower = better)\n"
                   f"$\\lambda$={args.lam:.2f} in {args.day_start:g}-{args.day_end:g} h, "
                   f"{args.lam_night:.2f} elsewhere  ($\\lambda$=1: only QoS, 0: only energy)"))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (val, title) in zip(axes, panels):
        for i, n in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * w, [val(n, p) for p in PERIODS], w,
                   color=colors[i], edgecolor="k" if n == sc.K_SCHED else "none", linewidth=1.2,
                   label=SHORT.get(n, n.replace(" (always)", "")))
        ax.set_xticks(x)
        ax.set_xticklabels(xt)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("% of total true activity")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Unmet activity vs over-provisioning - horizon {args.horizon * 10} min - "
                 f"window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other hours "
                 f"({1 - args.night_eps:.0%}) - test set", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_stacked(res, args, path):
    """ONE bar chart: per period and approach a stacked bar, unmet (solid, bottom) + over-provisioning
    (hatched, top). Total height = unmet + overprov = normalised absolute error. Read the solid part
    first (safety), then the total (cost of that safety): a low bar is only good if unmet is low."""
    names = list(res.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"][:len(names)]
    xt = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]
    x = np.arange(len(PERIODS))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for i, n in enumerate(names):
        pos = x + (i - (len(names) - 1) / 2) * w
        u = np.array([res[n][p]["unmet_%"] for p in PERIODS])
        o = np.array([res[n][p]["overprov_%"] for p in PERIODS])
        ax.bar(pos, u, w, color=colors[i], edgecolor="k", linewidth=0.4)
        ax.bar(pos, o, w, bottom=u, facecolor=to_rgba(colors[i], 0.4), edgecolor=colors[i],
               hatch="//", linewidth=0)
        for xp, t in zip(pos, u + o):
            ax.text(xp, t + 0.25, f"{t:.1f}", ha="center", va="bottom", fontsize=7)
    handles = [Patch(facecolor=colors[i], edgecolor="k", linewidth=0.4,
                     label=SHORT.get(n, n.replace(" (always)", "")).replace("\n", " "))
               for i, n in enumerate(names)]
    handles += [Patch(facecolor="gray", edgecolor="k", linewidth=0.4, label="unmet (solid, bottom)"),
                Patch(facecolor=to_rgba("gray", 0.4), edgecolor="gray", hatch="//",
                      label="over-provisioning (hatched, top)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(xt)
    ax.set_ylabel("% of total true activity (bar height = unmet + over-provisioning)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Unmet + over-provisioning (stacked) - horizon {args.horizon * 10} min - "
                 f"window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other hours "
                 f"({1 - args.night_eps:.0%}) - test set", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def pareto_front(pts):
    """Indices (sorted by x) of the non-dominated points; lower x and lower y are both better."""
    front = [i for i, (xi, yi) in enumerate(pts)
             if not any(xj <= xi and yj <= yi and (xj < xi or yj < yi)
                        for j, (xj, yj) in enumerate(pts) if j != i)]
    return sorted(front, key=lambda i: pts[i][0])


def figure_merged(res, args, path):
    """ONE figure: per period a scatter (x = over-provisioning, y = unmet), one marker per approach.
    Bottom-left is better; the dashed line joins the non-dominated approaches (Pareto front)."""
    names = list(res.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"][:len(names)]
    markers = ["o", "s", "^", "D", "*"][:len(names)]
    titles = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, p, t in zip(axes, PERIODS, titles):
        pts = [(res[n][p]["overprov_%"], res[n][p]["unmet_%"]) for n in names]
        front = pareto_front(pts)
        ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], "--", color="gray", lw=1, zorder=1)
        for i, n in enumerate(names):
            prop = n == sc.K_SCHED
            ax.scatter(*pts[i], s=300 if prop else 110, c=colors[i], marker=markers[i],
                       edgecolors="k", linewidths=0.8, zorder=3,
                       label=SHORT.get(n, n.replace(" (always)", "")).replace("\n", " "))
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("over-provisioning, % of total (lower = less waste)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("unmet traffic, % of total (lower = safer)")
    axes[-1].annotate("better", xy=(0.62, 0.62), xytext=(0.93, 0.93), xycoords="axes fraction",
                      textcoords="axes fraction", ha="right", va="top", fontsize=9, color="gray",
                      arrowprops=dict(arrowstyle="->", color="gray"))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(names), fontsize=9, frameon=False)
    fig.suptitle(f"Unmet traffic vs over-provisioning - horizon {args.horizon * 10} min - "
                 f"window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other hours "
                 f"({1 - args.night_eps:.0%}) - test set", fontsize=12)
    plt.tight_layout(rect=(0, 0.07, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def print_cost(res, args):
    """Print the weighted cost J = lam * unmet + (1 - lam) * overprov per approach and period."""
    cost = weighted_cost(res, args)
    print(f"\nweighted cost J = lam*unmet + (1-lam)*overprov  (lam = {args.lam:.2f} in window, "
          f"{args.lam_night:.2f} outside)")
    for n, per in cost.items():
        tag = "  <- proposal" if n == sc.K_SCHED else ""
        print(f"{n:40s} | " + " | ".join(f"{per[p]:9.2f}" for p in PERIODS) + tag)


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
    if not (0 <= args.lam <= 1 and 0 <= args.lam_night <= 1):
        raise SystemExit("--lam and --lam_night must be in [0, 1]")
    os.makedirs(args.out_dir, exist_ok=True)
    all_res = []   # one {"horizon_min", "window", "results"} entry per combination

    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | "
              f"calibration windows {cal['pred'].shape[0]}, evaluation windows {ev['pred'].shape[0]}")
        for start, end in args.windows:
            args.day_start, args.day_end = start, end
            print(f"\n--- horizon {h * 10} min, window {start:g}-{end:g} h ---")
            try:
                res = run_one(cal, ev, args)
            except SystemExit as e:
                print(f"SKIPPED: {e}")
                continue
            print_tables(res, args)
            print_cost(res, args)
            base = os.path.join(args.out_dir, f"schedule_final_metrics_h{h * 10}_w{start:g}-{end:g}")
            figure(res, args, base + ".png")
            figure_stacked(res, args, base + "_stacked.png")
            figure_merged(res, args, base + "_merged.png")
            all_res.append({"horizon_min": h * 10, "window": f"{start:g}-{end:g}", "results": res,
                            "cost": weighted_cost(res, args)})

    csv_path = os.path.join(args.out_dir, "schedule_final_metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_min", "window", "approach", "period", "unmet_%", "overprov_%", "cost_J"])
        for r in all_res:
            for n, per in r["results"].items():
                for p, m in per.items():
                    w.writerow([r["horizon_min"], r["window"], n, p, m["unmet_%"], m["overprov_%"],
                                r["cost"][n][p]])
    json_path = os.path.join(args.out_dir, "schedule_final_metrics_all.json")
    with open(json_path, "w") as f:
        json.dump({"eps_window": args.day_eps, "eps_outside": args.night_eps,
                   "lam_window": args.lam, "lam_outside": args.lam_night, "runs": all_res}, f, indent=2)
    print(f"\nSaved {csv_path}, {json_path} ({len(all_res)} combinations)")


if __name__ == "__main__":
    main()
