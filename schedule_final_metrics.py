"""
Final metrics: distance of each approach from the TRUE traffic (companion of schedule_conformal.py).

Approaches, on the test set and in ORIGINAL units:
    STID point prediction (no confidence)   ensemble mean, the "before"
    constant margin per period              pred + ONE margin per period (no deciles): ablation
    fixed 90 % / fixed 75 %                 conformal by level, same eps all day
    scheduled, calibrated per period        the proposal

Metrics (all hours / window hours / other hours):
  Symmetric distance (reported for honesty: a bound sits ABOVE the prediction on purpose,
  so it cannot beat the point prediction here):
    MAE, RMSE, WAPE_%
  How far the estimate falls BELOW the true traffic (the error that hurts):
    unmet_%                  sum(max(truth - est, 0)) / sum(truth)
    shortfall_mean           mean shortfall over the points where truth > est
    shortfall_p95, _p99      95th / 99th percentile of max(truth - est, 0) over all points
  Capacity reserved but unused:
    overprov_%               sum(max(est - truth, 0)) / sum(truth)
  Asymmetric absolute error, "MAE where under-estimating costs w times more":
    cost_w<w>                100 * (w * sum(shortfall) + sum(excess)) / sum(truth)
                             (w = 1 is the WAPE). break-even w = weight above which an approach beats STID.
  Calibration:
    coverage, target, pinball (loss at the target quantile: 0.90 in window, 0.75 outside)

Saved (prefix = --out_prefix, default "schedule_final_metrics"):
    schedule_final_metrics.png / .csv / .json

Usage (same arguments as schedule_conformal.py, plus --out_prefix and --weights):
    python schedule_final_metrics.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python schedule_final_metrics.py --weights 1 2 3 5 10
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
SHORT = {K_POINT: "STID point\n(no confidence)", K_CONST: "const. margin\n(no deciles)",
         sc.K_SCHED: "scheduled\n(proposal)"}


def parse_all():
    """Own options + every option of schedule_conformal.py."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out_prefix", default="schedule_final_metrics")
    ap.add_argument("--weights", type=float, nargs="*", default=[1, 2, 3, 5, 10],
                    help="weights w of the asymmetric error (shortfall counts w times the excess)")
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = sc.parse_args()
    args.out_prefix, args.weights = extra.out_prefix, extra.weights
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


def compute(est, truth, tau, weights):
    """All metrics of one estimate against the truth. tau (target quantile) broadcasts on est."""
    d = est - truth
    short, over = np.maximum(-d, 0), np.maximum(d, 0)
    ts = truth.sum()
    q = -d
    out = {
        "MAE": float(np.abs(d).mean()),
        "RMSE": float(np.sqrt((d ** 2).mean())),
        "WAPE_%": float(100 * np.abs(d).sum() / ts),
        "unmet_%": float(100 * short.sum() / ts),
        "shortfall_mean": float(short[short > 0].mean()) if (short > 0).any() else 0.0,
        "shortfall_p95": float(np.quantile(short, 0.95)),
        "shortfall_p99": float(np.quantile(short, 0.99)),
        "overprov_%": float(100 * over.sum() / ts),
    }
    for w in weights:
        out[f"cost_w{w:g}"] = float(100 * (w * short.sum() + over.sum()) / ts)
    out["coverage"] = float((d >= 0).mean())
    out["target"] = float(np.mean(tau))
    out["pinball"] = float(np.maximum(tau * q, (tau - 1) * q).mean())
    return out


def evaluate(estimates, ev, c, args):
    """{approach: {period: metrics}}."""
    day = c.day_e[:, 0]
    tau = np.where(day, 1 - args.day_eps, 1 - args.night_eps)[:, None] * np.ones_like(ev["truth"])
    masks = {"all hours": np.ones_like(day, dtype=bool), "window": day, "other hours": ~day}
    return {name: {p: compute(est[m], ev["truth"][m], tau[m], args.weights) for p, m in masks.items()}
            for name, est in estimates.items()}


def break_even(ref, other):
    """Weight w above which `other` has lower asymmetric error than `ref` (None if never / always)."""
    d_short = ref["unmet_%"] - other["unmet_%"]       # shortfall removed by `other`
    d_over = other["overprov_%"] - ref["overprov_%"]  # excess added by `other`
    if d_short <= 0:
        return None
    return max(d_over, 0.0) / d_short


def print_tables(res, args):
    cols = ["MAE", "RMSE", "WAPE_%", "unmet_%", "shortfall_mean", "shortfall_p95", "shortfall_p99",
            "overprov_%", "coverage", "pinball"]
    for p in PERIODS:
        print(f"\n=== {p} ===")
        print(f"{'approach':40s} " + " ".join(f"{k:>14s}" for k in cols))
        for n, per in res.items():
            print(f"{n:40s} " + " ".join(f"{per[p][k]:14.3f}" for k in cols))
        wk = [f"cost_w{w:g}" for w in args.weights]
        print(f"\n  asymmetric error (% of traffic), shortfall weighted w times:")
        print(f"  {'approach':40s} " + " ".join(f"{'w=' + format(w, 'g'):>8s}" for w in args.weights))
        for n, per in res.items():
            print(f"  {n:40s} " + " ".join(f"{per[p][k]:8.2f}" for k in wk))


def print_summary(res, args):
    pt = res[K_POINT]
    print("\n=== proposal vs STID point prediction ===")
    for p in PERIODS:
        a, b = pt[p], res[sc.K_SCHED][p]
        be = break_even(a, b)
        be_txt = "never (no shortfall gain)" if be is None else f"w > {be:.2f}"
        print(f"{p:12s} MAE {a['MAE']:.2f} -> {b['MAE']:.2f} (worse, by construction) | "
              f"unmet {a['unmet_%']:.2f}% -> {b['unmet_%']:.2f}% | "
              f"shortfall p95 {a['shortfall_p95']:.1f} -> {b['shortfall_p95']:.1f} | "
              f"p99 {a['shortfall_p99']:.1f} -> {b['shortfall_p99']:.1f}")
        print(f"{'':12s} proposal has lower asymmetric error than STID when {be_txt} "
              f"(shortfall costing more than that many times the excess)")
    print("\n=== proposal vs constant margin (does the decile structure help?) ===")
    for p in PERIODS:
        a, b = res[K_CONST][p], res[sc.K_SCHED][p]
        print(f"{p:12s} overprov {a['overprov_%']:.2f}% -> {b['overprov_%']:.2f}% | "
              f"unmet {a['unmet_%']:.2f}% -> {b['unmet_%']:.2f}% | coverage {a['coverage']:.3f} -> {b['coverage']:.3f} | "
              f"pinball {a['pinball']:.3f} -> {b['pinball']:.3f}")
    print("\nHow to read it: a bound is above the prediction by design, so symmetric MAE/RMSE favour the point\n"
          "prediction. The proposal's advantage is in the shortfall columns, in the asymmetric error when\n"
          "under-estimating costs more than over-estimating, and in the coverage guarantee STID does not have.")


def save_tables(res, args):
    first = next(iter(next(iter(res.values())).values()))
    fields = ["approach", "period"] + list(first.keys())
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, per in res.items():
            for p, m in per.items():
                w.writerow({"approach": n, "period": p, **m})
    be = {p: {n: break_even(res[K_POINT][p], res[n][p]) for n in res if n != K_POINT} for p in PERIODS}
    meta = {"horizon_min": (args.horizon + 1) * 10, "window_hours": [args.day_start, args.day_end],
            "eps_window": args.day_eps, "eps_outside": args.night_eps, "weights": args.weights}
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump({"meta": meta, "results": res, "break_even_weight_vs_stid": be}, f, indent=2)
    print(f"Saved {args.out_prefix}.csv, {args.out_prefix}.json")


def grouped(ax, res, key, names, colors):
    x = np.arange(len(PERIODS))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, [res[n][p][key] for p in PERIODS], w,
               color=colors[i], label=SHORT.get(n, n.replace(" (always)", "")))
    return x


def figure(res, args):
    names = list(res.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"][:len(names)]
    xt = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]
    panels = [("MAE", "MAE vs true traffic\n(symmetric: lower = closer; bounds are above on purpose)"),
              ("shortfall_p95", "shortfall P95: how far BELOW the traffic\nin the bad cases (lower = better)"),
              ("unmet_%", "unmet traffic, % of total (lower = safer)"),
              ("overprov_%", "over-provisioning, % of total (lower = less waste)"),
              ("coverage", "coverage (dashed = target of the period)")]
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for ax, (key, title) in zip(axes.ravel()[:5], panels):
        x = grouped(ax, res, key, names, colors)
        ax.set_xticks(x)
        ax.set_xticklabels(xt)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if key == "coverage":
            for xi, p in zip(x, PERIODS):
                if p != "all hours":
                    ax.hlines(res[names[0]][p]["target"], xi - 0.4, xi + 0.4, color="k", ls="--", lw=1.5)
    axes[0, 0].legend(fontsize=7, loc="upper left")

    ax = axes[1, 2]
    wg = np.linspace(1, max(args.weights + [10]), 100)
    for n, col in zip(names, colors):
        m = res[n]["all hours"]
        ax.plot(wg, wg * m["unmet_%"] + m["overprov_%"], color=col, lw=2, label=SHORT.get(n, n))
    ax.set_xlabel("w = cost of under-estimating / cost of over-estimating")
    ax.set_ylabel("asymmetric error, % of traffic (all hours)")
    ax.set_title("asymmetric error vs w (lower = better)", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Distance from the true traffic - horizon {(args.horizon + 1) * 10} min - test set", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
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

    res = evaluate(estimates, ev, c, args)
    print_tables(res, args)
    print_summary(res, args)
    save_tables(res, args)
    figure(res, args)


if __name__ == "__main__":
    main()
