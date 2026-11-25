"""
Adaptive traffic-conditioned conformal schedule + filtered growth-ramp penalty (extension of
schedule_conformal.py), evaluated on ALL cells against the 10-decile schedule and an iso-overprovisioning baseline.

    adaptive conformal : Mondrian margins per period x --n_bins quantile bins of the predicted level
    adaptive + ramp    : adaptive bound + lambda * w(bin) * max(0, pred[t] - pred[t-1]), w(bin) = normalised
                         underestimation rate of the bin; lambda tuned on a chronological val split of the
                         calibration data minimising J = cost_w * unmet + (1 - cost_w) * overprov
    iso-overprov       : the 10-decile schedule with its margin scaled by ONE constant alpha, matched on that val
                         split to the over-provisioning of adaptive + ramp (isolates allocation from budget)

Output (--out_dir, default final_metrics/adaptive_ramp): ONE csv, adaptive_ramp_metrics.csv, with the schema of
metrics_common.py (scopes global / tier; switch_% = ON/OFF transitions % of cell-steps with hysteresis
[off_thr, on_thr]), plus one summary figure per horizon. Units: unmet / over-provisioning are % of the total TRUE
activity of the group (activity index [a.u.], NOT energy); cover is a fraction (0-1); margins in [a.u.].

Usage (from the repository root; --horizons are 1-based, 1 = 10 min):
    python src/analysis/schedule_adaptive_ramp.py --horizons 1 3 6 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
The per-run options of schedule_conformal.py are accepted (--day_eps, --night_eps, --cal_data, ...); window 9-18 h.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metrics_common as mc
import schedule_conformal as sc
from analyse_result import kept_columns, load_series
from conformal_eval import conf_q

METHOD = "adaptive_ramp"
K_ISO = "iso-overprov baseline"
K_ADAPT = "adaptive conformal (no ramp)"
K_RAMP = "adaptive + ramp"
APPROACHES = (mc.K_POINT, mc.K_REF, K_ISO, K_ADAPT, K_RAMP)
MIN_GROUP = 100
COLORS = {mc.K_POINT: "tab:blue", mc.K_REF: "tab:purple", K_ISO: "tab:gray", K_ADAPT: "tab:orange",
          K_RAMP: "tab:green"}
MARKERS = {mc.K_POINT: "o", mc.K_REF: "D", K_ISO: "X", K_ADAPT: "s", K_RAMP: "*"}
LABELS = {mc.K_POINT: "STID point (no margin)", mc.K_REF: "Scheduled 10-deciles (orig)"}


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def parse_all(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="final_metrics/adaptive_ramp")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1])
    ap.add_argument("--n_bins", type=int, default=5,
                    help="number of predicted-traffic quantile bins for adaptive margins")
    ap.add_argument("--val_frac", type=float, default=0.3,
                    help="fraction of calibration data held out for tuning lambda (prevents leakage)")
    ap.add_argument("--ramp_lambda", default="auto",
                    help="'auto' = tuned on val split via weighted cost J, or a fixed float >= 0")
    ap.add_argument("--cost_w", type=float, default=0.80,
                    help="weight of unmet%% in J = cost_w * unmet%% + (1 - cost_w) * overprov%% for tuning lambda")
    ap.add_argument("--on_thr", type=float, default=None,
                    help="activation threshold [a.u.] for the ON/OFF switching metric (default: 30th pct of train)")
    ap.add_argument("--off_thr", type=float, default=None,
                    help="deactivation threshold [a.u.] for hysteresis (default: 0.7 * on_thr)")
    mc.add_args(ap)
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.ramp_lambda != "auto":
        args.ramp_lambda = float(args.ramp_lambda)
    mc.set_lambdas(args)
    return args


# ----------------------------------------------------------------------
# Data splitting and adaptive conformal calibration
# ----------------------------------------------------------------------
def split_cal_val(cal, val_frac):
    """Chronological split of the calibration set into pure calibration and validation."""
    n = cal["pred"].shape[0]
    cut = int(n * (1.0 - val_frac))
    if cut < 50 or (n - cut) < 20:
        return cal, cal
    c_sub, v_sub = {}, {}
    for k, v in cal.items():
        if isinstance(v, np.ndarray) and v.shape[0] == n:
            c_sub[k], v_sub[k] = v[:cut], v[cut:]
        else:
            c_sub[k] = v_sub[k] = v
    return c_sub, v_sub


class AdaptiveCalib:
    """Traffic-conditioned conformal margins + empirical bin underestimation weights."""

    def __init__(self, cal, args):
        self.args = args
        self.day_c = sc.in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        self.res_c = cal["truth"] - cal["pred"]
        self.n_bins = args.n_bins
        while True:
            self.edges = np.quantile(cal["pred"], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.lb_c = np.searchsorted(self.edges, cal["pred"], side="right")
            small = min(((self.lb_c == b) & (self.day_c == p)).sum()
                        for b in range(self.n_bins) for p in (True, False))
            if small >= MIN_GROUP or self.n_bins == 1:
                break
            self.n_bins -= 1
        self.q_day = self._table(args.day_eps, True)
        self.q_night = self._table(args.night_eps, False)
        self.bin_underest_weight = self._underest_weights(cal)

    def _table(self, eps, period):
        margins = []
        for b in range(self.n_bins):
            sel = (self.lb_c == b) & (self.day_c == period)
            q = float(conf_q(self.res_c[sel], eps)) if sel.any() else 0.0
            margins.append(max(0.0, q))
        return np.array(margins)

    def _underest_weights(self, cal):
        """Normalised underestimation tendency per bin (scales the ramp penalty where STID underestimates most)."""
        rates = []
        for b in range(self.n_bins):
            sel = self.lb_c == b
            if sel.any() and cal["truth"][sel].sum() > 0:
                rates.append(np.maximum(0.0, self.res_c[sel]).sum() / cal["truth"][sel].sum())
            else:
                rates.append(0.0)
        rates = np.array(rates)
        mean_r = rates.mean()
        return rates / mean_r if mean_r > 1e-8 else np.ones_like(rates)

    def predict(self, data):
        day_d = sc.in_window(data["slot"], self.args.day_start, self.args.day_end)[:, None]
        lb_d = np.searchsorted(self.edges, data["pred"], side="right")
        margin = np.where(day_d, self.q_day[lb_d], self.q_night[lb_d])
        return data["pred"] + margin, margin, lb_d, self.bin_underest_weight[lb_d]


def apply_ramp_penalty(pred, base_bound, lam, bin_weights=None):
    """Add the filtered positive-derivative ramp penalty: lambda * w(bin) * max(0, pred[t] - pred[t-1])."""
    if lam <= 0.0:
        return base_bound.copy()
    delta = np.zeros_like(pred)
    delta[1:] = np.maximum(0.0, pred[1:] - pred[:-1])
    if bin_weights is not None:
        delta = delta * bin_weights
    return base_bound + lam * delta


def tune_lambda(calib_model, val_data, args):
    """Select lambda on the disjoint validation split minimising J = cost_w * unmet + (1 - cost_w) * overprov."""
    base_val, _, _, w_val = calib_model.predict(val_data)
    best_lam, best_cost = 0.0, float("inf")
    for lam in np.linspace(0.0, 1.0, 41):
        m = mc.compute(apply_ramp_penalty(val_data["pred"], base_val, lam, w_val), val_data["truth"])
        cost = args.cost_w * m["unmet_%"] + (1.0 - args.cost_w) * m["overprov_%"]
        if cost < best_cost - 1e-5:
            best_cost, best_lam = cost, float(lam)
    return best_lam


def match_iso_overprov(pred, orig_bound, target_overprov, truth):
    """Constant alpha such that pred + alpha * (orig_bound - pred) has over-provisioning target_overprov (on val)."""
    margin = np.maximum(0.0, orig_bound - pred)
    lo, hi = 0.0, 5.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        if mc.compute(pred + mid * margin, truth)["overprov_%"] < target_overprov:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# ON/OFF switching with hysteresis (extra metric of this method)
# ----------------------------------------------------------------------
def switching_transitions(est, on_thr, off_thr):
    """[T, C] bool: ON/OFF state change at step t of a cell, with hysteresis [off_thr, on_thr]."""
    state = est[0] >= on_thr
    tr = np.zeros(est.shape, dtype=bool)
    for t in range(1, est.shape[0]):
        nxt = np.where(state, est[t] >= off_thr, est[t] >= on_thr)
        tr[t] = nxt != state
        state = nxt
    return tr


def switch_rate(tr, mask):
    """% of cell-steps (inside the mask) with an ON/OFF transition."""
    if tr.shape[0] < 2:
        return 0.0
    valid = mask[1:]
    return float(100.0 * tr[1:][valid].mean()) if valid.any() else float("nan")


# ----------------------------------------------------------------------
# Evaluation and report
# ----------------------------------------------------------------------
def evaluate(ev, calib_model, est, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    trans = {n: switching_transitions(e, args.on_thr_used, args.off_thr_used) for n, e in est.items()}
    scopes = {"global": {"all cells": None}, "tier": mc.tier_groups(tier, tier_names, truth.shape)}
    res = mc.evaluate_scopes(est, truth, mc.day_mask(ev, args), args, scopes,
                             extra=lambda n, m: {"switch_%": switch_rate(trans[n], m)})
    info = {"tier": mc.tier_info(tier, tier_mean, tier_names, truth)}

    # correction profile by predicted traffic bin (all hours)
    _, margin_ev, lb_ev, _ = calib_model.predict(ev)
    up_ref, up_ramp = est[mc.K_REF], est[K_RAMP]
    bins = {}
    for b in range(calib_model.n_bins):
        m = lb_ev == b
        if not m.any():
            continue
        t, p = truth[m], ev["pred"][m]
        bins[f"bin_{b + 1}"] = {
            "pred_mean_au": float(p.mean()), "truth_mean_au": float(t.mean()),
            "underest_point_au": float(np.maximum(0.0, t - p).mean()),
            "orig_margin_au": float((up_ref[m] - p).mean()),
            "adaptive_margin_au": float(margin_ev[m].mean()),
            "final_margin_au": float((up_ramp[m] - p).mean()),
            "unmet_point_%": float(100.0 * np.maximum(0.0, t - p).sum() / t.sum()),
            "unmet_orig_%": float(100.0 * np.maximum(0.0, t - up_ref[m]).sum() / t.sum()),
            "unmet_final_%": float(100.0 * np.maximum(0.0, t - up_ramp[m]).sum() / t.sum())}
    return res, info, bins


def fmt(m):
    return (f"{m['cover']:6.3f} {m['unmet_%']:8.2f} {m['overprov_%']:8.2f} {m['cost_J']:8.2f} "
            f"{m['switch_%']:7.2f}")


def print_report(res, bins, lam, iso_alpha, args, calib_model):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    glob = res["global"]["all cells"]
    print(f"\nAdaptive bins: {calib_model.n_bins} | ramp lambda = {lam:.3f} (cost_w={args.cost_w:.2f}) | "
          f"iso-overprov scale alpha = {iso_alpha:.3f} | hysteresis = [{args.off_thr_used:.1f}, "
          f"{args.on_thr_used:.1f}] a.u.")

    print("\n== 1) GLOBAL, ALL CELLS (cover, unmet %, over-prov. %, cost J, ON/OFF switch %) ==")
    print(f"{'approach':40s} | " + " | ".join(f"{(w if p == 'window' else p):^41s}" for p in mc.PERIODS))
    print(f"{'':40s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'over%':>8s} {'J':>8s} {'sw%':>7s}"
                                      for _ in mc.PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_RAMP else ("  <- fair ref" if n == K_ISO else "")
        print(f"{n:40s} | " + " | ".join(fmt(glob[n][p]) for p in mc.PERIODS) + tag)

    g, io, ref = glob[K_RAMP]["all hours"], glob[K_ISO]["all hours"], glob[mc.K_REF]["all hours"]
    print("\n[Efficiency Check - All Hours]")
    print(f"  vs scheduled 10-deciles (orig): unmet {g['unmet_%'] - ref['unmet_%']:+.2f} pp | "
          f"overprov {g['overprov_%'] - ref['overprov_%']:+.2f} pp | "
          f"switch {g['switch_%'] - ref['switch_%']:+.3f} pp")
    print(f"  vs iso-overprov baseline      : unmet {g['unmet_%'] - io['unmet_%']:+.2f} pp "
          f"at equal over-provisioning (~{g['overprov_%']:.2f}%)")

    print("\n== 2) CORRECTION PROFILE BY PREDICTED TRAFFIC BIN (all hours) ==")
    print(f"{'bin':7s} {'pred_avg':>9s} {'underest':>9s} {'orig_m':>9s} {'new_m':>9s} "
          f"{'unmet_pt%':>10s} {'unmet_orig%':>12s} {'unmet_new%':>11s}")
    for name, b in bins.items():
        print(f"{name:7s} {b['pred_mean_au']:9.1f} {b['underest_point_au']:9.1f} "
              f"{b['orig_margin_au']:9.1f} {b['final_margin_au']:9.1f} "
              f"{b['unmet_point_%']:10.2f} {b['unmet_orig_%']:12.2f} {b['unmet_final_%']:11.2f}")


# ----------------------------------------------------------------------
# Summary figure (3 panels, with Pareto front)
# ----------------------------------------------------------------------
def pareto_front(pts):
    front = [i for i, (xi, yi) in enumerate(pts)
             if not any(xj <= xi and yj <= yi and (xj < xi or yj < yi)
                        for j, (xj, yj) in enumerate(pts) if j != i)]
    return sorted(front, key=lambda i: pts[i][0])


def figure_summary(res, info, bins, args, path):
    glob = res["global"]["all cells"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    # 1. margin allocation by predicted-traffic bin
    ax = axes[0]
    xb = np.arange(len(bins))
    ax.bar(xb - 0.18, [b["underest_point_au"] for b in bins.values()], 0.35, color="tab:red", alpha=0.75,
           label="STID mean underestimation")
    ax.plot(xb + 0.18, [b["orig_margin_au"] for b in bins.values()], "D--", color=COLORS[mc.K_REF],
            label=LABELS[mc.K_REF])
    ax.plot(xb + 0.18, [b["final_margin_au"] for b in bins.values()], "*-", color=COLORS[K_RAMP], markersize=9,
            linewidth=2, label="Adaptive + ramp (new)")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"Bin {i + 1}\n({b['pred_mean_au']:.0f})" for i, b in enumerate(bins.values())], fontsize=8)
    ax.set_xlabel("Predicted traffic bin (mean pred [a.u.])")
    ax.set_ylabel("Activity [a.u.]")
    ax.set_title("1. Margin Allocation vs Underestimation", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 2. trade-off (bottom-left is better)
    ax = axes[1]
    pts = [(glob[n]["all hours"]["overprov_%"], glob[n]["all hours"]["unmet_%"]) for n in APPROACHES]
    front = pareto_front(pts)
    ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], "--", color="gray", lw=1.2,
            label="Pareto frontier", zorder=1)
    for i, n in enumerate(APPROACHES):
        ax.scatter(*pts[i], s=240 if n == K_RAMP else 110, c=COLORS[n], marker=MARKERS[n],
                   edgecolors="k", linewidths=0.8, zorder=3, label=LABELS.get(n, n))
    ax.set_xlabel("Over-provisioning [% of total activity] (lower = less waste)")
    ax.set_ylabel("Unmet activity [% of total activity] (lower = safer)")
    ax.set_title("2. Efficiency Frontier (All Cells, All Hours)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    # 3. unmet per static tier
    ax = axes[2]
    tnames = list(info["tier"].keys())
    xt = np.arange(len(tnames))
    keys = (mc.K_REF, K_ISO, K_RAMP)
    w = 0.26
    for i, n in enumerate(keys):
        ax.bar(xt + (i - 1) * w, [res["tier"][t][n]["all hours"]["unmet_%"] for t in tnames], w,
               color=COLORS[n], label=LABELS.get(n, n))
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t.replace('cells ', '')}\n({info['tier'][t]['share_activity_%']:.0f}% act.)"
                        for t in tnames], fontsize=8)
    ax.set_xlabel("Static cell tier (share of total traffic)")
    ax.set_ylabel("Unmet activity [%]")
    ax.set_title("3. Unmet Activity by Tier (vs Fair Baselines)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(f"Efficiency Comparison - Horizon {args.horizon * 10} min | "
                 f"Window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}/{1 - args.night_eps:.0%}) | "
                 f"Bins={args.bins_used}, lambda={args.lambda_used:.2f}", fontsize=11.5)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run_one(cal, ev, args, cols):
    tier, tier_mean, tier_names, train = mc.static_tiers(args, cols)
    args.on_thr_used = args.on_thr if args.on_thr is not None else float(np.quantile(train, 0.30))
    args.off_thr_used = args.off_thr if args.off_thr is not None else 0.70 * args.on_thr_used

    # reference: 10-decile schedule of schedule_conformal.py
    bounds, _, _ = sc.build_bounds(sc.Calib(cal, ev, args), ev, args)
    up_ref = bounds[sc.K_SCHED]

    # lambda tuned on a disjoint validation split of the calibration data
    cal_pure, val_pure = split_cal_val(cal, args.val_frac)
    calib_model = AdaptiveCalib(cal_pure, args)
    args.bins_used = calib_model.n_bins
    lam = tune_lambda(calib_model, val_pure, args) if args.ramp_lambda == "auto" else float(args.ramp_lambda)
    args.lambda_used = lam

    # margins re-fitted on the full calibration set once lambda is fixed
    full_calib = AdaptiveCalib(cal, args)
    up_adapt, _, _, w_ev = full_calib.predict(ev)
    up_ramp = apply_ramp_penalty(ev["pred"], up_adapt, lam, w_ev)

    # iso-overprov baseline: alpha matched on the validation split
    b_ref_val, _, _ = sc.build_bounds(sc.Calib(cal_pure, val_pure, args), val_pure, args)
    up_adapt_val, _, _, w_val = calib_model.predict(val_pure)
    up_ramp_val = apply_ramp_penalty(val_pure["pred"], up_adapt_val, lam, w_val)
    target = mc.compute(up_ramp_val, val_pure["truth"])["overprov_%"]
    iso_alpha = match_iso_overprov(val_pure["pred"], b_ref_val[sc.K_SCHED], target, val_pure["truth"])
    up_iso = ev["pred"] + iso_alpha * np.maximum(0.0, up_ref - ev["pred"])

    est = {mc.K_POINT: ev["pred"], mc.K_REF: up_ref, K_ISO: up_iso, K_ADAPT: up_adapt, K_RAMP: up_ramp}
    res, info, bins = evaluate(ev, full_calib, est, args, tier, tier_mean, tier_names)
    print_report(res, bins, lam, iso_alpha, args, full_calib)

    h, window = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}"
    figure_summary(res, info, bins, args, os.path.join(args.out_dir, f"adaptive_h{h}_w{window}_summary.png"))
    params = {K_RAMP: f"lambda={lam:.3f};bins={calib_model.n_bins};cost_w={args.cost_w}",
              K_ADAPT: f"bins={calib_model.n_bins}", K_ISO: f"alpha={iso_alpha:.3f}"}
    return mc.to_rows(METHOD, h, window, res, params)


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
        print(f"\n##### Horizon {h * 10} min | cells {ev['truth'].shape[1]} | window {args.day_start:g}-{args.day_end:g} h")
        rows += run_one(cal, ev, args, cols)
    mc.write_csv(os.path.join(args.out_dir, "adaptive_ramp_metrics.csv"), rows)


if __name__ == "__main__":
    main()