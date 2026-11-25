"""
Adaptive traffic-conditioned conformal schedule + filtered growth-ramp penalty,
evaluated against the original schedule_conformal.py and an iso-overprovisioning baseline.

Key additions for fair evaluation:
  1. Direct comparison with `schedule_conformal.py` (10-decile Mondrian schedule).
  2. Iso-overprovisioning baseline: scales the conformal margin on validation so it matches the
     exact global over-provisioning % of the new approach, isolating allocation efficiency from
     mere budget increase.
  3. Underestimation-weighted ramp penalty: weights delta_pred = max(0, pred[t] - pred[t-1]) by the
     empirical underestimation ratio of each traffic bin, and tunes lambda via weighted cost J.

Usage:
  python src/analysis/schedule_adaptive_ramp.py --horizons 1 3 6 --day_start 9 --day_end 18 \
      --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
"""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schedule_conformal as sc
from analyse_result import kept_columns, load_series
from conformal_eval import conf_q

K_POINT = "STID point (no margin)"
K_ORIG = "Scheduled 10-deciles (orig)"
K_ISO = "Iso-overprov baseline"
K_ADAPT = "Adaptive conformal (5 bins)"
K_RAMP = "Adaptive + ramp (new)"
APPROACHES = (K_POINT, K_ORIG, K_ISO, K_ADAPT, K_RAMP)
PERIODS = ("all hours", "window", "other hours")
MIN_GROUP = 100


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
                    help="activation threshold [a.u.] for ON/OFF switching metric (default: 30th pct of train)")
    ap.add_argument("--off_thr", type=float, default=None,
                    help="deactivation threshold [a.u.] for hysteresis (default: 0.7 * on_thr)")
    ap.add_argument("--tiers", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.97],
                    help="train-mean quantiles splitting cells into static tiers")
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.ramp_lambda != "auto":
        args.ramp_lambda = float(args.ramp_lambda)
    return args


# ----------------------------------------------------------------------
# Data Splitting & Adaptive Conformal Calibration
# ----------------------------------------------------------------------
def split_cal_val(cal, val_frac):
    """Chronological split of calibration set into pure calibration and validation."""
    n = cal["pred"].shape[0]
    cut = int(n * (1.0 - val_frac))
    if cut < 50 or (n - cut) < 20:
        return cal, cal
    c_sub, v_sub = {}, {}
    for k, v in cal.items():
        if isinstance(v, np.ndarray) and v.shape[0] == n:
            c_sub[k] = v[:cut]
            v_sub[k] = v[cut:]
        else:
            c_sub[k] = v
            v_sub[k] = v
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
        """Normalized underestimation tendency per bin (scales ramp penalty where STID underestimates most)."""
        rates = []
        for b in range(self.n_bins):
            sel = self.lb_c == b
            if sel.any() and cal["truth"][sel].sum() > 0:
                u_rate = np.maximum(0.0, self.res_c[sel]).sum() / cal["truth"][sel].sum()
            else:
                u_rate = 0.0
            rates.append(u_rate)
        rates = np.array(rates)
        mean_r = rates.mean()
        return rates / mean_r if mean_r > 1e-8 else np.ones_like(rates)

    def predict(self, data):
        day_d = sc.in_window(data["slot"], self.args.day_start, self.args.day_end)[:, None]
        lb_d = np.searchsorted(self.edges, data["pred"], side="right")
        margin = np.where(day_d, self.q_day[lb_d], self.q_night[lb_d])
        weights = self.bin_underest_weight[lb_d]
        return data["pred"] + margin, margin, lb_d, weights


def apply_ramp_penalty(pred, base_bound, lam, bin_weights=None):
    """Add filtered positive-derivative ramp penalty: lambda * w(bin) * max(0, pred[t] - pred[t-1])."""
    if lam <= 0.0:
        return base_bound.copy()
    delta = np.zeros_like(pred)
    delta[1:] = np.maximum(0.0, pred[1:] - pred[:-1])
    if bin_weights is not None:
        delta = delta * bin_weights
    return base_bound + lam * delta


def tune_lambda(calib_model, val_data, args):
    """Select optimal ramp_lambda on the disjoint validation split minimizing J = w*unmet + (1-w)*overprov."""
    base_val, _, _, w_val = calib_model.predict(val_data)
    candidates = np.linspace(0.0, 1.0, 41)

    best_lam = 0.0
    best_cost = float("inf")
    for lam in candidates:
        est = apply_ramp_penalty(val_data["pred"], base_val, lam, w_val)
        m = global_metrics(est, val_data["truth"], args.on_thr_used, args.off_thr_used)
        cost = args.cost_w * m["unmet_%"] + (1.0 - args.cost_w) * m["overprov_%"]
        if cost < best_cost - 1e-5:
            best_cost = cost
            best_lam = float(lam)
    return best_lam


def match_iso_overprov(pred, orig_bound, target_overprov, truth):
    """Scale the margin of orig_bound by a constant alpha so its global overprov_% matches target_overprov."""
    orig_margin = np.maximum(0.0, orig_bound - pred)
    lo, hi = 0.0, 5.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        cand = pred + mid * orig_margin
        ov = 100.0 * np.maximum(cand - truth, 0.0).sum() / truth.sum()
        if ov < target_overprov:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return pred + alpha * orig_margin, alpha


# ----------------------------------------------------------------------
# Metrics (including hysteresis ON/OFF switching rate)
# ----------------------------------------------------------------------
def compute_switching_rate(est, on_thr, off_thr, mask=None):
    """Compute ON/OFF transition percentage per cell-step using hysteresis [off_thr, on_thr]."""
    T, C = est.shape
    if T < 2:
        return 0.0
    state = est[0] >= on_thr
    transitions = np.zeros((T, C), dtype=bool)
    for t in range(1, T):
        next_state = np.where(state, est[t] >= off_thr, est[t] >= on_thr)
        transitions[t] = next_state != state
        state = next_state

    if mask is None:
        return float(100.0 * transitions[1:].mean())
    valid = mask[1:]
    return float(100.0 * transitions[1:][valid].mean()) if valid.any() else float("nan")


def global_metrics(est, truth, on_thr, off_thr, mask=None):
    """cover (0-1), unmet %, overprov %, and ON/OFF switch % inside mask."""
    t, e = (truth, est) if mask is None else (truth[mask], est[mask])
    if t.size == 0 or t.sum() <= 0:
        return {"cover": float("nan"), "unmet_%": float("nan"),
                "overprov_%": float("nan"), "switch_%": float("nan"), "n": int(t.size)}
    d = e - t
    sw = compute_switching_rate(est, on_thr, off_thr, mask)
    return {
        "cover": float((t <= e).mean()),
        "unmet_%": float(100.0 * np.maximum(-d, 0.0).sum() / t.sum()),
        "overprov_%": float(100.0 * np.maximum(d, 0.0).sum() / t.sum()),
        "switch_%": sw,
        "n": int(t.size),
    }


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def static_tiers(args, cols):
    """Tier index (0 = quietest) of every cell from TRAIN mean activity, plus default hysteresis thresholds."""
    train = load_series(Path(args.data) / "train_data.npy")[:, cols]
    mean = train.mean(0)
    edges = np.quantile(mean, args.tiers)
    tier = np.searchsorted(edges, mean, side="right")
    cuts = [0.0] + list(args.tiers) + [1.0]
    names = [f"cells {cuts[i]:.0%}-{cuts[i + 1]:.0%}" for i in range(len(cuts) - 1)]

    on_thr = args.on_thr if args.on_thr is not None else float(np.quantile(train, 0.30))
    off_thr = args.off_thr if args.off_thr is not None else 0.70 * on_thr
    return tier, mean, names, on_thr, off_thr


def evaluate(ev, calib_model, bounds_dict, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    day = np.broadcast_to(sc.in_window(ev["slot"], args.day_start, args.day_end)[:, None], truth.shape)
    periods = {"all hours": np.ones_like(day), "window": day, "other hours": ~day}
    out = {"global": {}, "tiers": {}, "bins": {}}

    # 1. Global metrics across periods
    for n, e in bounds_dict.items():
        out["global"][n] = {p: global_metrics(e, truth, args.on_thr_used, args.off_thr_used, m)
                            for p, m in periods.items()}

    # 2. Static cell tiers (all hours)
    tier_pairs = tier[None, :] * np.ones(truth.shape, dtype=int)
    for i, name in enumerate(tier_names):
        m_t = tier_pairs == i
        out["tiers"][name] = {
            "n_cells": int((tier == i).sum()),
            "train_mean_activity_au": float(tier_mean[tier == i].mean()),
            "share_activity_%": float(100.0 * truth[m_t].sum() / truth.sum()),
            "approaches": {
                n: global_metrics(e, truth, args.on_thr_used, args.off_thr_used, m_t)
                for n, e in bounds_dict.items()
            },
        }

    # 3. Behavior by predicted traffic bin (all hours)
    _, margin_ev, lb_ev, _ = calib_model.predict(ev)
    up_ramp = bounds_dict[K_RAMP]
    up_orig = bounds_dict[K_ORIG]
    for b in range(calib_model.n_bins):
        m_b = lb_ev == b
        if not m_b.any():
            continue
        out["bins"][f"bin_{b + 1}"] = {
            "pred_mean_au": float(ev["pred"][m_b].mean()),
            "truth_mean_au": float(truth[m_b].mean()),
            "underest_point_au": float(np.maximum(0.0, truth[m_b] - ev["pred"][m_b]).mean()),
            "orig_margin_au": float((up_orig[m_b] - ev["pred"][m_b]).mean()),
            "adaptive_margin_au": float(margin_ev[m_b].mean()),
            "final_margin_au": float((up_ramp[m_b] - ev["pred"][m_b]).mean()),
            "unmet_point_%": float(100.0 * np.maximum(0.0, truth[m_b] - ev["pred"][m_b]).sum() / truth[m_b].sum()),
            "unmet_orig_%": float(100.0 * np.maximum(0.0, truth[m_b] - up_orig[m_b]).sum() / truth[m_b].sum()),
            "unmet_final_%": float(100.0 * np.maximum(0.0, truth[m_b] - up_ramp[m_b]).sum() / truth[m_b].sum()),
        }
    return out


def print_report(res, lam, iso_alpha, args, calib_model):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    print(f"\nAdaptive bins: {calib_model.n_bins} | ramp lambda = {lam:.3f} (cost_w={args.cost_w:.2f}) | "
          f"iso-overprov scale alpha = {iso_alpha:.3f} | hysteresis = [{args.off_thr_used:.1f}, {args.on_thr_used:.1f}] a.u.")

    print(f"\n== 1) GLOBAL, ALL CELLS (cover, unmet %, over-prov. %, ON/OFF switch %) ==")
    print(f"{'approach':30s} | " + " | ".join(f"{(w if p == 'window' else p):^33s}" for p in PERIODS))
    print(f"{'':30s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'over%':>8s} {'sw%':>7s}" for _ in PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_RAMP else ("  <- fair ref" if n == K_ISO else "")
        print(f"{n:30s} | " + " | ".join(
            f"{res['global'][n][p]['cover']:6.3f} {res['global'][n][p]['unmet_%']:8.2f} "
            f"{res['global'][n][p]['overprov_%']:8.2f} {res['global'][n][p]['switch_%']:7.2f}"
            for p in PERIODS) + tag)

    # Direct efficiency verdict on test (all hours)
    g_iso = res["global"][K_ISO]["all hours"]
    g_new = res["global"][K_RAMP]["all hours"]
    g_orig = res["global"][K_ORIG]["all hours"]
    print(f"\n[Efficiency Check — All Hours]")
    print(f"  vs Scheduled 10-deciles (orig): unmet {g_new['unmet_%'] - g_orig['unmet_%']:+.2f} pp | "
          f"overprov {g_new['overprov_%'] - g_orig['overprov_%']:+.2f} pp | "
          f"switch {g_new['switch_%'] - g_orig['switch_%']:+.3f} pp")
    print(f"  vs Iso-overprov baseline      : unmet {g_new['unmet_%'] - g_iso['unmet_%']:+.2f} pp "
          f"at equal over-provisioning (~{g_new['overprov_%']:.2f}%)")

    print("\n== 2) CORRECTION PROFILE BY PREDICTED TRAFFIC BIN (all hours) ==")
    print(f"{'bin':7s} {'pred_avg':>9s} {'underest':>9s} {'orig_m':>9s} {'new_m':>9s} "
          f"{'unmet_pt%':>10s} {'unmet_orig%':>12s} {'unmet_new%':>11s}")
    for bname, b in res["bins"].items():
        print(f"{bname:7s} {b['pred_mean_au']:9.1f} {b['underest_point_au']:9.1f} "
              f"{b['orig_margin_au']:9.1f} {b['final_margin_au']:9.1f} "
              f"{b['unmet_point_%']:10.2f} {b['unmet_orig_%']:12.2f} {b['unmet_final_%']:11.2f}")


# ----------------------------------------------------------------------
# Unified 3-Panel Summary Figure (with Pareto Frontier)
# ----------------------------------------------------------------------
def pareto_front(pts):
    front = [i for i, (xi, yi) in enumerate(pts)
             if not any(xj <= xi and yj <= yi and (xj < xi or yj < yi)
                        for j, (xj, yj) in enumerate(pts) if j != i)]
    return sorted(front, key=lambda i: pts[i][0])


def figure_summary(res, args, path):
    """Single comprehensive 3-panel representation with Pareto efficiency comparison."""
    colors = {
        K_POINT: "tab:blue",
        K_ORIG: "tab:purple",
        K_ISO: "tab:gray",
        K_ADAPT: "tab:orange",
        K_RAMP: "tab:green",
    }
    markers = {K_POINT: "o", K_ORIG: "D", K_ISO: "X", K_ADAPT: "s", K_RAMP: "*"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    # Panel 1: Margin allocation by traffic level (Orig 10-dec vs Adaptive+Ramp)
    ax = axes[0]
    bnames = list(res["bins"].keys())
    xb = np.arange(len(bnames))
    underest = [res["bins"][b]["underest_point_au"] for b in bnames]
    orig_m = [res["bins"][b]["orig_margin_au"] for b in bnames]
    final_m = [res["bins"][b]["final_margin_au"] for b in bnames]
    ax.bar(xb - 0.18, underest, 0.35, color="tab:red", alpha=0.75, label="STID mean underestimation")
    ax.plot(xb + 0.18, orig_m, "D--", color="tab:purple", label="Scheduled 10-deciles (orig)")
    ax.plot(xb + 0.18, final_m, "*-", color="tab:green", markersize=9, linewidth=2, label="Adaptive + ramp (new)")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"Bin {i + 1}\n({res['bins'][b]['pred_mean_au']:.0f})" for i, b in enumerate(bnames)], fontsize=8)
    ax.set_xlabel("Predicted traffic bin (mean pred [a.u.])")
    ax.set_ylabel("Activity [a.u.]")
    ax.set_title("1. Margin Allocation vs Underestimation", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Pareto Trade-off (Over-provisioning % vs Unmet % — bottom-left is better)
    ax = axes[1]
    pts = [(res["global"][n]["all hours"]["overprov_%"], res["global"][n]["all hours"]["unmet_%"])
           for n in APPROACHES]
    front = pareto_front(pts)
    ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], "--", color="gray", lw=1.2,
            label="Pareto frontier", zorder=1)
    for i, n in enumerate(APPROACHES):
        ax.scatter(*pts[i], s=240 if n == K_RAMP else 110, c=colors[n], marker=markers[n],
                   edgecolors="k", linewidths=0.8, zorder=3, label=n)
    ax.set_xlabel("Over-provisioning [% of total activity] (lower = less waste)")
    ax.set_ylabel("Unmet activity [% of total activity] (lower = safer)")
    ax.set_title("2. Efficiency Frontier (All Cells, All Hours)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    # Panel 3: Unmet % across static cell tiers (comparing Orig, Iso-overprov, and New)
    ax = axes[2]
    tnames = list(res["tiers"].keys())
    xt = np.arange(len(tnames))
    cmp_keys = (K_ORIG, K_ISO, K_RAMP)
    w = 0.26
    for i, n in enumerate(cmp_keys):
        vals = [res["tiers"][t]["approaches"][n]["unmet_%"] for t in tnames]
        ax.bar(xt + (i - 1) * w, vals, w, color=colors[n], label=n)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t.replace('cells ', '')}\n({res['tiers'][t]['share_activity_%']:.0f}% act.)"
                        for t in tnames], fontsize=8)
    ax.set_xlabel("Static cell tier (share of total traffic)")
    ax.set_ylabel("Unmet activity [%]")
    ax.set_title("3. Unmet Activity by Tier (vs Fair Baselines)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Efficiency Comparison — Horizon {args.horizon * 10} min | "
        f"Window {args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}/{1 - args.night_eps:.0%}) | "
        f"Bins={args.bins_used}, lambda={args.lambda_used:.2f}",
        fontsize=11.5
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------
def run_one(cal, ev, args, cols):
    tier, tier_mean, tier_names, on_thr, off_thr = static_tiers(args, cols)
    args.on_thr_used, args.off_thr_used = on_thr, off_thr

    # 1. Original schedule_conformal.py baseline (10 deciles per period)
    c_orig = sc.Calib(cal, ev, args)
    bounds_orig, _, _ = sc.build_bounds(c_orig, ev, args)
    up_orig = bounds_orig[sc.K_SCHED]

    # 2. Separate calibration and validation splits for tuning lambda
    cal_pure, val_pure = split_cal_val(cal, args.val_frac)
    calib_model = AdaptiveCalib(cal_pure, args)
    args.bins_used = calib_model.n_bins

    if args.ramp_lambda == "auto":
        lam = tune_lambda(calib_model, val_pure, args)
    else:
        lam = float(args.ramp_lambda)
    args.lambda_used = lam

    # 3. Re-fit adaptive margins on full calibration set once lambda is fixed
    full_calib = AdaptiveCalib(cal, args)
    up_adapt, _, _, w_ev = full_calib.predict(ev)
    up_ramp = apply_ramp_penalty(ev["pred"], up_adapt, lam, w_ev)

    # 4. Iso-overprovisioning baseline: scale original schedule margin on VAL to match target overprov%
    c_orig_val = sc.Calib(cal_pure, val_pure, args)
    b_orig_val, _, _ = sc.build_bounds(c_orig_val, val_pure, args)
    up_adapt_val, _, _, w_val = calib_model.predict(val_pure)
    up_ramp_val = apply_ramp_penalty(val_pure["pred"], up_adapt_val, lam, w_val)
    target_ov_val = 100.0 * np.maximum(up_ramp_val - val_pure["truth"], 0.0).sum() / val_pure["truth"].sum()
    _, iso_alpha = match_iso_overprov(val_pure["pred"], b_orig_val[sc.K_SCHED], target_ov_val, val_pure["truth"])
    up_iso = ev["pred"] + iso_alpha * np.maximum(0.0, up_orig - ev["pred"])

    bounds_dict = {
        K_POINT: ev["pred"],
        K_ORIG: up_orig,
        K_ISO: up_iso,
        K_ADAPT: up_adapt,
        K_RAMP: up_ramp,
    }

    res = evaluate(ev, full_calib, bounds_dict, args, tier, tier_mean, tier_names)
    print_report(res, lam, iso_alpha, args, full_calib)
    return res, full_calib, lam, iso_alpha


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    os.makedirs(args.out_dir, exist_ok=True)
    runs = []

    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
        print(f"\n##### Horizon {h * 10} min | cells {ev['truth'].shape[1]} | "
              f"window {args.day_start:g}-{args.day_end:g} h")

        res, calib_model, lam, iso_alpha = run_one(cal, ev, args, cols)

        base = os.path.join(args.out_dir, f"adaptive_h{h * 10}_w{args.day_start:g}-{args.day_end:g}")
        figure_summary(res, args, base + "_summary.png")

        mpath = Path("results/tables") / f"adaptive_margins_h{h * 10}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump({
                "horizon_min": h * 10,
                "window_hours": [args.day_start, args.day_end],
                "eps_window": args.day_eps,
                "eps_outside": args.night_eps,
                "ramp_lambda": lam,
                "bin_edges": [float(e) for e in calib_model.edges],
                "bin_underest_weights": [float(w) for w in calib_model.bin_underest_weight],
                "margin_window": [float(q) for q in calib_model.q_day],
                "margin_outside": [float(q) for q in calib_model.q_night],
                "formula": "b = searchsorted(bin_edges, pred(t), side='right'); "
                           "bound(t) = pred(t) + margin_<period>[b] "
                           "+ ramp_lambda * bin_underest_weights[b] * max(0, pred(t) - pred(t-1))"
            }, f, indent=2)
        print(f"Saved {mpath}")

        runs.append({
            "horizon_min": h * 10,
            "window": f"{args.day_start:g}-{args.day_end:g}",
            "n_bins": calib_model.n_bins,
            "ramp_lambda": lam,
            "iso_alpha": iso_alpha,
            "results": res
        })

    with open(os.path.join(args.out_dir, "adaptive_all.json"), "w") as f:
        json.dump({"eps_window": args.day_eps, "eps_outside": args.night_eps, "runs": runs}, f, indent=2)

    with open(os.path.join(args.out_dir, "adaptive_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_min", "window", "scope", "group", "approach", "period",
                    "cover", "unmet_%", "overprov_%", "switch_%"])
        for r in runs:
            for n, per in r["results"]["global"].items():
                for p, m in per.items():
                    w.writerow([r["horizon_min"], r["window"], "global", "all cells", n, p,
                                m["cover"], m["unmet_%"], m["overprov_%"], m["switch_%"]])
            for g, t in r["results"]["tiers"].items():
                for n, m in t["approaches"].items():
                    w.writerow([r["horizon_min"], r["window"], "tiers", g, n, "all hours",
                                m["cover"], m["unmet_%"], m["overprov_%"], m["switch_%"]])
    print(f"\nSaved {args.out_dir}/adaptive_all.json and .csv")


if __name__ == "__main__":
    main()