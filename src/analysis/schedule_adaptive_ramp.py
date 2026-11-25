"""
Adaptive traffic-conditioned conformal schedule + growth-ramp penalty.

Idea
  Replaces binary busy/quiet gating with a two-component, low-overhead correction:
    1. Traffic-conditioned conformal margin (AdaptiveCalib):
       Predictions are binned into N_BINS (default 5) volume quantiles on a pure calibration split.
       For each (period, bin), the upper conformal quantile of residuals (truth - pred) is computed
       and clamped at >= 0:
           y_adj(t, c) = pred(t, c) + max(0, Q_eps(e | period, bin(pred)))
    2. Growth-ramp penalty (ramp_lambda):
       To protect against sharp traffic surges and stabilize cell ON/OFF states without complex
       state-space optimization, a simple positive-derivative term is added:
           y_final(t, c) = y_adj(t, c) + lambda * max(0, pred(t, c) - pred(t-1, c))
       where lambda >= 0 is tuned on a disjoint validation split of calibration data (zero test leakage).

Evaluation (test split, ORIGINAL scale [a.u.]) on ALL cells:
  - Approaches compared:
      1) STID point (no margin)
      2) Adaptive conformal (5 bins)
      3) Adaptive + ramp penalty (new)
  - Metrics:
      * cover (0-1): fraction of (window, cell) pairs where truth <= bound
      * unmet_%: total under-provisioned volume / total true volume [%]
      * overprov_%: total over-provisioned volume / total true volume [%]
      * switch_%: ON/OFF state transitions per 100 cell-steps (proxy for switching cost, computed
                  with hysteresis thresholds --on_thr and --off_thr)

Saved in --out_dir (default final_metrics/adaptive_ramp):
  - adaptive_h<min>_w<start>-<end>_summary.png   3-panel comprehensive summary figure
  - adaptive_all.json / adaptive_all.csv        full metrics and parameter logs
  - results/tables/adaptive_margins_h<min>.json runtime lookup tables

Usage:
  python src/analysis/schedule_adaptive_ramp.py --horizons 1 --day_start 9 --day_end 18 \
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
K_ADAPT = "Adaptive conformal (5 bins)"
K_RAMP = "Adaptive + ramp penalty (new)"
APPROACHES = (K_POINT, K_ADAPT, K_RAMP)
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
                    help="'auto' = tuned on val split, or a fixed float >= 0 (e.g. 0.25)")
    ap.add_argument("--max_overprov_increase", type=float, default=2.0,
                    help="auto mode: max tolerated over-provisioning increase [pp] on val when tuning lambda")
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
    """Chronological split of calibration set into pure calibration and validation (for lambda tuning)."""
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
    """Traffic-conditioned conformal margins across predicted volume bins and time periods."""

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

    def _table(self, eps, period):
        margins = []
        for b in range(self.n_bins):
            sel = (self.lb_c == b) & (self.day_c == period)
            q = float(conf_q(self.res_c[sel], eps)) if sel.any() else 0.0
            margins.append(max(0.0, q))
        return np.array(margins)

    def predict(self, data):
        day_d = sc.in_window(data["slot"], self.args.day_start, self.args.day_end)[:, None]
        lb_d = np.searchsorted(self.edges, data["pred"], side="right")
        margin = np.where(day_d, self.q_day[lb_d], self.q_night[lb_d])
        return data["pred"] + margin, margin, lb_d


def apply_ramp_penalty(pred, base_bound, lam):
    """Add positive-derivative ramp penalty: lambda * max(0, pred[t] - pred[t-1])."""
    if lam <= 0.0:
        return base_bound.copy()
    delta = np.zeros_like(pred)
    delta[1:] = np.maximum(0.0, pred[1:] - pred[:-1])
    return base_bound + lam * delta


def tune_lambda(calib_model, val_data, args):
    """Select optimal ramp_lambda on the disjoint validation split."""
    base_val, _, _ = calib_model.predict(val_data)
    base_m = global_metrics(base_val, val_data["truth"], args.on_thr, args.off_thr)
    candidates = np.linspace(0.0, 1.0, 21)

    best_lam = 0.0
    best_unmet = base_m["unmet_%"]
    for lam in candidates:
        est = apply_ramp_penalty(val_data["pred"], base_val, lam)
        m = global_metrics(est, val_data["truth"], args.on_thr, args.off_thr)
        if m["overprov_%"] <= base_m["overprov_%"] + args.max_overprov_increase:
            if m["unmet_%"] < best_unmet - 1e-4:
                best_unmet = m["unmet_%"]
                best_lam = float(lam)
    return best_lam


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


def evaluate(ev, calib_model, up_adapt, up_ramp, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    est = {K_POINT: ev["pred"], K_ADAPT: up_adapt, K_RAMP: up_ramp}
    day = np.broadcast_to(sc.in_window(ev["slot"], args.day_start, args.day_end)[:, None], truth.shape)
    periods = {"all hours": np.ones_like(day), "window": day, "other hours": ~day}
    out = {"global": {}, "tiers": {}, "bins": {}, "per_cell": {}}

    # 1. Global metrics across periods
    for n, e in est.items():
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
                for n, e in est.items()
            },
        }

    # 3. Behavior by predicted traffic bin (all hours)
    _, margin_ev, lb_ev = calib_model.predict(ev)
    for b in range(calib_model.n_bins):
        m_b = lb_ev == b
        if not m_b.any():
            continue
        out["bins"][f"bin_{b + 1}"] = {
            "pred_mean_au": float(ev["pred"][m_b].mean()),
            "truth_mean_au": float(truth[m_b].mean()),
            "underest_point_au": float(np.maximum(0.0, truth[m_b] - ev["pred"][m_b]).mean()),
            "adaptive_margin_au": float(margin_ev[m_b].mean()),
            "final_margin_au": float((up_ramp[m_b] - ev["pred"][m_b]).mean()),
            "unmet_point_%": float(100.0 * np.maximum(0.0, truth[m_b] - ev["pred"][m_b]).sum() / truth[m_b].sum()),
            "unmet_final_%": float(100.0 * np.maximum(0.0, truth[m_b] - up_ramp[m_b]).sum() / truth[m_b].sum()),
        }

    # 4. Per-cell distribution
    for n, e in est.items():
        d = e - truth
        u = 100.0 * np.maximum(-d, 0.0).sum(0) / np.maximum(truth.sum(0), 1e-12)
        cov = (truth <= e).mean(0)
        out["per_cell"][n] = {
            "unmet_%_median": float(np.median(u)),
            "unmet_%_p90": float(np.percentile(u, 90)),
            "unmet_%_max": float(u.max()),
            "cover_median": float(np.median(cov)),
            "cover_p10": float(np.percentile(cov, 10)),
        }
    return out


def print_report(res, lam, args, calib_model):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    print(f"\nAdaptive bins: {calib_model.n_bins} | ramp lambda = {lam:.3f} | "
          f"hysteresis ON/OFF = [{args.off_thr_used:.1f}, {args.on_thr_used:.1f}] a.u.")

    print(f"\n== 1) GLOBAL, ALL CELLS (unmet %, over-prov. %, ON/OFF switch %) ==")
    print(f"{'approach':32s} | " + " | ".join(f"{(w if p == 'window' else p):^33s}" for p in PERIODS))
    print(f"{'':32s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'over%':>8s} {'sw%':>7s}" for _ in PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_RAMP else ""
        print(f"{n:32s} | " + " | ".join(
            f"{res['global'][n][p]['cover']:6.3f} {res['global'][n][p]['unmet_%']:8.2f} "
            f"{res['global'][n][p]['overprov_%']:8.2f} {res['global'][n][p]['switch_%']:7.2f}"
            for p in PERIODS) + tag)

    print("\n== 2) BY STATIC TIER of cells (all hours) ==")
    print(f"{'tier':16s} {'cells':>6s} {'share%':>7s} | "
          + " | ".join(f"{k[:18]:^23s}" for k in APPROACHES))
    for name, t in res["tiers"].items():
        print(f"{name:16s} {t['n_cells']:6d} {t['share_activity_%']:7.1f} | " + " | ".join(
            f"{t['approaches'][n]['cover']:6.3f} {t['approaches'][n]['unmet_%']:7.2f} "
            f"{t['approaches'][n]['overprov_%']:7.2f}" for n in APPROACHES))

    print("\n== 3) CORRECTION PROFILE BY PREDICTED TRAFFIC BIN (all hours) ==")
    print(f"{'bin':8s} {'pred_avg':>10s} {'true_avg':>10s} {'underest':>10s} {'adapt_m':>10s} "
          f"{'final_m':>10s} {'unmet_pt%':>10s} {'unmet_new%':>10s}")
    for bname, b in res["bins"].items():
        print(f"{bname:8s} {b['pred_mean_au']:10.1f} {b['truth_mean_au']:10.1f} "
              f"{b['underest_point_au']:10.1f} {b['adaptive_margin_au']:10.1f} "
              f"{b['final_margin_au']:10.1f} {b['unmet_point_%']:10.2f} {b['unmet_final_%']:10.2f}")


# ----------------------------------------------------------------------
# Unified 3-Panel Summary Figure
# ----------------------------------------------------------------------
def figure_summary(res, args, path):
    """Single comprehensive 3-panel representation grouping all key insights."""
    colors = {K_POINT: "tab:blue", K_ADAPT: "tab:orange", K_RAMP: "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # Panel 1: Correction profile across predicted traffic bins
    ax = axes[0]
    bnames = list(res["bins"].keys())
    xb = np.arange(len(bnames))
    underest = [res["bins"][b]["underest_point_au"] for b in bnames]
    adapt_m = [res["bins"][b]["adaptive_margin_au"] for b in bnames]
    final_m = [res["bins"][b]["final_margin_au"] for b in bnames]
    ax.bar(xb - 0.18, underest, 0.35, color="tab:red", alpha=0.8, label="STID mean underestimation")
    ax.plot(xb + 0.18, adapt_m, "o--", color="tab:orange", label="Adaptive conformal margin")
    ax.plot(xb + 0.18, final_m, "s-", color="tab:green", linewidth=2, label="Final margin (+ ramp)")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"Bin {i + 1}\n({res['bins'][b]['pred_mean_au']:.0f})" for i, b in enumerate(bnames)], fontsize=8)
    ax.set_xlabel("Predicted traffic bin (mean pred [a.u.])")
    ax.set_ylabel("Activity [a.u.]")
    ax.set_title("1. Correction vs STID Underestimation", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Global Trade-off (Unmet %, Over-provisioning %, ON/OFF Switching %)
    ax = axes[1]
    metrics = ("unmet_%", "overprov_%", "switch_%")
    mlabs = ("Unmet Activity [%]\n(lower = safer)", "Over-provisioning [%]\n(lower = less waste)",
             "ON/OFF Switch [%]\n(lower = more stable)")
    xm = np.arange(len(metrics))
    w = 0.25
    for i, n in enumerate(APPROACHES):
        vals = [res["global"][n]["all hours"][m] for m in metrics]
        ax.bar(xm + (i - 1) * w, vals, w, color=colors[n], label=n)
    ax.set_xticks(xm)
    ax.set_xticklabels(mlabs, fontsize=8)
    ax.set_ylabel("Percentage [%]")
    ax.set_title("2. Global Trade-off (All Cells, All Hours)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # Panel 3: Unmet % across static cell tiers
    ax = axes[2]
    tnames = list(res["tiers"].keys())
    xt = np.arange(len(tnames))
    for i, n in enumerate(APPROACHES):
        vals = [res["tiers"][t]["approaches"][n]["unmet_%"] for t in tnames]
        ax.bar(xt + (i - 1) * w, vals, w, color=colors[n], label=n)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t.replace('cells ', '')}\n({res['tiers'][t]['share_activity_%']:.0f}% act.)"
                        for t in tnames], fontsize=8)
    ax.set_xlabel("Static cell tier (share of total traffic)")
    ax.set_ylabel("Unmet activity [%]")
    ax.set_title("3. Unmet Activity by Cell Tier", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Adaptive Conformal + Ramp Penalty — Horizon {args.horizon * 10} min | "
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

    # Separate calibration and validation splits to prevent parameter overfitting
    cal_pure, val_pure = split_cal_val(cal, args.val_frac)
    calib_model = AdaptiveCalib(cal_pure, args)
    args.bins_used = calib_model.n_bins

    if args.ramp_lambda == "auto":
        lam = tune_lambda(calib_model, val_pure, args)
    else:
        lam = float(args.ramp_lambda)
    args.lambda_used = lam

    # Re-fit adaptive margins on full calibration set once lambda is fixed
    full_calib = AdaptiveCalib(cal, args)
    up_adapt, _, _ = full_calib.predict(ev)
    up_ramp = apply_ramp_penalty(ev["pred"], up_adapt, lam)

    res = evaluate(ev, full_calib, up_adapt, up_ramp, args, tier, tier_mean, tier_names)
    print_report(res, lam, args, full_calib)
    return res, full_calib, lam


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

        res, calib_model, lam = run_one(cal, ev, args, cols)

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
                "margin_window": [float(q) for q in calib_model.q_day],
                "margin_outside": [float(q) for q in calib_model.q_night],
                "formula": "bound(t) = pred(t) + margin_<period>[searchsorted(bin_edges, pred(t), side='right')] "
                           "+ ramp_lambda * max(0, pred(t) - pred(t-1))"
            }, f, indent=2)
        print(f"Saved {mpath}")

        runs.append({
            "horizon_min": h * 10,
            "window": f"{args.day_start:g}-{args.day_end:g}",
            "n_bins": calib_model.n_bins,
            "ramp_lambda": lam,
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