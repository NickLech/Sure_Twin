"""
Busy-gated conformal schedule (extension of schedule_conformal.py)..

Idea: spend margin only where the predicted traffic is high. Pairs (window, cell) are split at PREDICTION time:
    busy   pred >= tau : full Mondrian margin (period x decile of the predicted level), calibrated ONLY on busy pairs
    quiet  pred <  tau : bound = pred (no margin at all)
tau: --busy_thr auto = largest candidate whose GLOBAL unmet on val grows by at most --max_unmet_increase pp w.r.t.
the ungated schedule (chosen on val, never on test), or a fixed number in [a.u.].

Fair verdict: on test the gate is compared with the ungated schedule whose margin is scaled by ONE constant alpha
(metrics_common.iso_baselines): iso-overprov = same over-provisioning as the gate, iso-unmet = same unmet.
The gate is useful only if it beats both (negative deltas).

Observed (h = 10 / 30 / 60 min): the gate loses on both. +0.15 / +0.14 / +0.15 pp unmet at equal over, and
+0.70 / +0.64 / +0.73 pp over at equal unmet. The quietest tier of cells (0-50%) is left near the point forecast
(unmet ~2.2% -> ~4.8%, worst cell > 100% of its activity). A 20-configuration sweep (ensemble-spread flags, reduced
quiet margin) never beat the frontier either, so that code was removed.

Output (--out_dir, default final_metrics/busy_gate): busy_gate_metrics.csv (schema of metrics_common.py), plus per
horizon a frontier figure and the tau-selection figure. Units: unmet / over-provisioning are % of the total TRUE
activity of the group (activity index [a.u.], NOT energy); cover is a fraction (0-1); tau and margins in [a.u.].

Usage (from the repository root; --horizons are 1-based, 1 = 10 min):
    python src/analysis/schedule_busy_gate.py --horizons 1 3 6 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
The per-run options of schedule_conformal.py are accepted (--day_eps, --night_eps, --cal_data, ...).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metrics_common as mc
import schedule_conformal as sc
from conformal_eval import N_BINS, conf_q

METHOD = "busy_gate"
K_ISOO, K_ISOU, K_GATE = "iso-overprov baseline", "iso-unmet baseline", "busy gate"
APPROACHES = (mc.K_POINT, mc.K_REF, K_ISOO, K_ISOU, K_GATE)
SHORT = {mc.K_POINT: "point", mc.K_REF: "all cells", K_ISOO: "iso-over", K_ISOU: "iso-unmet", K_GATE: "gate"}
MIN_GROUP = 200        # minimum busy calibration pairs per (period, decile) group


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def parse_all(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="final_metrics/busy_gate")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1])
    ap.add_argument("--busy_thr", default="auto",
                    help="'auto' = chosen on val (see --max_unmet_increase) or a number in [a.u.]")
    ap.add_argument("--max_unmet_increase", type=float, default=0.25,
                    help="auto mode: tolerated increase of the GLOBAL unmet %% on val, in percentage points")
    ap.add_argument("--n_tau", type=int, default=24, help="number of candidate thresholds of the auto search")
    mc.add_args(ap)
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.busy_thr != "auto":
        args.busy_thr = float(args.busy_thr)
    mc.set_lambdas(args)
    return args


# ----------------------------------------------------------------------
# Gated conformal margins
# ----------------------------------------------------------------------
class GatedCalib:
    """Mondrian margins [period x decile of the busy predicted level], calibrated ONLY on busy calibration pairs."""

    def __init__(self, cal, tau, args):
        self.tau, self.args = tau, args
        busy = cal["pred"] >= tau
        if not busy.any():
            raise ValueError("no busy calibration pair: lower --busy_thr")
        day = sc.in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        res = cal["truth"] - cal["pred"]
        n = N_BINS
        while True:                         # fewer bins when the busy set is too small for deciles
            self.edges = np.quantile(cal["pred"][busy], np.linspace(0, 1, n + 1)[1:-1])
            lb = np.searchsorted(self.edges, cal["pred"], side="right")
            sel = [[busy & (lb == b) & (day == p) for p in (True, False)] for b in range(n)]   # [bin][window/other]
            small = min(s.sum() for pair in sel for s in pair)
            if small >= MIN_GROUP or n == 1:
                break
            n -= 1
        if small < MIN_GROUP:
            raise ValueError(f"busy set too small ({small} pairs in the smallest group): lower --busy_thr")
        self.n_bins = n
        self.q_day = np.array([conf_q(res[pair[0]], args.day_eps) for pair in sel])
        self.q_night = np.array([conf_q(res[pair[1]], args.night_eps) for pair in sel])

    def bound(self, d):
        """(bound, busy mask) on data d: pred + margin on busy pairs, pred alone on quiet ones."""
        a = self.args
        day = sc.in_window(d["slot"], a.day_start, a.day_end)[:, None]
        lb = np.searchsorted(self.edges, d["pred"], side="right")
        margin = np.where(day, self.q_day[lb], self.q_night[lb])
        busy = d["pred"] >= self.tau
        return d["pred"] + np.where(busy, margin, 0.0), busy


def search_tau(cal, args):
    """Global metrics on VAL (all cells) for every candidate tau, and the largest tau within the unmet budget.
    Calibration and evaluation are both on val here (in-sample): it only selects tau, never reports performance."""
    qs = np.quantile(cal["pred"], np.linspace(0, 0.97, args.n_tau))
    curve = []
    for tau in np.concatenate([[-np.inf], np.unique(qs[1:])]):      # -inf = every pair busy = ungated
        try:
            bound, busy = GatedCalib(cal, tau, args).bound(cal)
        except ValueError:
            continue
        curve.append({"tau": float(tau), **mc.compute(bound, cal["truth"]),
                      "busy_pairs_%": float(100 * busy.mean()),
                      "busy_activity_%": float(100 * cal["truth"][busy].sum() / cal["truth"].sum())})
    if not curve:
        raise ValueError("no valid tau candidate")
    base = curve[0]["unmet_%"]
    ok = [r for r in curve if r["unmet_%"] <= base + args.max_unmet_increase]
    return curve, max(ok, key=lambda r: r["tau"])


# ----------------------------------------------------------------------
# Evaluation and report
# ----------------------------------------------------------------------
def evaluate(ev, busy, est, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    dynamic = {"busy (pred >= tau)": busy, "quiet (no margin)": ~busy}
    dynamic = {k: m for k, m in dynamic.items() if m.any()}
    scopes = {"global": {"all cells": None}, "tier": mc.tier_groups(tier, tier_names, truth.shape),
              "dynamic": dynamic}
    res = mc.evaluate_scopes(est, truth, mc.day_mask(ev, args), args, scopes)
    info = {"tier": mc.tier_info(tier, tier_mean, tier_names, truth),
            "dynamic": {k: {"share_pairs_%": float(100 * m.mean()),
                            "share_activity_%": float(100 * truth[m].sum() / truth.sum())}
                        for k, m in dynamic.items()}}
    for i, name in enumerate(tier_names):
        info["tier"][name]["busy_pairs_%"] = float(100 * busy[:, tier == i].mean())
    return res, info, {n: mc.per_cell_stats(e, truth) for n, e in est.items()}


def uo(m):
    return f"{m['unmet_%']:5.2f}/{m['overprov_%']:5.2f}"


def print_report(res, info, per_cell, tau, args, busy, alphas):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    glob = res["global"]["all cells"]
    print(f"\ntau = {tau:.1f} a.u. | busy pairs {100 * busy.mean():.1f}% | "
          f"iso alpha: overprov {alphas[0]:.3f}, unmet {alphas[1]:.3f}")

    print("\n== 1) GLOBAL, ALL CELLS (cover 0-1; unmet / over-prov. in % of total true activity) ==")
    print(f"{'approach':38s} | " + " | ".join(f"{(w if p == 'window' else p):^24s}" for p in mc.PERIODS))
    print(f"{'':38s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'over%':>8s}" for _ in mc.PERIODS))
    for n in APPROACHES:
        print(f"{n:38s} | " + " | ".join(mc.fmt(glob[n][p]) for p in mc.PERIODS))

    print("\n== 2) BY STATIC TIER of cells (train-mean activity), all hours: unmet% / over% ==")
    print(f"{'tier':14s} {'cells':>5s} {'mean':>6s} {'act%':>5s} {'busy%':>5s} | "
          + " | ".join(f"{SHORT[n]:^11s}" for n in APPROACHES))
    for name, t in info["tier"].items():
        print(f"{name:14s} {t['n_cells']:5d} {t['train_mean_au']:6.0f} {t['share_activity_%']:5.1f} "
              f"{t['busy_pairs_%']:5.1f} | "
              + " | ".join(uo(res["tier"][name][n]["all hours"]) for n in APPROACHES))

    print("\n== 3) BY DYNAMIC STATE at prediction time, all hours: unmet% / over% ==")
    for name, t in info["dynamic"].items():
        print(f"{name}: {t['share_pairs_%']:.1f}% of pairs, {t['share_activity_%']:.1f}% of total activity")
        for n in APPROACHES:
            print(f"   {n:38s} {uo(res['dynamic'][name][n]['all hours'])}")

    print("\n== 4) PER-CELL distribution (unmet % of each cell's own activity; cover = per-cell fraction) ==")
    print(f"{'approach':38s} {'unmet med':>9s} {'p90':>7s} {'p99':>7s} {'max':>7s} | "
          f"{'cov med':>7s} {'p10':>6s} {'min':>6s}")
    for n in APPROACHES:
        p = per_cell[n]
        print(f"{n:38s} {p['unmet_med']:9.2f} {p['unmet_p90']:7.2f} {p['unmet_p99']:7.2f} {p['unmet_max']:7.2f} | "
              f"{p['cover_med']:7.3f} {p['cover_p10']:6.3f} {p['cover_min']:6.3f}")

    g, a = glob[K_GATE]["all hours"], glob[mc.K_REF]["all hours"]
    io, iu = glob[K_ISOO]["all hours"], glob[K_ISOU]["all hours"]
    print(f"\nGATE vs ALL-CELLS    (all hours): over {g['overprov_%'] - a['overprov_%']:+.2f} pp, "
          f"unmet {g['unmet_%'] - a['unmet_%']:+.2f} pp")
    print(f"GATE vs ISO-OVERPROV (all hours): unmet {g['unmet_%'] - io['unmet_%']:+.2f} pp at equal over   "
          f"[negative = gate better]")
    print(f"GATE vs ISO-UNMET    (all hours): over  {g['overprov_%'] - iu['overprov_%']:+.2f} pp at equal unmet  "
          f"[negative = gate better]")


def figure_threshold(curve, tau, args, path):
    """Global metrics on VAL vs tau: what the gate saves (over) and what it costs (unmet)."""
    t = np.array([r["tau"] for r in curve[1:]])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, lab in zip(axes[:2], ("overprov_%", "unmet_%"), ("over-provisioning [%]", "unmet [%]")):
        ax.plot(t, [r[key] for r in curve[1:]], "o-", color="tab:red")
        ax.axhline(curve[0][key], color="tab:green", ls="--", label="ungated (all cells)")
        ax.axvline(tau, color="k", ls=":", label=f"chosen tau = {tau:.0f} a.u.")
        ax.set_ylabel(lab + " of total activity")
        ax.legend(fontsize=8)
    axes[2].plot(t, [r["busy_pairs_%"] for r in curve[1:]], "o-", label="busy pairs, % of pairs")
    axes[2].plot(t, [r["busy_activity_%"] for r in curve[1:]], "s-", label="activity of busy pairs, % of total")
    axes[2].axvline(tau, color="k", ls=":")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("busy threshold tau on predicted activity [a.u.]")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Choice of tau on the VAL split (all cells) - horizon {args.horizon * 10} min", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run_one(cal, ev, args, cols):
    if args.busy_thr == "auto":
        curve, best = search_tau(cal, args)
        tau = best["tau"]
        print(f"[auto] tau = {tau:.1f} a.u. chosen on val (global unmet <= ungated + {args.max_unmet_increase} pp): "
              f"busy pairs {best['busy_pairs_%']:.1f}%, activity {best['busy_activity_%']:.1f}%")
    else:
        curve, tau = None, args.busy_thr

    up_gate, busy = GatedCalib(cal, tau, args).bound(ev)
    up_ref = mc.ref_bounds(cal, ev, args)
    up_io, up_iu, a_o, a_u = mc.iso_baselines(ev["pred"], up_ref, ev["truth"], up_gate)
    est = {mc.K_POINT: ev["pred"], mc.K_REF: up_ref, K_ISOO: up_io, K_ISOU: up_iu, K_GATE: up_gate}

    tier, tier_mean, tier_names, _ = mc.static_tiers(args, cols)
    res, info, per_cell = evaluate(ev, busy, est, args, tier, tier_mean, tier_names)
    print_report(res, info, per_cell, tau, args, busy, (a_o, a_u))

    h, window = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}"
    base = os.path.join(args.out_dir, f"busy_gate_h{h}_w{window}")
    mc.figure_frontier(base + "_frontier.png", ev["pred"], up_ref, ev["truth"], est,
                       f"Busy gate vs scaled reference - horizon {h} min - tau = {tau:.0f} a.u.")
    if curve is not None:
        figure_threshold(curve, tau, args, base + "_threshold.png")
    params = {K_GATE: f"tau={tau:.1f}", K_ISOO: f"alpha={a_o:.3f}", K_ISOU: f"alpha={a_u:.3f}"}
    return mc.to_rows(METHOD, h, window, res, params)


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for h in args.horizons:
        args.horizon = h
        cal, ev, cols = mc.load_horizon(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | cells {ev['truth'].shape[1]} | "
              f"window {args.day_start:g}-{args.day_end:g} h | calibration windows {cal['pred'].shape[0]}, "
              f"evaluation windows {ev['pred'].shape[0]}")
        try:
            rows += run_one(cal, ev, args, cols)
        except ValueError as e:
            raise SystemExit(str(e))
    mc.write_csv(os.path.join(args.out_dir, "busy_gate_metrics.csv"), rows)


if __name__ == "__main__":
    main()