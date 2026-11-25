"""
Busy-gated conformal schedule (extension of schedule_conformal.py), evaluated on ALL cells.

Pairs (window, cell) are split at PREDICTION time:
    busy            : pred >= tau  (| signal >= tau_spread with --spread_mode busy)
                      -> full Mondrian margin (period x decile of the predicted level), calibrated ONLY on busy pairs
    protected quiet : only with --quiet_eps: ONE reduced margin per period for the quiet pairs
                      (all of them, or only those with signal >= tau_spread with --spread_mode quiet_only)
    other quiet     : bound = pred
signal = ensemble spread, or spread / max(pred, 1) with --spread_rel.
tau: --busy_thr auto = largest tau whose GLOBAL unmet on val grows by at most --max_unmet_increase pp w.r.t.
the ungated schedule (never chosen on test), or a fixed number in [a.u.]. tau_spread = --spread_q quantile of the
val signal (over all pairs in mode busy, over the quiet pairs in mode quiet_only).

Fair verdict: on test the gate is compared with the ungated schedule whose margin is scaled by ONE constant
alpha chosen on val: iso-overprov (same over-provisioning as the gate) and iso-unmet (same unmet as the gate).
The gate is useful only if it beats both (negative deltas).

--sweep runs a grid spread_mode x spread_q x quiet_eps (tau auto each time), ranked by the VAL delta vs
iso-overprov; the TEST columns are not used by the selection.

Output (--out_dir, default final_metrics/busy_gate): ONE csv with the schema of metrics_common.py
(scopes global / tier / dynamic). Single run: busy_gate_metrics.csv + two figures. Sweep: busy_gate_sweep_metrics.csv
(every configuration is an approach named "busy gate [<config>]", with its two iso baselines).
Units: unmet / over-provisioning are % of the total TRUE activity of the group (activity index [a.u.], NOT
energy); cover is a fraction (0-1); tau and margins in [a.u.].

Usage (from the repository root; --horizons are 1-based, 1 = 10 min):
    python src/analysis/schedule_busy_gate.py --horizons 1 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python src/analysis/schedule_busy_gate.py ... --quiet_eps 0.3
    python src/analysis/schedule_busy_gate.py ... --spread_mode quiet_only --spread_q 0.9 --spread_rel --quiet_eps 0.3
    python src/analysis/schedule_busy_gate.py ... --sweep --sweep_quiet_eps 0.4 0.3 0.2 --sweep_spread_q 0.9 0.95
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
from conformal_eval import N_BINS, conf_q

METHOD = "busy_gate"
K_ISOO = "iso-overprov baseline"
K_ISOU = "iso-unmet baseline"
K_GATE = "busy gate"
APPROACHES = (mc.K_POINT, mc.K_REF, K_ISOO, K_ISOU, K_GATE)
SHORT = {mc.K_POINT: "point", mc.K_REF: "all cells", K_ISOO: "iso-overprov", K_ISOU: "iso-unmet", K_GATE: "gate"}
COLORS = {mc.K_POINT: "tab:blue", mc.K_REF: "tab:green", K_ISOO: "tab:gray", K_ISOU: "tab:orange", K_GATE: "tab:red"}
MIN_GROUP = 200        # minimum calibration pairs per (period, decile) group
MIN_QUIET = 50         # minimum calibration pairs to calibrate a quiet margin


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
    ap.add_argument("--quiet_eps", type=float, default=None,
                    help="violation rate for the protected quiet pairs (coverage 1 - eps, must exceed the point "
                         "coverage ~0.52 to add margin). Default: no margin")
    ap.add_argument("--spread_q", type=float, default=None,
                    help="quantile (0-1) of the val ensemble-spread signal that defines tau_spread")
    ap.add_argument("--spread_mode", choices=("busy", "quiet_only"), default="busy",
                    help="busy: high spread => busy (full margin). quiet_only: high spread only flags quiet pairs "
                         "that receive the reduced --quiet_eps margin")
    ap.add_argument("--spread_rel", action="store_true",
                    help="use spread / max(pred, 1) (relative uncertainty) instead of the raw spread")
    ap.add_argument("--n_tau", type=int, default=24, help="number of candidate thresholds of the auto search")
    ap.add_argument("--sweep", action="store_true",
                    help="grid over spread_mode x spread_q x quiet_eps instead of a single run")
    ap.add_argument("--sweep_spread_q", type=float, nargs="*", default=[0.9, 0.95, 0.98])
    ap.add_argument("--sweep_quiet_eps", type=float, nargs="*", default=[0.4, 0.3, 0.2])
    ap.add_argument("--sweep_n_tau", type=int, default=10, help="candidate thresholds per config in the sweep")
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
# Busy gate
# ----------------------------------------------------------------------
class Gate:
    """busy: pred >= tau (| signal >= tau_spread in 'busy' mode).
    protected quiet: risky quiet pairs in 'quiet_only' mode (signal >= tau_spread), all quiet pairs otherwise."""

    def __init__(self, tau, cal, args):
        self.tau, self.mode, self.rel = tau, args.spread_mode, args.spread_rel
        self.ts = None
        if args.spread_q is not None:
            sig = self.signal(cal)
            pool = sig if self.mode == "busy" else sig[cal["pred"] < tau]
            self.ts = float(np.quantile(pool, args.spread_q)) if pool.size else float("inf")

    def signal(self, d):
        return d["spread"] / np.maximum(d["pred"], 1.0) if self.rel else d["spread"]

    def busy(self, d):
        m = d["pred"] >= self.tau
        if self.ts is not None and self.mode == "busy":
            m = m | (self.signal(d) >= self.ts)
        return m

    def protected_quiet(self, d):
        if self.ts is not None and self.mode == "quiet_only":
            return (d["pred"] < self.tau) & (self.signal(d) >= self.ts)
        return ~self.busy(d)


class GatedCalib:
    """Mondrian conformal margins calibrated ONLY on the busy calibration pairs:
    margin[period][decile of the busy predicted level]; optional margin for protected quiet pairs."""

    def __init__(self, cal, ev, gate, args):
        self.gate, self.args = gate, args
        self.busy_c, self.busy_e = gate.busy(cal), gate.busy(ev)
        self.prot_c, self.prot_e = gate.protected_quiet(cal), gate.protected_quiet(ev)
        self.day_c = sc.in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        self.day_e = sc.in_window(ev["slot"], args.day_start, args.day_end)[:, None]
        self.res_c = cal["truth"] - cal["pred"]
        bp = cal["pred"][self.busy_c]
        if bp.size == 0:
            raise SystemExit("no busy calibration pair: lower --busy_thr")
        self.n_bins = N_BINS
        while True:
            self.edges = np.quantile(bp, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.lb_c = np.searchsorted(self.edges, cal["pred"], side="right")
            self.lb_e = np.searchsorted(self.edges, ev["pred"], side="right")
            small = min(((self.busy_c & (self.lb_c == b) & (self.day_c == p)).sum()
                         for b in range(self.n_bins) for p in (True, False)))
            if small >= MIN_GROUP or self.n_bins == 1:
                break
            self.n_bins -= 1          # too few busy pairs for 10 deciles: use fewer bins
        if small < MIN_GROUP:
            raise SystemExit(f"busy set too small ({small} pairs in the smallest group): lower --busy_thr")

    def table(self, eps, period):
        """Margin per decile from busy calibration pairs of one period (True = window, False = other hours)."""
        return np.array([conf_q(self.res_c[self.busy_c & (self.lb_c == b) & (self.day_c == period)], eps)
                         for b in range(self.n_bins)])

    def quiet_margin(self, eps, period):
        """One margin per period from the protected quiet calibration pairs (never negative)."""
        sel = self.prot_c & (self.day_c == period)
        if sel.sum() < MIN_QUIET:
            return 0.0
        return max(0.0, float(conf_q(self.res_c[sel], eps)))

    def bound(self, ev):
        a = self.args
        q_day, q_night = self.table(a.day_eps, True), self.table(a.night_eps, False)
        busy_margin = np.where(self.day_e, q_day[self.lb_e], q_night[self.lb_e])
        if a.quiet_eps is None:
            quiet_margin = 0.0
        else:
            qq = (self.quiet_margin(a.quiet_eps, True), self.quiet_margin(a.quiet_eps, False))
            quiet_margin = np.where(self.day_e, qq[0], qq[1]) * self.prot_e
        return ev["pred"] + np.where(self.busy_e, busy_margin, quiet_margin)


# ----------------------------------------------------------------------
# Iso baselines (alpha chosen on VAL) and threshold search (on VAL)
# ----------------------------------------------------------------------
def match_alpha(pred, margin, truth, target, key):
    """Constant alpha such that the metric `key` of pred + alpha * margin equals target (bisection, on val)."""
    lo, hi = 0.0, 5.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        m = mc.compute(pred + mid * margin, truth)[key]
        below = m < target if key == "overprov_%" else m > target     # overprov grows, unmet falls with alpha
        lo, hi = (mid, hi) if below else (lo, mid)
    return 0.5 * (lo + hi)


def iso_baselines(cal, up_gate_val, up_all_val, ev, up_all_test):
    """Scale the margin of the ungated schedule by a constant alpha, matched on VAL to the gate."""
    mv = np.maximum(up_all_val - cal["pred"], 0.0)
    mt = np.maximum(up_all_test - ev["pred"], 0.0)
    gv = mc.compute(up_gate_val, cal["truth"])
    a_ov = match_alpha(cal["pred"], mv, cal["truth"], gv["overprov_%"], "overprov_%")
    a_un = match_alpha(cal["pred"], mv, cal["truth"], gv["unmet_%"], "unmet_%")
    return ev["pred"] + a_ov * mt, ev["pred"] + a_un * mt, a_ov, a_un


def tau_candidates(cal, n):
    qs = np.quantile(cal["pred"], np.linspace(0, 0.97, n))
    return np.concatenate([[-np.inf], np.unique(qs[1:])])      # -inf = every pair busy = ungated


def search_tau(cal, args):
    """Curve of the global metrics on VAL (all cells) for every candidate tau, and the chosen tau."""
    curve = []
    for tau in tau_candidates(cal, args.n_tau):
        try:
            c = GatedCalib(cal, cal, Gate(tau, cal, args), args)
        except SystemExit:
            continue
        m = mc.compute(c.bound(cal), cal["truth"])
        busy = c.busy_e
        curve.append({"tau": float(tau), **m,
                      "busy_pairs_%": float(100 * busy.mean()),
                      "busy_activity_%": float(100 * cal["truth"][busy].sum() / cal["truth"].sum())})
    base = curve[0]["unmet_%"]
    ok = [r for r in curve if r["unmet_%"] <= base + args.max_unmet_increase]
    return curve, max(ok, key=lambda r: r["tau"])


def all_cells_bounds(cal, ev, args):
    """Ungated schedule (the current proposal): bounds on ev, calibrated on cal."""
    bounds, _, _ = sc.build_bounds(sc.Calib(cal, ev, args), ev, args)
    return bounds[sc.K_SCHED]


# ----------------------------------------------------------------------
# Evaluation and report
# ----------------------------------------------------------------------
def per_cell_stats(est, truth):
    """Distribution over cells of unmet % (of each cell's own activity) and coverage."""
    u = 100 * np.maximum(truth - est, 0).sum(0) / np.maximum(truth.sum(0), 1e-12)
    cov = (truth <= est).mean(0)
    return {"unmet_med": float(np.median(u)), "unmet_p90": float(np.percentile(u, 90)),
            "unmet_p99": float(np.percentile(u, 99)), "unmet_max": float(u.max()),
            "cover_med": float(np.median(cov)), "cover_p10": float(np.percentile(cov, 10)),
            "cover_min": float(cov.min())}


def evaluate(ev, c, est, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    day = mc.day_mask(ev, args)
    prot = (c.prot_e & ~c.busy_e) if args.quiet_eps is not None else np.zeros_like(c.busy_e)
    dynamic = {"busy (pred >= tau)": c.busy_e, "quiet, protected": prot, "quiet, calm (no margin)": ~c.busy_e & ~prot}
    dynamic = {k: m for k, m in dynamic.items() if m.any()}
    scopes = {"global": {"all cells": None}, "tier": mc.tier_groups(tier, tier_names, truth.shape),
              "dynamic": dynamic}
    res = mc.evaluate_scopes(est, truth, day, args, scopes)

    info = {"tier": mc.tier_info(tier, tier_mean, tier_names, truth),
            "dynamic": {k: {"share_pairs_%": float(100 * m.mean()),
                            "share_activity_%": float(100 * truth[m].sum() / truth.sum())}
                        for k, m in dynamic.items()}}
    for i, name in enumerate(tier_names):
        info["tier"][name]["busy_pairs_%"] = float(100 * c.busy_e[:, tier == i].mean())
    return res, info, {n: per_cell_stats(e, truth) for n, e in est.items()}


def describe_gate(args, c):
    q = "no margin" if args.quiet_eps is None else f"coverage {1 - args.quiet_eps:.0%}"
    if c.gate.ts is None:
        sp = "spread: off"
    else:
        sp = (f"spread({'rel' if args.spread_rel else 'raw'}) q={args.spread_q:g} -> "
              f"{c.gate.ts:.3g} [{args.spread_mode}]")
    return sp, q


def fmt(m):
    return f"{m['cover']:6.3f} {m['unmet_%']:8.2f} {m['overprov_%']:10.2f} {m['cost_J']:8.2f}"


def print_report(res, info, per_cell, tau, args, c, alphas):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    sp, q = describe_gate(args, c)
    glob = res["global"]["all cells"]
    print(f"\nbusy threshold tau = {tau:.1f} a.u. | busy pairs {100 * c.busy_e.mean():.1f}% | "
          f"deciles used: {c.n_bins}\n{sp} | protected quiet pairs get: {q} | "
          f"iso alpha: overprov {alphas[0]:.3f}, unmet {alphas[1]:.3f}")

    print(f"\n== 1) GLOBAL, ALL {sum(t['n_cells'] for t in info['tier'].values())} CELLS "
          f"(unmet / over-prov. in % of total true activity; cover = fraction; J = weighted cost) ==")
    print(f"{'approach':40s} | " + " | ".join(f"{(w if p == 'window' else p):^35s}" for p in mc.PERIODS))
    print(f"{'':40s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'overprov%':>10s} {'J':>8s}" for _ in mc.PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_GATE else ("  <- fair ref" if n in (K_ISOO, K_ISOU) else "")
        print(f"{n:40s} | " + " | ".join(fmt(glob[n][p]) for p in mc.PERIODS) + tag)

    print("\n== 2) BY STATIC TIER of cells (train-mean activity), period = all hours ==")
    print(f"{'tier':16s} {'cells':>6s} {'mean[a.u.]':>10s} {'share act%':>10s} {'busy%':>6s} | "
          + " | ".join(f"{SHORT[k]:^21s}" for k in APPROACHES))
    print(f"{'':16s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} | "
          + " | ".join(f"{'cover':>5s} {'unmet%':>6s} {'over%':>6s}" for _ in APPROACHES))
    for name, t in info["tier"].items():
        print(f"{name:16s} {t['n_cells']:6d} {t['train_mean_au']:10.1f} {t['share_activity_%']:10.2f} "
              f"{t['busy_pairs_%']:6.1f} | " + " | ".join(
                  f"{res['tier'][name][n]['all hours']['cover']:5.3f} {res['tier'][name][n]['all hours']['unmet_%']:6.2f} "
                  f"{res['tier'][name][n]['all hours']['overprov_%']:6.2f}" for n in APPROACHES))

    print("\n== 3) BY DYNAMIC STATE at prediction time, period = all hours ==")
    for name, t in info["dynamic"].items():
        print(f"{name}: {t['share_pairs_%']:.1f}% of pairs, {t['share_activity_%']:.1f}% of total activity")
        for n in APPROACHES:
            m = res["dynamic"][name][n]["all hours"]
            print(f"   {n:40s} cover {m['cover']:.3f} | unmet {m['unmet_%']:6.2f}% | overprov {m['overprov_%']:6.2f}%")

    print("\n== 4) PER-CELL distribution (unmet % of each cell's own activity; cover = per-cell fraction) ==")
    print(f"{'approach':40s} {'unmet med':>9s} {'p90':>7s} {'p99':>7s} {'max':>7s} | "
          f"{'cov med':>7s} {'p10':>6s} {'min':>6s}")
    for n in APPROACHES:
        p = per_cell[n]
        print(f"{n:40s} {p['unmet_med']:9.2f} {p['unmet_p90']:7.2f} {p['unmet_p99']:7.2f} {p['unmet_max']:7.2f} | "
              f"{p['cover_med']:7.3f} {p['cover_p10']:6.3f} {p['cover_min']:6.3f}")

    g, a = glob[K_GATE]["all hours"], glob[mc.K_REF]["all hours"]
    io, iu = glob[K_ISOO]["all hours"], glob[K_ISOU]["all hours"]
    print(f"\nGATE vs ALL-CELLS    (all hours): over-provisioning {g['overprov_%'] - a['overprov_%']:+.2f} pp, "
          f"unmet {g['unmet_%'] - a['unmet_%']:+.2f} pp")
    print(f"GATE vs ISO-OVERPROV (all hours): unmet {g['unmet_%'] - io['unmet_%']:+.2f} pp "
          f"(over {g['overprov_%']:.2f} vs {io['overprov_%']:.2f})   [negative = gate better]")
    print(f"GATE vs ISO-UNMET    (all hours): over  {g['overprov_%'] - iu['overprov_%']:+.2f} pp "
          f"(unmet {g['unmet_%']:.2f} vs {iu['unmet_%']:.2f})   [negative = gate better]")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figure_tiers(res, info, args, path):
    """Unmet % and over-provisioning % per static tier of cells, for all approaches."""
    names = list(info["tier"].keys())
    x = np.arange(len(names) + 1)
    labels = [f"{n.replace('cells ', '')}\n({info['tier'][n]['train_mean_au']:.0f} a.u.,\n"
              f"{info['tier'][n]['share_activity_%']:.1f}% of act.)" for n in names] + ["ALL CELLS"]
    w = 0.85 / len(APPROACHES)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6))
    for ax, key, title in zip(axes, ("unmet_%", "overprov_%"),
                              ("unmet activity (lower = safer)", "over-provisioning (lower = less waste)")):
        for i, n in enumerate(APPROACHES):
            vals = [res["tier"][t][n]["all hours"][key] for t in names] + [res["global"]["all cells"][n]["all hours"][key]]
            ax.bar(x + (i - (len(APPROACHES) - 1) / 2) * w, vals, w, color=COLORS[n], label=n,
                   edgecolor="k" if n == K_GATE else "none", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("cell tier by mean train activity [a.u.] (share of total test activity)")
        ax.set_ylabel("% of the total true activity of the tier [%]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    sp = "" if args.spread_q is None else f" - spread {args.spread_mode} q={args.spread_q:g}{' rel' if args.spread_rel else ''}"
    qe = "" if args.quiet_eps is None else f" - quiet cov {1 - args.quiet_eps:.0%}"
    fig.suptitle(f"Evaluation on ALL cells by tier - horizon {args.horizon * 10} min - window "
                 f"{args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other ({1 - args.night_eps:.0%})"
                 f" - tau = {args.tau_used:.0f} a.u.{sp}{qe} - test set, all hours", fontsize=10.5)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_threshold(curve, tau, args, path):
    """Global metrics on VAL vs tau: how much the gate saves and what it costs."""
    t = np.array([r["tau"] for r in curve[1:]])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, lab in zip(axes[:2], ("overprov_%", "unmet_%"),
                            ("over-provisioning, % of total activity [%]", "unmet activity, % of total activity [%]")):
        ax.plot(t, [r[key] for r in curve[1:]], "o-", color="tab:red")
        ax.axhline(curve[0][key], color="tab:green", ls="--", label="ungated (all cells)")
        ax.axvline(tau, color="k", ls=":", label=f"chosen tau = {tau:.0f} a.u.")
        ax.set_xlabel("busy threshold tau on predicted activity [a.u.]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(t, [r["busy_pairs_%"] for r in curve[1:]], "o-", label="busy pairs, % of all (window, cell) pairs")
    ax.plot(t, [r["busy_activity_%"] for r in curve[1:]], "s-", label="activity carried by busy pairs, % of total")
    ax.axvline(tau, color="k", ls=":")
    ax.set_xlabel("busy threshold tau on predicted activity [a.u.]")
    ax.set_ylabel("[%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Choice of tau on the VAL split (all cells) - horizon {args.horizon * 10} min - window "
                 f"{args.day_start:g}-{args.day_end:g} h", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Single run and sweep (both return CSV rows)
# ----------------------------------------------------------------------
def run_one(cal, ev, args, cols):
    if args.busy_thr == "auto":
        curve, best = search_tau(cal, args)
        tau = best["tau"]
        print(f"[auto] tau = {tau:.1f} a.u. chosen on val (global unmet <= ungated + {args.max_unmet_increase} pp): "
              f"busy pairs {best['busy_pairs_%']:.1f}%, activity {best['busy_activity_%']:.1f}%")
    else:
        curve, tau = None, float(args.busy_thr)
    args.tau_used = tau

    gate = Gate(tau, cal, args)
    c = GatedCalib(cal, ev, gate, args)
    up_gate = c.bound(ev)
    up_gate_val = GatedCalib(cal, cal, gate, args).bound(cal)
    up_all, up_all_val = all_cells_bounds(cal, ev, args), all_cells_bounds(cal, cal, args)
    up_io, up_iu, a_ov, a_un = iso_baselines(cal, up_gate_val, up_all_val, ev, up_all)

    est = {mc.K_POINT: ev["pred"], mc.K_REF: up_all, K_ISOO: up_io, K_ISOU: up_iu, K_GATE: up_gate}
    tier, tier_mean, tier_names, _ = mc.static_tiers(args, cols)
    res, info, per_cell = evaluate(ev, c, est, args, tier, tier_mean, tier_names)
    print_report(res, info, per_cell, tau, args, c, (a_ov, a_un))

    h, window = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}"
    base = os.path.join(args.out_dir, f"busy_gate_h{h}_w{window}")
    figure_tiers(res, info, args, base + "_tiers.png")
    if curve is not None:
        figure_threshold(curve, tau, args, base + "_threshold.png")
    params = {K_GATE: f"tau={tau:.1f};quiet_eps={args.quiet_eps};spread_mode={args.spread_mode};"
                      f"spread_q={args.spread_q};spread_rel={int(args.spread_rel)}",
              K_ISOO: f"alpha={a_ov:.3f}", K_ISOU: f"alpha={a_un:.3f}"}
    return mc.to_rows(METHOD, h, window, res, params)


def sweep(cal, ev, args):
    """Grid of gate configurations, ranked by the VAL delta vs iso-overprov (TEST columns not used to select)."""
    h, window, truth = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}", ev["truth"]
    day = mc.day_mask(ev, args)
    up_all_val, up_all = all_cells_bounds(cal, cal, args), all_cells_bounds(cal, ev, args)
    ref = mc.evaluate({mc.K_POINT: ev["pred"], mc.K_REF: up_all}, truth, day, args)
    rows = mc.to_rows(METHOD, h, window, {"global": {"all cells": ref}})

    qe_grid = [None] + list(args.sweep_quiet_eps)
    configs = [("busy", None, False, q) for q in qe_grid]                       # plain gate (+ quiet margin)
    for rel in (False, True):
        for sq in args.sweep_spread_q:
            configs.append(("busy", sq, rel, None))                              # spread makes pairs busy
            configs += [("quiet_only", sq, rel, q) for q in args.sweep_quiet_eps]  # spread flags quiet pairs
    keep = (args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps, args.n_tau)
    args.n_tau = args.sweep_n_tau
    table = []
    for i, (mode, sq, rel, qe) in enumerate(configs, 1):
        args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps = mode, sq, rel, qe
        label = f"{mode} q={sq} rel={int(rel)} qeps={qe}"
        try:
            _, best = search_tau(cal, args)
            gate = Gate(best["tau"], cal, args)
            up_v = GatedCalib(cal, cal, gate, args).bound(cal)
            up_t = GatedCalib(cal, ev, gate, args).bound(ev)
        except SystemExit as e:
            print(f"[{i}/{len(configs)}] skipped {label}: {e}")
            continue
        up_io, up_iu, a_ov, a_un = iso_baselines(cal, up_v, up_all_val, ev, up_all)
        iso_v = cal["pred"] + a_ov * np.maximum(up_all_val - cal["pred"], 0.0)
        val_d = mc.compute(up_v, cal["truth"])["unmet_%"] - mc.compute(iso_v, cal["truth"])["unmet_%"]

        n_g, n_io, n_iu = (f"{k} [{label}]" for k in (K_GATE, K_ISOO, K_ISOU))
        res = mc.evaluate({n_g: up_t, n_io: up_io, n_iu: up_iu}, truth, day, args)
        g, io, iu = res[n_g]["all hours"], res[n_io]["all hours"], res[n_iu]["all hours"]
        params = {n_g: f"tau={best['tau']:.1f};busy_pairs={best['busy_pairs_%']:.1f}%;val_d_unmet_vs_isoover={val_d:+.3f}",
                  n_io: f"alpha={a_ov:.3f}", n_iu: f"alpha={a_un:.3f}"}
        rows += mc.to_rows(METHOD, h, window, {"global": {"all cells": res}}, params)
        table.append({"mode": mode, "q": sq, "rel": rel, "qe": qe, "tau": best["tau"], "busy": best["busy_pairs_%"],
                      "val_d": val_d, "unmet": g["unmet_%"], "over": g["overprov_%"],
                      "d_unmet": g["unmet_%"] - io["unmet_%"], "d_over": g["overprov_%"] - iu["overprov_%"]})
        r = table[-1]
        print(f"[{i}/{len(configs)}] {label} -> tau {r['tau']:.0f}, test unmet {r['unmet']:.2f} over {r['over']:.2f} | "
              f"d_unmet@iso-over {r['d_unmet']:+.3f} | d_over@iso-unmet {r['d_over']:+.3f}")
    args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps, args.n_tau = keep

    table.sort(key=lambda r: r["val_d"])
    print("\n== SWEEP (ranked by VAL delta; TEST columns are untouched by the selection; negative = gate better) ==")
    print(f"{'mode':10s} {'q':>5s} {'rel':>3s} {'q_eps':>5s} {'tau':>6s} {'busy%':>6s} | {'valD_unmet':>10s} | "
          f"{'unmet%':>6s} {'over%':>6s} {'D_unmet@isoOv':>13s} {'D_over@isoUn':>12s}")
    for r in table:
        print(f"{r['mode']:10s} {str(r['q']):>5s} {int(r['rel']):>3d} {str(r['qe']):>5s} {r['tau']:6.0f} "
              f"{r['busy']:6.1f} | {r['val_d']:+10.3f} | {r['unmet']:6.2f} {r['over']:6.2f} "
              f"{r['d_unmet']:+13.3f} {r['d_over']:+12.3f}")
    return rows


# ----------------------------------------------------------------------
def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    if args.spread_mode == "quiet_only" and args.spread_q is not None and args.quiet_eps is None and not args.sweep:
        print("[warning] --spread_mode quiet_only without --quiet_eps: flagged quiet pairs get no margin, "
              "so the run equals the plain gate")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | cells {ev['truth'].shape[1]} | "
              f"window {args.day_start:g}-{args.day_end:g} h | calibration windows {cal['pred'].shape[0]}, "
              f"evaluation windows {ev['pred'].shape[0]}")
        rows += sweep(cal, ev, args) if args.sweep else run_one(cal, ev, args, cols)
    name = "busy_gate_sweep_metrics.csv" if args.sweep else "busy_gate_metrics.csv"
    mc.write_csv(os.path.join(args.out_dir, name), rows)


if __name__ == "__main__":
    main()