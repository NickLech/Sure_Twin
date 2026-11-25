"""
Busy-gated conformal schedule + GLOBAL evaluation on ALL cells  (v2: spread signal, quiet_eps, iso baselines).

Three groups of (window, cell) pairs, decided at PREDICTION time:
    busy            : pred >= tau                                  -> full Mondrian margin (period x decile)
    quiet-protected : quiet pairs that get a reduced margin        -> margin_quiet[period] with --quiet_eps
    quiet-calm      : all other quiet pairs                        -> bound = pred (no margin)

Which quiet pairs are "protected" depends on --spread_mode:
    busy        (default, as v1) : spread >= tau_spread makes a pair BUSY (full margin). Quiet pairs are all
                                   protected by --quiet_eps (if given), otherwise no margin.
    quiet_only  (new)            : spread >= tau_spread does NOT make a pair busy. It only flags a QUIET pair
                                   as "risky"; only risky quiet pairs get the reduced margin (--quiet_eps).
                                   = "quiet cells with high variability get a smaller margin instead of zero".
tau_spread is the --spread_q quantile of the VAL signal (over all pairs in 'busy' mode, over quiet pairs in
'quiet_only' mode). --spread_rel uses spread / max(pred, 1) instead of the raw spread: the raw spread grows with
the level, so it is mostly a proxy of pred; the relative spread is the genuinely new information.

Fair verdict (new): the gate is compared on TEST with two baselines built from the ungated schedule, with the
margin scaled by ONE constant alpha chosen on VAL (never on test):
    iso-overprov : same over-provisioning as the gate (on val)  -> who has less unmet?
    iso-unmet    : same unmet as the gate (on val)              -> who has less over-provisioning?
The gate is useful only if it beats BOTH. Negative deltas = gate better.

--sweep runs a grid (spread_mode x spread_q x quiet_eps, tau auto each time), ranks the configurations by the
VAL delta w.r.t. iso-overprov and reports the TEST deltas (no selection on test).

Units: unmet % and over-provisioning % are % of the total TRUE activity of the group they refer to
(activity index in [a.u.], NOT energy); coverage is a fraction (0-1); tau and margins in [a.u.].

Usage (repository root; no change to the checkpoint arguments):
    python src/analysis/schedule_busy_gate.py --horizons 1 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*" --sweep
    # single runs
    python src/analysis/schedule_busy_gate.py ... --quiet_eps 0.5                       # all quiet pairs, 50% cov.
    python src/analysis/schedule_busy_gate.py ... --spread_mode quiet_only --spread_q 0.9 --spread_rel --quiet_eps 0.5
    python src/analysis/schedule_busy_gate.py ... --spread_mode busy --spread_q 0.98
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
from conformal_eval import N_BINS, conf_q

K_POINT = "STID point (no margin)"
K_ALL = "scheduled, all cells (current)"
K_ISOO = "iso-overprov baseline"
K_ISOU = "iso-unmet baseline"
K_GATE = "scheduled + busy gate (new)"
APPROACHES = (K_POINT, K_ALL, K_ISOO, K_ISOU, K_GATE)
SHORT = {K_POINT: "point", K_ALL: "all cells", K_ISOO: "iso-overprov", K_ISOU: "iso-unmet", K_GATE: "gate"}
COLORS = {K_POINT: "tab:blue", K_ALL: "tab:green", K_ISOO: "tab:gray", K_ISOU: "tab:orange", K_GATE: "tab:red"}
PERIODS = ("all hours", "window", "other hours")
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
                    help="violation rate for the protected quiet pairs (one margin per period). Default: no margin")
    ap.add_argument("--spread_q", type=float, default=None,
                    help="quantile (0-1) of the val ensemble-spread signal that defines tau_spread")
    ap.add_argument("--spread_mode", choices=("busy", "quiet_only"), default="busy",
                    help="busy: high spread => busy (full margin). quiet_only: high spread only flags quiet pairs "
                         "that receive the reduced --quiet_eps margin")
    ap.add_argument("--spread_rel", action="store_true",
                    help="use spread / max(pred, 1) (relative uncertainty) instead of the raw spread")
    ap.add_argument("--n_tau", type=int, default=24, help="number of candidate thresholds of the auto search")
    ap.add_argument("--tiers", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.97],
                    help="train-mean quantiles that split the cells into static tiers")
    ap.add_argument("--sweep", action="store_true",
                    help="grid over spread_mode x spread_q x quiet_eps instead of a single run")
    ap.add_argument("--sweep_spread_q", type=float, nargs="*", default=[0.9, 0.95, 0.98])
    ap.add_argument("--sweep_quiet_eps", type=float, nargs="*", default=[0.5, 0.7, 0.85])
    ap.add_argument("--sweep_n_tau", type=int, default=10, help="candidate thresholds per config in the sweep")
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.busy_thr != "auto":
        args.busy_thr = float(args.busy_thr)
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
    """Mondrian conformal margins calibrated ONLY on the busy calibration pairs.
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
            qq = (0.0, 0.0)
            quiet_margin = 0.0
        else:
            qq = (self.quiet_margin(a.quiet_eps, True), self.quiet_margin(a.quiet_eps, False))
            quiet_margin = np.where(self.day_e, qq[0], qq[1]) * self.prot_e
        margin = np.where(self.busy_e, busy_margin, quiet_margin)
        return ev["pred"] + margin, {"window": q_day, "outside": q_night, "quiet": qq}


def global_metrics(est, truth, mask=None):
    """cover (0-1), unmet %, overprov % of the total true activity inside the mask (all pairs if None)."""
    t, e = (truth, est) if mask is None else (truth[mask], est[mask])
    if t.size == 0 or t.sum() <= 0:
        return {"cover": float("nan"), "unmet_%": float("nan"), "overprov_%": float("nan"), "n": int(t.size)}
    d = e - t
    return {"cover": float((t <= e).mean()),
            "unmet_%": float(100 * np.maximum(-d, 0).sum() / t.sum()),
            "overprov_%": float(100 * np.maximum(d, 0).sum() / t.sum()),
            "n": int(t.size)}


# ----------------------------------------------------------------------
# Iso baselines (alpha chosen on VAL)
# ----------------------------------------------------------------------
def match_alpha(pred, margin, truth, target, key):
    """Constant alpha such that the metric `key` of pred + alpha * margin equals target (bisection, on val)."""
    lo, hi = 0.0, 5.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        m = global_metrics(pred + mid * margin, truth)[key]
        below = m < target if key == "overprov_%" else m > target     # overprov grows, unmet falls with alpha
        lo, hi = (mid, hi) if below else (lo, mid)
    return 0.5 * (lo + hi)


def iso_baselines(cal, up_gate_val, up_all_val, ev, up_all_test):
    """Scale the margin of the ungated schedule by a constant alpha, matched on VAL to the gate."""
    mv = np.maximum(up_all_val - cal["pred"], 0.0)
    mt = np.maximum(up_all_test - ev["pred"], 0.0)
    gv = global_metrics(up_gate_val, cal["truth"])
    a_ov = match_alpha(cal["pred"], mv, cal["truth"], gv["overprov_%"], "overprov_%")
    a_un = match_alpha(cal["pred"], mv, cal["truth"], gv["unmet_%"], "unmet_%")
    return ev["pred"] + a_ov * mt, ev["pred"] + a_un * mt, a_ov, a_un


# ----------------------------------------------------------------------
# Threshold selection (on val only)
# ----------------------------------------------------------------------
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
        up, _ = c.bound(cal)
        m = global_metrics(up, cal["truth"])
        busy = c.busy_e
        curve.append({"tau": float(tau), **m,
                      "busy_pairs_%": float(100 * busy.mean()),
                      "busy_activity_%": float(100 * cal["truth"][busy].sum() / cal["truth"].sum())})
    base = curve[0]["unmet_%"]
    ok = [r for r in curve if r["unmet_%"] <= base + args.max_unmet_increase]
    return curve, max(ok, key=lambda r: r["tau"])


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def static_tiers(args, cols):
    """Tier index (0 = quietest) of every evaluated cell, from the TRAIN mean activity (no test leakage)."""
    train = load_series(Path(args.data) / "train_data.npy")[:, cols]
    mean = train.mean(0)
    edges = np.quantile(mean, args.tiers)
    tier = np.searchsorted(edges, mean, side="right")
    cuts = [0.0] + list(args.tiers) + [1.0]
    names = [f"cells {cuts[i]:.0%}-{cuts[i + 1]:.0%}" for i in range(len(cuts) - 1)]
    return tier, mean, names


def evaluate(ev, c, est, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    day = np.broadcast_to(c.day_e, truth.shape)
    periods = {"all hours": np.ones_like(day), "window": day, "other hours": ~day}
    out = {"global": {}, "tiers": {}, "dynamic": {}, "per_cell": {}}

    for n, e in est.items():
        out["global"][n] = {p: global_metrics(e, truth, m) for p, m in periods.items()}

    tier_pairs = tier[None, :] * np.ones(truth.shape, dtype=int)
    for i, name in enumerate(tier_names):
        m_t = tier_pairs == i
        out["tiers"][name] = {
            "n_cells": int((tier == i).sum()),
            "train_mean_activity_au": float(tier_mean[tier == i].mean()),
            "share_activity_%": float(100 * truth[m_t].sum() / truth.sum()),
            "busy_pairs_%": float(100 * c.busy_e[m_t].mean()),
            "approaches": {n: {p: global_metrics(e, truth, m_t & pm) for p, pm in periods.items()}
                           for n, e in est.items()}}

    calm = ~c.busy_e & ~c.prot_e
    groups = (("busy (pred >= tau)", c.busy_e), ("quiet, protected", c.prot_e & ~c.busy_e),
              ("quiet, calm (no margin)", calm))
    for label, m_b in groups:
        if not m_b.any():
            continue
        out["dynamic"][label] = {
            "share_pairs_%": float(100 * m_b.mean()),
            "share_activity_%": float(100 * truth[m_b].sum() / truth.sum()),
            "approaches": {n: {p: global_metrics(e, truth, m_b & pm) for p, pm in periods.items()}
                           for n, e in est.items()}}

    for n, e in est.items():       # per-cell unmet % (own activity) and coverage
        d = e - truth
        u = 100 * np.maximum(-d, 0).sum(0) / np.maximum(truth.sum(0), 1e-12)
        cov = (truth <= e).mean(0)
        out["per_cell"][n] = {"unmet_%_median": float(np.median(u)), "unmet_%_p90": float(np.percentile(u, 90)),
                              "unmet_%_p99": float(np.percentile(u, 99)), "unmet_%_max": float(u.max()),
                              "cover_median": float(np.median(cov)), "cover_p10": float(np.percentile(cov, 10)),
                              "cover_min": float(cov.min())}
    return out


def describe_gate(args, c):
    q = "no margin" if args.quiet_eps is None else f"coverage {1 - args.quiet_eps:.0%}"
    if c.gate.ts is None:
        sp = "spread: off"
    else:
        sp = (f"spread({'rel' if args.spread_rel else 'raw'}) q={args.spread_q:g} -> "
              f"{c.gate.ts:.3g} [{args.spread_mode}]")
    return sp, q


def print_report(res, tau, args, c, alphas):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    sp, q = describe_gate(args, c)
    print(f"\nbusy threshold tau = {tau:.1f} a.u. | busy pairs {100 * c.busy_e.mean():.1f}% | "
          f"protected quiet pairs {100 * (c.prot_e & ~c.busy_e).mean():.1f}% | deciles used: {c.n_bins}\n"
          f"{sp} | protected quiet pairs get: {q} | iso alpha: overprov {alphas[0]:.3f}, unmet {alphas[1]:.3f}")

    print(f"\n== 1) GLOBAL, ALL {sum(t['n_cells'] for t in res['tiers'].values())} CELLS "
          f"(unmet % and over-prov. % of total true activity; cover = fraction 0-1) ==")
    print(f"{'approach':34s} | " + " | ".join(f"{(w if p == 'window' else p):^27s}" for p in PERIODS))
    print(f"{'':34s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'overprov%':>10s}" for _ in PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_GATE else ("  <- fair ref" if n in (K_ISOO, K_ISOU) else "")
        print(f"{n:34s} | " + " | ".join(
            f"{res['global'][n][p]['cover']:6.3f} {res['global'][n][p]['unmet_%']:8.2f} "
            f"{res['global'][n][p]['overprov_%']:10.2f}" for p in PERIODS) + tag)

    print("\n== 2) BY STATIC TIER of cells (train-mean activity), period = all hours ==")
    print(f"{'tier':16s} {'cells':>6s} {'mean[a.u.]':>10s} {'share act%':>10s} {'busy%':>6s} | "
          + " | ".join(f"{SHORT[k]:^21s}" for k in APPROACHES))
    print(f"{'':16s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} | "
          + " | ".join(f"{'cover':>5s} {'unmet%':>6s} {'over%':>6s}" for _ in APPROACHES))
    for name, t in res["tiers"].items():
        print(f"{name:16s} {t['n_cells']:6d} {t['train_mean_activity_au']:10.1f} {t['share_activity_%']:10.2f} "
              f"{t['busy_pairs_%']:6.1f} | " + " | ".join(
                  f"{t['approaches'][n]['all hours']['cover']:5.3f} {t['approaches'][n]['all hours']['unmet_%']:6.2f} "
                  f"{t['approaches'][n]['all hours']['overprov_%']:6.2f}" for n in APPROACHES))

    print("\n== 3) BY DYNAMIC STATE at prediction time, period = all hours ==")
    for name, t in res["dynamic"].items():
        print(f"{name}: {t['share_pairs_%']:.1f}% of pairs, {t['share_activity_%']:.1f}% of total activity")
        for n in APPROACHES:
            m = t["approaches"][n]["all hours"]
            print(f"   {n:34s} cover {m['cover']:.3f} | unmet {m['unmet_%']:6.2f}% | overprov {m['overprov_%']:6.2f}%")

    print("\n== 4) PER-CELL distribution (unmet % of each cell's own activity; cover = per-cell fraction) ==")
    print(f"{'approach':34s} {'unmet med':>9s} {'p90':>7s} {'p99':>7s} {'max':>7s} | "
          f"{'cov med':>7s} {'p10':>6s} {'min':>6s}")
    for n in APPROACHES:
        p = res["per_cell"][n]
        print(f"{n:34s} {p['unmet_%_median']:9.2f} {p['unmet_%_p90']:7.2f} {p['unmet_%_p99']:7.2f} "
              f"{p['unmet_%_max']:7.2f} | {p['cover_median']:7.3f} {p['cover_p10']:6.3f} {p['cover_min']:6.3f}")

    g = res["global"][K_GATE]["all hours"]
    a = res["global"][K_ALL]["all hours"]
    io, iu = res["global"][K_ISOO]["all hours"], res["global"][K_ISOU]["all hours"]
    print(f"\nGATE vs ALL-CELLS   (all hours): over-provisioning {g['overprov_%'] - a['overprov_%']:+.2f} pp, "
          f"unmet {g['unmet_%'] - a['unmet_%']:+.2f} pp")
    print(f"GATE vs ISO-OVERPROV (all hours): unmet {g['unmet_%'] - io['unmet_%']:+.2f} pp "
          f"(over {g['overprov_%']:.2f} vs {io['overprov_%']:.2f})   [negative = gate better]")
    print(f"GATE vs ISO-UNMET    (all hours): over  {g['overprov_%'] - iu['overprov_%']:+.2f} pp "
          f"(unmet {g['unmet_%']:.2f} vs {iu['unmet_%']:.2f})   [negative = gate better]")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figure_tiers(res, args, path):
    """Unmet % and over-provisioning % per static tier of cells, for all approaches."""
    names = list(res["tiers"].keys())
    x = np.arange(len(names) + 1)
    labels = [f"{n.replace('cells ', '')}\n({res['tiers'][n]['train_mean_activity_au']:.0f} a.u.,\n"
              f"{res['tiers'][n]['share_activity_%']:.1f}% of act.)" for n in names] + ["ALL CELLS"]
    w = 0.85 / len(APPROACHES)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6))
    for ax, key, title in zip(axes, ("unmet_%", "overprov_%"),
                              ("unmet activity (lower = safer)", "over-provisioning (lower = less waste)")):
        for i, n in enumerate(APPROACHES):
            vals = [res["tiers"][t]["approaches"][n]["all hours"][key] for t in names] + \
                   [res["global"][n]["all hours"][key]]
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
# Single run
# ----------------------------------------------------------------------
def all_cells_bounds(cal, ev, args):
    """Ungated schedule (the current proposal): bounds on ev, calibrated on cal."""
    c_all = sc.Calib(cal, ev, args)
    bounds, _, _ = sc.build_bounds(c_all, ev, args)
    return bounds[sc.K_SCHED]


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
    up_gate, tables = c.bound(ev)
    up_gate_val, _ = GatedCalib(cal, cal, gate, args).bound(cal)

    up_all = all_cells_bounds(cal, ev, args)
    up_all_val = all_cells_bounds(cal, cal, args)
    up_io, up_iu, a_ov, a_un = iso_baselines(cal, up_gate_val, up_all_val, ev, up_all)

    est = {K_POINT: ev["pred"], K_ALL: up_all, K_ISOO: up_io, K_ISOU: up_iu, K_GATE: up_gate}
    tier, tier_mean, tier_names = static_tiers(args, cols)
    res = evaluate(ev, c, est, args, tier, tier_mean, tier_names)
    print_report(res, tau, args, c, (a_ov, a_un))
    return res, curve, tau, c, tables, gate


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def sweep(cal, ev, args):
    """Grid of gate configurations. Ranked by the VAL delta vs iso-overprov, reported on TEST."""
    up_all_val = all_cells_bounds(cal, cal, args)
    up_all = all_cells_bounds(cal, ev, args)
    qe_grid = [None] + list(args.sweep_quiet_eps)
    configs = [("busy", None, False, q) for q in qe_grid]                       # v1 behaviour (+ quiet margin)
    for rel in (False, True):
        for sq in args.sweep_spread_q:
            configs += [("busy", sq, rel, None)]                                 # spread makes pairs busy
            configs += [("quiet_only", sq, rel, q) for q in args.sweep_quiet_eps]  # spread flags quiet pairs
    keep = (args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps, args.n_tau)
    args.n_tau = args.sweep_n_tau
    rows = []
    for i, (mode, sq, rel, qe) in enumerate(configs, 1):
        args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps = mode, sq, rel, qe
        try:
            _, best = search_tau(cal, args)
            gate = Gate(best["tau"], cal, args)
            up_v, _ = GatedCalib(cal, cal, gate, args).bound(cal)
            up_t, _ = GatedCalib(cal, ev, gate, args).bound(ev)
        except SystemExit as e:
            print(f"[{i}/{len(configs)}] skipped {mode} q={sq} rel={rel} quiet_eps={qe}: {e}")
            continue
        up_io, up_iu, a_ov, a_un = iso_baselines(cal, up_v, up_all_val, ev, up_all)
        iso_v_up = cal["pred"] + a_ov * np.maximum(up_all_val - cal["pred"], 0.0)
        gv, iv = global_metrics(up_v, cal["truth"]), global_metrics(iso_v_up, cal["truth"])
        g, io, iu = (global_metrics(x, ev["truth"]) for x in (up_t, up_io, up_iu))
        rows.append({"spread_mode": mode, "spread_q": sq, "spread_rel": rel, "quiet_eps": qe,
                     "tau_au": best["tau"], "busy_pairs_%": best["busy_pairs_%"],
                     "val_d_unmet_vs_isoover": gv["unmet_%"] - iv["unmet_%"],
                     "test_unmet_%": g["unmet_%"], "test_overprov_%": g["overprov_%"], "test_cover": g["cover"],
                     "test_d_unmet_vs_isoover": g["unmet_%"] - io["unmet_%"],
                     "test_d_over_vs_isounmet": g["overprov_%"] - iu["overprov_%"]})
        print(f"[{i}/{len(configs)}] {mode:10s} q={sq} rel={int(rel)} quiet_eps={qe} -> tau {best['tau']:.0f}, "
              f"test unmet {g['unmet_%']:.2f} over {g['overprov_%']:.2f} | d_unmet@iso-over "
              f"{g['unmet_%'] - io['unmet_%']:+.3f} | d_over@iso-unmet {g['overprov_%'] - iu['overprov_%']:+.3f}")
    args.spread_mode, args.spread_q, args.spread_rel, args.quiet_eps, args.n_tau = keep
    rows.sort(key=lambda r: r["val_d_unmet_vs_isoover"])
    print("\n== SWEEP (ranked by VAL delta; TEST columns are untouched by the selection; negative = gate better) ==")
    print(f"{'mode':10s} {'q':>5s} {'rel':>3s} {'q_eps':>5s} {'tau':>6s} {'busy%':>6s} | {'valD_unmet':>10s} | "
          f"{'unmet%':>6s} {'over%':>6s} {'D_unmet@isoOv':>13s} {'D_over@isoUn':>12s}")
    for r in rows:
        print(f"{r['spread_mode']:10s} {str(r['spread_q']):>5s} {int(r['spread_rel']):>3d} {str(r['quiet_eps']):>5s} "
              f"{r['tau_au']:6.0f} {r['busy_pairs_%']:6.1f} | {r['val_d_unmet_vs_isoover']:+10.3f} | "
              f"{r['test_unmet_%']:6.2f} {r['test_overprov_%']:6.2f} {r['test_d_unmet_vs_isoover']:+13.3f} "
              f"{r['test_d_over_vs_isounmet']:+12.3f}")
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
    runs = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | cells {ev['truth'].shape[1]} | "
              f"window {args.day_start:g}-{args.day_end:g} h | calibration windows {cal['pred'].shape[0]}, "
              f"evaluation windows {ev['pred'].shape[0]}")
        if args.sweep:
            rows = sweep(cal, ev, args)
            path = os.path.join(args.out_dir, f"busy_gate_sweep_h{h * 10}_w{args.day_start:g}-{args.day_end:g}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Saved {path}")
            continue

        res, curve, tau, c, tables, gate = run_one(cal, ev, args, cols)
        base = os.path.join(args.out_dir, f"busy_gate_h{h * 10}_w{args.day_start:g}-{args.day_end:g}")
        figure_tiers(res, args, base + "_tiers.png")
        if curve is not None:
            figure_threshold(curve, tau, args, base + "_threshold.png")
        mpath = Path("results/tables") / f"busy_gate_margins_h{h * 10}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump({"horizon_min": h * 10, "window_hours": [args.day_start, args.day_end],
                       "eps_window": args.day_eps, "eps_outside": args.night_eps,
                       "busy_threshold_au": tau,
                       "spread_mode": args.spread_mode, "spread_relative": bool(args.spread_rel),
                       "spread_quantile": args.spread_q, "spread_threshold": gate.ts,
                       "quiet_eps": args.quiet_eps,
                       "decile_edges": [float(e) for e in c.edges],
                       "margin_window": [float(q) for q in tables["window"]],
                       "margin_outside": [float(q) for q in tables["outside"]],
                       "quiet_margin_window": tables["quiet"][0], "quiet_margin_outside": tables["quiet"][1],
                       "usage": "signal = spread (or spread/max(pred,1) if spread_relative). "
                                "busy = pred >= busy_threshold_au (| signal >= spread_threshold if mode 'busy'); "
                                "busy: bound = pred + margin_<period>[searchsorted(decile_edges, pred, side='right')]. "
                                "protected quiet = quiet & signal >= spread_threshold (mode 'quiet_only') or every "
                                "quiet pair (mode 'busy'): bound = pred + quiet_margin_<period> (if quiet_eps). "
                                "other quiet pairs: bound = pred"},
                      f, indent=2)
        print(f"Saved {mpath}")
        runs.append({"horizon_min": h * 10, "window": f"{args.day_start:g}-{args.day_end:g}",
                     "busy_threshold_au": tau, "busy_pairs_%": float(100 * c.busy_e.mean()),
                     "threshold_curve_val": curve, "results": res})

    if args.sweep:
        return
    with open(os.path.join(args.out_dir, "busy_gate_all.json"), "w") as f:
        json.dump({"eps_window": args.day_eps, "eps_outside": args.night_eps, "quiet_eps": args.quiet_eps,
                   "spread_mode": args.spread_mode, "spread_q": args.spread_q, "spread_rel": bool(args.spread_rel),
                   "max_unmet_increase_pp": args.max_unmet_increase, "runs": runs}, f, indent=2)
    with open(os.path.join(args.out_dir, "busy_gate_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_min", "window", "scope", "group", "approach", "period", "cover", "unmet_%", "overprov_%"])
        for r in runs:
            for n, per in r["results"]["global"].items():
                for p, m in per.items():
                    w.writerow([r["horizon_min"], r["window"], "global", "all cells", n, p,
                                m["cover"], m["unmet_%"], m["overprov_%"]])
            for scope in ("tiers", "dynamic"):
                for g, t in r["results"][scope].items():
                    for n, per in t["approaches"].items():
                        for p, m in per.items():
                            w.writerow([r["horizon_min"], r["window"], scope, g, n, p,
                                        m["cover"], m["unmet_%"], m["overprov_%"]])
    print(f"\nSaved {args.out_dir}/busy_gate_all.json / .csv")


if __name__ == "__main__":
    main()