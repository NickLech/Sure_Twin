"""
Busy-gated conformal schedule + GLOBAL evaluation on ALL cells.

Idea
  The conformal margin of schedule_conformal.py is added to every (window, cell) pair. Here it is added
  only where the cell is BUSY, and "busy" is decided at PREDICTION time from the predicted volume:

        busy(t, cell) = pred(t, cell) >= tau            [optionally: or ensemble spread >= tau_spread]

  A cell is therefore not busy "for ever": a busy cell can become quiet at night and vice versa.
  Everything else is unchanged: schedule (high coverage in the window, lower outside), Mondrian margins
  per period and per predicted-level decile. The deciles and the margins are now calibrated ONLY on the
  busy calibration pairs, so the coverage target holds inside the busy set.

      busy pair   : bound = pred + margin[period, decile(pred)]
      quiet pair  : bound = pred                    (default)
                    bound = pred + margin_quiet[period]   with --quiet_eps (smaller coverage target)

  The threshold tau is a number in activity units [a.u.]. With --busy_thr auto (default) it is chosen on
  the VAL split (never on test): the largest tau such that the GLOBAL unmet activity on val (all cells,
  volume-weighted) grows by at most --max_unmet_increase percentage points w.r.t. the ungated schedule.
  With --busy_thr 1000 (any number) tau is fixed by hand.

Evaluation (test split, ORIGINAL scale [a.u.]) is ALWAYS on all cells, never only on the busy ones:
    1) global table: all cells, per period (all hours / window / other hours)
    2) breakdown by static tier of cells (cells ranked by their mean activity on the TRAIN split)
    3) breakdown by dynamic state at prediction time (busy / quiet pairs)
    4) per-cell distribution of unmet % (is any quiet cell left unprotected?)
  Approaches compared: STID point | scheduled (all cells, current proposal) | scheduled + busy gate (new).

Units: unmet % and over-provisioning % are % of the total TRUE activity of the group they refer to
(dimensionless, activity index in [a.u.], NOT energy); coverage is a fraction (0-1); tau and margins in [a.u.].

Saved in --out_dir (default final_metrics/busy_gate):
    busy_gate_h<min>_w<start>-<end>_tiers.png       unmet % / over-provisioning % per static tier
    busy_gate_h<min>_w<start>-<end>_threshold.png   global metrics on val vs tau (chosen tau marked)
    busy_gate_all.json / busy_gate_all.csv          every number of every run
and the margin tables for run-time use in results/tables/busy_gate_margins_h<min>.json

Usage (from the repository root):
    python src/analysis/schedule_busy_gate.py --horizons 1 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python src/analysis/schedule_busy_gate.py --busy_thr 1000                 # fixed threshold [a.u.]
    python src/analysis/schedule_busy_gate.py --max_unmet_increase 0.5        # more aggressive gate
    python src/analysis/schedule_busy_gate.py --quiet_eps 0.5                 # quiet cells: 50% coverage
    python src/analysis/schedule_busy_gate.py --spread_q 0.98                 # also busy if spread is very high

--horizons are 1-based (1 = 10 min). The per-run options of schedule_conformal.py are accepted
(--day_start, --day_end, --day_eps, --night_eps, --cal_data, ...); defaults here: window 9-18 h.
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
import schedule_final_metrics as sfm
from analyse_result import kept_columns, load_series
from conformal_eval import N_BINS, conf_q

K_POINT = "STID point (no margin)"
K_ALL = "scheduled, all cells (current)"
K_GATE = "scheduled + busy gate (new)"
APPROACHES = (K_POINT, K_ALL, K_GATE)
PERIODS = ("all hours", "window", "other hours")
MIN_GROUP = 200        # minimum calibration pairs per (period, decile) group
N_TAU = 24             # candidate thresholds of the automatic search


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
                    help="violation rate for quiet pairs (one margin per period). Default: no margin")
    ap.add_argument("--spread_q", type=float, default=None,
                    help="quantile (0-1) of the val ensemble spread: pairs above it are also busy")
    ap.add_argument("--tiers", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.97],
                    help="train-mean quantiles that split the cells into static tiers")
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
    """Busy rule decided at prediction time: pred >= tau (and/or spread >= tau_spread)."""

    def __init__(self, tau, tau_spread=None):
        self.tau, self.tau_spread = tau, tau_spread

    def mask(self, d):
        m = d["pred"] >= self.tau
        if self.tau_spread is not None:
            m = m | (d["spread"] >= self.tau_spread)
        return m


class GatedCalib:
    """Mondrian conformal margins calibrated ONLY on the busy calibration pairs.
    margin[period][decile of the busy predicted level]; optional margin for quiet pairs."""

    def __init__(self, cal, ev, gate, args):
        self.gate = gate
        self.busy_c, self.busy_e = gate.mask(cal), gate.mask(ev)
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
        self.args = args

    def table(self, eps, period):
        """Margin per decile from busy calibration pairs of one period (True = window, False = other hours)."""
        return np.array([conf_q(self.res_c[self.busy_c & (self.lb_c == b) & (self.day_c == period)], eps)
                         for b in range(self.n_bins)])

    def quiet_margin(self, eps, period):
        sel = ~self.busy_c & (self.day_c == period)
        return float(conf_q(self.res_c[sel], eps)) if sel.any() else 0.0

    def bound(self, ev):
        a = self.args
        q_day, q_night = self.table(a.day_eps, True), self.table(a.night_eps, False)
        busy_margin = np.where(self.day_e, q_day[self.lb_e], q_night[self.lb_e])
        if a.quiet_eps is None:
            quiet_margin = 0.0
        else:
            quiet_margin = np.where(self.day_e, self.quiet_margin(a.quiet_eps, True),
                                    self.quiet_margin(a.quiet_eps, False))
        margin = np.where(self.busy_e, busy_margin, quiet_margin)
        return ev["pred"] + margin, (q_day, q_night)


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
# Threshold selection (on val only)
# ----------------------------------------------------------------------
def tau_candidates(cal):
    qs = np.quantile(cal["pred"], np.linspace(0, 0.97, N_TAU))
    return np.concatenate([[-np.inf], np.unique(qs[1:])])      # -inf = every pair busy = ungated


def search_tau(cal, args, tau_spread):
    """Curve of the global metrics on VAL (all cells) for every candidate tau, and the chosen tau."""
    curve = []
    for tau in tau_candidates(cal):
        try:
            c = GatedCalib(cal, cal, Gate(tau, tau_spread), args)
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
def static_tiers(args, n_cols, cols):
    """Tier index (0 = quietest) of every evaluated cell, from the TRAIN mean activity (no test leakage)."""
    train = load_series(Path(args.data) / "train_data.npy")[:, cols]
    mean = train.mean(0)
    edges = np.quantile(mean, args.tiers)
    tier = np.searchsorted(edges, mean, side="right")
    names = []
    cuts = [0.0] + list(args.tiers) + [1.0]
    for i in range(len(cuts) - 1):
        names.append(f"cells {cuts[i]:.0%}-{cuts[i + 1]:.0%}")
    return tier, mean, names


def evaluate(ev, c, up_gate, up_all, args, tier, tier_mean, tier_names):
    truth = ev["truth"]
    est = {K_POINT: ev["pred"], K_ALL: up_all, K_GATE: up_gate}
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

    for label, m_b in (("busy (pred >= tau)", c.busy_e), ("quiet (pred < tau)", ~c.busy_e)):
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


def print_report(res, tau, args, c):
    w = f"{args.day_start:g}-{args.day_end:g} h"
    print(f"\nbusy threshold tau = {tau:.1f} a.u. | busy pairs {100 * c.busy_e.mean():.1f}% | "
          f"deciles used: {c.n_bins} | quiet pairs get "
          f"{'no margin' if args.quiet_eps is None else f'coverage {1 - args.quiet_eps:.0%}'}")

    print(f"\n== 1) GLOBAL, ALL {sum(t['n_cells'] for t in res['tiers'].values())} CELLS "
          f"(unmet % and over-prov. % of total true activity; cover = fraction 0-1) ==")
    print(f"{'approach':34s} | " + " | ".join(f"{(w if p == 'window' else p):^27s}" for p in PERIODS))
    print(f"{'':34s} | " + " | ".join(f"{'cover':>6s} {'unmet%':>8s} {'overprov%':>10s}" for _ in PERIODS))
    for n in APPROACHES:
        tag = "  <- new" if n == K_GATE else ""
        print(f"{n:34s} | " + " | ".join(
            f"{res['global'][n][p]['cover']:6.3f} {res['global'][n][p]['unmet_%']:8.2f} "
            f"{res['global'][n][p]['overprov_%']:10.2f}" for p in PERIODS) + tag)

    print("\n== 2) BY STATIC TIER of cells (train-mean activity), period = all hours ==")
    print(f"{'tier':16s} {'cells':>6s} {'mean[a.u.]':>10s} {'share act%':>10s} {'busy%':>6s} | "
          + " | ".join(f"{k.split(',')[0][:16]:^24s}" for k in APPROACHES))
    print(f"{'':16s} {'':>6s} {'':>10s} {'':>10s} {'':>6s} | "
          + " | ".join(f"{'cover':>6s} {'unmet%':>7s} {'over%':>7s}" for _ in APPROACHES))
    for name, t in res["tiers"].items():
        print(f"{name:16s} {t['n_cells']:6d} {t['train_mean_activity_au']:10.1f} {t['share_activity_%']:10.2f} "
              f"{t['busy_pairs_%']:6.1f} | " + " | ".join(
                  f"{t['approaches'][n]['all hours']['cover']:6.3f} {t['approaches'][n]['all hours']['unmet_%']:7.2f} "
                  f"{t['approaches'][n]['all hours']['overprov_%']:7.2f}" for n in APPROACHES))

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
    g, a = res["global"][K_GATE]["all hours"], res["global"][K_ALL]["all hours"]
    print(f"\nGATE vs ALL-CELLS (global, all hours): over-provisioning {g['overprov_%'] - a['overprov_%']:+.2f} pp, "
          f"unmet {g['unmet_%'] - a['unmet_%']:+.2f} pp")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figure_tiers(res, args, path):
    """Unmet % and over-provisioning % per static tier of cells, for the three approaches."""
    names = list(res["tiers"].keys())
    colors = {K_POINT: "tab:blue", K_ALL: "tab:green", K_GATE: "tab:red"}
    x = np.arange(len(names) + 1)
    labels = [f"{n.replace('cells ', '')}\n({res['tiers'][n]['train_mean_activity_au']:.0f} a.u.,\n"
              f"{res['tiers'][n]['share_activity_%']:.1f}% of act.)" for n in names] + ["ALL CELLS"]
    w = 0.8 / len(APPROACHES)
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, key, title in zip(axes, ("unmet_%", "overprov_%"),
                              ("unmet activity (lower = safer)", "over-provisioning (lower = less waste)")):
        for i, n in enumerate(APPROACHES):
            vals = [res["tiers"][t]["approaches"][n]["all hours"][key] for t in names] + \
                   [res["global"][n]["all hours"][key]]
            ax.bar(x + (i - 1) * w, vals, w, color=colors[n], label=n,
                   edgecolor="k" if n == K_GATE else "none", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("cell tier by mean train activity [a.u.] (share of total test activity)")
        ax.set_ylabel("% of the total true activity of the tier [%]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Evaluation on ALL cells by tier - horizon {args.horizon * 10} min - window "
                 f"{args.day_start:g}-{args.day_end:g} h ({1 - args.day_eps:.0%}) / other ({1 - args.night_eps:.0%})"
                 f" - busy gate tau = {args.tau_used:.0f} a.u. - test set, all hours", fontsize=11)
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
def run_one(cal, ev, args, cols):
    tau_spread = None if args.spread_q is None else float(np.quantile(cal["spread"], args.spread_q))
    if args.busy_thr == "auto":
        curve, best = search_tau(cal, args, tau_spread)
        tau = best["tau"]
        print(f"[auto] tau = {tau:.1f} a.u. chosen on val (global unmet <= ungated + {args.max_unmet_increase} pp): "
              f"busy pairs {best['busy_pairs_%']:.1f}%, activity {best['busy_activity_%']:.1f}%")
    else:
        curve, tau = None, float(args.busy_thr)
    args.tau_used = tau

    gate = Gate(tau, tau_spread)
    c = GatedCalib(cal, ev, gate, args)
    up_gate, tables = c.bound(ev)

    # reference: the current proposal (same period schedule, deciles over ALL pairs)
    c_all = sc.Calib(cal, ev, args)
    bounds, _, _ = sc.build_bounds(c_all, ev, args)
    up_all = bounds[sc.K_SCHED]

    tier, tier_mean, tier_names = static_tiers(args, ev["truth"].shape[1], cols)
    res = evaluate(ev, c, up_gate, up_all, args, tier, tier_mean, tier_names)
    print_report(res, tau, args, c)
    return res, curve, tau, c, tables, tau_spread


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
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | cells {ev['truth'].shape[1]} | "
              f"window {args.day_start:g}-{args.day_end:g} h | calibration windows {cal['pred'].shape[0]}, "
              f"evaluation windows {ev['pred'].shape[0]}")
        res, curve, tau, c, tables, tau_spread = run_one(cal, ev, args, cols)
        base = os.path.join(args.out_dir, f"busy_gate_h{h * 10}_w{args.day_start:g}-{args.day_end:g}")
        figure_tiers(res, args, base + "_tiers.png")
        if curve is not None:
            figure_threshold(curve, tau, args, base + "_threshold.png")
        mpath = Path("results/tables") / f"busy_gate_margins_h{h * 10}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump({"horizon_min": h * 10, "window_hours": [args.day_start, args.day_end],
                       "eps_window": args.day_eps, "eps_outside": args.night_eps,
                       "busy_threshold_au": tau, "busy_spread_threshold_au": tau_spread,
                       "quiet_eps": args.quiet_eps,
                       "decile_edges": [float(e) for e in c.edges],
                       "margin_window": [float(q) for q in tables[0]],
                       "margin_outside": [float(q) for q in tables[1]],
                       "usage": "busy = prediction >= busy_threshold_au (or spread >= busy_spread_threshold_au); "
                                "busy: bound = prediction + margin_<period>[searchsorted(decile_edges, prediction, "
                                "side='right')]; quiet: bound = prediction (+ quiet margin if quiet_eps)"},
                      f, indent=2)
        print(f"Saved {mpath}")
        runs.append({"horizon_min": h * 10, "window": f"{args.day_start:g}-{args.day_end:g}",
                     "busy_threshold_au": tau, "busy_pairs_%": float(100 * c.busy_e.mean()),
                     "threshold_curve_val": curve, "results": res})

    with open(os.path.join(args.out_dir, "busy_gate_all.json"), "w") as f:
        json.dump({"eps_window": args.day_eps, "eps_outside": args.night_eps, "quiet_eps": args.quiet_eps,
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
