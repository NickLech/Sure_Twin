"""
Shared metrics and utilities of the extension scripts (schedule_busy_gate.py, schedule_adaptive_bins.py), so that
their results can be compared with each other and with the reference (schedule_conformal.py / schedule_final_metrics.py).

Definitions (ORIGINAL scale [a.u.], identical to schedule_final_metrics.py):
    unmet_%     100 * sum(max(truth - est, 0)) / sum(truth)
    overprov_%  100 * sum(max(est - truth, 0)) / sum(truth)
    cover       fraction of (window, cell) pairs with truth <= est
    cost_J      lam * unmet_% + (1 - lam) * overprov_%, lam = --lam inside the window, --lam_night outside
                (defaults 1 - day_eps and 1 - night_eps); "all hours" = s * J_window + (1 - s) * J_other,
                s = share of the true activity that falls inside the window (exact)
Every metric is computed per period (all hours / window / other hours) inside a group of (window, cell) pairs:
    scope "global"   group "all cells"
    scope "tier"     one group per static tier of cells (ranked by their TRAIN mean activity, no test leakage)
    scope "dynamic"  groups decided at prediction time (method specific)

Fair reference ("frontier"): the reference schedule with its margin scaled by ONE constant alpha,
    bound = pred + alpha * max(reference_bound - pred, 0).
iso_baselines() finds the alpha that gives the same over-provisioning (iso-overprov) or the same unmet (iso-unmet)
as an evaluated method, on the evaluated data itself. It is a diagnostic read-off of the frontier, not a deployable
method: a method is useful only if it lies below that curve (less unmet at equal over, or less over at equal unmet).

CSV schema (one file per method, NaN where a metric does not apply):
    method, horizon_min, window, scope, group, approach, period, cover, unmet_%, overprov_%, cost_J, switch_%, params
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schedule_conformal as sc
from analyse_result import kept_columns, load_series

PERIODS = ("all hours", "window", "other hours")
K_POINT = "STID point prediction (no confidence)"   # same label as schedule_final_metrics.py
K_REF = sc.K_SCHED                                    # "scheduled, calibrated per period" (the proposal)
COLUMNS = ["method", "horizon_min", "window", "scope", "group", "approach", "period",
           "cover", "unmet_%", "overprov_%", "cost_J", "switch_%", "params"]
NAN = float("nan")


# ----------------------------------------------------------------------
# Arguments and data loading
# ----------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--lam", type=float, default=None,
                    help="weight of unmet in J = lam*unmet + (1-lam)*overprov inside the window; default 1 - day_eps")
    ap.add_argument("--lam_night", type=float, default=None,
                    help="same, outside the window; default 1 - night_eps")
    ap.add_argument("--tiers", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.97],
                    help="train-mean quantiles that split the cells into static tiers")


def set_lambdas(args):
    """Resolve the lambda defaults (call after args.day_eps / args.night_eps are known)."""
    if args.lam is None:
        args.lam = 1 - args.day_eps
    if args.lam_night is None:
        args.lam_night = 1 - args.night_eps
    if not (0 <= args.lam <= 1 and 0 <= args.lam_night <= 1):
        raise SystemExit("--lam and --lam_night must be in [0, 1]")


def load_horizon(args):
    """Calibration data, evaluation data and kept cell columns for args.horizon."""
    cal, ev = sc.load_cal_ev(args)
    cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
    return cal, ev, cols


def ref_bounds(cal, ev, args):
    """Reference (ungated 10-decile) schedule: bound on `ev`, calibrated on `cal`."""
    bounds, _, _ = sc.build_bounds(sc.Calib(cal, ev, args), ev, args)
    return bounds[sc.K_SCHED]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def day_mask(ev, args):
    """[windows, cells] bool: True where the forecast target is inside the high-coverage window."""
    return np.broadcast_to(sc.in_window(ev["slot"], args.day_start, args.day_end)[:, None], ev["truth"].shape)


def compute(est, truth, mask=None):
    """cover (0-1), unmet % and over-provisioning % of the total true activity inside the mask."""
    t, e = (truth, est) if mask is None else (truth[mask], est[mask])
    if t.size == 0 or t.sum() <= 0:
        return {"cover": NAN, "unmet_%": NAN, "overprov_%": NAN}
    d = e - t
    return {"cover": float((t <= e).mean()),
            "unmet_%": float(100 * np.maximum(-d, 0).sum() / t.sum()),
            "overprov_%": float(100 * np.maximum(d, 0).sum() / t.sum())}


def evaluate(estimates, truth, day, args, group=None, extra=None):
    """{approach: {period: metrics}} inside `group` (bool [windows, cells]; None = all pairs).
    extra(approach, mask) may return more metrics (e.g. switch_%) that are merged into each period."""
    g = np.ones(truth.shape, dtype=bool) if group is None else group
    masks = {"all hours": g, "window": g & day, "other hours": g & ~day}
    total = truth[g].sum()
    share = truth[masks["window"]].sum() / total if total > 0 else NAN
    lam = {"window": args.lam, "other hours": args.lam_night}
    out = {}
    for n, est in estimates.items():
        per = {}
        for p, m in masks.items():
            per[p] = compute(est, truth, m)
            if extra is not None:
                per[p].update(extra(n, m))
        j = {p: lam[p] * per[p]["unmet_%"] + (1 - lam[p]) * per[p]["overprov_%"] for p in lam}
        per["window"]["cost_J"], per["other hours"]["cost_J"] = j["window"], j["other hours"]
        per["all hours"]["cost_J"] = share * j["window"] + (1 - share) * j["other hours"]
        out[n] = per
    return out


def evaluate_scopes(estimates, truth, day, args, scopes, extra=None):
    """scopes = {scope: {group: mask or None}} -> {scope: {group: {approach: {period: metrics}}}}."""
    return {s: {g: evaluate(estimates, truth, day, args, m, extra) for g, m in groups.items()}
            for s, groups in scopes.items()}


def per_cell_stats(est, truth):
    """Distribution over cells of unmet % (of each cell's own activity) and coverage."""
    u = 100 * np.maximum(truth - est, 0).sum(0) / np.maximum(truth.sum(0), 1e-12)
    cov = (truth <= est).mean(0)
    return {"unmet_med": float(np.median(u)), "unmet_p90": float(np.percentile(u, 90)),
            "unmet_p99": float(np.percentile(u, 99)), "unmet_max": float(u.max()),
            "cover_med": float(np.median(cov)), "cover_p10": float(np.percentile(cov, 10)),
            "cover_min": float(cov.min())}


def fmt(m):
    """'cover unmet% over%' of one metrics dict (24 chars)."""
    return f"{m['cover']:6.3f} {m['unmet_%']:8.2f} {m['overprov_%']:8.2f}"


# ----------------------------------------------------------------------
# Frontier: reference schedule with a scaled margin
# ----------------------------------------------------------------------
def match_alpha(pred, margin, truth, key, target):
    """alpha such that `key` of pred + alpha * margin equals target (bisection).
    overprov grows with alpha, unmet falls with alpha."""
    lo, hi = 0.0, 5.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        m = compute(pred + mid * margin, truth)[key]
        below = m < target if key == "overprov_%" else m > target
        lo, hi = (mid, hi) if below else (lo, mid)
    return 0.5 * (lo + hi)


def iso_baselines(pred, ref_bound, truth, est):
    """Scaled reference with the same over-provisioning / the same unmet as `est`.
    Returns (iso_overprov_bound, iso_unmet_bound, alpha_overprov, alpha_unmet)."""
    margin = np.maximum(ref_bound - pred, 0.0)
    m = compute(est, truth)
    a_o = match_alpha(pred, margin, truth, "overprov_%", m["overprov_%"])
    a_u = match_alpha(pred, margin, truth, "unmet_%", m["unmet_%"])
    return pred + a_o * margin, pred + a_u * margin, a_o, a_u


def figure_frontier(path, pred, ref_bound, truth, est, title):
    """Unmet vs over-provisioning (all cells, all hours): scaled reference (dashed) vs the evaluated approaches.
    Points below / left of the dashed curve beat the reference."""
    margin = np.maximum(ref_bound - pred, 0.0)
    pts = [compute(pred + a * margin, truth) for a in np.linspace(0.0, 1.6, 33)]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([p["overprov_%"] for p in pts], [p["unmet_%"] for p in pts], "--", color="gray",
            label="reference schedule x alpha")
    for n, e in est.items():
        m = compute(e, truth)
        ax.scatter(m["overprov_%"], m["unmet_%"], s=70, zorder=3, edgecolors="k", label=n)
    ax.set_xlabel("over-provisioning [% of total true activity]")
    ax.set_ylabel("unmet [% of total true activity]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Static tiers of cells (train mean activity: no test leakage)
# ----------------------------------------------------------------------
def static_tiers(args, cols):
    """Tier index (0 = quietest) of every evaluated cell, its train mean, tier names, and the train series."""
    train = load_series(Path(args.data) / "train_data.npy")[:, cols]
    mean = train.mean(0)
    tier = np.searchsorted(np.quantile(mean, args.tiers), mean, side="right")
    cuts = [0.0] + list(args.tiers) + [1.0]
    names = [f"cells {cuts[i]:.0%}-{cuts[i + 1]:.0%}" for i in range(len(cuts) - 1)]
    return tier, mean, names, train


def tier_groups(tier, names, shape):
    return {n: np.broadcast_to((tier == i)[None, :], shape) for i, n in enumerate(names)}


def tier_info(tier, mean, names, truth):
    out = {}
    for i, n in enumerate(names):
        cell = tier == i
        out[n] = {"n_cells": int(cell.sum()), "train_mean_au": float(mean[cell].mean()),
                  "share_activity_%": float(100 * truth[:, cell].sum() / truth.sum())}
    return out


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def to_rows(method, horizon_min, window, res, params=None):
    """Flatten {scope: {group: {approach: {period: metrics}}}} into CSV rows. params: {approach: text}."""
    params = params or {}
    return [{"method": method, "horizon_min": horizon_min, "window": window, "scope": scope, "group": group,
             "approach": approach, "period": p, "params": params.get(approach, ""), **per[p]}
            for scope, groups in res.items() for group, apps in groups.items()
            for approach, per in apps.items() for p in PERIODS]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval=NAN)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")