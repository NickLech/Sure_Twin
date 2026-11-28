"""
Conformal risk control (CRC) on the true sleep cost, instead of a coverage level 1 - eps.

You choose a target alpha for the operational risk (--risk unmet: unmet traffic as % of total y; or
--risk viol_all: sleeping cell-steps with y >= theta as % of ALL cell-steps) and the method returns the
SMALLEST margin lambda >= 0 whose risk is guaranteed (in expectation) below alpha.

Setup (per theta, per horizon, calibration = val split):
    est      = prediction + lambda * shape        (shape = 1: constant margin; shape = level-conformal table: "level-shaped")
    asleep   = est < theta                         (instant rule)
    loss_t   = per calibration window t, in [0, B], non-increasing in lambda:
                 unmet:     sum_cells max(y - theta, 0) * asleep / Tbar        (Tbar = mean over windows of sum_cells y,
                                                                                so the mean loss IS the unmet fraction)
                 viol_all:  mean_cells (asleep & y >= theta)
    pick     = smallest lambda with  n/(n+1) * mean_t loss_t(lambda) + B/(n+1) <= alpha      (Angelopoulos et al., CRC)

Notes
  * B must bound the loss a priori. Here B = max_t loss_t(0), the largest window loss seen on calibration
    (loss is maximal at lambda = 0). This is a heuristic (--B overrides it): it assumes no unseen window is worse.
  * Smallest certifiable target is about B/(n+1); a target below that is "infeasible" and lambda is set so that
    nobody sleeps.
  * The minimum dwell time (hysteresis, see schedule_hysteresis.py) only removes sleeping cell-steps, so it can
    only lower the risk: the same lambda stays valid with any dwell.
  * Windows are time-correlated: the guarantee is marginal, not per day. Calibration (val) and test are
    different periods, so the realised risk can drift.

Saved in --out_dir (default results/costs):
    crc_frontier_h<min>.png   rows = dwell, columns = theta: net sleep vs realised risk (CRC vs level conformal)
    crc_control_h<min>.png    realised risk on test vs target alpha (the diagonal = perfect control)
    crc.csv                   every (horizon, kappa, dwell, theta, approach, alpha / eps)

Usage (from the repository root; other options as in schedule_costs.py):
    python src/analysis/schedule_crc.py --horizons 1 3 --risk unmet --kappas 3 6 12 --kappa 6 --dwell 1 3 6 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
Needs schedule_hysteresis.py, schedule_costs.py, schedule_conformal.py, costs.py (same folder).
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import costs
import schedule_conformal as sc
import schedule_costs as scost
from analyse_result import STEPS_PER_DAY
from analysis.discarded_cost.schedule_hysteresis import ORACLE_LABEL, hysteresis

CRC_C, CRC_L = "STID + CRC constant margin", "STID + CRC level-shaped margin"
KEEP = {scost.point("STID"), scost.level("STID"), scost.point("last value"), scost.ORACLE}


def own_args():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--risk", choices=["unmet", "viol_all"], default="unmet")
    ap.add_argument("--alpha", type=float, nargs="*",
                    default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2],
                    help="target risk in PERCENT (of total y for unmet, of all cell-steps for viol_all)")
    ap.add_argument("--shape_eps", type=float, default=0.1, help="eps of the level-conformal table used as shape")
    ap.add_argument("--B", type=float, default=None, help="loss bound (fraction); default = max window loss at lambda=0")
    ap.add_argument("--grid", type=int, default=150, help="number of candidate lambdas")
    ap.add_argument("--kappas", type=float, nargs="*", default=[0, 3, 6, 12])
    ap.add_argument("--kappa", type=float, default=6, help="kappa of the frontier figure")
    ap.add_argument("--dwell", type=int, nargs="*", default=[1, 3, 6])
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return own


def candidate_lambdas(dist, mult, grid):
    """lambda values where the decision of some cell changes: quantiles of the positive (theta - pred) / mult."""
    mask = dist > 0
    r = dist[mask] / (mult[mask] if isinstance(mult, np.ndarray) else mult)
    r = r.ravel()[::max(1, r.size // 300_000)]
    lam = np.quantile(r, np.linspace(0, 1, grid)) if r.size else np.array([0.0])
    return np.unique(np.concatenate([[0.0], lam]))


def loss_curve(cal, theta, mult, lams, risk, tbar):
    """[len(lams), windows] loss of every calibration window for every candidate lambda."""
    pred = cal["pred"].astype(np.float32)
    truth = cal["truth"].astype(np.float32)
    excess = np.maximum(truth - theta, 0)
    viol = truth >= theta
    out = np.empty((len(lams), pred.shape[0]))
    for i, lam in enumerate(lams):
        asleep = pred + lam * mult < theta
        out[i] = (excess * asleep).sum(1) / tbar if risk == "unmet" else (asleep & viol).mean(1)
    return out


def crc_pick(curve, lams, alpha, b):
    """Smallest lambda with n/(n+1) * mean loss + B/(n+1) <= alpha. Returns (lambda, feasible)."""
    n = curve.shape[1]
    ok = (n / (n + 1)) * curve.mean(1) + b / (n + 1) <= alpha
    if ok.any():
        return float(lams[np.argmax(ok)]), True
    return float(lams[-1]), False


def run(cal, ev, args, own):
    truth = ev["truth"]
    steps = ev["day"] * STEPS_PER_DAY + ev["slot"]
    thetas = {q: np.quantile(cal["truth"], q) for q in args.theta_q}
    base = [(n, e, est) for n, e, est in scost.estimates(cal, ev, args) if n in KEEP]
    c = sc.Calib(cal, ev, args)
    shape = np.maximum(c.table(own.shape_eps), 0.0)
    shape = np.maximum(shape, 1e-3 * shape.max())            # avoid division by 0 in the lambda grid
    shape_c, shape_e = shape[c.lb_c].astype(np.float32), shape[c.lb_e].astype(np.float32)
    tbar = cal["truth"].sum(1).mean()
    rows, picks = [], []
    for q, theta in thetas.items():
        dist = (theta - cal["pred"]).astype(np.float32)
        ests = [(n, e, None, est) for n, e, est in base]
        for label, mult_c, mult_e in ((CRC_C, 1.0, 1.0), (CRC_L, shape_c, shape_e)):
            lams = candidate_lambdas(dist, mult_c, own.grid)
            curve = loss_curve(cal, theta, mult_c, lams, own.risk, tbar)
            b = own.B if own.B is not None else max(float(curve[0].max()), 1e-12)
            for a in own.alpha:
                lam, feas = crc_pick(curve, lams, a / 100, b)
                picks.append({"theta_q": q, "approach": label, "alpha_%": a, "lambda": lam,
                              "feasible": feas, "B": b, "cal_risk_%": float(100 * curve[np.searchsorted(lams, lam)].mean())})
                ests.append((label, None, (a, lam, feas), ev["pred"] + lam * mult_e))
        for name, eps, meta, est in ests:
            for n in own.dwell:
                asleep = hysteresis(est, theta, steps, n)
                cst = costs.sleep_cost(truth, asleep, theta)
                wk = costs.wakeups_per_cell_day(asleep, steps)
                for k in own.kappas:
                    rows.append({"theta_q": q, "theta": theta, "approach": name, "eps": eps,
                                 "alpha_%": None if meta is None else meta[0],
                                 "lambda": None if meta is None else meta[1],
                                 "feasible": None if meta is None else meta[2],
                                 "dwell": n, "kappa": k, **cst, "wakeups_day": wk,
                                 "viol_all_%": cst["violation_%"] * cst["sleep_%"] / 100,
                                 "net_sleep_%": cst["sleep_%"] - 100 * k * wk / STEPS_PER_DAY})
    return rows, picks


def xkey_of(own):
    return "unmet_%" if own.risk == "unmet" else "viol_all_%"


def curve_xy(sub, approach, xkey):
    pts = sorted((r for r in sub if r["approach"] == approach), key=lambda r: r[xkey])
    return np.array([r[xkey] for r in pts]), np.array([r["net_sleep_%"] for r in pts])


def mean_gain(sub, a, b, xkey):
    """Mean of (a - b) net sleep at equal x over the x range covered by both curves."""
    x1, y1 = curve_xy(sub, a, xkey)
    x2, y2 = curve_xy(sub, b, xkey)
    if x1.size < 2 or x2.size < 2:
        return float("nan")
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return float("nan")
    g = np.linspace(lo, hi, 25)
    return float(np.mean(np.interp(g, x1, y1) - np.interp(g, x2, y2)))


def summary(rows, picks, args, own):
    xkey = xkey_of(own)
    print(f"\n== CRC picks (risk = {own.risk}, target alpha in %) ==")
    for q in args.theta_q:
        for label in (CRC_C, CRC_L):
            print(f"  theta quantile {q:g} | {label}")
            for p in picks:
                if p["theta_q"] == q and p["approach"] == label:
                    print(f"    alpha {p['alpha_%']:6.3f}%  lambda {p['lambda']:8.2f}  "
                          f"cal risk {p['cal_risk_%']:6.3f}%  B {100 * p['B']:.3f}%  "
                          f"{'' if p['feasible'] else 'INFEASIBLE (nobody sleeps)'}")
    n0 = own.dwell[0]
    print(f"\n== realised risk on test vs target (dwell {n0}) ==")
    for q in args.theta_q:
        print(f"  theta quantile {q:g}")
        for label in (CRC_C, CRC_L):
            sub = [r for r in rows if r["approach"] == label and r["theta_q"] == q
                   and r["dwell"] == n0 and r["kappa"] == own.kappa]
            print(f"    {label}")
            for r in sorted(sub, key=lambda r: r["alpha_%"]):
                print(f"      alpha {r['alpha_%']:6.3f}%  realised {r[xkey]:6.3f}%  sleep {r['sleep_%']:6.2f}  "
                      f"net {r['net_sleep_%']:6.2f}")
    for k in own.kappas:
        for n in own.dwell:
            print(f"\nkappa = {k:g}, dwell = {n}: mean net-sleep gain at equal {xkey}")
            for q in args.theta_q:
                sub = [r for r in rows if (r["kappa"], r["dwell"], r["theta_q"]) == (k, n, q)]
                print(f"  theta q {q:g}: CRC const - level conformal {mean_gain(sub, CRC_C, scost.level('STID'), xkey):+.2f} | "
                      f"CRC level-shaped - level conformal {mean_gain(sub, CRC_L, scost.level('STID'), xkey):+.2f}")


def figure_frontier(rows, args, own, path):
    xkey = xkey_of(own)
    nd, nt = len(own.dwell), len(args.theta_q)
    fig, axes = plt.subplots(nd, nt, figsize=(5.5 * nt, 4 * nd), squeeze=False)
    for i, n in enumerate(own.dwell):
        for j, q in enumerate(args.theta_q):
            ax = axes[i, j]
            sub = [r for r in rows if r["kappa"] == own.kappa and r["dwell"] == n and r["theta_q"] == q]
            ax.plot(*curve_xy(sub, scost.level("STID"), xkey), "o-", color="tab:orange", label=scost.level("STID"))
            ax.plot(*curve_xy(sub, CRC_C, xkey), "s-", color="tab:red", label=CRC_C)
            ax.plot(*curve_xy(sub, CRC_L, xkey), "^--", color="tab:purple", label=CRC_L)
            ax.plot(*curve_xy(sub, scost.point("STID"), xkey), "*", color="tab:orange", ms=12, label=scost.point("STID"))
            ax.plot(*curve_xy(sub, scost.point("last value"), xkey), "*", color="tab:green", ms=12,
                    label=scost.point("last value"))
            ax.axhline(curve_xy(sub, scost.ORACLE, xkey)[1][0], color="gray", ls=":", label=ORACLE_LABEL)
            ax.set_title(f"theta = {sub[0]['theta']:.0f} a.u. (q {q:g}), dwell n = {n}, kappa = {own.kappa:g}", fontsize=9)
            ax.set_xlabel(f"realised risk on test [{xkey}]")
            ax.set_ylabel("net sleep [% of cell-steps]")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    sc.prepare(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_control(rows, args, own, path):
    xkey = xkey_of(own)
    fig, axes = plt.subplots(1, len(args.theta_q), figsize=(5.5 * len(args.theta_q), 4.5), squeeze=False)
    for j, q in enumerate(args.theta_q):
        ax = axes[0, j]
        lim = [min(own.alpha) * 0.8, max(own.alpha) * 1.3]
        ax.plot(lim, lim, "k:", label="realised = target")
        for label, ls in ((CRC_C, "-"), (CRC_L, "--")):
            for n in own.dwell:
                sub = sorted((r for r in rows if r["approach"] == label and r["theta_q"] == q
                              and r["dwell"] == n and r["kappa"] == own.kappa), key=lambda r: r["alpha_%"])
                ax.plot([r["alpha_%"] for r in sub], [max(r[xkey], 1e-4) for r in sub], "o" + ls,
                        label=f"{label}, n = {n}")
        ax.set(xscale="log", yscale="log", xlabel="target alpha [%]", ylabel=f"realised on test [{xkey}]",
               title=f"theta quantile {q:g}")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=6)
    sc.prepare(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    own = own_args()
    args = scost.parse_all()
    if own.kappa not in own.kappas:
        own.kappas.append(own.kappa)
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for h in args.horizons:
        args.horizon = h
        cal, ev = sc.load_cal_ev(args)
        print(f"\n##### horizon {h * 10} min | {ev['members']} members | calibration windows {cal['pred'].shape[0]}")
        rows, picks = run(cal, ev, args, own)
        summary(rows, picks, args, own)
        figure_frontier(rows, args, own, os.path.join(args.out_dir, f"crc_frontier_h{h * 10}.png"))
        figure_control(rows, args, own, os.path.join(args.out_dir, f"crc_control_h{h * 10}.png"))
        out += [{"horizon_min": h * 10, **r} for r in rows]
    path = os.path.join(args.out_dir, "crc.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
