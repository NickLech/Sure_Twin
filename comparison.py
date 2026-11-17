"""
Compares upper-bound approaches at several coverage levels (default 95 / 90 / 85 / 80 / 75 %),
to decide which one is best. Calibration on the val split, evaluation on the whole test.

Approaches (same definitions as conformal_eval.py):
    point (no margin), gaussian ensemble (uncalibrated), conformal global,
    conformal by level, conformal / spread, conformal by level / spread

Per (approach, level) it reports, on the test set (ORIGINAL units, whole ~5 days):
    cover     fraction of (window, cell) pairs with truth <= bound        (should be >= target)
    gap       cover - target                                              (< 0 = promise broken)
    margin    mean (bound - prediction), absolute and as % of mean traffic (energy cost of safety)
    MAE_b     mean |bound - truth|     closeness of the bound to the real traffic
    WAPE_b    sum |bound - truth| / sum truth
    pinball   pinball loss at quantile (1 - eps), as % of mean traffic.  Proper score for an upper
              quantile: it penalises BOTH violations and needless slack, so one number ranks the
              approaches at a given level (lower = better).
"best" at each level = lowest pinball among the approaches that keep the promise
(cover >= target - --tol). The plain prediction (point) is the reference for closeness:
its MAE_b / WAPE_b are the MAE / WAPE of the prediction itself.

Also printed: per-day coverage and WAPE of the chosen approach (--day_method) at every level.
Saved: comparison.png, comparison.csv.

Usage:
    python comparison.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python comparison.py --horizon 2 --levels 0.95 0.90 0.85 0.80 0.75 --tol 0.01

--horizon is 0-based (0 = 10 min ahead, 2 = 30 min). Needs only analyse_result.py and conformal_eval.py.
"""
import argparse
import csv
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_result import WEEKDAYS
from conformal_eval import MIN_DAY_WINDOWS, N_BINS, SPREAD_FLOOR, conf_q, load_split, subset

POINT = "point (no margin)"
METHODS = [POINT, "gaussian ensemble (uncalibrated)", "conformal global", "conformal by level",
           "conformal / spread", "conformal by level / spread"]
COLORS = {POINT: "0.4", "gaussian ensemble (uncalibrated)": "tab:purple", "conformal global": "tab:gray",
          "conformal by level": "tab:green", "conformal / spread": "tab:orange",
          "conformal by level / spread": "tab:brown"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*", help="runs of the EVALUATION split (test)")
    ap.add_argument("--data", default="datasets/MILAN", help="dataset of the EVALUATION split (test)")
    ap.add_argument("--cal_ckpts", default=None, help="runs of the CALIBRATION split (e.g. val view)")
    ap.add_argument("--cal_data", default=None, help="dataset view of the calibration split")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=0, help="0 = first step = 10 min ahead")
    ap.add_argument("--cal_frac", type=float, default=0.5, help="only used without --cal_data")
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--csv", default="milan_raw.csv")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.95, 0.90, 0.85, 0.80, 0.75],
                    help="target coverages to compare")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="an approach keeps the promise if cover >= target - tol")
    ap.add_argument("--day_method", default="conformal by level", choices=METHODS,
                    help="approach used in the per-day tables and panel (d)")
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--out_csv", default="comparison.csv")
    return ap.parse_args()


# ----------------------------------------------------------------------
# Bounds (same recipe as conformal_eval.py)
# ----------------------------------------------------------------------
class Calib:
    """Everything that does not depend on eps: level bins and calibration residuals."""

    def __init__(self, cal, ev):
        edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
        self.lb_c = np.searchsorted(edges, cal["pred"], side="right")
        self.lb_e = np.searchsorted(edges, ev["pred"], side="right")
        self.res_c = cal["truth"] - cal["pred"]
        self.sp_c = cal["spread"] + SPREAD_FLOOR
        self.sp_e = ev["spread"] + SPREAD_FLOOR
        self.sc = self.res_c / self.sp_c


def build_bounds(cal_info, ev, eps):
    c, pred = cal_info, ev["pred"]
    z = NormalDist().inv_cdf(1 - eps)
    qb = np.array([conf_q(c.res_c[c.lb_c == b], eps) for b in range(N_BINS)])
    qnb = np.array([conf_q(c.sc[c.lb_c == b], eps) for b in range(N_BINS)])
    return {
        POINT: pred,
        "gaussian ensemble (uncalibrated)": pred + z * ev["spread"],
        "conformal global": pred + conf_q(c.res_c, eps),
        "conformal by level": pred + qb[c.lb_e],
        "conformal / spread": pred + conf_q(c.sc, eps) * c.sp_e,
        "conformal by level / spread": pred + qnb[c.lb_e] * c.sp_e,
    }


def score(up, ev, target, mean_true):
    truth, pred = ev["truth"], ev["pred"]
    ok = truth <= up
    d = truth - up
    q = target                                           # upper quantile = 1 - eps
    pin = np.mean(np.maximum(q * d, (q - 1) * d))
    return {
        "cover": ok.mean(),
        "gap": ok.mean() - target,
        "margin": (up - pred).mean(),
        "margin_pct": 100 * (up - pred).mean() / mean_true,
        "mae_b": np.abs(up - truth).mean(),
        "wape_b": np.abs(up - truth).sum() / truth.sum(),
        "pinball_pct": 100 * pin / mean_true,
    }


def per_day(up, ev, min_windows=1):
    """coverage and WAPE of the bound for each day -> list of dicts."""
    out = []
    for d in np.unique(ev["day"]):
        m = ev["day"] == d
        if m.sum() < min_windows:
            continue
        t = ev["truth"][m]
        out.append({"day": int(d), "dow": int(ev["dow"][m][0]), "n": int(m.sum()),
                    "cover": (t <= up[m]).mean(),
                    "wape": np.abs(up[m] - t).sum() / max(t.sum(), 1e-9),
                    "partial": bool(m.sum() < MIN_DAY_WINDOWS)})
    return out


# ----------------------------------------------------------------------
def compare(cal, ev, args):
    info = Calib(cal, ev)
    mean_true = ev["truth"].mean()
    levels = sorted(args.levels, reverse=True)
    rows, day_rows, best = [], {}, {}

    for cov in levels:
        eps = 1 - cov
        bounds = build_bounds(info, ev, eps)
        print(f"\n== TARGET COVERAGE {cov:.0%} (eps = {eps:.2f}) ==")
        print(f"{'approach':34s} {'cover':>6s} {'gap':>7s} {'margin':>7s} {'marg%':>6s} "
              f"{'MAE_b':>7s} {'WAPE_b':>7s} {'pinball%':>9s}")
        sc = {m: score(bounds[m], ev, cov, mean_true) for m in METHODS}
        valid = [m for m in METHODS if sc[m]["cover"] >= cov - args.tol and m != POINT]
        best[cov] = min(valid, key=lambda m: sc[m]["pinball_pct"]) if valid else None
        for m in METHODS:
            s = sc[m]
            flag = "  <- best" if m == best[cov] else ("" if s["cover"] >= cov - args.tol else "  (promise broken)")
            print(f"{m:34s} {s['cover']:6.3f} {s['gap']:+7.3f} {s['margin']:7.2f} {s['margin_pct']:5.1f}% "
                  f"{s['mae_b']:7.2f} {s['wape_b']:7.3f} {s['pinball_pct']:8.2f}%{flag}")
            rows.append({"target": cov, "approach": m, **s, "best": m == best[cov]})
        day_rows[cov] = per_day(bounds[args.day_method], ev)

    # ---- per-day tables for the chosen approach
    days = day_rows[levels[0]]
    pred_days = per_day(ev["pred"], ev)
    print(f"\n== PER DAY, approach: {args.day_method} ==")
    print("coverage per day (target in the header) | WAPE of the bound per day | WAPE of the plain prediction")
    head = f"{'day':>4s} {'wkday':>6s} {'win':>5s} | " + " ".join(f"{c:>6.0%}" for c in levels) + " | " \
           + " ".join(f"{c:>6.0%}" for c in levels) + f" | {'pred':>6s}"
    print(head)
    for i, d in enumerate(days):
        note = "  (partial day)" if d["partial"] else ""
        cv = " ".join(f"{day_rows[c][i]['cover']:6.3f}" for c in levels)
        wp = " ".join(f"{day_rows[c][i]['wape']:6.3f}" for c in levels)
        print(f"{d['day'] - days[0]['day']:4d} {WEEKDAYS[d['dow']]:>6s} {d['n']:5d} | {cv} | {wp} | "
              f"{pred_days[i]['wape']:6.3f}{note}")

    # ---- summary
    print("\n== BEST APPROACH PER LEVEL (lowest pinball among those that keep the promise) ==")
    for cov in levels:
        print(f"  {cov:.0%}: {best[cov] if best[cov] else 'none keeps the promise'}")
    print("\nHow to read it:\n"
          "  gap < 0 (below -tol)      : more violations than promised -> unsafe for sleep mode.\n"
          "  marg% / MAE_b / WAPE_b    : cost of the safety. Lower at equal cover = more energy saved.\n"
          "  pinball%                  : one number that ranks approaches at a given level.\n"
          "  moving to a higher level  : buys fewer violations, costs a larger margin (see panel b).")

    save_csv(rows, args.out_csv)
    figure(rows, levels, day_rows, days, args, best)
    return rows, best


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def figure(rows, levels, day_rows, days, args, best):
    lv = sorted(levels)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    h_min = (args.horizon + 1) * 10
    fig.suptitle(f"Upper-bound approaches vs coverage level - horizon {h_min} min - test set", fontsize=12)

    def series(m, key):
        return [next(r[key] for r in rows if r["approach"] == m and r["target"] == c) for c in lv]

    ax = axes[0, 0]                                       # (a) achieved vs target
    ax.plot([min(lv) - .02, max(lv) + .02], [min(lv) - .02, max(lv) + .02], "k--", lw=1, label="ideal")
    for m in METHODS[1:]:
        ax.plot(lv, series(m, "cover"), "o-", color=COLORS[m], label=m)
    ax.set_xlabel("target coverage")
    ax.set_ylabel("achieved coverage on test")
    ax.set_title("(a) promise kept? (on or above the dashed line)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]                                       # (b) margin
    for m in METHODS[1:]:
        ax.plot(lv, series(m, "margin_pct"), "o-", color=COLORS[m], label=m)
    ax.set_xlabel("target coverage")
    ax.set_ylabel("mean margin (% of mean traffic)")
    ax.set_title("(b) cost of the margin", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]                                       # (c) pinball
    for m in METHODS:
        ax.plot(lv, series(m, "pinball_pct"), "o-" if m != POINT else "s--", color=COLORS[m], label=m)
    ax.set_xlabel("target coverage")
    ax.set_ylabel("pinball loss (% of mean traffic)")
    ax.set_title("(c) overall score, lower = better (point shown as reference)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]                                       # (d) coverage per day
    x = np.arange(len(days))
    cmap = plt.get_cmap("viridis")
    for k, c in enumerate(levels):
        col = cmap(k / max(len(levels) - 1, 1) * 0.85)
        ax.plot(x, [d["cover"] for d in day_rows[c]], "o-", color=col, label=f"target {c:.0%}")
        ax.axhline(c, color=col, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{WEEKDAYS[d['dow']]}" + ("*" if d["partial"] else "")
                        for i, d in enumerate(days)], fontsize=8)
    ax.set_xlabel("day (* = partial day)")
    ax.set_ylabel("coverage")
    ax.set_title(f"(d) coverage per day - {args.day_method} (dotted = target)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")


def main():
    args = parse_args()
    if not all(0.5 < c < 1 for c in args.levels):
        raise SystemExit("--levels must be coverages in (0.5, 1), e.g. 0.95 0.90")
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
    print(f"\n{ev['members']} members | horizon {(args.horizon + 1) * 10} min | ORIGINAL units | "
          f"calibration windows {cal['pred'].shape[0]}, evaluation windows {ev['pred'].shape[0]}")
    compare(cal, ev, args)


if __name__ == "__main__":
    main()
