"""
Compares ways of building an UPPER bound on traffic (for the sleep-mode decision).

Two modes:
  1) separate calibration set (recommended): calibrate on the val split, evaluate on the whole test
         python conformal_eval.py --horizon 2 \
             --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
  2) fallback: no --cal_data -> first --cal_frac of the test = calibration, the rest = evaluation

Loading is shared with analyse_result.py (same scaler, last-batch fix, --exclude), so
everything is in ORIGINAL units. Horizon 0 = 10 min ahead, 2 = 30 min.
"""
import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from analyse_result import (
    STEPS_PER_DAY,
    WEEKDAYS,
    cross_check_with_basicts,
    find_run_dirs,
    fix_misplaced_last_batch,
    kept_columns,
    load_results,
    load_series,
)

EPS_LIST = [0.10, 0.05, 0.01]   # target violation rate
N_BINS = 10                     # predicted-level deciles
TOD_GROUPS = 6                  # 6 groups of 4 hours (real clock time)
SPREAD_FLOOR = 1e-3             # avoids dividing by ~0 spread (original units)
MIN_DAY_WINDOWS = 36            # days with fewer windows (partial days) are ignored in day min/max
DAY_TABLE_EPS = 0.05            # eps used for the per-day table


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
    return ap.parse_args()


def conf_q(scores, eps):
    """Finite-sample conformal quantile (upper), numpy-version agnostic."""
    scores = np.asarray(scores).ravel()
    n = scores.size
    idx = min(n - 1, math.ceil((n + 1) * (1 - eps)) - 1)
    return np.partition(scores, idx)[idx]


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def load_split(data_dir, ckpts, args, tag):
    """Ensemble prediction, spread and truth at one horizon, original units."""
    IN, OUT, H = args.input_len, args.output_len, args.horizon
    data = Path(data_dir)
    train = load_series(data / "train_data.npy")
    test = load_series(data / "test_data.npy")
    test_ts = np.load(data / "test_timestamps.npy")
    mean, std = train.mean(0), train.std(0) + 1e-8      # per-node z-score
    N = test.shape[1]
    S_all = test.shape[0] - IN - OUT + 1

    win = sliding_window_view(test, IN + OUT, axis=0)    # [S, N, IN+OUT]
    true = win[:, :, IN:].transpose(0, 2, 1)             # [S, OUT, N]

    run_dirs = find_run_dirs(ckpts)
    print(f"\n[{tag}] {len(run_dirs)} run(s) from {ckpts}")
    shape = (S_all, OUT, N)
    targets = load_results(Path(run_dirs[0]) / "test_results" / "targets.npy", shape, mean, std)
    preds = np.stack([load_results(Path(d) / "test_results" / "prediction.npy", shape, mean, std)
                      for d in run_dirs])                # [M, S, OUT, N]
    targets, preds, true, keep = fix_misplaced_last_batch(targets, preds, true)
    print(f"[{tag}] windows evaluated: {true.shape[0]} of {S_all}; "
          f"max |targets - raw| = {np.abs(targets - true).max():.4g}")
    cross_check_with_basicts(run_dirs, preds, true, std)

    cols = kept_columns(args, N)
    P = preds[:, :, H, :][..., cols].astype(np.float64)  # [M, S, cells]
    n_m = P.shape[0]
    out = {
        "pred": P.mean(0),
        "spread": P.std(0),
        "truth": true[:, H][:, cols].astype(np.float64),
        "members": n_m,
    }
    del preds, P

    # target time of each kept window: row of test_ts it predicts
    t = np.flatnonzero(keep) + IN + H
    off = int(round(test_ts[0, 0] * STEPS_PER_DAY))      # time-of-day slot of the first test row
    out["slot"] = (t + off) % STEPS_PER_DAY
    out["day"] = (t + off) // STEPS_PER_DAY
    out["dow"] = np.round(test_ts[t, 1] * 7).astype(int) % 7
    return out


def subset(d, sl):
    return {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def main():
    args = parse_args()
    H = args.horizon

    ev = load_split(args.data, args.ckpts, args, "evaluation")
    if args.cal_data:
        if not args.cal_ckpts:
            raise SystemExit("--cal_data needs --cal_ckpts")
        cal = load_split(args.cal_data, args.cal_ckpts, args, "calibration")
        if cal["members"] != ev["members"]:
            raise SystemExit(f"different number of members: calibration {cal['members']} vs evaluation {ev['members']}")
        mode = "calibration = separate split, evaluation = whole test"
    else:
        S = ev["pred"].shape[0]
        cut = int(S * args.cal_frac)
        cal, ev = subset(ev, slice(0, cut)), subset(ev, slice(cut, S))
        mode = f"FALLBACK: calibration = first {args.cal_frac:.0%} of test, evaluation = the rest"
    print(f"\n{ev['members']} members | horizon {(H + 1) * 10} min | ORIGINAL units | {mode}")
    print(f"calibration windows: {cal['pred'].shape[0]} | evaluation windows: {ev['pred'].shape[0]}")

    # predicted-level bins, defined on the calibration set
    edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    lb_c = np.searchsorted(edges, cal["pred"], side="right")
    lb_e = np.searchsorted(edges, ev["pred"], side="right")
    res_c = cal["truth"] - cal["pred"]                   # positive = underestimation
    res_e = ev["truth"] - ev["pred"]

    # ------------------------------------------------ does spread help beyond the level?
    print("\n== SPREAD vs |ERROR| WITHIN each predicted-level decile (evaluation set) ==")
    rng = np.random.default_rng(0)
    within = []
    for b in range(N_BINS):
        m = lb_e == b
        sp, ae = ev["spread"][m], np.abs(res_e)[m]
        idx = rng.choice(sp.size, size=min(100_000, sp.size), replace=False)
        within.append(spearman(sp[idx], ae[idx]))
    print("  per decile:", " ".join(f"{w:.2f}" for w in within))
    print(f"  mean within-level Spearman = {np.mean(within):.2f}")
    print("  near 0 -> spread adds nothing beyond the level;  clearly > 0 -> spread is useful")

    # ------------------------------------------------ methods (calibrated on cal, applied to ev)
    def build(eps):
        out = {"point (no margin)": ev["pred"]}
        z = NormalDist().inv_cdf(1 - eps)
        out["gaussian ensemble (uncalibrated)"] = ev["pred"] + z * ev["spread"]
        out["conformal global"] = ev["pred"] + conf_q(res_c, eps)
        qb = np.array([conf_q(res_c[lb_c == b], eps) for b in range(N_BINS)])
        out["conformal by level"] = ev["pred"] + qb[lb_e]
        sp_c, sp_e = cal["spread"] + SPREAD_FLOOR, ev["spread"] + SPREAD_FLOOR
        sc = res_c / sp_c
        out["conformal / spread"] = ev["pred"] + conf_q(sc, eps) * sp_e
        qnb = np.array([conf_q(sc[lb_c == b], eps) for b in range(N_BINS)])
        out["conformal by level / spread"] = ev["pred"] + qnb[lb_e] * sp_e
        return out

    tod_e = ev["slot"] // (STEPS_PER_DAY // TOD_GROUPS)
    days_e = np.unique(ev["day"])
    mean_true = ev["truth"].mean()

    def group_cov(ok, labels, values, min_windows=1):
        covs = []
        for g in values:
            m = labels == g
            if m.sum() >= min_windows:
                covs.append(ok[m].mean())
        return (min(covs), max(covs)) if covs else (np.nan, np.nan)

    def evaluate(up):
        ok = ev["truth"] <= up
        width = (up - ev["pred"]).mean()
        lo_l, hi_l = group_cov(ok, lb_e, range(N_BINS))
        lo_t, hi_t = group_cov(ok, np.broadcast_to(tod_e[:, None], ok.shape), range(TOD_GROUPS))
        day_lab = ev["day"][:, None]
        lo_d, hi_d = (np.nan, np.nan)
        covs = [ok[ev["day"] == d].mean() for d in days_e if (ev["day"] == d).sum() >= MIN_DAY_WINDOWS]
        if covs:
            lo_d, hi_d = min(covs), max(covs)
        return ok.mean(), width, width / mean_true, lo_l, hi_l, lo_t, hi_t, lo_d, hi_d

    for eps in EPS_LIST:
        print(f"\n== TARGET COVERAGE {1 - eps:.0%}  (eps = {eps}) ==")
        print(f"{'method':34s} {'cover':>6s} {'margin':>7s} {'marg%':>6s} | {'level min/max':>13s} | "
              f"{'clock min/max':>13s} | {'day min/max':>13s}")
        for name, up in build(eps).items():
            cov, w, wr, lo_l, hi_l, lo_t, hi_t, lo_d, hi_d = evaluate(up)
            print(f"{name:34s} {cov:6.3f} {w:7.2f} {100 * wr:5.1f}% | {lo_l:6.3f}/{hi_l:6.3f} | "
                  f"{lo_t:6.3f}/{hi_t:6.3f} | {lo_d:6.3f}/{hi_d:6.3f}")

    # ------------------------------------------------ per-day table (anomalous days?)
    up = build(DAY_TABLE_EPS)["conformal by level"]
    ok = ev["truth"] <= up
    print(f"\n== PER-DAY on the evaluation set: conformal by level, eps = {DAY_TABLE_EPS} (target coverage {1 - DAY_TABLE_EPS:.0%}) ==")
    print(f"{'day':>5s} {'weekday':>8s} {'windows':>8s} {'MAE':>8s} {'mean traffic':>13s} {'coverage':>9s}")
    for d in days_e:
        m = ev["day"] == d
        wd = WEEKDAYS[int(ev["dow"][m][0])]
        note = "" if m.sum() >= MIN_DAY_WINDOWS else "  (partial day)"
        print(f"{int(d - days_e[0]):5d} {wd:>8s} {int(m.sum()):8d} {np.abs(res_e[m]).mean():8.2f} "
              f"{ev['truth'][m].mean():13.1f} {ok[m].mean():9.3f}{note}")

    print(
        "\nHow to read it:\n"
        "  cover        : should be ~ 1-eps. Below = more QoS violations than promised.\n"
        "  margin       : mean (upper bound - prediction), original units; marg% = relative to mean traffic.\n"
        "                 Smaller at equal coverage = more energy saved.\n"
        "  level / clock / day min/max : worst and best coverage across predicted-level deciles,\n"
        "                 4-hour blocks of the clock, and days (partial days ignored). Should be uniform.\n"
        "  per-day table: shows whether unusual days (weekend, holidays) lose the guarantee."
    )


if __name__ == "__main__":
    main()
