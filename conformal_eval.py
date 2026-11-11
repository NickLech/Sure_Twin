"""
Compares ways of building an UPPER bound on traffic (for the sleep-mode decision).

Data loading is shared with analyse_result.py (same scaler, same last-batch fix,
same --exclude), so results are in ORIGINAL units and on the same windows.
Split: first CAL_FRAC of the evaluated windows = calibration, the rest = evaluation
(temporal split, no shuffling).

Usage (from the project folder, next to analyse_result.py):
    python conformal_eval.py
    python conformal_eval.py --exclude 858 808 --horizon 0
"""
import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from analyse_result import (
    STEPS_PER_DAY,
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*")
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=0, help="0 = first step = 10 min ahead")
    ap.add_argument("--cal_frac", type=float, default=0.5)
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


def main():
    args = parse_args()
    IN, OUT, H = args.input_len, args.output_len, args.horizon
    data = Path(args.data)

    # ------------------------------------------------ same loading as analyse_result.py
    train = load_series(data / "train_data.npy")
    test = load_series(data / "test_data.npy")
    test_ts = np.load(data / "test_timestamps.npy")
    mean, std = train.mean(0), train.std(0) + 1e-8      # per-node z-score
    N = test.shape[1]
    S_all = test.shape[0] - IN - OUT + 1

    win = sliding_window_view(test, IN + OUT, axis=0)    # [S, N, IN+OUT]
    true = win[:, :, IN:].transpose(0, 2, 1)             # [S, OUT, N]

    run_dirs = find_run_dirs(args.ckpts)
    print(f"Found {len(run_dirs)} run(s)")
    shape = (S_all, OUT, N)
    targets = load_results(Path(run_dirs[0]) / "test_results" / "targets.npy", shape, mean, std)
    preds = np.stack([load_results(Path(d) / "test_results" / "prediction.npy", shape, mean, std)
                      for d in run_dirs])                # [M, S, OUT, N]

    targets, preds, true, keep = fix_misplaced_last_batch(targets, preds, true)
    S = true.shape[0]
    print(f"[check] windows evaluated: {S} of {S_all}; "
          f"max |targets - raw test| = {np.abs(targets - true).max():.4g}")
    cross_check_with_basicts(run_dirs, preds, true, std)

    cols = kept_columns(args, N)
    print(f"[check] cells evaluated: {len(cols)} of {N}")

    # ------------------------------------------------ keep one horizon, original units
    P = preds[:, :, H, :][..., cols].astype(np.float64)  # [M, S, cells]
    del preds
    pred = P.mean(0)
    spread = P.std(0)
    truth = true[:, H][:, cols].astype(np.float64)
    res = truth - pred                                   # positive = underestimation
    n_m = P.shape[0]

    # real clock time of the TARGET step: slot of window start + horizon
    slot0 = (test_ts[IN:IN + S_all, 0] * STEPS_PER_DAY).round().astype(int)[keep]
    slot = (slot0 + H) % STEPS_PER_DAY
    tod_step = slot // (STEPS_PER_DAY // TOD_GROUPS)
    tod = np.broadcast_to(tod_step[:, None], pred.shape)

    cut = int(S * args.cal_frac)
    cal, ev = slice(0, cut), slice(cut, S)
    print(f"{n_m} members | {S} windows | calibration: 0-{cut - 1} | evaluation: {cut}-{S - 1} "
          f"| horizon {(H + 1) * 10} min | ORIGINAL units")

    edges = np.quantile(pred[cal], np.linspace(0, 1, N_BINS + 1)[1:-1])
    lb = np.searchsorted(edges, pred, side="right")

    # ------------------------------------------------ does spread help beyond the level?
    print("\n== SPREAD vs |ERROR| WITHIN each predicted-level decile ==")
    rng = np.random.default_rng(0)
    abserr = np.abs(res)
    within = []
    for b in range(N_BINS):
        m = lb == b
        sp, ae = spread[m], abserr[m]
        idx = rng.choice(sp.size, size=min(100_000, sp.size), replace=False)
        within.append(spearman(sp[idx], ae[idx]))
    print("  per decile:", " ".join(f"{w:.2f}" for w in within))
    print(f"  mean within-level Spearman = {np.mean(within):.2f}")
    print("  near 0 -> spread adds nothing beyond the level;  clearly > 0 -> spread is useful")

    # ------------------------------------------------ methods
    def build(eps):
        out = {"point (no margin)": pred}
        z = NormalDist().inv_cdf(1 - eps)
        out["gaussian ensemble (uncalibrated)"] = pred + z * spread
        out["conformal global"] = pred + conf_q(res[cal], eps)
        qb = np.array([conf_q(res[cal][lb[cal] == b], eps) for b in range(N_BINS)])
        out["conformal by level"] = pred + qb[lb]
        sp = spread + SPREAD_FLOOR
        sc = res / sp
        out["conformal / spread"] = pred + conf_q(sc[cal], eps) * sp
        qnb = np.array([conf_q(sc[cal][lb[cal] == b], eps) for b in range(N_BINS)])
        out["conformal by level / spread"] = pred + qnb[lb] * sp
        return out

    mean_true_ev = truth[ev].mean()

    def evaluate(up):
        t, u, p = truth[ev], up[ev], pred[ev]
        ok = t <= u
        lev, td = lb[ev], tod[ev]
        cov_lev = [ok[lev == b].mean() for b in range(N_BINS) if (lev == b).any()]
        cov_tod = [ok[td == g].mean() for g in range(TOD_GROUPS) if (td == g).any()]
        width = (u - p).mean()
        return ok.mean(), width, width / mean_true_ev, min(cov_lev), max(cov_lev), min(cov_tod), max(cov_tod)

    for eps in EPS_LIST:
        print(f"\n== TARGET COVERAGE {1 - eps:.0%}  (eps = {eps}) ==")
        print(f"{'method':36s} {'coverage':>8s} {'margin':>8s} {'margin%':>8s} | "
              f"{'level cov min/max':>17s} | {'clock-group cov min/max':>23s}")
        for name, up in build(eps).items():
            cov, w, wr, lo_l, hi_l, lo_t, hi_t = evaluate(up)
            print(f"{name:36s} {cov:8.3f} {w:8.2f} {100 * wr:7.1f}% | "
                  f"{lo_l:8.3f} /{hi_l:7.3f} | {lo_t:11.3f} /{hi_t:10.3f}")

    print(
        "\nHow to read it:\n"
        "  coverage     : should be ~ 1-eps. Below = more QoS violations than promised.\n"
        "  margin       : mean (upper bound - prediction), original units; margin% = relative to mean traffic.\n"
        "                 Smaller at equal coverage = more energy saved.\n"
        "  level cov    : coverage in the worst/best predicted-level decile -> should be uniform.\n"
        "  clock-group  : same by 4h block of the real clock (00-04, 04-08, ...)."
    )


if __name__ == "__main__":
    main()
