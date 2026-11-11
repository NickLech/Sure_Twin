"""
Compares ways of building an UPPER bound on traffic (for the sleep-mode decision).

Split: first CAL_FRAC of the test steps = calibration, the rest = evaluation
(temporal split, no shuffling). All numbers in NORMALIZED units unless STD is set.

Run from the project folder:   python conformal_eval.py
"""
import glob
import math
import os
from statistics import NormalDist

import numpy as np

ROOT = "checkpoints/STID_MILAN"
OUTPUT_LEN = 12
NUM_CELLS = 2500
HORIZON = 0
STEPS_PER_DAY = 144
CAL_FRAC = 0.5
EPS_LIST = [0.10, 0.05, 0.01]   # target violation rate: upper bound should be below true traffic <= eps of the time
N_BINS = 10
TOD_GROUPS = 6                  # time-of-day groups (24 slots = 4h each); grouping is consistent even if test start is unaligned
SPREAD_FLOOR = 1e-4
STD = None                      # set to the scaler std to print margins in original units


def load_raw(path):
    a = np.fromfile(path, dtype=np.float32)
    per_sample = OUTPUT_LEN * NUM_CELLS
    assert a.size % per_sample == 0, f"{path}: bad size"
    return a.reshape(-1, OUTPUT_LEN, NUM_CELLS)[:, HORIZON].astype(np.float64)


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


# ------------------------------------------------------------------ load
files = sorted(glob.glob(os.path.join(ROOT, "seed*", "*", "test_results", "prediction.npy")))
assert files, f"no prediction.npy under {ROOT} (run from the project folder)"
s = s2 = None
for f in files:
    p = load_raw(f)
    s = p.copy() if s is None else s + p
    s2 = p ** 2 if s2 is None else s2 + p ** 2
n_m = len(files)
pred = s / n_m
spread = np.sqrt(np.maximum(s2 / n_m - pred ** 2, 0.0))
true = load_raw(files[0].replace("prediction.npy", "targets.npy"))
B = true.shape[0]
res = true - pred                       # positive = underestimation
cut = int(B * CAL_FRAC)
cal, ev = slice(0, cut), slice(cut, B)
unit = STD if STD is not None else 1.0
print(f"{n_m} members | {B} steps | calibration: steps 0-{cut - 1} | evaluation: steps {cut}-{B - 1}")

# level bins (from calibration predictions)
edges = np.quantile(pred[cal], np.linspace(0, 1, N_BINS + 1)[1:-1])
lb = np.searchsorted(edges, pred, side="right")
tod = np.broadcast_to(((np.arange(B) % STEPS_PER_DAY) // (STEPS_PER_DAY // TOD_GROUPS))[:, None], pred.shape)

# ----------------------------------------- does spread help beyond the level?
print("\n== SPREAD vs |ERROR| WITHIN each predicted-level decile (removes the level effect) ==")
rng = np.random.default_rng(0)
abserr = np.abs(res)
within = []
for b in range(N_BINS):
    m = lb == b
    sp, ae = spread[m], abserr[m]
    idx = rng.choice(sp.size, size=min(100_000, sp.size), replace=False)
    within.append(spearman(sp[idx], ae[idx]))
print("  per decile:", " ".join(f"{w:.2f}" for w in within))
print(f"  mean within-level Spearman = {np.mean(within):.2f}   (overall was ~0.32)")
print("  near 0 -> spread adds nothing beyond the level;  clearly > 0 -> spread is useful")


# ------------------------------------------------------------------ methods
def build(eps):
    out = {}
    out["point (no margin)"] = pred
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


def evaluate(up):
    t, u, p = true[ev], up[ev], pred[ev]
    ok = t <= u
    cov = ok.mean()
    width = (u - p).mean()
    lev = lb[ev]
    cov_lev = [ok[lev == b].mean() for b in range(N_BINS) if (lev == b).any()]
    td = tod[ev]
    cov_tod = [ok[td == g].mean() for g in range(TOD_GROUPS) if (td == g).any()]
    return cov, width, min(cov_lev), max(cov_lev), min(cov_tod), max(cov_tod)


for eps in EPS_LIST:
    print(f"\n== TARGET COVERAGE {1 - eps:.0%}  (eps = {eps}) ==")
    print(f"{'method':36s} {'coverage':>8s} {'margin':>8s} | {'level cov min/max':>18s} | {'time-of-day cov min/max':>24s}")
    for name, up in build(eps).items():
        cov, w, lo_l, hi_l, lo_t, hi_t = evaluate(up)
        print(f"{name:36s} {cov:8.3f} {w * unit:8.3f} | {lo_l:8.3f} /{hi_l:7.3f} | {lo_t:11.3f} /{hi_t:10.3f}")

print(
    "\nHow to read it:\n"
    "  coverage     : should be ~ 1-eps. Below = you would violate QoS more often than promised.\n"
    "  margin       : mean (upper bound - prediction). Smaller at equal coverage = more energy saved.\n"
    "  level cov    : coverage in the worst/best predicted-level decile -> should be uniform.\n"
    "  time-of-day  : same by 4h block -> shows whether the calibration holds across the day."
)
