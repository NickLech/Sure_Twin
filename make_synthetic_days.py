"""
Builds datasets/MILAN_synth_<scenario>: the REAL test series with ONE day modified,
so existing checkpoints can be evaluated on "strange" days (everything else is untouched).
train/val files are linked unchanged (same scaler), test_timestamps are unchanged.

Scenarios (all in original traffic units, values clipped at >= 0):
  noisy : same daily shape, high-frequency variation multiplied by --k (more variance)
  burst : event (e.g. a final streamed by many people): traffic of some cells is multiplied by
          up to --mult for a few hours (rise -> hold -> exponential decay)
  flat  : the daily cycle is lost: smooth profile replaced by the day mean, noise kept
  shift : the daily cycle happens --shift_hours later than usual

--day indexes the days of the test series: 0 = first day (only used as model input),
1 = second day, ... With the current split: 0=Mon 25, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat.
NOTE: the day AFTER the modified one still uses it as input history, so its first windows
are partly affected too.

test_data.npy can be [T, N] or [T, N, 1]: the output keeps the same shape and dtype.

Usage:
    python make_synthetic_days.py --scenario burst --day 3
    python make_synthetic_days.py --scenario noisy --day 3 --k 3
    python make_synthetic_days.py --scenario flat  --day 3
    python make_synthetic_days.py --scenario shift --day 3 --shift_hours 3
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

STEPS_PER_DAY = 144
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/MILAN")
    ap.add_argument("--scenario", choices=["noisy", "burst", "flat", "shift"], required=True)
    ap.add_argument("--day", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=7, help="moving-average window (steps) separating cycle and noise")
    ap.add_argument("--edge", type=int, default=12, help="steps used to blend flat/shift into the neighbouring days")
    ap.add_argument("--k", type=float, default=3.0, help="noisy: factor on the high-frequency part")
    ap.add_argument("--mult", type=float, default=2.5, help="burst: peak multiplier of the traffic")
    ap.add_argument("--start", type=float, default=17.0, help="burst: start hour of the day")
    ap.add_argument("--rise", type=float, default=1.0, help="burst: hours to reach the peak")
    ap.add_argument("--hold", type=float, default=2.0, help="burst: hours at the peak")
    ap.add_argument("--decay", type=float, default=1.5, help="burst: decay time constant (hours)")
    ap.add_argument("--frac", type=float, default=0.15, help="burst: fraction of cells involved")
    ap.add_argument("--cells", type=int, nargs="*", default=None, help="burst: explicit column indices")
    ap.add_argument("--shift_hours", type=float, default=3.0, help="shift: delay of the daily cycle (hours)")
    return ap.parse_args()


def moving_avg(X, w):
    pad = w // 2
    Xp = np.pad(X, ((pad, w - 1 - pad), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), Xp]), axis=0)
    return (c[w:] - c[:-w]) / w


def ramp(n, edge):
    """1 in the middle, linear 0->1 / 1->0 on the first/last `edge` steps."""
    a = np.ones(n)
    edge = min(edge, n // 2)
    if edge > 0:
        r = np.linspace(0, 1, edge + 2)[1:-1]
        a[:edge] = r
        a[-edge:] = r[::-1]
    return a[:, None]


def link(target, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(target, dst)
    except OSError:
        shutil.copy(target, dst)


def main():
    args = parse_args()
    src = Path(args.data).resolve()
    X = np.load(src / "test_data.npy")
    ts = np.load(src / "test_timestamps.npy")
    has_feat_dim = X.ndim == 3                      # [T, N, 1] vs [T, N]
    X2 = (X[..., 0] if has_feat_dim else X).astype(np.float64)
    if X2.ndim != 2:
        raise SystemExit(f"unexpected test_data shape {X.shape}: expected [T, N] or [T, N, F]")
    T, N = X2.shape

    off = int(round(ts[0, 0] * STEPS_PER_DAY))
    day_id = (np.arange(T) + off) // STEPS_PER_DAY
    n_days = int(day_id.max()) + 1
    if not 0 <= args.day < n_days:
        raise SystemExit(f"--day must be in 0..{n_days - 1}")
    rows = np.flatnonzero(day_id == args.day)
    wd = int(round(ts[rows[0], 1] * 7)) % 7
    dom = int(round(ts[rows[0], 2] * 31)) + 1

    rng = np.random.default_rng(args.seed)
    Y = X2.copy()
    sm = moving_avg(X2, args.smooth)
    info = {"scenario": args.scenario, "day": args.day, "weekday": WEEKDAYS[wd], "day_of_month": dom,
            "seed": args.seed}

    if args.scenario == "noisy":
        Y[rows] = sm[rows] + args.k * (X2[rows] - sm[rows])
        info["k"] = args.k

    elif args.scenario == "burst":
        mean_cell = X2.mean(0)
        if args.cells:
            cells = np.array(args.cells)
        else:  # random subset of the busier half of the cells, so the effect is visible
            pool = np.flatnonzero(mean_cell >= np.median(mean_cell))
            n = max(1, int(round(args.frac * N)))
            cells = rng.choice(pool, size=min(n, pool.size), replace=False)
        t = np.arange(T, dtype=float)
        t0 = rows[0] + args.start * 6
        rise, hold, dec = max(args.rise * 6, 1), args.hold * 6, max(args.decay * 6, 1)
        g = np.zeros(T)
        m1 = (t >= t0) & (t < t0 + rise)
        m2 = (t >= t0 + rise) & (t < t0 + rise + hold)
        m3 = t >= t0 + rise + hold
        g[m1] = (t[m1] - t0) / rise
        g[m2] = 1.0
        g[m3] = np.exp(-(t[m3] - (t0 + rise + hold)) / dec)
        Y[:, cells] = X2[:, cells] * (1 + (args.mult - 1) * g[:, None])
        info.update(mult=args.mult, start_hour=args.start, rise_h=args.rise, hold_h=args.hold,
                    decay_h=args.decay, cells=[int(c) for c in cells])

    elif args.scenario == "flat":
        a = ramp(len(rows), args.edge)
        mean_day = sm[rows].mean(0)
        Y[rows] = (1 - a) * X2[rows] + a * (mean_day + X2[rows] - sm[rows])

    elif args.scenario == "shift":
        S = int(round(args.shift_hours * 6))
        a = ramp(len(rows), args.edge)
        Y[rows] = (1 - a) * X2[rows] + a * X2[np.clip(rows - S, 0, T - 1)]
        info["shift_hours"] = args.shift_hours

    Y = np.clip(Y, 0, None)

    dst = src.parent / f"{src.name}_synth_{args.scenario}"
    dst.mkdir(exist_ok=True)
    for name in ("train_data", "train_timestamps", "val_data", "val_timestamps"):
        link(src / f"{name}.npy", dst / f"{name}.npy")
    link(src / "test_timestamps.npy", dst / "test_timestamps.npy")
    out = Y[..., None] if has_feat_dim else Y       # same shape as the original
    np.save(dst / "test_data.npy", out.astype(X.dtype))
    meta = json.load(open(src / "meta.json"))
    meta["name"] = dst.name
    json.dump(meta, open(dst / "meta.json", "w"), indent=2)
    json.dump(info, open(dst / "synth_info.json", "w"), indent=2)

    b, a_ = X2[rows], Y[rows]
    print(f"created {dst}")
    print(f"test_data shape {X.shape}, dtype {X.dtype} (kept)")
    print(f"modified day {args.day} = {WEEKDAYS[wd]} {dom}: scenario '{args.scenario}'")
    print(f"  total traffic of that day: {b.sum():.3g} -> {a_.sum():.3g} ({100 * (a_.sum() / b.sum() - 1):+.1f}%)")
    print(f"  mean |change| per cell and step: {np.abs(a_ - b).mean():.2f} "
          f"(mean traffic {b.mean():.1f}) | peak (max over cells and steps): {b.max():.0f} -> {a_.max():.0f}")
    if args.scenario == "burst":
        print(f"  burst cells: {len(info['cells'])}")


if __name__ == "__main__":
    main()
