import glob
import os

import numpy as np

ROOT = "checkpoints/STID_MILAN"
OUTPUT_LEN = 12      # steps in the forecast window
NUM_CELLS = 2500     # N * C
HORIZON = 0          # 0 = first step = 10 min ahead
STEPS_PER_DAY = 144  # 10-min slots

# The saved arrays are z-score NORMALIZED. To get the original scale put here the
# mean and std used by the scaler (same ones your plotting script uses), else leave None.
MEAN = None
STD = None


def load_raw(path):
    """BasicTS writes raw float32 without header: shape = [B, OUTPUT_LEN, NUM_CELLS]."""
    a = np.fromfile(path, dtype=np.float32)
    per_sample = OUTPUT_LEN * NUM_CELLS
    assert a.size % per_sample == 0, f"{path}: size {a.size} not divisible by {per_sample}"
    return a.reshape(-1, OUTPUT_LEN, NUM_CELLS)[:, HORIZON].astype(np.float64)


def denorm(x):
    return x * STD + MEAN if (MEAN is not None and STD is not None) else x


pred_files = sorted(glob.glob(os.path.join(ROOT, "seed*", "*", "test_results", "prediction.npy")))
assert pred_files, f"no prediction.npy found under {ROOT} (run from the project folder)"
print(f"found {len(pred_files)} ensemble members")

preds = [denorm(load_raw(f)) for f in pred_files]
true = denorm(load_raw(pred_files[0].replace("prediction.npy", "targets.npy")))
B = true.shape[0]
scale = "ORIGINAL scale" if MEAN is not None else "NORMALIZED scale (z-score): absolute values not comparable to your plot"
print(f"{B} test steps x {true.shape[1]} cells | {scale}")

pred = np.mean(preds, axis=0)  # ensemble mean
err = pred - true

# 1) global: is it systematic?
print("\n== 1. GLOBAL ==")
print(f"bias {err.mean():.4f} | MAE {np.abs(err).mean():.4f} | median err {np.median(err):.4f}")
print(f"|bias|/MAE = {abs(err.mean()) / np.abs(err).mean():.2f}")
if MEAN is not None:
    print(f"pred/true = {pred.sum() / true.sum():.3f}")
print("single member bias:")
for f, p in zip(pred_files, preds):
    print(f"  {(p - true).mean():9.4f}  {f.split('/')[2]}")

print("\n== 2. BIAS PER PREDICTED-LEVEL DECILE ==")
edges = np.quantile(pred, np.linspace(0, 1, 11))
bins = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, 9)
for b in range(10):
    m = bins == b
    print(f"decile {b}: mean pred {pred[m].mean():8.3f} | mean true {true[m].mean():8.3f} | bias {err[m].mean():8.4f}")

# 3) structured? first half vs second half of the test, by time of day
print("\n== 3. STABILITY OVER TIME ==")
days = B // STEPS_PER_DAY
if days >= 2:
    prof = err[: days * STEPS_PER_DAY].mean(axis=1).reshape(days, STEPS_PER_DAY)
    h = days // 2
    a, b = prof[:h].mean(0), prof[h:].mean(0)
    print(f"{days} full days, split {h} / {days - h}")
    print(f"profile corr first vs second half: {np.corrcoef(a, b)[0, 1]:.2f}")
    print(f"bias first half {a.mean():.4f} | second half {b.mean():.4f}")
else:
    print(f"only {B} steps ({B / STEPS_PER_DAY:.1f} days): not enough to split in halves")
