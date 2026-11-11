"""
Export predictions of one split (train / val / test) to CSV, in a format
that diagnose_bias.py understands.

Columns written: timestamp, cell, pred, true  (ORIGINAL scale, not normalized)
"""
import numpy as np
import pandas as pd


def save_predictions(path, pred, true, timestamps, cells=None):
    """
    pred, true : arrays of shape (T, N)  -> T target time steps, N cells.
                 Must already be inverse-transformed (original scale) and
                 pred must be the ENSEMBLE MEAN.
    timestamps : sequence of length T, timestamp of the TARGET step
                 (the time being predicted, not the end of the input window).
    cells      : sequence of length N (cell ids). Defaults to 0..N-1.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    assert pred.shape == true.shape, f"shape mismatch {pred.shape} vs {true.shape}"
    T, N = pred.shape
    timestamps = pd.to_datetime(pd.Series(list(timestamps)))
    assert len(timestamps) == T, f"{len(timestamps)} timestamps for {T} steps"
    if cells is None:
        cells = np.arange(N)
    assert len(cells) == N

    df = pd.DataFrame({
        "timestamp": np.repeat(timestamps.values, N),
        "cell": np.tile(np.asarray(cells), T),
        "pred": pred.reshape(-1),
        "true": true.reshape(-1),
    })
    df.to_csv(path, index=False)
    print(f"saved {len(df):,} rows -> {path}")


# Example of use (adapt the first lines to your own code):
#
#   preds_members = [m.predict(X) for m in ensemble]          # each (T, N), normalized
#   pred_norm = np.mean(preds_members, axis=0)
#   pred = scaler.inverse_transform(pred_norm)                # original scale
#   true = scaler.inverse_transform(y_norm)                   # original scale
#   save_predictions("preds_val.csv", pred, true, target_timestamps)
#
# Do it for train, val and test with the SAME horizon (e.g. 10 min).
