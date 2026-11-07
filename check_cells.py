import numpy as np
import pandas as pd

x = np.load("datasets/MILAN/test_data.npy")
x = x[..., 0] if x.ndim == 3 else x
df = pd.read_csv("milan_raw.csv", parse_dates=["time"])

ids = sorted(df.cell_id.unique())
times = pd.date_range(df.time.min(), df.time.max(), freq="10min")
test_start = times[int(len(times) * 0.6) + int(len(times) * 0.2)]

zeros = (x == 0).sum(0)
for col in np.argsort(zeros)[-8:][::-1]:   # 8 cells with the most zeros
    sub = df[(df.cell_id == ids[col]) & (df.time >= test_start)]
    print(f"cell {ids[col]}: zeros in test={zeros[col]}, "
          f"rows present={len(sub)}/{x.shape[0]}, present with y=0: {(sub.y == 0).sum()}")
