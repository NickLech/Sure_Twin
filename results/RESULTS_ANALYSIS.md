# Results analysis (run of 14 Nov 2026)

What the numbers printed by `analyse_result.py`, `plot_signed_error.py` and `conformal_eval.py` show, and what they do and do not prove.

Setup: 10 STID seeds, 2500 cells, test = 709 windows (about 4.9 days: Tue, Wed, Thu, Fri, partial Sat), calibration = val split (also 709 windows). All values in original traffic units; mean traffic on the test is about 155 per cell and step.

> This analysis is based on the console output only. The figures (`analysis.png`, `timeseries.png`, `signed_error.png`) are not described here.

---

## 1. Is the pipeline consistent?

| Check | Result | Reading |
|---|---|---|
| `max abs(targets - raw test)` | 0.0005 | The inverse z-score reproduces the raw data up to float32 rounding: the scaler is correct. |
| Windows evaluated | 709 of 709 | The last-batch fix dropped nothing. |
| Cells evaluated | 2500 of 2500 | No cell excluded. |
| Normalized MAE recomputed vs BasicTS | identical for all seeds | Predictions and targets loaded by our scripts are the ones BasicTS evaluated. |

Conclusion: the numbers below come from a consistent pipeline. The `no log files match: out_*.log` warning does not affect them.

Calibration (val) vs evaluation (test) difficulty: the normalized MAE is about 0.296 on val against about 0.268 on test (about 10% higher). Val is therefore a harder split than test. This matters in section 5.

---

## 2. Forecast accuracy

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| last value | 27.04 | 65.71 | 0.173 |
| seasonal naive (24 h) | 28.61 | 73.49 | 0.183 |
| seeds 0-9 | 16.30 - 16.76 | 38.59 - 40.06 | 0.105 - 0.107 |
| ensemble mean | 16.32 | 38.84 | 0.105 |

What it shows
- STID beats the best naive baseline by about 40% in MAE (27.04 to 16.32), and the seasonal baseline by about 43%. The model learns something beyond persistence.
- The seasonal naive is worse than last value: the traffic of the same hour the day before is a poorer guess than the last observation, at least on this short test where the day of the week changes (Tue to Sat, with Saturday about 22% lower).
- Seeds are consistent (MAE spread about 2.8%), with two groups: seeds 4, 5, 7, 9 are slightly worse (16.63 - 16.76) than the rest (16.30 - 16.39).
- The ensemble is about 1% better than the average member (average MAE of the seeds is 16.48) but **not better than the best seed** (seed 6: MAE 16.30, RMSE 38.59). Members differ only by initialization, so their errors are highly correlated and averaging gives little.

What it does not show: which horizon the baseline and seed table refers to is decided by `analyse_result.py` (not reproduced here); the conformal results below are at fixed horizons.

---

## 3. Where does the point forecast err?

From `plot_signed_error.py` (horizon 10 min, error = pred - true):
- Mean over all cells: bias -1.95, MAE of the cell-averaged error 2.32. The model **under-estimates on average** by about 2 units (about 1 to 2% of mean traffic). The 2.32 is the error of the average over cells, not the per-cell MAE (about 16).
- Largest mean error by time of day: -6.82 at 11:10 (all days), -7.87 at 01:00 (weekdays only).
- These clock profiles are averages over only 4 to 5 days, so single points are noisy; read them as "under-estimation appears at some times of day", not as a stable pattern.

Link with section 4: the point forecast is under the truth only 48% of the time (coverage 0.518), yet the mean error is negative. The error distribution is therefore skewed: typical errors are small and balanced, but the rare large positive deviations (spikes) are under-estimated and pull the mean down. This is why a symmetric margin is not enough.

---

## 4. Can the point forecast or the ensemble be used as an upper bound?

Target coverage 90 / 95 / 99%, test set:

| Method | h = 10 min | h = 30 min |
|---|---|---|
| point (no margin) | 0.518 | 0.520 |
| gaussian ensemble (mean + z * std) | 0.566 / 0.578 / 0.601 | 0.575 / 0.590 / 0.616 |

- The point forecast is above the truth about half of the time, as expected: using it as a bound violates QoS in almost half of the cases.
- The Gaussian bound built on the ensemble spread is far from the target (57% when 90% is promised, 60% when 99% is promised). Its margin is tiny (1.6 to 2.9 units) because the seeds agree with each other.
- The two-sided band mean +/- 1.645 std covers only 0.171 of the data (nominal 0.90). This is the clearest evidence: ensemble disagreement measures only the variability from initialization, not the real prediction uncertainty (noise, bursts), which all members miss in the same way.

Spread is weakly informative:
- Overall correlation between std and |error| is 0.375, but inside a predicted-level decile the Spearman correlation drops to 0.11 (10 min) and 0.15 (30 min). Most of the 0.375 comes from the level: busy cells have both larger spread and larger errors.
- Only the extreme deciles retain some signal (about 0.24 to 0.30 for the lowest and highest).

Conclusion: an uncalibrated bound is unsafe and the ensemble spread is not a reliable uncertainty measure. Calibration on held-out data is required.

---

## 5. Conformal upper bounds

### 5.1 Horizon 10 min

| Method | cover 90% | margin | cover 95% | margin | cover 99% | margin |
|---|---|---|---|---|---|---|
| conformal global | 0.917 | 23.5 (15.1%) | 0.961 | 49.1 (31.5%) | 0.993 | 144.0 (92.6%) |
| **conformal by level** | 0.918 | 27.5 (17.6%) | 0.963 | 41.6 (26.7%) | 0.993 | 85.6 (55.0%) |
| conformal / spread | 0.915 | 32.6 (20.9%) | 0.961 | 52.0 (33.4%) | 0.993 | 116.6 (74.9%) |
| conformal by level / spread | 0.914 | 34.7 (22.3%) | 0.960 | 54.0 (34.7%) | 0.993 | 114.9 (73.8%) |

### 5.2 Horizon 30 min

| Method | cover 90% | margin | cover 95% | margin | cover 99% | margin |
|---|---|---|---|---|---|---|
| conformal global | 0.919 | 26.5 (17.0%) | 0.962 | 55.3 (35.5%) | 0.993 | 160.4 (103.0%) |
| **conformal by level** | 0.919 | 30.8 (19.8%) | 0.962 | 46.9 (30.1%) | 0.994 | 96.0 (61.7%) |
| conformal / spread | 0.917 | 36.6 (23.5%) | 0.962 | 58.9 (37.8%) | 0.993 | 132.2 (84.9%) |
| conformal by level / spread | 0.917 | 38.0 (24.4%) | 0.962 | 59.4 (38.2%) | 0.993 | 126.1 (81.0%) |

### 5.3 What this shows

1. **Calibration on val transfers to test.** Every conformal method reaches at least the target on test. It slightly over-covers (for example 0.918 vs 0.90, 0.963 vs 0.95, 0.993 vs 0.99). This is the safe direction and is consistent with val being the harder split (section 1): the calibration errors are a bit larger than the test errors.

2. **The global margin hides a strong heterogeneity.** The overall coverage is right, but by predicted level it ranges from 0.666 to 0.999 at 90% (0.782 to 1.000 at 95%, 0.947 to 1.000 at 99%). A constant margin is far too large for low-traffic predictions (coverage about 100%) and too small for high ones. Presumably the undercovered group is the highest-level cells, which are also the most critical for QoS. The extremes (min/max) are read from the table; the script does not say which decile is which.

3. **Level-conditional calibration fixes it.** `conformal by level` brings the level coverage to 0.905 - 0.924 (90%), 0.956 - 0.965 (95%) and 0.992 - 0.995 (99%), and also keeps the clock-block range (0.906 - 0.941 at 90%) and day range (0.912 - 0.926) narrow.

4. **It is also the cheapest among the uniform methods.** At 95% and 99% it has a smaller margin than the global one (41.6 vs 49.1; 85.6 vs 144.0 at 10 min), because it does not waste margin on low-traffic cells. At 90% the global margin is smaller (23.5 vs 27.5), but only at the price of the 0.67 coverage in the top level.

5. **Dividing by the ensemble spread does not help.** Margins are larger (for example 52.0 and 54.0 vs 41.6 at 95%, 10 min) and uniformity is not better: the clock-block range gets wider (0.883 - 0.965 vs 0.906 - 0.941 at 90%). This is what section 4 predicted: spread carries little information beyond the level.

6. **The price of safety grows fast with the target.** At 10 min the margin is 17.6% of mean traffic at 90%, 26.7% at 95% and 55.0% at 99%. At 30 min it is 19.8%, 30.1% and 61.7%. Going from 95% to 99% roughly doubles the margin: the error distribution has a heavy tail. The dependence on horizon is mild (about 10% more margin from 10 to 30 min).

7. **Per-day coverage is uniform, including Saturday.** At 95%, 10 min: Tue 0.960, Wed 0.957, Thu 0.965, Fri 0.969, Sat 0.963. Saturday has about 22% less traffic (129 vs 166 on Tue) and lower MAE (11.9 vs 14.2 to 15.1) and keeps the coverage. Per-day numbers at 30 min are equivalent (0.957 - 0.969).

---

## 6. What this does and does not prove

Supported
- The ensemble is clearly better than naive forecasts, and the pipeline is internally consistent.
- Neither the point forecast nor the ensemble spread gives a valid upper bound; a calibrated margin is necessary.
- With level-conditional conformal calibration on a held-out split, the empirical coverage on the test matches the target, uniformly across levels, hours of the day and days.
- The cost of the bound is quantified: about 18% to 20% extra over the prediction for 90%, 27% to 30% for 95%, 55% to 62% for 99%.

Not supported (limits)
- **Short test.** Only about 5 days, one partial weekend day, no holidays or events. Coverage on rare or anomalous days is not tested; the synthetic days (`make_synthetic_days.py`) are the planned tool and have not been rerun after the dataset refresh.
- **Dependence.** Neighbouring windows and cells are strongly correlated, so the effective sample size is much smaller than the 709 x 2500 values; the finite-sample conformal guarantee assumes exchangeability, which does not hold exactly for time series. The 99% results in particular rely on few independent tail events.
- **Marginal, not per-cell.** Coverage is computed over all cells and windows together (with checks per level, hour and day). Coverage of an individual cell was not examined.
- **Distribution shift.** Here val is harder than test, which favours safety. If the shift went the other way (test harder than val), coverage would fall below target.
- **No energy metric yet.** The margin is a proxy for energy saved; the actual sleep-mode decision rule (threshold on the bound) was not simulated.

---

## 7. Suggested next steps

1. Rerun the synthetic days (`burst`, `noisy`, `flat`, `shift`) on the new checkpoints and compare per-day coverage against the table of section 5.
2. Look at coverage per cell (or per group of cells) to check that no group is systematically under-covered.
3. Simulate the sleep-mode rule with the `conformal by level` bound to turn the margin into an energy-versus-violation curve.
4. If more data become available, extend the test (holidays, events) and consider an online/adaptive recalibration against drift.
