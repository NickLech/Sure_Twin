# MILAN data: meaning, checks and decisions

## What `y` measures

The variable `y` is an **aggregate cell activity index**: the overall usage of a cell in terms of Internet traffic, calls and SMS, acquired every 10 minutes for a 50x50 grid of cells (2500 cells) in Milan, from 2013-11-01 00:00 (4320 steps, 30 days).

- **Unit:** arbitrary units, written **[a.u.]** in this repository. It is **not** a number of bytes, and the data set does not state a physical unit.
- **Not power:** the data set does **not** contain power-consumption measurements (confirmed by the data provider). Estimating the power of the base stations requires a separate traffic-to-power model, calibrated on these data or on a different data set. In this repository, "margin", "over-provisioning" and "unmet activity" are therefore proxies expressed in activity units (or as percentages of total activity), not in energy.
- **Dimensionless quantities:** coverage, WAPE, unmet % and over-provisioning % are ratios and have no unit.

## Data format

The original data set (`milan_raw.csv`) is in **long** format: one row per (timestamp, cell) measurement, with columns `time`, `cell_id`, `y`.

For training spatio-temporal models (BasicTS), `src/data/convert_milan_to_basicts.py` pivots the table into a **wide** matrix `[T, N]`:
- **Time steps (T):** 4320 rows (regular 10-minute grid).
- **Variables/cells (N):** 2500 columns.

| Format | Structure | Description |
| :--- | :--- | :--- |
| **Long (`milan_raw.csv`)** | Columns: `time`, `cell_id`, `y` | One row per (timestamp, cell) pair. Suitable for storage, not for model inputs. |
| **Wide (`*_data.npy`)** | Shape `[T, N]` | Each row holds the activity of all 2500 cells at the same timestamp. |

The converter also saves `datasets/MILAN/cell_ids.npy`: the `cell_id` of each column, needed by `--exclude` in the analysis scripts.

## Temporal features (`*_timestamps.npy`)

The temporal features have shape `[T, 4]` and provide cyclical context to the model. All values are normalized in the range [0, 1):

1. **Time of day:** seconds elapsed since midnight / 86400 (daily cycle).
2. **Day of week:** day of week (0-6) / 7 (weekdays vs weekend).
3. **Day of month:** (day of month - 1) / 31.
4. **Day of year:** (day of year - 1) / 366.

## Dataset and splits

- Source: `milan_raw.csv` (`time`, `cell_id`, `y`), converted to `[T, N]` arrays on a regular 10-minute grid (N = 2500).
- Chronological split 60/20/20: train T=2592, val T=864, test T=864 (4320 steps in total, that is 18 + 6 + 6 days: train 1-18 Nov, val 19-24 Nov, test 25-30 Nov 2013).
- Missing (time, cell) pairs are filled with 0 by the converter.
- Windows: input 144 steps (24 h), output 12 steps (2 h). Each of val and test (864 steps) gives 864 - 144 - 12 + 1 = **709 windows**.
- Test period: 6 days (Mon-Sat). The first day is only used as input history, so the 709 evaluated windows predict about 4.9 days (Tue, Wed, Thu, Fri, partial Sat). No holidays or special events.
- Val is a harder split than test: normalized MAE about 0.296 on val against about 0.268 on test (about 10% higher). Val is used to calibrate the conformal upper bounds, so this difference pushes coverage on the test slightly above target (the safe direction).

## Missing data and zeros

These checks were done with two ad-hoc scripts that are no longer in the repository (they were removed as obsolete); the results are kept here. If `datasets/MILAN` is regenerated, they should be recomputed.

- No missing timestamps (4320 / 4320). Missing (time, cell) rows: 2705 of 10,800,000 (0.025%).
- Zeros: 0.04% in train and val, 0.17% in test. No time step has more than 50% of cells at zero, so there are no network-wide gaps filled with zeros.
- In test, 8 cells have a zero run of at least 6 hours (longest: 345 steps). Comparing with the raw CSV:
  - **Cells 858 and 808**: the zeros are missing rows, not activity (only 83/864 and 420/864 rows present, none with `y=0`). They carry no information in test. They are **kept in the reported evaluation** (2500 cells, 0.17% zeros in test overall); `src/analysis/analyse_result.py --exclude 858 808` evaluates without them.
  - **Cells 1194, 1243, 1343, 1293, 1292, 1244**: the zeros are mostly recorded `y=0` (221-313 per cell). They are genuine low-activity cells and are **kept**: they are natural sleep candidates.
- Training used all 2500 cells; zeros there are too few to matter, so no retraining was needed.

## Saved test results (BasicTS)

- In an earlier run, the last partial test batch was written at the wrong offset in `test_results` (5 windows overwritten, 5 containing garbage). `analyse_result.py` still contains a workaround (`fix_misplaced_last_batch`) that moves the block back and drops the lost windows.
- In the run of 14 Nov 2026 (10 seeds) the workaround was not triggered: **709 of 709 windows evaluated**, on both the test split and the val view used for calibration.
- Checks: max |targets - raw test| = 4.9e-4 (float32 rounding), and the recomputed normalized MAE equals `test_metrics.json` for all 10 seeds (test and val).

## Evaluation conventions

- Metrics are computed in the original scale [a.u.] (de-normalized with the per-cell train mean/std); BasicTS logs are in normalized units.
- The point forecast is the mean of the 10-seed ensemble (STID, one model per seed).
- Baselines: last value, and seasonal naive (same slot 24 h earlier).
- Upper bounds are calibrated with split conformal prediction: calibration on the val split (same checkpoints, same z-score statistics), evaluation on the whole test split (`src/analysis/conformal_eval.py`). Best method so far: conformal by predicted level. See [`results/RESULTS_ANALYSIS.md`](../results/RESULTS_ANALYSIS.md).

## Open points

- Check that excluding cells 858 and 808 does not change the conclusions (run with `--exclude 858 808`).
- The raw ensemble interval is not calibrated (coverage of mean +/- 1.645 std = 0.171, nominal 0.90). This is addressed with conformal calibration, see `conformal_eval.py` and [`results/RESULTS_ANALYSIS.md`](../results/RESULTS_ANALYSIS.md).
- Find or build a traffic-to-power model (or a power data set) to convert activity into energy for the digital twin.
