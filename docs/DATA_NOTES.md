# MILAN data: checks and decisions
 
## Data Format
The original dataset (`milan_raw.csv`) is structured in a **Long** (vertical) format, where each row represents a single cell measurement at a given timestamp.
 
For training spatio-temporal models (BasicTS), `convert_milan_to_basicts.py` pivots the table into a **Wide** format matrix (`[T, N]`):
- **Time steps ($T$):** 4320 rows (regular 10-minute grid).
- **Variables/Cells ($N$):** 2500 columns.
| Format | Structure | Description |
| :--- | :--- | :--- |
| **Long (`milan_raw.csv`)** | Columns: `time`, `cell_id`, `y` | Each row is a (timestamp, cell) pair. Suitable for storage, not for model inputs. |
| **Wide (`*_data.npy`)** | Shape: `[T, N]` | Each row contains the status of all 2500 cells at the exact same timestamp $T$. |
 
## Temporal Features (`*_timestamps.npy`)
The temporal features have shape `[T, 4]` and provide cyclical background information to the model. All values are normalized in the range $[0, 1)$:
 
1. **Time of day:** $\frac{\text{Seconds elapsed since midnight}}{86400}$ (captures daily 24h cycles).
2. **Day of week:** $\frac{\text{Day of week (0-6)}}{7.0}$ (distinguishes weekdays from weekends).
3. **Day of month:** $\frac{\text{Day of month - 1}}{31.0}$ (captures monthly trends).
4. **Day of year:** $\frac{\text{Day of year - 1}}{366.0}$ (captures annual seasonality).
## Dataset
- Source: `milan_raw.csv` (`time`, `cell_id`, `y`), converted with `convert_milan_to_basicts.py`
  to `[T, N]` arrays on a regular 10-minute grid (N = 2500 cells).
- Chronological split 60/20/20: train T=2592, val T=864, test T=864 (4320 steps in total).
- Missing (time, cell) pairs are filled with 0 by the converter.
- Windows: input 144 steps (24 h), output 12 steps (2 h). Each of val and test (864 steps) gives
  864 - 144 - 12 + 1 = **709 windows**.
- Test period: 6 days (Mon-Sat). The first day is only used as input history, so the 709 evaluated
  windows predict about 4.9 days (Tue, Wed, Thu, Fri, partial Sat). No holidays or special events.
- Val is a harder split than test: normalized MAE about 0.296 on val against about 0.268 on test
  (about 10% higher). Val is used to calibrate the conformal upper bounds, so this difference
  pushes coverage on the test slightly above target (the safe direction).
## Missing data and zeros
These checks were done with `check_zeros.py` and `check_cells.py`, now archived in `old_files/checks/`.
- No missing timestamps (4320 / 4320). Missing (time, cell) rows: 2705 of 10,800,000 (0.025%).
- Zeros: 0.04% in train and val, 0.17% in test. No time step has more than 50% of cells at zero,
  so there are no network-wide gaps filled with zeros.
- In test, 8 cells have a zero run of at least 6 hours (longest: 345 steps). Comparing with the raw CSV:
  - **Cells 858 and 808**: zeros are missing rows, not traffic (only 83/864 and 420/864 rows present,
    none with `y=0`). They carry no information in test. They are **kept in the reported evaluation**
    (2500 cells, 0.17% zeros in test overall); `analyse_result.py --exclude 858 808` evaluates
    without them (needs `datasets/MILAN/cell_ids.npy` or the raw csv).
  - **Cells 1194, 1243, 1343, 1293, 1292, 1244**: zeros are mostly recorded `y=0`
    (221-313 per cell). They are genuine low-traffic cells and are **kept**: they are natural
    sleep candidates.
- Training used all 2500 cells; zeros there are too few to matter, so no retraining was needed.
## Saved test results (BasicTS)
- In an earlier run, the last partial test batch was written at the wrong offset in `test_results`
  (5 windows overwritten, 5 containing garbage). `analyse_result.py` still contains a workaround
  (`fix_misplaced_last_batch`) that moves the block back and drops the lost windows.
- In the run of 14 Nov 2026 (10 seeds) the workaround was not triggered: **709 of 709 windows evaluated**,
  on both the test split and the val view used for calibration.
- Checks: max |targets - raw test| = 4.9e-4 (float32 rounding), and the recomputed normalized MAE
  equals `test_metrics.json` for all 10 seeds (test and val).
## Evaluation conventions
- Metrics are computed in original units (de-normalized with per-cell train mean/std);
  BasicTS logs are in normalized units.
- The point forecast is the mean of the 10-seed ensemble (STID, one model per seed).
- Baselines: last value, and seasonal naive (same slot 24 h earlier).
- Upper bounds are calibrated with split conformal prediction: calibration on the val split
  (same checkpoints, same z-score statistics), evaluation on the whole test split
  (`conformal_eval.py`). Best method so far: conformal by predicted level.
  See `results/RESULTS_ANALYSIS.md`.
## Open points
- Check that excluding cells 858 and 808 does not change the conclusions (run with `--exclude 858 808`).
- The raw ensemble interval is not calibrated (coverage of mean +/- 1.645 std = 0.171, nominal 0.90).
  This is addressed with conformal calibration, see `conformal_eval.py` and `results/RESULTS_ANALYSIS.md`.
- The missing-data and zero statistics above come from the check scripts. If `datasets/MILAN` has been
  regenerated since they were run, they should be recomputed.
 