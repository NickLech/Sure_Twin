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

## Missing data and zeros (`check_zeros.py`, `check_cells.py`)
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
- The last partial test batch was written at the wrong offset in `test_results`
  (5 windows overwritten, 5 containing garbage).
- `analyse_result.py` moves the block back and drops the 5 lost windows: **704 of 709 windows evaluated**.
- Checks after the fix: max |targets - raw test| = 5e-4, and the recomputed normalized MAE matches
  `test_metrics.json` within 0.2% for all 5 seeds.

## Evaluation conventions
- Metrics are computed in original units (de-normalized with per-cell train mean/std);
  BasicTS logs are in normalized units.
- Baselines: last value, and seasonal naive (same slot 24 h earlier).

## Open points
- Check that excluding cells 858 and 808 does not change the conclusions (run with `--exclude 858 808`).
- Ensemble intervals are not calibrated: coverage of mean +/- 1.645 std is 0.149 (nominal 0.90).
