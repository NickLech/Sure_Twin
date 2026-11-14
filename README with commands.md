# Sure_Twin: STID traffic forecasting on Milan, with conformal upper bounds

Pipeline for forecasting mobile-network traffic per cell (2500 cells, 10-minute steps) with an ensemble of STID models (BasicTS), and for building a calibrated **upper bound** on traffic for the sleep-mode decision.

- Input window: 144 steps (24 h). Output: 12 steps (10 to 120 min ahead).
- Split: chronological 60/20/20 (train/val/test).
- Ensemble: one STID per seed. Everything is evaluated in ORIGINAL traffic units.

---

## 1. Setup

Tested on the UniTrento `baldo` SLURM GPU cluster.

```bash
chmod +x setup.sh && ./setup.sh      # installs uv, creates .venv (Python 3.11), installs requirements.txt
source .venv/bin/activate
python -c "import torch, basicts; print(torch.cuda.is_available())"   # False on the login node is normal
```

`requirements.txt`: `torch` and `basicts` pinned to commit `af71025`.

---

## 2. Files

| File | What it does |
|---|---|
| `setup.sh` | Creates `.venv` and installs the dependencies. |
| `requirements.txt` | Python dependencies. |
| `convert_milan_to_basicts.py` | Converts `milan_raw.csv` (`time, cell_id, y`) into the BasicTS/ETTh1 format in `datasets/MILAN/` (`*_data.npy`, `*_timestamps.npy`, `meta.json`). Fills gaps and NaN with 0. |
| `train_STID.py` | Trains one STID for one seed. Saves to `checkpoints/STID_MILAN/seed<k>/`, including test predictions. |
| `run_train.sh` | SLURM array job: trains the seeds in parallel (2 h each). |
| `make_split_view.py` | Creates `datasets/MILAN_val` or `MILAN_train`: that split exposed as "test", with the same z-score statistics. Symlinks, no data copied. |
| `eval_split.py` | Evaluates an existing checkpoint (best val MAE) of one seed on the val or train view. Output goes to `checkpoints_eval/MILAN_<split>/seed<k>/`. |
| `run_eval.sh` | SLURM array job: runs `eval_split.py` on seeds 0-9 (30 min each). |
| `analyse_result.py` | Shared helpers (loading, last-batch fix, clock axis) plus the main analysis and `analysis.png`. Imported by the scripts below. |
| `plot_timeseries.py` | True vs predicted traffic for a few cells, one panel each, whole test period. |
| `plot_signed_error.py` | Signed error (pred - true) over time and folded on the 24 h clock (all days vs weekdays). |
| `conformal_eval.py` | Compares upper-bound methods (point, Gaussian ensemble, conformal global / by level / scaled by spread) on coverage, margin and uniformity. |
| `make_synthetic_days.py` | Builds `datasets/MILAN_synth_<scenario>`: the real test series with ONE modified day (`noisy`, `burst`, `flat`, `shift`) to stress-test existing checkpoints. |

---

## 3. Commands

### 3.1 Data

```bash
python convert_milan_to_basicts.py                      # milan_raw.csv -> datasets/MILAN
python convert_milan_to_basicts.py --csv datasets/MILAN/milan_raw.csv --ref datasets/ETTh1
```
Expected: the printed time step (10 min), the number of filled timestamps, and shapes such as `train: data (T_train, 2500, 1)`. It also prints ETTh1's `meta.json` for comparison.

### 3.2 Training

```bash
sbatch run_train.sh                  # array 0-4 = seeds 0..4
python train_STID.py --seed 0        # single seed, local
python train_STID.py --seed 0 --hidden 64 --layers 3
```
Expected: `checkpoints/STID_MILAN/seed<k>/<run>/` containing `STID_best_val_MAE.pt` and `test_results/{prediction,targets}.npy`. Test metrics are logged at h=1, 6, 12 (10, 60, 120 min).

### 3.3 Calibration split (val)

```bash
python make_split_view.py --split val          # creates datasets/MILAN_val
sbatch run_eval.sh val                         # all seeds 0-9
sbatch --array=0 run_eval.sh val               # seed 0 only, as a first test
python eval_split.py --split val --seed 0      # local, one seed
```
Expected: `checkpoints_eval/MILAN_val/seed<k>/...` with the same files as in training. Use the same `--hidden/--layers` as in training. `--split train` works the same way (diagnostic only).

### 3.4 Analysis and plots

```bash
python analyse_result.py
python plot_timeseries.py                                  # busiest / median / quietest cell
python plot_timeseries.py --horizon 6 --start 0 --length 144 --cells 12 345 1800 --band 1.645
python plot_signed_error.py                                # mean over all cells
python plot_signed_error.py --horizon 6 --cell 1800 --seed 0
```
Expected: `analysis.png`, `timeseries.png`, `signed_error.png`, plus console summaries (bias, MAE, time of largest mean error).

### 3.5 Conformal upper bounds

```bash
# recommended: calibrate on val, evaluate on the whole test
python conformal_eval.py --horizon 2 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"

# fallback: first 50% of test = calibration, rest = evaluation
python conformal_eval.py --horizon 2
```
`--horizon 0` is 10 min ahead, `2` is 30 min. The number of members must match between calibration and evaluation.

Expected output, for each target coverage (90/95/99%): a table with `cover` (should be about 1-eps), `margin` (smaller at equal coverage = more energy saved), and min/max coverage across predicted-level deciles, 4-hour blocks and days (should be uniform). It also prints the within-level Spearman of spread vs |error| and a per-day table.

### 3.6 Stress tests on synthetic days

```bash
python make_synthetic_days.py --scenario burst --day 3
python make_synthetic_days.py --scenario noisy --day 3 --k 3
python make_synthetic_days.py --scenario flat  --day 3
python make_synthetic_days.py --scenario shift --day 3 --shift_hours 3
```
Expected: `datasets/MILAN_synth_<scenario>/` (train/val linked unchanged, only `test_data.npy` modified) and `synth_info.json`, plus a summary of the change in traffic. Day 0 is the first test day (input only); with the current split 0=Mon 25 ... 5=Sat. The day after the modified one is also partly affected (it uses it as history). Evaluate it by pointing `--data` of the analysis scripts at the new folder and re-running the checkpoints on it.

---

## 4. Current state

> Numbers below are to be filled in from the latest runs; they were not recorded when this file was written.

**Done**
- Data conversion to the BasicTS format, chronological split.
- STID ensemble trained per seed, with test predictions saved.
- Val-split evaluation of the same checkpoints (same scaler) to provide a separate calibration set.
- Analysis/plots: per-cell time series, signed-error profile (bias by time of day, weekdays vs all days).
- Conformal comparison with a separate calibration split; per-day coverage table to spot anomalous days.
- Synthetic anomaly generator (`noisy`, `burst`, `flat`, `shift`).

**Results** (fill in)
- Test MAE / MSE at h=1, 6, 12: _TBD_
- Conformal coverage vs target (eps = 0.10, 0.05, 0.01) and margin %, best method: _TBD_
- Within-level Spearman (does ensemble spread help?): _TBD_
- Coverage on the synthetic days: _TBD_

**Known issues and open points**
- `run_train.sh` trains seeds 0-4, while `run_eval.sh` uses seeds 0-9 and `conformal_eval.py` needs matching members. Either train 10 seeds (`--array=0-9`) or set `run_eval.sh` to `--array=0-4`.
- Python version: the old README said 3.9.21, but `setup.sh` creates 3.11.
- Dropped last-batch windows are handled by `fix_misplaced_last_batch`; gaps in plots are those windows.
- The Gaussian ensemble band is uncalibrated, so use the conformal bounds for decisions.
- Coverage can degrade on unusual days (weekends, holidays, events); the per-day table and the synthetic scenarios are meant to quantify this.
