# Sure_Twin: STID traffic forecasting with conformal upper bounds

Goal: forecast mobile-network traffic per cell and build a **calibrated upper bound** on it, to decide safely when a cell can go to sleep.

- Data: 2500 cells, 10-minute steps, split chronologically 60/20/20 (train/val/test).
- Model: STID (BasicTS), input 144 steps (24 h), output 12 steps (10 to 120 min ahead).
- Ensemble: 10 STID models, one per seed (0-9).
- Everything is evaluated in ORIGINAL traffic units, on the test split (709 windows, about 4.9 days, Tue-Sat).

Detailed interpretation of the results: see [`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md).

---

## 1. Pipeline at a glance

```
milan_raw.csv --(convert)--> datasets/MILAN --(train x10 seeds)--> checkpoints/
                                   |                                     |
                                   +--(make_split_view val)--> datasets/MILAN_val
                                                                         |
                                  checkpoints --(eval_split on val)--> checkpoints_eval/
                                                                         |
   checkpoints (test preds) + checkpoints_eval (val preds) --> analysis, plots, conformal_eval
```

Why a val view? The conformal margin must be calibrated on data the model did not use for the final test. The same 10 checkpoints are run on the val split (exposed as "test" in a view), so calibration and evaluation use the same models and the same z-score statistics.

---

## 2. Setup

Tested on the UniTrento `baldo` SLURM GPU cluster, Python 3.11, CUDA 12.4.

```bash
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
python -c "import torch, basicts; print(torch.cuda.is_available())"   # False on the login node is normal
```

---

## 3. Files

| File | Role |
|---|---|
| `setup.sh`, `requirements.txt` | Create `.venv` and install `torch` + `basicts` (commit `af71025`). |
| `convert_milan_to_basicts.py` | `milan_raw.csv` (`time, cell_id, y`) to `datasets/MILAN/` in the ETTh1 format. **Needs the raw CSV.** |
| `train_STID.py` | Trains one STID for one seed. |
| `run_train.sh` | SLURM array 0-9: trains the 10 seeds. |
| `make_split_view.py` | Creates `datasets/MILAN_val` (symlinks): the val split exposed as "test". |
| `eval_split.py` | Evaluates one seed's best-val-MAE checkpoint on the val view. |
| `run_eval.sh` | SLURM array 0-9: runs `eval_split.py` for all seeds. |
| `analyse_result.py` | Shared helpers (loading, last-batch fix, clock axis) and the main accuracy analysis. |
| `plot_timeseries.py` | True vs predicted traffic for 3 cells (busiest, median, quietest). |
| `plot_signed_error.py` | Signed error (pred - true) over time and folded on the 24 h clock. |
| `conformal_eval.py` | Compares upper-bound methods: coverage, margin, uniformity. |
| `make_synthetic_days.py` | Builds test sets with ONE modified day (`noisy`, `burst`, `flat`, `shift`). Optional, see section 5. |

---

## 4. Step-by-step run

Run everything from `~/Sure_Twin` with the venv active.

### Step 0 - Clean previous runs (optional)

```bash
rm -rf checkpoints checkpoints_eval datasets/MILAN_val datasets/MILAN_train
rm -f out_*.log eval_*.log analysis.png timeseries.png signed_error.png
```
**Do NOT delete `datasets/MILAN`** unless you still have `milan_raw.csv`: the `.npy` files there are the only copy of the data. 
Keep a backup (`cp -r datasets/MILAN ~/MILAN_backup`).

### Step 1 - Convert the data (only if you have the CSV)

```bash
python convert_milan_to_basicts.py
```
- What: pivots the CSV to `[T, N]`, fills missing timestamps and NaN with 0, builds 4 time features, splits 60/20/20.
- Output:`datasets/MILAN/{train,val,test}_{data,timestamps}.npy`,`meta.json`. Printed: step of 10 min, shapes like `(T, 2500, 1)` and `(T, 4)`

### Step 2 - Create the val view

```bash
python make_split_view.py --split val
```
- Output: `datasets/MILAN_val`, and a line "`'test' of this view = original val split: ... steps`".

### Step 3 - Train the ensemble, then evaluate it on val

```bash
TRAIN=$(sbatch --parsable run_train.sh)
sbatch --dependency=afterok:$TRAIN run_eval.sh val
squeue -u $USER
```
- What: 10 independent trainings (best checkpoint = lowest validation MAE, early stopping, 100 epochs max), then the same checkpoints are evaluated on val.
- Output: `checkpoints/STID_MILAN/seed<k>/<run>/` (checkpoint + `test_results/{prediction,targets}.npy`) and `checkpoints_eval/MILAN_val/seed<k>/...`.
- Check:
```bash
ls checkpoints/STID_MILAN/seed*/*/STID_best_val_MAE.pt | wc -l                    # 10
ls checkpoints_eval/MILAN_val/seed*/*/test_results/prediction.npy | wc -l         # 10
```
- A single failed seed: read `out_<jobid>_<seed>.log`, rerun with `sbatch --array=<seed> run_train.sh`. If the checkpoint already existed, delete that seed's folder first, otherwise `eval_split.py` finds more than one.

### Step 4 - Accuracy analysis and plots

```bash
python analyse_result.py
python plot_timeseries.py                  # add --horizon 6, --cells ..., --band 1.645 if needed
python plot_signed_error.py                # add --horizon 6, --cell 1800, --seed 0 if needed
```
- `analyse_result.py` prints sanity checks, then a table (MAE, RMSE, WAPE) of two baselines, each seed and the ensemble mean, then how informative the ensemble spread is. Saves `analysis.png`.
- `plot_timeseries.py` saves `timeseries.png`; `plot_signed_error.py` saves `signed_error.png` and prints bias, MAE and the time of the largest mean error.
- Sanity checks that must pass: `max |targets - raw test|` about 0 (here 0.0005), `windows evaluated: 709 of 709`, `cells evaluated: 2500 of 2500`, and recomputed normalized MAE equal to the BasicTS one for every seed.

### Step 5 - Conformal upper bounds

```bash
python conformal_eval.py --horizon 0 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python conformal_eval.py --horizon 2 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
`--horizon 0` = 10 min ahead, `2` = 30 min. Calibration = val split, evaluation = whole test.

Output, for target coverage 90 / 95 / 99%, one row per method:

| Column | Meaning | Good value |
|---|---|---|
| `cover` | fraction of (window, cell) pairs with truth <= bound | about 1 - eps |
| `margin`, `marg%` | mean (bound - prediction), absolute and relative to mean traffic | small at equal coverage (more energy saved) |
| `level / clock / day min/max` | worst and best coverage over predicted-level deciles, 4-hour blocks, days | min close to max |

Also printed: within-level Spearman between ensemble spread and |error|, and a per-day table (MAE, traffic, coverage).

Methods: `point`, `gaussian ensemble (uncalibrated)`, `conformal global`, `conformal by level`, `conformal / spread`, `conformal by level / spread`.

---

## 5. Optional - stress test with synthetic days (not rerun yet)

```bash
python make_synthetic_days.py --scenario burst --day 3
python make_synthetic_days.py --scenario noisy --day 3 --k 3
python make_synthetic_days.py --scenario flat  --day 3
python make_synthetic_days.py --scenario shift --day 3 --shift_hours 3
```
Creates `datasets/MILAN_synth_<scenario>/` (train/val linked unchanged, one test day modified) plus `synth_info.json`. Existing checkpoints must then be evaluated on it to see whether coverage survives anomalous days. Old synthetic folders were built from previous test data and must be regenerated.

---

## 6. Current state (run of 14 Nov 2026)

Full discussion in [`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md). Summary:

**Accuracy** (original units, test, ensemble of 10)

| | MAE | RMSE | WAPE |
|---|---|---|---|
| last value | 27.04 | 65.71 | 0.173 |
| seasonal naive (24 h) | 28.61 | 73.49 | 0.183 |
| single seeds | 16.30 - 16.76 | 38.59 - 40.06 | 0.105 - 0.107 |
| **ensemble mean** | **16.32** | **38.84** | **0.105** |

**Upper bound** (`conformal by level`, calibrated on val, evaluated on test)

| Horizon | Target | Coverage | Margin (% of mean traffic) |
|---|---|---|---|
| 10 min | 90% | 0.918 | 27.5 (17.6%) |
| 10 min | 95% | 0.963 | 41.6 (26.7%) |
| 10 min | 99% | 0.993 | 85.6 (55.0%) |
| 30 min | 90% | 0.919 | 30.8 (19.8%) |
| 30 min | 95% | 0.962 | 46.9 (30.1%) |
| 30 min | 99% | 0.994 | 96.0 (61.7%) |

**Main takeaways**
- The ensemble cuts MAE by about 40% against the naive baselines.
- The point forecast and the Gaussian ensemble bound are not safe (coverage 52% to 62%). Ensemble spread underestimates uncertainty.
- Conformal calibrated on val reaches the target on test; level-conditional calibration is the best compromise (uniform coverage across levels, hours and days, smallest margin among the uniform methods).
- Scaling by ensemble spread does not help (wider margins, no better uniformity).

**Open points**
- Test covers only about 5 days (Tue-Sat, one partial weekend day): no holidays or events. Robustness to anomalous days still to be tested with the synthetic days.
- Coverage is marginal (over all cells and windows), not guaranteed per cell, and windows are strongly autocorrelated, so the finite-sample guarantee is only approximate.