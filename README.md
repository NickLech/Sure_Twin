# Sure_Twin: STID traffic forecasting with conformal upper bounds

Goal: forecast mobile-network activity per cell and build a **calibrated upper bound** on it, to decide safely when a cell can go to sleep (input for a digital twin of a base-station grid).

- **Data:** 2500 cells (50x50 grid, Milan, November 2013), 10-minute steps, split chronologically 60/20/20 (train/val/test). See [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md).
- **Variable:** `y` is an aggregate **cell activity index** (Internet traffic, calls and SMS combined), in arbitrary units **[a.u.]**. It is **not** bytes and **not** power consumption.
- **Model:** STID (BasicTS), input 144 steps (24 h), output 12 steps (10 to 120 min ahead).
- **Ensemble:** 10 STID models, one per seed (0-9).
- **Evaluation:** in the ORIGINAL activity-index scale [a.u.] (de-normalized), on the test split (709 windows, about 4.9 days, Tue-Sat).

Detailed interpretation of the results: [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md).
Final configuration and its motivation: [`final_metrics/README.md`](final_metrics/README.md).

---

## 1. Pipeline at a glance

```
milan_raw.csv --(convert)--> datasets/MILAN --(train x10 seeds)--> checkpoints/
                                   |                                     |
                                   +--(make_split_view val)--> datasets/MILAN_val
                                                                         |
                                  checkpoints --(eval_split on val)--> checkpoints_eval/
                                                                         |
   checkpoints (test preds) + checkpoints_eval (val preds) --> analysis, plots, conformal bounds, schedule
```

Why a val view? The conformal margin must be calibrated on data the model did not use for the final test. The same 10 checkpoints are run on the val split (exposed as "test" in a view), so calibration and evaluation use the same models and the same z-score statistics.

---

## 2. Repository layout

```
Sure_Twin/
├── README.md                 this file
├── requirements.txt          direct dependencies
├── requirements.lock         pinned environment (used by scripts/setup.sh)
├── datasets/                 MILAN, MILAN_val (+ ETTh1 as format reference)
├── final_metrics/            chosen configuration: tables, figures, README
├── docs/
│   ├── SETUP.md              installation
│   ├── CLUSTER_GUIDE.md      UniTrento GPU cluster (SLURM)
│   └── DATA_NOTES.md         what the data is, checks and decisions
├── src/
│   ├── data/                 convert_milan_to_basicts.py, make_split_view.py
│   ├── training/             train_STID.py, eval_split.py
│   ├── analysis/             analyse_result.py, conformal_eval.py,
│   │                         schedule_conformal.py, schedule_final_metrics.py
│   └── plotting/             plot_timeseries.py, plot_signed_error.py, plot_concentration.py,
│                             plot_confident.py, busiest_levels.py
├── scripts/
│   ├── setup.sh              create .venv and install dependencies
│   ├── git_update.sh         helper to commit and push everything
│   └── slurm/                run_train.sh, run_eval.sh (job arrays, seeds 0-9)
└── results/
    ├── RESULTS_ANALYSIS.md   written interpretation of the results
    ├── logs/                 raw console output of analysis runs
    ├── tables/               machine-readable outputs (schedule_margins.json); created by schedule_conformal.py
    └── figures/
        ├── accuracy/         point-forecast accuracy (analysis, timeseries, signed_error)
        ├── conformal/        upper-bound plots (busiest_levels, busiest_confident, confident_pred)
        ├── schedule/         time-of-day schedule (schedule_summary, schedule_busiest)
        └── data/             data exploration (concentration)
```

Checkpoints (`checkpoints/`, `checkpoints_eval/`) and job logs (`out_*.log`, `eval_*.log`) are produced by the runs and are git-ignored. The scripts in `src/analysis/` create their output folders when missing.

---

## 3. Setup

Tested on the UniTrento `baldo` SLURM GPU cluster, Python 3.11, CUDA 12.4.

```bash
chmod +x scripts/setup.sh && ./scripts/setup.sh
source .venv/bin/activate
python -c "import torch, basicts; print(torch.cuda.is_available())"   # False on the login node is normal
```

Full instructions: [`docs/SETUP.md`](docs/SETUP.md). Cluster specifics: [`docs/CLUSTER_GUIDE.md`](docs/CLUSTER_GUIDE.md).

---

## 4. Step-by-step run

Run everything from the repository root (`~/Sure_Twin`) with the venv active.

### Step 0 - Clean previous runs (optional)

```bash
rm -rf checkpoints checkpoints_eval datasets/MILAN_val
rm -f out_*.log eval_*.log
```
**Do NOT delete `datasets/MILAN`** unless you still have `milan_raw.csv`: the `.npy` files there are the only copy of the data. Keep a backup (`cp -r datasets/MILAN ~/MILAN_backup`).

### Step 1 - Convert the data (only if you have the CSV)

```bash
python src/data/convert_milan_to_basicts.py
```
- What: pivots the CSV (`time, cell_id, y`) to `[T, N]`, fills missing timestamps and NaN with 0, builds 4 time features, splits 60/20/20.
- Output: `datasets/MILAN/{train,val,test}_{data,timestamps}.npy` and `meta.json`. Printed: step of 10 min and the shapes of data `(T, 2500)` and timestamps `(T, 4)`.
- `cell_ids.npy` (cell id of each column) is **not** written by the converter. It is only needed for `--exclude`; without it the analysis scripts read the ids from `milan_raw.csv`.

### Step 2 - Create the val view

```bash
python src/data/make_split_view.py --split val
```
- Output: `datasets/MILAN_val`, and a line "`'test' of this view = original val split: ... steps`".

### Step 3 - Train the ensemble, then evaluate it on val

```bash
TRAIN=$(sbatch --parsable scripts/slurm/run_train.sh)
sbatch --dependency=afterok:$TRAIN scripts/slurm/run_eval.sh val
squeue -u $USER
```
- What: 10 independent trainings (best checkpoint = lowest validation MAE, early stopping, 100 epochs max), then the same checkpoints are evaluated on val.
- Output: `checkpoints/STID_MILAN/seed<k>/<run>/` (checkpoint + `test_results/{prediction,targets}.npy`) and `checkpoints_eval/MILAN_val/seed<k>/...`.
- Check:
```bash
ls checkpoints/STID_MILAN/seed*/*/STID_best_val_MAE.pt | wc -l                    # 10
ls checkpoints_eval/MILAN_val/seed*/*/test_results/prediction.npy | wc -l         # 10
```
- A single failed seed: read `out_<jobid>_<seed>.log`, rerun with `sbatch --array=<seed> scripts/slurm/run_train.sh`. If the checkpoint already existed, delete that seed's folder first, otherwise `eval_split.py` finds more than one.

### Step 4 - Accuracy analysis and plots

```bash
python src/analysis/analyse_result.py
python src/plotting/plot_timeseries.py        # add --horizon 6, --cells ..., --band 1.645 if needed
python src/plotting/plot_signed_error.py      # add --horizon 6, --cell 1800, --seed 0 if needed
```
- `analyse_result.py` prints sanity checks, then a table (MAE, RMSE, WAPE) of two baselines, each seed and the ensemble mean, then how informative the ensemble spread is. The learning-curve panel reads the training logs `out_*.log` from the current directory (`--logs` to change the pattern).
- `plot_signed_error.py` also prints bias, MAE and the time of the largest mean error.
- Sanity checks that must pass: `max |targets - raw test|` about 0 (here 0.0005), `windows evaluated: 709 of 709`, `cells evaluated: 2500 of 2500`, and recomputed normalized MAE equal to the BasicTS one for every seed.

### Step 5 - Conformal upper bounds

```bash
python src/analysis/conformal_eval.py --horizon 0 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/analysis/conformal_eval.py --horizon 2 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
`--horizon 0` = 10 min ahead, `2` = 30 min. Calibration = val split, evaluation = whole test.

Output, for target coverage 90 / 95 / 99%, one row per method:

| Column | Meaning | Good value |
|---|---|---|
| `cover` | fraction of (window, cell) pairs with truth <= bound (dimensionless) | about 1 - eps |
| `margin`, `marg%` | mean (bound - prediction), in [a.u.] and relative to mean activity | small at equal coverage (more energy saved) |
| `level / clock / day min/max` | worst and best coverage over predicted-level deciles, 4-hour blocks, days | min close to max |

Also printed: within-level Spearman between ensemble spread and |error|, and a per-day table (MAE, mean activity, coverage).

Methods: `point`, `gaussian ensemble (uncalibrated)`, `conformal global`, `conformal by level`, `conformal / spread`, `conformal by level / spread`.

### Step 6 - Time-of-day schedule and final metrics

```bash
python src/analysis/schedule_conformal.py --horizon 0 --day_start 9 --day_end 18 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/analysis/schedule_final_metrics.py \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
- `schedule_conformal.py`: coverage 90% inside a daytime window and 75% outside it, with margins calibrated separately per period and per predicted-level decile. Defaults: window 8-21 h, `--day_eps 0.10`, `--night_eps 0.25`; the command above uses the chosen window (section 5).
- `schedule_final_metrics.py`: repeats the comparison for 3 horizons x 3 windows (`--horizons`, `--windows` to restrict, for example `--horizons 0 --windows 9-18`) and writes the results to `final_metrics/` (unmet activity % and over-provisioning %, both dimensionless percentages of total activity). The per-run options of `schedule_conformal.py` (`--horizon`, `--day_start`, `--day_end`) are overridden there.

### Optional plots

```bash
python src/plotting/plot_confident.py --horizon 0 --eps 0.05 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/plotting/busiest_levels.py --horizon 0 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/plotting/plot_concentration.py
```

### Where the outputs go

| Script | Output |
|---|---|
| `analyse_result.py` | `results/figures/accuracy/analysis.png` |
| `plot_timeseries.py` | `results/figures/accuracy/timeseries.png` |
| `plot_signed_error.py` | `results/figures/accuracy/signed_error.png` |
| `plot_confident.py` | `results/figures/conformal/confident_pred.png`, `busiest_confident.png` |
| `busiest_levels.py` | `results/figures/conformal/busiest_levels.png` |
| `schedule_conformal.py` | `results/figures/schedule/schedule_summary.png`, `schedule_busiest.png`; margin tables in `results/tables/schedule_margins.json` |
| `schedule_final_metrics.py` | `final_metrics/` |
| `plot_concentration.py` | `results/figures/data/concentration.png` |

Save the console output of an analysis run under `results/logs/` (for example `python ... | tee results/logs/analysis_<date>.txt`).

---

## 5. Chosen configuration

**Horizon 10 min, window 9-18 h** (90% coverage inside the window, 75% outside), approach *scheduled, calibrated per period*. Motivation, full table and limitations: [`final_metrics/README.md`](final_metrics/README.md).

---

## 6. Current state (run of 14 Nov 2026)

Full discussion in [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md). Summary:

**Accuracy** (original scale, test, ensemble of 10; MAE and RMSE in [a.u.], WAPE dimensionless)

| | MAE [a.u.] | RMSE [a.u.] | WAPE |
|---|---|---|---|
| last value | 27.04 | 65.71 | 0.173 |
| seasonal naive (24 h) | 28.61 | 73.49 | 0.183 |
| single seeds | 16.30 - 16.76 | 38.59 - 40.06 | 0.105 - 0.107 |
| **ensemble mean** | **16.32** | **38.84** | **0.105** |

**Upper bound** (`conformal by level`, calibrated on val, evaluated on test)

| Horizon | Target | Coverage | Margin [a.u.] (% of mean activity) |
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
- The test covers only about 5 days (Tue-Sat, one partial weekend day): no holidays or events, so robustness to anomalous days is not tested.
- Coverage is marginal (over all cells and windows), not guaranteed per cell, and windows are strongly autocorrelated, so the finite-sample guarantee is only approximate.
- The margin and the over-provisioning / unmet percentages are expressed in activity units, not in energy. Turning them into power or energy requires a traffic-to-power model, which is not part of this dataset.

---

## 7. Documentation index

| Document | Content |
|---|---|
| [`docs/SETUP.md`](docs/SETUP.md) | Installation (automatic and manual) |
| [`docs/CLUSTER_GUIDE.md`](docs/CLUSTER_GUIDE.md) | UniTrento GPU cluster: SLURM, CUDA, job scripts |
| [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | Meaning of the variable, data format, checks and decisions |
| [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md) | Interpretation of accuracy and conformal results |
| [`final_metrics/README.md`](final_metrics/README.md) | Choice of horizon and window for the schedule |