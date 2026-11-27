# Sure_Twin: STID traffic forecasting with conformal upper bounds

Goal: forecast mobile-network activity per cell and build a **calibrated upper bound** on it, to decide safely when a cell can go to sleep (input for a digital twin of a base-station grid).

- **Data:** 2500 cells (50x50 grid, Milan, November 2013), 10-minute steps, split chronologically 60/20/20 (train/val/test). See [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md).
- **Variable:** `y` is an aggregate **cell activity index** (Internet traffic, calls and SMS combined), in arbitrary units **[a.u.]**. It is **not** bytes and **not** power consumption.
- **Model:** STID (BasicTS), input 144 steps (24 h), output 12 steps (10 to 120 min ahead).
- **Ensemble:** 10 STID models, one per seed (0-9).
- **Evaluation:** in the ORIGINAL activity-index scale [a.u.] (de-normalized), on the test split (709 windows, about 4.9 days, Tue-Sat).

Detailed interpretation of the results: [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md).
Final configuration and its motivation: [`final_metrics/schedule_fm_horizon_windows/README.md`](final_metrics/schedule_fm_horizon_windows/README.md).
Finer bins on the busiest decile (3 horizons x 3 windows): [`final_metrics/adaptive_bins_all/README.md`](final_metrics/adaptive_bins_all/README.md).

### Metrics and units

| Quantity | Definition | Unit |
|---|---|---|
| `y`, prediction, true value | activity index of one cell in one 10-minute step | [a.u.] |
| MAE, RMSE | mean absolute / root-mean-square error, original scale | [a.u.] |
| bias | mean signed error (prediction - true); negative = under-estimate | [a.u.] |
| margin | upper bound - prediction | [a.u.] |
| `marg%` | mean margin / mean true activity | % |
| WAPE | sum of absolute errors / sum of true values | dimensionless |
| coverage (`cover`) | fraction of (window, cell) pairs with true value <= upper bound | dimensionless, 0-1 |
| unmet % | sum(max(true - bound, 0)) / sum(true) | % of total true activity |
| over-provisioning % | sum(max(bound - true, 0)) / sum(true) | % of total true activity |
| weighted cost J | `lam * unmet + (1 - lam) * over-provisioning`; `lam = 1` only QoS, `lam = 0` only energy. Default `lam = 1 - eps` per band (0.90 in the window, 0.75 outside) | % of total true activity |
| normalized MAE (BasicTS logs) | MAE on z-scored data | dimensionless |
| horizon | steps ahead x 10 min. `--horizon` is 1-based in every script (1 = 10 min, 3 = 30 min) | min |

Plots show the activity index on the y axis as `activity [a.u.]` and errors or margins as `[a.u.]`. None of these quantities is power or energy (see [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md)).

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
├── final_metrics/            results of the schedule analyses (tables, figures, README)
│   ├── schedule_fm_horizon_windows/   3 horizons x 3 windows: chosen configuration, README
│   ├── adaptive_bins_all/    finer bins on the busiest decile, 3 horizons x 3 windows, README
│   ├── busy_gate/            busy gate results
│   └── schedule_final_metrics_lamsweep*   lambda sweep (10 min, window 9-18); adaptive_bins/ and
│                             adaptive_bins_topsplit/ hold earlier single runs
├── docs/
│   ├── SETUP.md              installation
│   ├── CLUSTER_GUIDE.md      UniTrento GPU cluster (SLURM)
│   └── DATA_NOTES.md         what the data is, checks and decisions
├── src/
│   ├── data/                 convert_milan_to_basicts.py, make_split_view.py
│   ├── training/             train_STID.py, eval_split.py
│   ├── analysis/             analyse_result.py, conformal_eval.py, metrics_common.py,
│   │                         schedule_conformal.py, schedule_final_metrics.py,
│   │                         schedule_busy_gate.py, schedule_adaptive_bins.py,
│   │                         make_bins_table.py
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
- Output: `datasets/MILAN/{train,val,test}_{data,timestamps}.npy`, `meta.json` and `cell_ids.npy` (cell id of each column, needed by `--exclude`). Printed: step of 10 min and the shapes of data `(T, 2500)` and timestamps `(T, 4)`.

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
python src/analysis/conformal_eval.py --horizon 1 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/analysis/conformal_eval.py --horizon 3 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
`--horizon 1` = 10 min ahead, `3` = 30 min. Calibration = val split, evaluation = whole test.

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
python src/analysis/schedule_conformal.py --horizon 1 --day_start 9 --day_end 18 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/analysis/schedule_final_metrics.py \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
- `schedule_conformal.py`: coverage 90% inside a daytime window and 75% outside it, with margins calibrated separately per period and per predicted-level decile. Defaults: window 8-21 h, `--day_eps 0.10`, `--night_eps 0.25`; the command above uses the chosen window (section 5).
- `schedule_final_metrics.py`: repeats the comparison for 3 horizons x 3 windows (`--horizons`, `--windows` to restrict, for example `--horizons 1 --windows 9-18`) and writes the results to `final_metrics/` (unmet activity % and over-provisioning %, both dimensionless percentages of total activity). The per-run options of `schedule_conformal.py` (`--horizon`, `--day_start`, `--day_end`) are overridden there.
- Weighted cost: `schedule_final_metrics.py` adds a third bar chart with `J = lam * unmet + (1 - lam) * over-provisioning` (`lam = 1` only QoS, `lam = 0` only energy) and a `cost_J` column in the CSV. `--lam` is the weight inside the window and `--lam_night` outside it, both in [0, 1]; defaults `1 - day_eps` (0.90) and `1 - night_eps` (0.75), the weights the schedule is calibrated for (a conformal bound at coverage `1 - eps` is the minimiser of J with `lam = 1 - eps`). For the same `lam` everywhere pass it twice, for example `--lam 0.8 --lam_night 0.8`. The cost of 'all hours' combines the two bands with their share of the true activity.
- Lambda sweep: `--lam_sweep` also computes J for `lam` from 0 to 1 (`--lam_points`, default 101) with one `lam` common to every band, and writes `<base>_lamsweep.png`, `schedule_final_metrics_lamsweep.csv` (J on the grid), `..._lamsweep_best.csv` (lam intervals where each approach is the cheapest) and `..._lamsweep_cross.csv` (exact lam where each approach crosses the proposal). It does not return a 'best lam': lam is the relative cost of unmet versus waste, a design choice. It shows for which lam each approach is the cheapest, for example `python src/analysis/schedule_final_metrics.py --horizons 1 --windows 9-18 --lam_sweep --cal_data ... --cal_ckpts ...`.

### Step 7 - Busy gate and global evaluation on all cells

```bash
python src/analysis/schedule_busy_gate.py --horizons 1 --day_start 9 --day_end 18 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
```
- What: the conformal margin is added only where a cell is **busy**, decided at prediction time as `prediction >= tau` (tau in [a.u.]; a cell can be busy at noon and quiet at night). Margins and deciles are calibrated only on busy calibration pairs; quiet pairs keep the point prediction (`--quiet_eps` gives them a smaller coverage target instead).
- `tau`: `--busy_thr auto` (default) picks on **val** the largest tau such that the global unmet % grows by at most `--max_unmet_increase` pp (default 0.25) w.r.t. the ungated schedule; `--busy_thr 1000` fixes it by hand. `--spread_q 0.98` also marks as busy the pairs with a very high ensemble spread.
- Evaluation is always on **all cells** (test): global table, breakdown by static tier of cells (train-mean activity quantiles, `--tiers`), by dynamic state (busy / quiet), and per-cell distribution of unmet %. Compared: STID point, scheduled on all cells (current), scheduled + busy gate (new).
- Output: `final_metrics/busy_gate/` (figures `*_tiers.png`, `*_threshold.png`, `busy_gate_all.json/.csv`) and `results/tables/busy_gate_margins_h<min>.json`. Unmet % and over-provisioning % are % of the total true activity of the group they refer to (dimensionless, not energy); tau and margins in [a.u.].
- Result: not adopted. At 10 min and window 9-18 the gate saves over-provisioning but loses coverage on the quiet pairs, and for the same unmet scaling all the margins by a constant gives less over-provisioning than the gate (14.4% against 15.7% in the window). The tables in `final_metrics/busy_gate/` show the busy / quiet breakdown.

### Step 8 - Finer bins on the busiest decile

```bash
for W in "9 18" "9 19" "8 21"; do
  set -- $W
  nohup python src/analysis/schedule_adaptive_bins.py --horizons 1 2 3 \
      --day_start $1 --day_end $2 \
      --n_bins_list 40 --top_splits 2 4 8 \
      --bootstrap_iters 200 \
      --out_dir final_metrics/adaptive_bins_all/w$1-$2 \
      --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*" \
      > results/logs/adaptive_bins_w$1-$2_<date>.txt 2>&1 &
done
python src/analysis/make_bins_table.py "final_metrics/adaptive_bins_all/w*/adaptive_bins_metrics.csv" --variant deciles+top8
```
- What: the margin is calibrated per period and per bin of the forecast level (the proposal uses 10 bins of the same size). `--top_splits K` keeps the first 9 deciles and splits the busiest one into K sub-bins (9 + K bins); `--n_bins_list N` uses N bins of the same size, as a reference. The script takes one window per run, so there is one run per window, each on the horizons given.
- Compared: STID point, the base schedule and each variant. Each variant is also compared with the base whose margins are scaled by a constant `alpha`, chosen to match the variant on over-provisioning or on unmet; the 95% intervals come from a block bootstrap on days (`--bootstrap_iters`, 0 disables it and makes the run much faster).
- Output: per window, `final_metrics/adaptive_bins_all/w<window>/` with `adaptive_bins_metrics.csv` (global, per cell tier and per busy / quiet state; includes `cost_J`) and the frontier and gain figures. `make_bins_table.py` prints the comparison tables (J per band, gain against the scaled-margin frontier) used in `final_metrics/adaptive_bins_all/README.md`.
- Result: about -2% of J in the window and less than -0.5% outside it in all 9 combinations (95% interval excludes 0 every time); `deciles+top8` (17 bins) matches 40 equal bins. The gain is small, so the base schedule stays the proposal.

### Optional plots

```bash
python src/plotting/plot_confident.py --horizon 1 --eps 0.05 \
    --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
python src/plotting/busiest_levels.py --horizon 1 \
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
| `schedule_final_metrics.py` | `final_metrics/` (with `--lam_sweep`: also `schedule_final_metrics_lamsweep*.csv` and `*_lamsweep.png`) |
| `schedule_busy_gate.py` | `final_metrics/busy_gate/`, `results/tables/busy_gate_margins_h<min>.json` |
| `schedule_adaptive_bins.py` | `final_metrics/adaptive_bins_all/w<window>/` (`adaptive_bins_metrics.csv`, frontier and gain figures) |
| `make_bins_table.py` | tables printed to the console (copied into `final_metrics/adaptive_bins_all/README.md`) |
| `plot_concentration.py` | `results/figures/data/concentration.png` |

Save the console output of an analysis run under `results/logs/` (for example `python ... | tee results/logs/analysis_<date>.txt`).

---

## 5. Chosen configuration

**Horizon 10 min, window 9-18 h** (90% coverage inside the window, 75% outside), approach *scheduled, calibrated per period*. Motivation, full table (unmet, over-provisioning and weighted cost J) and limitations: [`final_metrics/schedule_fm_horizon_windows/README.md`](final_metrics/schedule_fm_horizon_windows/README.md). Adding the weighted cost J does not change the choice.

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

## 6b. Further results (27 Nov 2026)

**Weighted cost J and lambda sweep** (10 min, window 9-18; details in `final_metrics/schedule_final_metrics_lamsweep*.csv`)
- With one `lam` for every band no value makes the scheduled approach the cheapest everywhere: it wins from `lam` about 0.92 inside the window and for `lam` between about 0.67 and 0.77 outside it. This supports a different `lam` per band (0.90 and 0.75 by default).
- At the default weights the scheduled approach ties with the best fixed level in each band (fixed 90% in the window, fixed 75% outside); its advantage is that it is the single policy that reaches both over the 24 hours.
- Below `lam` about 0.66 the point forecast without margins is the cheapest: margins are worth it only if unmet costs more than about twice the waste.
- Caveat: `lam = 1 - eps` is the loss the schedule is designed to minimise, so this is a consistency check, not an independent validation.

**Finer bins on the busiest decile** (3 horizons x 3 windows; `final_metrics/adaptive_bins_all/README.md`): small, consistent gain (about -2% of J in the window), not adopted as the proposal.

---

## 7. Documentation index

| Document | Content |
|---|---|
| [`docs/SETUP.md`](docs/SETUP.md) | Installation (automatic and manual) |
| [`docs/CLUSTER_GUIDE.md`](docs/CLUSTER_GUIDE.md) | UniTrento GPU cluster: SLURM, CUDA, job scripts |
| [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | Meaning of the variable, data format, checks and decisions |
| [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md) | Interpretation of accuracy and conformal results |
| [`final_metrics/schedule_fm_horizon_windows/README.md`](final_metrics/schedule_fm_horizon_windows/README.md) | Choice of horizon and window for the schedule (with weighted cost J) |
| [`final_metrics/adaptive_bins_all/README.md`](final_metrics/adaptive_bins_all/README.md) | Finer bins on the busiest decile: all horizons and windows, how they were produced |
