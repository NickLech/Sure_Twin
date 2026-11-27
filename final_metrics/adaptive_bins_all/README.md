# Finer bins on the busiest decile: all horizons and windows

Base schedule = *scheduled, calibrated per period* with 10 deciles of the forecast level (the proposal). Tables 1 and 2 compare it with the variant **deciles+top8**; the summary of all the variants is in the section "All variants".
All values on the test set; costs and metrics are percentages of the total true activity index (arbitrary units, [a.u.]); they are not energy.

J = λ · unmet + (1 − λ) · over-provisioning, with λ = 1 − ε in each band (defaults: 0.90 in the window, 0.75 outside). Lower is better.

## How these results were produced

Three runs of `src/analysis/schedule_adaptive_bins.py`, one per window (the script takes one window per run), each on the 3 horizons (10, 20, 30 min), with 200 bootstrap iterations (resampling days) for the confidence intervals:

```bash
for W in "9 18" "9 19" "8 21"; do
  set -- $W
  nohup python src/analysis/schedule_adaptive_bins.py --horizons 1 2 3 \
      --day_start $1 --day_end $2 \
      --n_bins_list 40 --top_splits 2 4 8 \
      --bootstrap_iters 200 \
      --out_dir final_metrics/adaptive_bins_all/w$1-$2 \
      --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*" \
      > log_bins_w$1-$2.txt 2>&1 &
done
```

The console logs are in `results/logs/adaptive_bins_w<window>_2026-11-27.txt`. Per window, `final_metrics/adaptive_bins_all/w<window>/` has `adaptive_bins_metrics.csv` (all variants, global / per cell tier / per dynamic state) and the frontier and gain figures. The tables below are built from the CSVs with:

```bash
python src/analysis/make_bins_table.py "final_metrics/adaptive_bins_all/w*/adaptive_bins_metrics.csv" --variant deciles+top8
```

The script prints Table 1 and Table 2 (use `--variant "40 bins"` or another variant to compare it instead); they are pasted below.

## What is compared

- **Base** (the proposal): *scheduled, calibrated per period*, a margin per period and per decile of the forecast level (10 bins of the same size).
- **deciles+topK**: the first 9 deciles stay as they are, the last one (the busiest 10% of the forecasts) is split into K equal sub-bins: 11 bins for K = 2, 13 for K = 4, 17 for K = 8.
- **40 bins**: 40 bins of the same size (reference for "just use more bins").

## All variants: summary over the 9 combinations (3 horizons x 3 windows)

Range over the 9 combinations. ΔJ is the relative change of the weighted cost J against the base, per time band (negative = better). The last two columns compare each variant with the base whose margins are scaled by a constant α (see Table 2): change of unmet at equal over-provisioning and of over-provisioning at equal unmet, in percentage points.

| Variant | Bins | ΔJ window | ΔJ other hours | Δ unmet at equal over-prov. | Δ over-prov. at equal unmet | 95% CI excludes 0 |
|---|---|---|---|---|---|---|
| deciles+top2 | 11 | -1.2% to -1.5% | -0.1% to -0.3% | -0.032 to -0.034 | -0.24 to -0.31 | 9 of 9 |
| deciles+top4 | 13 | -1.7% to -2.1% | -0.2% to -0.4% | -0.045 to -0.048 | -0.34 to -0.45 | 9 of 9 |
| **deciles+top8** | **17** | **-2.1% to -2.5%** | **-0.3% to -0.5%** | **-0.053 to -0.056** | **-0.40 to -0.53** | **9 of 9** |
| 40 bins | 40 | -2.0% to -2.4% | -0.3% to -0.5% | -0.057 to -0.060 | -0.44 to -0.57 | 9 of 9 |

**Reading.** Splitting the busiest decile gives a small but consistent gain in every combination: about 2% of J in the window and less than 0.5% outside it. It grows with the number of sub-bins and with K = 8 reaches the level of 40 equal bins, using 17 bins instead of 40: the gain comes from the busiest forecasts. The gain is modest: for example at 10 min and window 9-18 the busiest 3% of the cells go from a coverage of 0.85 to about 0.91 (target 0.90), but the cost of the other cells is untouched. We keep it as a result and the base schedule stays the proposal.

## Table 1: weighted cost J

| Horizon | Window | J window base | J window deciles+top8 | Δ J window | J other hours base | J other hours deciles+top8 | Δ J other hours |
|---|---|---|---|---|---|---|---|
| 10 min | 9-18 | 2.62 | 2.55 | -2.5% | 4.23 | 4.21 | -0.5% |
| 10 min | 9-19 | 2.63 | 2.56 | -2.5% | 4.26 | 4.24 | -0.4% |
| 10 min | 8-21 | 2.68 | 2.62 | -2.4% | 4.24 | 4.22 | -0.4% |
| 20 min | 9-18 | 2.83 | 2.77 | -2.2% | 4.56 | 4.54 | -0.4% |
| 20 min | 9-19 | 2.84 | 2.77 | -2.3% | 4.59 | 4.58 | -0.3% |
| 20 min | 8-21 | 2.90 | 2.84 | -2.1% | 4.57 | 4.56 | -0.3% |
| 30 min | 9-18 | 2.93 | 2.87 | -2.2% | 4.72 | 4.70 | -0.3% |
| 30 min | 9-19 | 2.94 | 2.87 | -2.3% | 4.75 | 4.74 | -0.3% |
| 30 min | 8-21 | 3.00 | 2.93 | -2.1% | 4.74 | 4.73 | -0.3% |

## Table 2: gain against the scaled-margin frontier

Each variant is compared with the base schedule whose margins are multiplied by a constant α, chosen so that it matches the variant either on over-provisioning or on unmet. Negative = the variant is better than just scaling the margin. A gain is considered real when the 95% bootstrap interval (resampling days) excludes 0. In the CSV the two measures are identical for the three time bands.

| Horizon | Window | Δ unmet at equal over-prov. [95% CI] | Δ over-prov. at equal unmet [95% CI] | Gain significant |
|---|---|---|---|---|
| 10 min | 9-18 | -0.055 [-0.066, -0.046] | -0.408 [-0.468, -0.357] | yes |
| 10 min | 9-19 | -0.055 [-0.065, -0.047] | -0.431 [-0.487, -0.382] | yes |
| 10 min | 8-21 | -0.055 [-0.066, -0.048] | -0.511 [-0.576, -0.442] | yes |
| 20 min | 9-18 | -0.053 [-0.068, -0.041] | -0.401 [-0.482, -0.330] | yes |
| 20 min | 9-19 | -0.053 [-0.069, -0.042] | -0.426 [-0.508, -0.353] | yes |
| 20 min | 8-21 | -0.055 [-0.068, -0.044] | -0.513 [-0.596, -0.423] | yes |
| 30 min | 9-18 | -0.054 [-0.071, -0.042] | -0.410 [-0.492, -0.339] | yes |
| 30 min | 9-19 | -0.054 [-0.070, -0.042] | -0.436 [-0.521, -0.362] | yes |
| 30 min | 8-21 | -0.056 [-0.072, -0.044] | -0.528 [-0.620, -0.432] | yes |

## Limitations

- J uses λ = 1 − ε, the loss the schedule is designed to minimise: it is consistent with the method but not an independent validation. The λ sweep shows how the ranking changes with λ.
- The test set covers a limited number of days, so the bootstrap intervals are indicative.
- J for "all hours" is not shown: it is not comparable across windows (changing the window changes which hours use each λ).
