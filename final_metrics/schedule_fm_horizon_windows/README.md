# Configuration choice with weighted cost J: horizon 10 min, window 9-18 h

**The choice does not change when the cost is added: horizon = 10 min, window = 9-18 h** (90% in window, 75% outside window), approach *scheduled, calibrated per period*.

## Criterion

Same as the previous table (unmet and over-provisioning per time band), plus the weighted cost of each band:

J = λ · unmet + (1 − λ) · over-provisioning

with λ = 0.90 in the window and λ = 0.75 outside the window (λ = 1 − ε, the weights the schedule is calibrated for; λ = 1: only QoS, λ = 0: only energy). Lower J is better. Values computed on the test set (`schedule_final_metrics_all.csv`, column `cost_J`), *scheduled* approach.

## Results

| Horizon | Window | Unmet window | Unmet other hours | Over-prov. window | Over-prov. other hours | J window (λ=0.90) | J other hours (λ=0.75) |
|---|---|---|---|---|---|---|---|
| **10 min** | **9-18** | **1.06** | **2.62** | **16.60** | 9.08 | **2.62** | **4.23** |
| 10 min | 9-19 | 1.06 | 2.65 | 16.72 | 9.09 | 2.63 | 4.26 |
| 10 min | 8-21 | 1.08 | 2.65 | 17.14 | 9.01 | 2.68 | 4.24 |
| 20 min | 9-18 | 1.14 | 2.83 | 18.09 | 9.76 | 2.83 | 4.56 |
| 20 min | 9-19 | 1.13 | 2.86 | 18.19 | 9.77 | 2.84 | 4.59 |
| 20 min | 8-21 | 1.16 | 2.88 | 18.60 | 9.66 | 2.90 | 4.57 |
| 30 min | 9-18 | 1.17 | 2.90 | 18.77 | 10.17 | 2.93 | 4.72 |
| 30 min | 9-19 | 1.17 | 2.94 | 18.85 | 10.20 | 2.94 | 4.75 |
| 30 min | 8-21 | 1.19 | 2.96 | 19.27 | 10.09 | 3.00 | 4.74 |

## Why horizon 10 min

With the window held fixed, going from 10 to 20 or 30 min worsens J in both bands (with 9-18: J window 2.62 → 2.83 → 2.93, J other hours 4.23 → 4.56 → 4.72). The same holds for the other two windows.

## Why window 9-18

At every horizon, 9-18 has the lowest J in both bands (at 10 min: J window 2.62 against 2.63 for 9-19 and 2.68 for 8-21; J other hours 4.23 against 4.26 and 4.24). Adding the cost therefore confirms the choice made on unmet and over-provisioning alone.

## Limitations

- The differences between the three windows are very small (≤ 0.07 in J at 10 min; 9-18 and 9-19 are practically tied in the window band). The choice of 9-18 is a marginal optimization; the choice of horizon 10 min is clear-cut.
- J for "all hours" is not shown on purpose: it is not comparable across windows, because changing the window also changes which hours are weighted with λ = 0.90 and which with λ = 0.75 (a wider window looks cheaper only for this reason).
- λ = 1 − ε is the loss the schedule is designed to minimise, so this comparison is consistent with the method but is not an independent validation. The λ sweep (`schedule_final_metrics_lamsweep*.csv`, horizon 10 min, window 9-18) shows for which λ the scheduled approach is the cheapest: λ ≥ 0.92 in the window, 0.67 ≤ λ ≤ 0.77 outside the window.
- Costs are percentages of the total true activity index (arbitrary units, [a.u.]); they are not energy.
