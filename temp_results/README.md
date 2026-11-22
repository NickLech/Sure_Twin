# Configuration choice: horizon 10 min, window 9-18 h

**Chosen configuration: horizon = 10 min, window = 9-18 h** (90% in window, 75% outside window), approach *scheduled, calibrated per period*.

## Criterion

For each time band (window and other hours) we look for the lowest bar on two metrics:

- **Unmet activity (%)**: activity not covered, as % of the total true activity (lower = safer)
- **Over-provisioning (%)**: capacity reserved but unused, as % of the total true activity (lower = less waste)

Both metrics are dimensionless percentages of the total true activity index (arbitrary units, [a.u.]); they are not energy.

The "all hours" column is ignored. Values computed on the test set (`schedule_final_metrics_all.csv`), *scheduled* approach.

## Results

| Horizon | Window | Unmet window | Unmet other hours | Over-prov. window | Over-prov. other hours |
|---|---|---|---|---|---|
| **10 min** | **9-18** | **1.06** | **2.62** | **16.60** | 9.08 |
| 10 min | 9-19 | 1.06 | 2.65 | 16.72 | 9.09 |
| 10 min | 8-21 | 1.08 | 2.65 | 17.14 | 9.01 |
| 20 min | 9-18 | 1.14 | 2.83 | 18.09 | 9.76 |
| 20 min | 9-19 | 1.13 | 2.86 | 18.19 | 9.77 |
| 20 min | 8-21 | 1.16 | 2.88 | 18.60 | 9.66 |
| 30 min | 9-18 | 1.17 | 2.90 | 18.77 | 10.17 |
| 30 min | 9-19 | 1.17 | 2.94 | 18.85 | 10.20 |
| 30 min | 8-21 | 1.19 | 2.96 | 19.27 | 10.09 |

## Why horizon 10 min

With the window held fixed, going from 10 to 20 or 30 min worsens **all four** metrics (with 9-18: unmet in window 1.06 → 1.14 → 1.17, over-provisioning in window 16.60 → 18.09 → 18.77). A shorter horizon gives more reliable forecasts, so there is no trade-off: 10 min is better for both coverage and waste.

## Why window 9-18

At horizon 10 min the three windows are very close, but 9-18 is the best or tied on 3 metrics out of 4:

- **Unmet in window**: 1.06, tied with 9-19 (8-21: 1.08)
- **Unmet other hours**: 2.62, the lowest (9-19 and 8-21: 2.65)
- **Over-provisioning in window**: 16.60, the lowest (9-19: 16.72, 8-21: 17.14)
- **Over-provisioning other hours**: 9.08, practically identical to the others (9.01-9.09), negligible difference

## Limitations

Differences between the three windows are small (≤ 0.55 percentage points in over-provisioning, ≤ 0.03 in unmet). The choice of 9-18 is therefore a marginal optimization; the choice of horizon 10 min is clear-cut.
