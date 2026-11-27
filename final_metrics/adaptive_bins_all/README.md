# Base schedule vs deciles+top8: all horizons and windows

Base schedule = *scheduled, calibrated per period* with 10 deciles of the forecast level (the proposal). Variant = **deciles+top8**.
All values on the test set; costs and metrics are percentages of the total true activity index (arbitrary units, [a.u.]); they are not energy.

J = λ · unmet + (1 − λ) · over-provisioning, with λ = 1 − ε in each band (defaults: 0.90 in the window, 0.75 outside). Lower is better.

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
