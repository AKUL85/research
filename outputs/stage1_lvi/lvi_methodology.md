# Stage 1 - Livelihood Vulnerability Index (LVI) construction

BIHS Round 3 | Generated 2026-09-19 | n = 5605 households

## 1. Approach

The LVI follows the balanced weighted-average composite-index approach.
Every indicator is rescaled to `[0, 1]`, oriented so a higher value always
means greater vulnerability, averaged with equal weights within its
dimension, and the three dimension scores are averaged with equal weights
into the composite.

## 2. Normalization

For an indicator where a higher raw value means **greater** vulnerability:

```
I = (x - min(x)) / (max(x) - min(x))
```

For an indicator where a higher raw value means **lower** vulnerability
(all Adaptive Capacity indicators except `adult_illiteracy_rate`, plus
`meals_per_person_week`):

```
I = (max(x) - x) / (max(x) - min(x))
```

`min(x)` and `max(x)` are the observed sample minimum and maximum, listed
per indicator in `lvi_indicator_summary.csv`. Missing values are preserved
as `NaN`. A constant indicator (`max == min`) is set to `0.0` rather than
dividing by zero and is flagged; no indicator in this dataset is constant.
No second winsorization is applied - the input is already cleaned.

## 3. Dimension scores

```
D_exposure    = mean(4 normalized Exposure indicators)
D_sensitivity = mean(4 normalized Sensitivity indicators)
D_adaptive    = mean(7 normalized Adaptive Capacity indicators)
```

Written as `exposure_score`, `sensitivity_score` and
`adaptive_capacity_vulnerability_score`. Each is accompanied by a
`*_n_indicators` column recording how many indicators were available for
that household, so partial scores are never silent.

## 4. Composite

```
LVI = (exposure_score + sensitivity_score
       + adaptive_capacity_vulnerability_score) / 3
```

Range `[0, 1]`; higher = more vulnerable. Households are not classified
into Low/Medium/High at this stage.

## 5. Indicator placement

### Exposure (4 indicators)

| Indicator | Direction | Normalization | Min | Max |
|---|---|---|---|---|
| `flood_affected` | higher = more vulnerable | `(x - min) / (max - min)` | 0.0000 | 1.0000 |
| `flood_depth_ft_mean` | higher = more vulnerable | `(x - min) / (max - min)` | 0.0000 | 21.0000 |
| `seasonal_wind_avg` | higher = more vulnerable | `(x - min) / (max - min)` | 1.0902 | 3.8285 |
| `seasonal_evapotranspiration_avg` | higher = more vulnerable | `(x - min) / (max - min)` | 2.8015 | 4.7719 |

### Sensitivity (4 indicators)

| Indicator | Direction | Normalization | Min | Max |
|---|---|---|---|---|
| `agricultural_occupation_dependence` | higher = more vulnerable | `(x - min) / (max - min)` | 0.0000 | 6.0000 |
| `dependency_ratio` | higher = more vulnerable | `(x - min) / (max - min)` | 0.0000 | 6.0000 |
| `hh_size` | higher = more vulnerable | `(x - min) / (max - min)` | 1.0000 | 18.0000 |
| `meals_per_person_week` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 38.0000 |

### Adaptive Capacity (7 indicators)

| Indicator | Direction | Normalization | Min | Max |
|---|---|---|---|---|
| `income_per_capita_monthly` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 12500.0000 |
| `mean_edu_years_adults` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 17.0000 |
| `adult_illiteracy_rate` | higher = more vulnerable | `(x - min) / (max - min)` | 0.0000 | 1.0000 |
| `livelihood_diversity` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 4.0000 |
| `any_nonfarm_agri_work` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 1.0000 |
| `land_cultivable_decimal` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 495.0000 |
| `has_current_loan_binary` | higher = less vulnerable (reversed) | `(max - x) / (max - min)` | 0.0000 | 1.0000 |

## 6. Constructed indicators

```
agricultural_occupation_dependence = n_occ_agri_own + n_occ_agri_labour
meals_per_person_week              = mealdays_total_7d / hh_size
seasonal_wind_avg                  = mean(clim_{aus,aman,boro}_wind_avg)
seasonal_evapotranspiration_avg    = mean(clim_{aus,aman,boro}_evapotrans_avg)
has_current_loan_binary            = 1 if has_current_loan == 1 else 0
adult_illiteracy_rate              = adult_literacy_rate  (renamed, v2)
```

## 7. Caveats carried into the paper

1. **`adult_literacy_rate` is inverted (resolved in v2).** It correlates
   **negatively** with adult schooling (r = -0.50) and with income, i.e. it
   measures illiteracy. Since v2 it enters the index as
   `adult_illiteracy_rate`, forward-normalized (higher = more vulnerable).
   See `lvi_validation_report.txt`, caveat C1 and the v2 changelog.
2. **`has_current_loan` was recoded** from the BIHS 1=Yes/2=No coding to a
   0/1 indicator. Without this the framework would have been inverted.
   Current borrowing proxies credit **access**, not debt burden.
3. **Flood indicators are structurally collinear**: `flood_affected == 1`
   exactly when `flood_depth_ft_mean > 0`, so flooding carries half the
   Exposure dimension.
4. **Climate variables are district-level context** (39 districts), not
   household weather; they vary only between districts.
5. **Equal weighting** means an Adaptive Capacity indicator carries 1/21 of
   the composite against 1/12 for an Exposure indicator.

## 8. Reproducing

```bash
python outputs/stage1_lvi/lvi_construction.py
```

The script is deterministic: no sampling, no randomness, no seed needed.
It reads the cleaned CSV read-only and never writes back to it.