# Extension 2 and 3: public-source welfare validation

Date: 2026-09-24. This record supersedes the **welfare-outcome availability** and **welfare-validation status** statements in `12_extension2_outcome_availability.md` and `14_extension3_validation_status.md`. The female X5/FIES module remains unavailable, so no food-insecurity outcome was tested.

## Source and critical key correction

The public [World Bank reproducibility archive](https://reproducibility.worldbank.org/catalog/540) (DOI [10.60572/1z28-h024](https://doi.org/10.60572/1z28-h024)) distributes BIHS Round 3 male Module A identification, Round 3 sampling weights, and a separate household expenditure aggregate. The three source `.dta` files used here are preserved in `data/extension3_public_source/`; SHA-256 digests are recorded in `outputs/extension3_outcome_validation/public_welfare_link_audit.json`. The underlying BIHS dataset is [IFPRI Round 3](https://doi.org/10.7910/DVN/NXKLZJ). The public archive does **not** include the female X5/FIES module.

The received local `hhid2` column is unusable for direct matching: **0/5,605** local IDs equal the true official ID for the same row. The local survey fields instead match the officially weighted Module A sample sorted by numeric `a01`: all **5,605/5,605** rows agree on 13 shared Module A fields (`hh_type`, `a10`, `a11`, `a13`–`a15`, `a23`–`a27`, interview month and year), and local household size agrees with the separately released expenditure file on all 5,605 rows. The derived `a01` and corrected `hhid2` were saved in `public_welfare_link.csv`; the original cleaned file was not modified. The linked source has 64 documented district codes and 275 community IDs. One of 5,605 linked records lacks the expenditure measure. The IFPRI news release's 5,604-household count versus the 5,605-row weighted source subset remains a distinct source-definition question; the local missing `hhid2` is an alignment artifact, not evidence by itself of an extra household.

## Extension 2: criterion association

The separate outcome is the public file's monthly per-capita expenditure aggregate `pcm1_fx_nfx_hrentq_uval`, analyzed unweighted and on the log scale. The Stata label sums food, nonfood, and rent/use-value components. This is contemporaneous welfare expenditure, not FIES or a later hardship outcome.

| Index specification group | Nonmissing outcome n | Median monthly per-capita expenditure |
|---|---:|---:|
| Strict robust High across all 11 variants | 705 | 3,413.95 |
| Switcher | 2,353 | 3,934.28 |
| Never High | 2,543 | 4,303.25 |
| Incomplete/uncertified | 3 | 3,510.49 |

The strict robust-High minus never-High mean log-expenditure difference is **−0.1948** (2,000-resample district-bootstrap 95% interval **−0.2733 to −0.1077**), corresponding to a geometric-mean ratio of **0.823**. The continuous LVI has Spearman correlation **−0.1259** with log expenditure. The direction is consistent with lower measured welfare in the robust-High group, but income and meals already enter the index, so conceptual overlap remains. These are descriptive sample associations.

## Extension 3: held-out-district baseline comparison

Five-fold GroupKFold holds each of the 64 source-verified districts wholly out of training. The same 5,604 nonmissing-outcome households enter every fixed-specification comparison. The outcome is log monthly per-capita expenditure; the demographic baseline uses head age, head sex, and household size. Each fitted model is median-imputed and ridge-regularized with fixed `alpha=1`; no outcome-driven hyperparameter search was performed.

| Comparison | Pooled log MAE | Pooled log R² |
|---|---:|---:|
| Training-mean outcome | 0.3723 | −0.0084 |
| Continuous LVI only | 0.3683 | 0.0113 |
| Demographics | 0.3578 | 0.0733 |
| Demographics + LVI | 0.3542 | 0.0879 |
| Demographics + strict robust-High flag | 0.3566 | 0.0779 |

The equal-district mean MAE change from adding LVI to demographics is **−0.00237** (district-bootstrap 95% interval **−0.00606 to 0.00148**); 44 of 64 districts improve. Adding strict robust High changes MAE by **−0.00065** (interval **−0.00243 to 0.00131**); 38 districts improve. Both intervals include zero. Thus there is a small point-estimate gain for continuous LVI and little incremental gain for the binary consensus class under this baseline, with uncertain geographic consistency.

## Interpretation limits

The analysis tests a separately prepared contemporaneous welfare measure and holdout by documented district code. It is not FIES validation, later-wave forecasting, or a verified PSU/design-weighted population analysis. The archived index uses full-sample normalization and classification, so the held-out-district estimates are **transductive** with respect to index construction. Climate/crop reference periods remain unverified. The source-data key correction must be incorporated before any new household-level link; earlier direct `hhid2` claims are superseded by this audit.

Artifacts: `outputs/extension2_robust_high/welfare_by_robustness.csv`, `welfare_criterion_summary.json`; `outputs/extension3_outcome_validation/public_welfare_link.csv`, `public_welfare_link_audit.json`, `welfare_oof_predictions.csv`, `welfare_baseline_comparison.csv`, `welfare_district_comparison.csv`, and `welfare_validation_summary.json`.
