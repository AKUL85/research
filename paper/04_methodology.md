# Methodology reconstructed from executed code

This is an audit description, not a paper draft. The cleaned CSV is the start of the reproducible chain; upstream construction of that CSV is unavailable.

## 1. Data preparation available in-repository

Stage 1 reads `bihs_r3_cleaned (1).csv` (5,605 x 137), checks required columns, retains all rows, and does not impute, re-winsorize, sample, or drop households. Raw missing values would remain NaN, dimension means would use observed indicators with a saved count, and LVI would require all three dimensions. In the saved dataset every LVI indicator is complete.

Constructed variables are:

```text
agricultural_occupation_dependence = n_occ_agri_own + n_occ_agri_labour
meals_per_person_week = mealdays_total_7d / hh_size
seasonal_wind_avg = mean(aus, aman, boro wind)
seasonal_evapotranspiration_avg = mean(aus, aman, boro evapotranspiration)
has_current_loan_binary = 1 if has_current_loan == 1, else 0 for code 2
adult_illiteracy_rate = adult_literacy_rate values unchanged, renamed
```

The literacy rename/direction is based on repository correlations, not an external data dictionary. Stage 3/Lite/ablation add columns from the cleaned file by row position rather than an ID merge.

## 2. LVI indicators and formulas

| Dimension | Indicator | Vulnerability direction |
|---|---|---|
| Exposure | `flood_affected` | Higher = more |
| Exposure | `flood_depth_ft_mean` | Higher = more |
| Exposure | `seasonal_wind_avg` | Higher = more |
| Exposure | `seasonal_evapotranspiration_avg` | Higher = more |
| Sensitivity | `agricultural_occupation_dependence` | Higher = more |
| Sensitivity | `dependency_ratio` | Higher = more |
| Sensitivity | `hh_size` | Higher = more |
| Sensitivity | `meals_per_person_week` | Higher = less |
| Adaptive Capacity | `income_per_capita_monthly` | Higher = less |
| Adaptive Capacity | `mean_edu_years_adults` | Higher = less |
| Adaptive Capacity | `adult_illiteracy_rate` | Higher = more |
| Adaptive Capacity | `livelihood_diversity` | Higher = less |
| Adaptive Capacity | `any_nonfarm_agri_work` | Higher = less |
| Adaptive Capacity | `land_cultivable_decimal` | Higher = less |
| Adaptive Capacity | `has_current_loan_binary` | Higher = less |

For higher-is-more indicators:

```text
I = (x - sample_min) / (sample_max - sample_min)
```

For higher-is-less indicators:

```text
I = (sample_max - x) / (sample_max - sample_min)
```

A constant indicator would be set to zero and flagged; none is constant. Dimension scores are unweighted arithmetic means of their 4, 4, and 7 normalized indicators:

```text
Exposure = mean(E1..E4)
Sensitivity = mean(S1..S4)
Adaptive-capacity vulnerability = mean(A1..A7)
LVI = mean(Exposure, Sensitivity, Adaptive-capacity vulnerability)
```

Thus dimensions have equal one-third weight, but individual indicators do not: Exposure/Sensitivity indicators each receive 1/12 and Adaptive Capacity indicators each receive 1/21.

## 3. Vulnerability classification

Stage 1 creates no class. Stage 2b and Stage 3 compute full-sample LVI terciles:

```text
Low:    LVI <= 0.4325604289795757                 n=1,869
Medium: 0.4325604289795757 < LVI <= 0.4937240489502903  n=1,868
High:   LVI > 0.4937240489502903                  n=1,868
```

The primary supervised target is High=1 vs Low+Medium=0. These are internal relative classes, not externally validated vulnerability thresholds.

Stage 4 also defines `socio_half = mean(sensitivity_score, adaptive_capacity_vulnerability_score)` and its High tercile using cut points 0.44067247665 and 0.49812172868. This removes the Exposure dimension from the corrected target.

## 4. PCA and clustering

Stage 2 uses all 15 normalized indicators, excluding LVI and dimension scores. Although already in [0,1], features are z-scored because their variances differ. Full PCA is fit on all 5,605 households; the smallest component count reaching 80% cumulative variance is retained: 9 PCs, 81.1714%.

K-Means uses these 9 scores with k=2..8, k-means++ initialization, 20 restarts, maximum 500 iterations, and seed 42. k is selected by majority preference among silhouette, Davies-Bouldin, and Calinski-Harabasz; k=2 wins two of three. Final labels are reordered by mean LVI. Ward linkage on the same PC space is cut at k=2 and compared by ARI.

Cluster profiles report means plus ANOVA and Kruskal-Wallis tests. Those tests describe variables used to build the clusters; they are not independent validation. Stage 2b creates LVI-tertile cross-tabs and six descriptive figures.

## 5. Original supervised ML

Stage 3 uses 25 features: household flood occurrence/depth, seasonal district climate/crop variables, two Stage 1 seasonal averages, and a numeric ID for 39 reconstructed climate profiles. Sensitivity and Adaptive Capacity variables are excluded, but four included Exposure variables are direct target components.

- Split: independent calls to 80/20 target-stratified household splits, seed 42; n_train=4,484, n_test=1,121.
- Training selection: 5-fold shuffled StratifiedKFold within training.
- Models: majority baseline; scaled Logistic Regression; Random Forest; XGBoost.
- Binary tuning score: ROC-AUC. Three-class tuning score: accuracy.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, binary Brier score, confusion matrix. Three-class precision/recall/F1 and OvR AUC are macro-averaged.
- SHAP: TreeExplainer for binary XGBoost on the test partition; two global and three selected local plots.

No train/validation predictions or CV tables are saved. The report's local SHAP prose does not match the plotted cases.

## 6. Stage 3 Lite and ablation

Lite expands to 39 features with six plot-level environmental variables, three aridity ratios, two flood interactions, and three yield-stress variables. Binary XGBoost hyperparameters are fixed. A 5-fold training OOF curve tests thresholds 0.20..0.70 in 0.005 steps and selects tau=0.285 for maximum F1. The ensemble tau=0.40 is hard-coded; it is not selected by that curve. Nominal XGBoost, Frank-Hall ordinal XGBoost, and soft voting are evaluated for three classes.

The ablation uses one binary-stratified 80/20 index split for both tasks:

- Tier 1: no features/majority.
- Tier 2: 39 exposure/environmental features.
- Tier 3: Tier 2 plus `head_age`, `head_sex`, `head_marital`, `school_distance_km_mean`, `hh_type` (p=44).
- Tier 4: Tier 3 plus the 11 Sensitivity/Adaptive Capacity indicators (p=55).

Fixed XGBoost models are fit once per tier. Because the target is computed from the variables added across Tiers 2 and 4, the progression is a reconstruction ablation, not a leakage-free prediction experiment.

## 7. Latest leakage audit/corrected experiment

Stage 4 verifies that the mean of the four Exposure inputs equals `exposure_score` and the mean of the three dimension scores equals LVI. It then uses 22 climate/agricultural variables that do not enter LVI directly and compares fixed LR/RF/XGBoost under:

- 5-fold shuffled stratified OOF prediction; and
- 5-fold `GroupKFold` holding out reconstructed climate-profile groups.

Targets are the High socioeconomic-half tercile and, for comparison, the High full-LVI tercile. Metrics are pooled OOF accuracy at 0.50 and ROC-AUC. There is no nested tuning, confidence interval, fold-level table, or final external/temporal test.

## 8. Reproducibility boundary

Stage 1 reports Python 3.12.10, pandas 3.0.6, and NumPy 2.5.2. Stage 2 additionally reports scikit-learn 1.9.1 and SciPy 1.18.1. Later stages import XGBoost, SHAP, Matplotlib and Seaborn but do not record/pin versions. Seeds are 42 where stochastic methods are used. Exact numerical provenance after the cleaned CSV is mostly traceable, but the raw-data, cleaning, merging, package environment, survey design, and source-license chain is incomplete.
