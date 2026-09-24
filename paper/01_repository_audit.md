# Repository audit

Audit date: 2026-09-22. Scope: all 46 primary repository files, hidden project configuration, Git history, and the hidden auxiliary worktree. The auxiliary `.kilo/worktrees/thinkable-comma` Stage 1/2 files are byte-identical to the main copies and are not separate analyses. No research files were modified.

## Research objective found in the repository

The implemented workflow (1) constructs a household Livelihood Vulnerability Index (LVI) from BIHS Round 3, (2) derives household vulnerability types using PCA and clustering, and (3) predicts LVI tertile classes using environmental and livelihood variables. The latest analysis, Stage 4, audits circular target leakage in Stage 3 and tests exogenous climate/agricultural variables against a socioeconomic-only vulnerability target using district-grouped validation.

## Data inventory and coverage

- Primary input: `bihs_r3_cleaned (1).csv`, described internally as Bangladesh Integrated Household Survey (BIHS) Round 3.
- Shape: 5,605 household rows x 137 columns; no duplicate full rows; one missing value in the entire CSV, the `hhid2` of row 0. There are 5,604 distinct non-missing household IDs and no duplicated non-missing ID.
- Geography: reports say Bangladesh. There is no explicit district identifier or geographic data dictionary. Code reconstructs 39 climate-profile groups and calls them districts; group sizes are 18 to 379 households (median 145). Administrative variables `a10`-`a27` cannot be decoded from repository contents.
- Time: date-like fields `a16_1_mm`/`a16_1_yy` contain 301 observations in 2018-11, 418 in 2018-12, 472 in 2019-01, 1,698 in 2019-02, 2,022 in 2019-03, and 694 in 2019-04. Their meaning is undocumented. Climate/crop reference periods are not recorded.
- Sources absent: no raw BIHS files, survey questionnaire/codebook, data dictionary, climate source, crop-statistics source, sampling weights/design documentation, merge keys, or upstream cleaning script.
- Preprocessing before this repository is unknown. Stage 1 states that the input was already cleaned and winsorized, but no code or metadata verifies that statement.

## Implemented analysis

- Stage 1: 15 indicators (4 Exposure, 4 Sensitivity, 7 Adaptive Capacity), min-max oriented so higher means more vulnerable; equal means within dimensions and equal mean across the three dimensions. All 5,605 rows are retained and complete.
- Stage 2: z-score the 15 normalized indicators; correlation-matrix PCA; retain 9 PCs reaching 81.1714% variance; K-Means for k=2..8; Ward clustering as a robustness comparison.
- Stage 2b: six figures, cluster summaries, and sample-tertile LVI classes.
- Stage 3: binary High-vs-Not-High and three-class Low/Medium/High models using Logistic Regression, Random Forest, and XGBoost; 80/20 stratified household split and 5-fold tuning on the training partition; SHAP for binary XGBoost.
- Stage 3 Lite: engineered environmental features, XGBoost threshold tuning, soft voting, nominal and ordinal three-class models.
- Stage 3 ablation: fixed XGBoost models across baseline, exposure, exposure+demographics, and full-LVI-input tiers.
- Stage 4: algebraic leakage audit plus 5-fold stratified and district-grouped out-of-fold tests using 22 exogenous climate/agricultural variables.

No serialized fitted model, fold assignments, household predictions/probabilities, CV search table, SHAP-value array, or preprocessing object is saved. Saved model outputs are aggregate metric CSVs and figures only.

## Latest completed analysis

- Latest overall: `outputs/stage4_leakage_audit/` (code timestamp 2026-09-20 18:00; results/log 2026-09-20 18:02). It is complete enough to have a results CSV and JSON log, but is untracked in Git.
- Latest tracked: Stage 3 Lite and ablation artifacts, commit `95e137a` dated 2026-09-19 21:08 +0600.
- Authoritative upstream files: Stage 1 v2 artifacts in `outputs/stage1_lvi/`, Stage 2 artifacts in `outputs/stage2_pca_clustering/`, and Stage 2b in `outputs/stage2b_visualization/`.
- Root Stage 3 scripts are exact SHA-256 duplicates of their copies inside the corresponding output folders; they do not represent additional executions.
- Reported input hashes match the current CSV contents after LF-to-CRLF conversion, so the hash differences are line-ending changes, not data-value differences.

## Existing figures, tables, and references

- Figures: 17 PNGs total; all contain 300-DPI metadata. Detailed inventory is in `03_result_registry.md`.
- Tables/results: Stage 1 indicator summary; Stage 2 cluster-profile table; Stage 2b cluster/LVI cross-tab and summary; Stage 3, Lite, ablation, and leakage-audit metric CSVs.
- Literature: only informal mentions of Hahn et al. (2009) and IPCC/livelihood-vulnerability frameworks. There is no bibliography, BibTeX, DOI, reference manager file, or full citation.
- Dependencies: no `requirements.txt`, `pyproject.toml`, Conda file, lockfile, or environment specification. Stage 1/2 reports record the Windows-run package versions; Stage 3/4 environments are not pinned.
- No notebooks or README/documentation outside the stage-specific reports were found.

## Serious methodological concerns

1. **Circular supervised targets.** Stage 3/Lite/Tier 2 include the four Exposure inputs that form exactly one-third of LVI. Tier 4 adds the remaining 11 indicators that algebraically form the target. Stage 4 verifies both identities to floating-point error (`2.22e-16`). The high Tier 4 scores are numerical facts but not evidence of external predictive validity or causal importance.
2. **Geographic leakage.** Original random household splits put households with identical district climate context in train and test. The corrected grouped Stage 4 AUCs are only 0.6220-0.6456 for the socioeconomic target and 0.6205-0.6579 for full LVI using exogenous features.
3. **Repeated use of one test split.** Stage 3, Lite, and ablation reuse the same seeded household split while configurations are iterated, leaving no untouched final test set.
4. **Full-sample target construction.** LVI normalization and tertile cut points are computed using all households before train/test splitting.
5. **Row-order merges.** Stage 3/Lite/ablation copy columns from the cleaned dataset into derived data by row position, not by household ID; one ID is missing. Any order change would silently misalign records.
6. **Weak cluster separation/stability.** k=2 silhouette is 0.1521; K-Means vs Ward ARI is 0.2775. Profiling tests reuse the variables that created the clusters and therefore do not independently validate them.
7. **Index construction choices.** Flood occurrence and depth are structurally collinear and occupy half of Exposure; district climate occupies the other half. Equal dimension weighting gives an Exposure indicator weight 1/12 and an Adaptive Capacity indicator weight 1/21. Min-max anchors are sample/outlier sensitive; 15 zero-meal households set the food indicator minimum.
8. **Unverified recodes.** `adult_literacy_rate` is renamed and treated as illiteracy based on internal correlations, without a codebook/raw recode. Current-loan holding is treated as adaptive capacity although it may also represent debt burden.
9. **Report inconsistencies.** `stage3_report.md` mixes the obsolete Stage 1 v1 LVI range with the v2 mean, swaps Low/Medium tertile counts, and its three waterfall narratives contradict the saved SHAP figures. Stage 1's report retains v1 sections before an appended v2 changelog.
10. **Stage 3 Lite claims exceed code.** The ensemble threshold 0.40 is hard-coded, not selected by the shown OOF procedure; “balanced accuracy” is not calculated. `best_t_acc` is computed but never evaluated or saved.
11. **Inference gaps.** No survey weights/complex design, confidence intervals for ML metrics, nested model selection, external validation, temporal holdout, or uncertainty/stability analysis is implemented.
12. **Stage 4 caveat.** Grouped CV is the most defensible saved experiment, but it uses fixed hyperparameters, non-stratified `GroupKFold`, aggregate OOF metrics only, and no confidence intervals. Its comment that every predictor is near-constant within district is inaccurate for the six plot-level variables, which vary within climate-profile groups.

## Major missing information

Dataset licenses/provenance; sampling frame and weights; geographic code labels; climate/crop sources and years; definitions/units for many cleaned variables; upstream exclusions, imputation, winsorization and merge procedure; prespecified LVI framework rationale; external vulnerability ground truth; model/fold artifacts; and complete references.
