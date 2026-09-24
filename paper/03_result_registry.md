# Result registry

Values below use report precision; cited CSVs retain full machine precision. `VERIFIED` confirms the saved numerical artifact and code consistency, not freedom from leakage or other design problems.

| ID | Result | Value | Source file | Cell/section | Status |
|---|---|---|---|---|---|
| DATA-01 | Cleaned sample shape | 5,605 rows x 137 columns | `bihs_r3_cleaned (1).csv` | Whole file | VERIFIED |
| DATA-02 | Household identifiers | 5,604 distinct; 1 missing; 0 duplicated non-missing IDs | `outputs/stage1_lvi/lvi_validation_report.txt` | Check 2 | VERIFIED |
| DATA-03 | Date-like coverage | 2018-11: 301; 2018-12: 418; 2019-01: 472; 2019-02: 1,698; 2019-03: 2,022; 2019-04: 694 | `bihs_r3_cleaned (1).csv` | `a16_1_mm`, `a16_1_yy` | VERIFIED |
| DATA-04 | Reconstructed climate-profile groups | 39; size min/median/max = 18/145/379 | `outputs/stage1_lvi/lvi_validation_report.txt` | Check 10 | VERIFIED |
| LVI-01 | Exposure score | mean 0.4521; SD 0.1489; range 0.1572-0.8482 | `outputs/stage1_lvi/lvi_validation_report.txt` | V2 Check 5 | VERIFIED |
| LVI-02 | Sensitivity score | mean 0.3008; SD 0.0594; range 0.0000-0.7176 | Same | V2 Check 5 | VERIFIED |
| LVI-03 | Adaptive-capacity vulnerability | mean 0.6381; SD 0.1390; range 0.1303-1.0000 | Same | V2 Check 5 | VERIFIED |
| LVI-04 | Composite LVI v2 | mean 0.4636; SD 0.0646; range 0.2490-0.6576; n=5,605 | Same | V2 Check 5 | VERIFIED |
| LVI-05 | LVI v2 percentiles | p25 0.4168; p50 0.4640; p75 0.5086; p95 0.5710 | Same | V2 Check 5 | VERIFIED |
| LVI-06 | Complete LVI rows | 5,605 complete; 0 NaN | Same | Check 9 | VERIFIED |
| LVI-07 | Current-loan recode | 3,946/5,605 hold a current loan | Same | Indicator preparation log/C2 | VERIFIED |
| LVI-08 | Zero-meal records | 15 households | Same | Caveat C5 | VERIFIED |
| LVI-09 | Flood structural overlap | 5,605/5,605 satisfy `flood_affected == 1 iff depth > 0` | Same | Caveat C3 | VERIFIED |
| CLASS-01 | LVI sample-tertile cuts/counts | Low <=0.43256042898 (n=1,869); Medium <=0.49372404895 (n=1,868); High >0.49372404895 (n=1,868) | `outputs/stage2_pca_clustering/bihs_r3_clustered.csv` | LVI quantiles; verified by Stage 2b script | VERIFIED |
| PCA-01 | Retained components | 9 PCs; cumulative variance 81.1714% | `outputs/stage2_pca_clustering/pca_clustering_report.txt` | Section 2 | VERIFIED |
| PCA-02 | Leading component variance | PC1 18.1553%; PC2 12.0447%; PC3 10.5051% | Same | Section 2 explained-variance table | VERIFIED |
| CLUS-01 | Selected K-Means solution | k=2; silhouette 0.1521; Davies-Bouldin 2.2347; Calinski-Harabasz 1000.2244 | Same | Section 3, k=2 row | VERIFIED |
| CLUS-02 | Alternative metric preference | Davies-Bouldin minimum at k=8: 1.8055 | Same | Section 3 | VERIFIED |
| CLUS-03 | Ward comparison | ARI 0.2775; matched 4,281/5,605 (76.4%); Ward silhouette 0.1060 | Same | Section 4 | VERIFIED |
| CLUS-04 | Cluster 1 | n=3,118 (55.6289%); LVI mean 0.45504764, SD 0.06379568 | `outputs/stage2b_visualization/stage2b_summary_stats.csv` | Cluster 1 row | VERIFIED |
| CLUS-05 | Cluster 2 | n=2,487 (44.3711%); LVI mean 0.47442464, SD 0.06401744 | Same | Cluster 2 row | VERIFIED |
| CLUS-06 | Cluster-profile tests | Kruskal-Wallis significant after Bonferroni for 12/15 indicators; ANOVA 13/15 | `outputs/stage2_pca_clustering/pca_clustering_report.txt` | Section 5 | VERIFIED |
| CLUS-07 | Cluster x tertile association | chi-square 177.95, df=2, p=2.28e-39; Cramer's V=0.178; Spearman rho=0.176, p=4.37e-40 | `outputs/stage2b_visualization/crosstab_cluster_vs_lvi_tertile.csv` | Footer | VERIFIED |
| CLUS-08 | Cluster x tertile counts | C1: 1,259/1,000/859; C2: 610/868/1,009 (Low/Medium/High) | Same | Rows 1-2 | VERIFIED |
| S3-B0 | Binary majority baseline | accuracy 0.6664; F1 0; AUC 0.5000; Brier 0.2223 | `outputs/stage3_supervised_ml/supervised_model_results.csv` | Binary baseline row | VERIFIED |
| S3-B1 | Binary Logistic Regression | accuracy 0.7948; precision 0.7250; recall 0.6203; F1 0.6686; AUC 0.8660; Brier 0.1395 | Same | Binary Logistic row | VERIFIED |
| S3-B2 | Binary Random Forest | accuracy 0.7966; precision 0.7401; recall 0.6016; F1 0.6637; AUC 0.8626; Brier 0.1418 | Same | Binary RF row | VERIFIED |
| S3-B3 | Binary XGBoost | accuracy 0.7975; precision 0.7155; recall 0.6524; F1 0.6825; AUC 0.8634; Brier 0.1389 | Same | Binary XGBoost row | VERIFIED |
| S3-M0 | Three-class majority baseline | accuracy 0.3336; macro F1 0.1668; macro AUC 0.5000 | Same | Three-class baseline row | VERIFIED |
| S3-M1 | Three-class Logistic Regression | accuracy 0.6137; macro precision 0.5978; recall 0.6139; F1 0.5992; AUC 0.7850 | Same | Three-class Logistic row | VERIFIED |
| S3-M2 | Three-class Random Forest | accuracy 0.6137; macro precision 0.5971; recall 0.6139; F1 0.5978; AUC 0.7824 | Same | Three-class RF row | VERIFIED |
| S3-M3 | Three-class XGBoost | accuracy 0.6030; macro precision 0.5827; recall 0.6032; F1 0.5844; AUC 0.7803 | Same | Three-class XGBoost row | VERIFIED |
| SHAP-01 | Top global mean absolute SHAP | flood depth 1.2913; Boro evapotranspiration 0.2871; Aman evapotranspiration 0.2664 | `outputs/stage3_supervised_ml/stage3_report.md`; SHAP bar PNG | Section 6.1 / plotted bars | VERIFIED |
| SHAP-02 | Saved high case | true High; P(High)=0.96; Aman rainfall 2448.54 contributes +1.06; flood depth 1 contributes +0.80 | `shap_waterfall_high_vulnerability.png` | Figure labels | VERIFIED |
| SHAP-03 | Saved low case | true Not-High; P(High)=0.03; flood depth 0 contributes -1.93 | `shap_waterfall_low_vulnerability.png` | Figure labels | VERIFIED |
| SHAP-04 | Saved borderline case | true Not-High; P(High)=0.52; flood depth 2.429 contributes +0.96 | `shap_waterfall_borderline_case.png` | Figure labels | VERIFIED |
| SHAP-05 | Waterfall prose values | Report says high 0.94/depth 7.5/+2.85; low 0.06/-1.15; borderline true High/0.51; these conflict with PNGs | `outputs/stage3_supervised_ml/stage3_report.md` | Section 6.2 | UNCERTAIN |
| LITE-B1 | Engineered XGBoost, tau=0.50 | accuracy 0.7930; recall 0.6658; F1 0.6822; AUC 0.8618; Brier 0.1380 | `outputs/stage3_lite/stage3_lite_results.csv` | Row 2 | VERIFIED |
| LITE-B2 | XGBoost OOF-F1 threshold | tau=0.285; accuracy 0.7806; precision 0.6231; recall 0.8663; F1 0.7248; AUC 0.8618 | Same | Row 3 | VERIFIED |
| LITE-B3 | Soft voting, tau=0.50 | accuracy 0.7984; recall 0.6497; F1 0.6826; AUC 0.8686; Brier 0.1363 | Same | Row 4 | VERIFIED |
| LITE-B4 | Soft voting, hard-coded tau=0.40 | accuracy 0.8055; precision 0.6757; recall 0.8021; F1 0.7335; AUC 0.8686 | Same | Row 5 | VERIFIED |
| LITE-M1 | Nominal XGBoost | accuracy 0.6120; macro F1 0.5993; AUC 0.7884 | Same | Row 6 | VERIFIED |
| LITE-M2 | Ordinal XGBoost | accuracy 0.6093; macro F1 0.5994; AUC 0.7896 | Same | Row 7 | VERIFIED |
| LITE-M3 | Three-class soft voting | accuracy 0.6182; macro F1 0.6057; AUC 0.7938 | Same | Row 8 | VERIFIED |
| LITE-X1 | Claimed ordinal catastrophic-error count | 21/373 High predicted Low (5.6%) | `outputs/stage3_lite/stage3_lite_report.md` | Section 3.3; no saved confusion matrix | UNCERTAIN |
| ABL-B1 | Exposure-only binary, p=39 | accuracy 0.7930; F1 0.6822; AUC 0.8618 | `outputs/stage3_ablation/stage3_ablation_results.csv` | Tier 2 binary row | VERIFIED |
| ABL-B2 | +Demographics binary, p=44 | accuracy 0.8109; F1 0.7151; AUC 0.8866 | Same | Tier 3 binary row | VERIFIED |
| ABL-B3 | Full livelihood binary, p=55 | accuracy 0.9500; F1 0.9255; AUC 0.9901 | Same | Tier 4 binary row | VERIFIED |
| ABL-M1 | Exposure-only three-class | accuracy 0.6280; macro F1 0.6176; AUC 0.7958 | Same | Tier 2 three-class row | VERIFIED |
| ABL-M2 | +Demographics three-class | accuracy 0.6646; macro F1 0.6568; AUC 0.8229 | Same | Tier 3 three-class row | VERIFIED |
| ABL-M3 | Full livelihood three-class | accuracy 0.8644; macro F1 0.8631; AUC 0.9702 | Same | Tier 4 three-class row | VERIFIED |
| AUD-01 | Algebraic reconstruction | max absolute error for Exposure and LVI = 2.220446049250313e-16; Exposure R2=1.0 | `outputs/stage4_leakage_audit/leakage_audit_log.json` | Identity fields | VERIFIED |
| AUD-02 | Closed-form exposure score predicting High-LVI | AUC 0.8504907663; flood depth alone 0.7954848758 | Same | Closed-form fields | VERIFIED |
| AUD-03 | Four direct Exposure inputs, stratified OOF | LR/RF/XGB AUC = 0.8522/0.8536/0.8533 | `outputs/stage4_leakage_audit/leakage_audit_results.csv` | Rows 1-3 | VERIFIED |
| AUD-04 | All 15 LVI inputs, stratified OOF | LR/RF/XGB AUC = 0.9971/0.9660/0.9885 | Same | Rows 4-6 | VERIFIED |
| AUD-05 | Exogenous climate -> socioeconomic target, stratified OOF | LR/RF/XGB AUC = 0.6282/0.6637/0.6656 | Same | Rows 7-9 | VERIFIED |
| AUD-06 | Exogenous climate -> socioeconomic target, grouped OOF | LR/RF/XGB accuracy = 0.6694/0.6667/0.6690; AUC = 0.6220/0.6389/0.6456 | Same | Rows 10-12 | VERIFIED |
| AUD-07 | Exogenous climate -> full LVI, grouped OOF | LR/RF/XGB accuracy = 0.6773/0.6546/0.6537; AUC = 0.6205/0.6564/0.6579 | Same | Rows 13-15 | VERIFIED |
| MISS-01 | External validation against observed vulnerability | Not present | Repository-wide | — | MISSING |
| E0-01 | One-to-one merge integrity | Raw -> LVI and LVI -> cluster each matched 5,605/5,605 keys; 0 duplicate, left-only, or right-only keys | `outputs/stage5_required_experiments/e0_merge_integrity_summary.csv` | Both join rows | VERIFIED |
| E0-02 | Shared-field preservation | 0 mismatched columns among 26 raw/LVI and 54 LVI/cluster shared columns | `outputs/stage5_required_experiments/e0_shared_column_checks.csv` | All rows | VERIFIED |
| E0-03 | Missing-ID surrogate | One missing `hhid2` assigned the same immutable SHA-256-based surrogate in all three files | `outputs/stage5_required_experiments/e0_surrogate_key_audit.csv` | All 3 rows | VERIFIED |
| E1-01 | Nested spatial validation sample | n=5,605; 1,875 outer-fold High targets; 39 reconstructed climate-profile groups; five outer folds | `outputs/stage5_required_experiments/e1_model_summary.csv`; `e1_fold_assignments.csv` | All rows | VERIFIED |
| E1-02 | Nested spatial XGBoost | AUC 0.6443 (group-bootstrap 95% CI 0.6189-0.6687); PR-AUC 0.4479 (0.4107-0.4801); balanced accuracy 0.6012 (0.5806-0.6199); Brier 0.2103 (0.1978-0.2209) | `outputs/stage5_required_experiments/e1_model_summary.csv`; `e1_group_bootstrap_ci.csv` | XGBoost rows | VERIFIED |
| E1-03 | Nested spatial Random Forest | AUC 0.6412 (0.6141-0.6661); PR-AUC 0.4439 (0.4067-0.4810); balanced accuracy 0.6093 (0.5895-0.6271); Brier 0.2121 (0.1993-0.2225) | Same | RandomForest rows | VERIFIED |
| E1-04 | Nested spatial Logistic Regression | AUC 0.6159 (0.5875-0.6455); PR-AUC 0.4267 (0.3827-0.4641); balanced accuracy 0.5857 (0.5622-0.6076); Brier 0.2150 (0.2024-0.2253) | Same | LogisticRegression rows | VERIFIED |
| E1-05 | Context-only Logistic baseline | AUC 0.5766 (0.5413-0.6066); PR-AUC 0.3928 (0.3553-0.4214); balanced accuracy 0.5528 (0.5249-0.5807); Brier 0.2197 (0.2076-0.2300) | Same | ContextOnlyLogistic rows | VERIFIED |
| E1-06 | Prevalence/majority baseline | AUC 0.5122; PR-AUC 0.3423; accuracy 0.6655; balanced accuracy 0.5000; Brier 0.2226 | `outputs/stage5_required_experiments/e1_model_summary.csv` | PrevalenceBaseline row | VERIFIED |
| E1-07 | E1 reproducibility artifacts | Saved immutable outer-fold IDs, OOF probabilities/classes, per-fold metrics, tuned parameters/thresholds, feature levels, calibration data, versions, and seeds | `outputs/stage5_required_experiments/` | `e1_*` artifacts | VERIFIED |
| E6-01 | k=2 household-bootstrap stability | 100 refits: mean ARI 0.9627 (2.5%-97.5% 0.9247-0.9865); mean matched Jaccard 0.9811 (0.9618-0.9932) | `outputs/stage5_required_experiments/e6_stability_summary.csv` | household_bootstrap, k=2 | VERIFIED |
| E6-02 | k=2 climate-profile-bootstrap stability | 100 refits: mean ARI 0.9199 (0.8345-0.9724); mean matched Jaccard 0.9594 (0.9160-0.9860) | Same | climate_profile_bootstrap, k=2 | VERIFIED |
| E6-03 | k=2 consensus separation | Mean within-baseline consensus 0.9692; between-baseline 0.0345; PAC(0.1,0.9)=0.1016 on fixed n=750 audit subset | `outputs/stage5_required_experiments/e6_consensus_summary.csv` | k=2 row | VERIFIED |
| E6-04 | k=2 flood-deduplication sensitivity | ARI vs baseline = 0.6708 omitting flood occurrence and 0.8172 omitting flood depth | `outputs/stage5_required_experiments/e6_method_sensitivity.csv` | k=2 flood rows | VERIFIED |
| E6-05 | k=2 geometry sensitivity | Robust-scaler ARI = -0.0025 on n=5,605; Gower average-linkage ARI = 0.0003 on fixed n=1,500 subset | Same | k=2 robust/Gower rows | VERIFIED |
| E7-01 | Literacy specification sensitivity | Reversing current inferred literacy direction: Spearman rho 0.9127, tertile kappa 0.6508, 686 High-status switches (12.24%); omitting literacy: rho 0.9787, kappa 0.8255, 370 switches (6.60%) | `outputs/stage5_required_experiments/e7_lvi_sensitivity_summary.csv` | Literacy rows | VERIFIED |
| E7-02 | Loan specification sensitivity | Burden direction: rho 0.7680, kappa 0.3644, 1,146 High-status switches (20.45%); omission: rho 0.9406, kappa 0.6778, 602 switches (10.74%) | Same | Loan rows | VERIFIED |
| E7-03 | Zero-meal specification sensitivity | Missing treatment: rho 0.9978, kappa 0.9930, 14 High-status switches (0.25%); excluding 15 households: kappa 0.9973, 5 switches among retained households (0.09%) | Same | Zero-meal rows | VERIFIED |
| E7-04 | Flood-deduplication LVI sensitivity | Omitting flood occurrence: rho 0.7611, kappa 0.2924, 1,210 High-status switches (21.59%); omitting depth: rho 0.9895, kappa 0.8710, 236 switches (4.21%) | Same | Flood rows | VERIFIED |
| E7-05 | Weight/bound sensitivity | Equal-indicator weights: rho 0.9341, kappa 0.6746, 598 High switches (10.67%); 1st/99th-percentile bounds: rho 0.9649, kappa 0.7806, 418 switches (7.46%) | Same | Weight/bound rows | VERIFIED |
| E7-06 | Downstream non-circular sensitivity | Fixed grouped-fold Logistic AUC ranges 0.6056-0.6812; PR-AUC 0.4440-0.5440; Brier 0.2014-0.2153 across 11 specifications | `outputs/stage5_required_experiments/e7_downstream_model_sensitivity.csv` | All rows | VERIFIED |
| REQ-01 | Conditional required experiments unavailable | E2: no independent external outcome; E3: no later outcome/time ordering; E5: no weights/strata/PSU documentation | `outputs/stage5_required_experiments/required_experiment_status.csv` | E2/E3/E5 rows | MISSING |
| REQ-02 | E7 documentation-dependent variants unavailable | Verified literacy recode, defensible fixed bounds, and theory-derived alternative weights cannot be run from available inputs | `outputs/stage5_required_experiments/e7_unresolved_specifications.csv` | All rows | MISSING |
| MISS-02 | Saved models/predictions/folds/CV search results | Present for E1; earlier Stage 3 fold/prediction/search artifacts remain absent | `outputs/stage5_required_experiments/e1_*.csv` | E1 artifacts | VERIFIED |

## Figure registry

| Figure ID | Filename | Source | What it shows | Data source | Verified consistency | Paper usefulness |
|---|---|---|---|---|---|---|
| F2B-01 | `fig1_pca_scatter.png` | `visualize_clusters.py` | PC1-PC2 points, clusters, centroids | `bihs_r3_clustered.csv` | Yes | Useful with low-silhouette/ARI caveat |
| F2B-02 | `fig2_radar_profiles.png` | Same | Cluster means on 15 normalized indicators | Same | Yes | Useful descriptive profile |
| F2B-03 | `fig3_mean_lvi_by_cluster.png` | Same | Mean LVI and 95% t CI | Same | Yes | Useful; truncated y-axis disclosed |
| F2B-04 | `fig4_cluster_sizes.png` | Same | Counts and shares | Same | Yes | Secondary |
| F2B-05 | `fig5_lvi_boxplots.png` | Same | Within-cluster LVI distributions | Same | Yes | Useful for overlap |
| F2B-06 | `fig6_crosstab_heatmap.png` | Same | Cluster x LVI-tertile row percentages | Same/cross-tab CSV | Yes | Useful descriptive comparison |
| FS3-01 | `shap_bar_global_importance.png` | `supervised_ml_stage3.py` | Binary XGBoost mean absolute SHAP | Unsaved test predictions/SHAP array | Yes vs report section 6.1 | Limited: target leakage |
| FS3-02 | `shap_beeswarm_summary.png` | Same | SHAP direction/distribution | Unsaved SHAP array | Yes | Limited: target leakage |
| FS3-03 | `shap_waterfall_high_vulnerability.png` | Same | Local true-High case, P=0.96 | Unsaved test row/SHAP array | No: report narrative conflicts | Do not use until prose is corrected |
| FS3-04 | `shap_waterfall_low_vulnerability.png` | Same | Local true-Not-High case, P=0.03 | Unsaved test row/SHAP array | No: report narrative conflicts | Do not use until prose is corrected |
| FS3-05 | `shap_waterfall_borderline_case.png` | Same | Local true-Not-High case, P=0.52 | Unsaved test row/SHAP array | No: report narrative conflicts | Do not use until prose is corrected |
| FL-01 | `roc_curve_comparison.png` | `supervised_ml_stage3_lite.py` | Engineered XGBoost and ensemble ROC | In-memory predictions; results CSV | Yes | Limited: circular target/random split |
| FL-02 | `threshold_tuning_curve.png` | Same | Training OOF XGBoost F1/accuracy vs threshold | In-memory OOF predictions | Yes | Useful for threshold method; not ensemble tau=0.40 |
| FL-03 | `confusion_matrices_comparison.png` | Same | Engineered XGB tau=.50 vs ensemble tau=.40 | In-memory test predictions | Yes | Limited: tau=.40 was hard-coded |
| FA-01 | `fig_ablation_metrics_progression.png` | `supervised_ml_stage3_ablation.py` | Metrics across four feature tiers | Ablation results CSV | Yes | Not inferential; higher tiers reconstruct target |
| FA-02 | `fig_ablation_roc_curves.png` | Same | Binary ROC curves across tiers | In-memory predictions/results CSV | Yes | Not inferential; circularity caveat essential |
| FA-03 | `fig_ablation_confusion_matrices.png` | Same | Three-class confusion matrices, tiers 2-4 | In-memory predictions | Yes | Not inferential; circularity caveat essential |
| F5-E0-01 | `e0_merge_integrity.png` | `plot_required_experiments.py` | Matched versus unmatched keys in both one-to-one joins | `e0_merge_integrity_summary.csv` | Yes | Merge-audit evidence |
| F5-E1-01 | `e1_discrimination_calibration.png` | Same | Outer-fold OOF ROC and calibration curves for E1 models/baselines | `e1_oof_predictions.csv`; `e1_calibration_curve.csv` | Yes | Primary corrected validation figure |
| F5-E1-02 | `e1_metric_confidence_intervals.png` | Same | Profile-group-bootstrap 95% CIs for AUC, PR-AUC, balanced accuracy and Brier | `e1_group_bootstrap_ci.csv` | Yes | Primary uncertainty figure |
| F5-E6-01 | `e6_bootstrap_stability.png` | Same | Mean ARI and matched Jaccard across k=2..8 under two bootstrap schemes | `e6_stability_summary.csv` | Yes | Cluster-stability evidence |
| F5-E6-02 | `e6_method_sensitivity.png` | Same | ARI versus baseline under flood removal, robust scaling and Gower hierarchy | `e6_method_sensitivity.csv` | Yes | Essential geometry-sensitivity caveat |
| F5-E7-01 | `e7_classification_sensitivity.png` | Same | Rank correlation, tertile kappa and High-status switching by LVI variant | `e7_lvi_sensitivity_summary.csv` | Yes | Primary specification-sensitivity figure |
| F5-E7-02 | `e7_score_distributions.png` | Same | LVI score distributions across 11 executable variants | `e7_household_variant_scores.csv` | Yes | Distributional sensitivity figure |

All 17 PNGs have approximately 300 x 300 DPI metadata. Stage 4 has no figure or narrative report.
