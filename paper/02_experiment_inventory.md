# Experiment inventory

`VERIFIED` means an executed artifact exists and agrees with its generating code/table; it does not mean the design is methodologically valid. No notebook experiments exist.

| Experiment | Implemented? | Executed? | Dataset | Target | Method | Metrics | Result source | Status |
|---|---:|---:|---|---|---|---|---|---|
| Stage 1 LVI v2 | Yes | Yes | Cleaned BIHS R3, n=5,605 | Continuous LVI | 15 indicators; min-max; dimension/composite means | Distribution, range, correlations, completeness | `outputs/stage1_lvi/bihs_r3_lvi_scored.csv`; validation report v2 changelog | VERIFIED |
| PCA | Yes | Yes | 15 normalized indicators, n=5,605 | None | Z-score + full PCA; >=80% cumulative rule | Eigenvalues, explained variance, loadings | `outputs/stage2_pca_clustering/pca_clustering_report.txt`, section 2 | VERIFIED |
| K-Means k search | Yes | Yes | 9 retained PCs | Cluster label | k=2..8, n_init=20, seed 42 | Silhouette, Davies-Bouldin, Calinski-Harabasz, inertia | Stage 2 report, section 3 | VERIFIED |
| Ward robustness | Yes | Yes | Same 9 PCs | Cluster label | Ward linkage, cut at k=2 | ARI, matched agreement, silhouette | Stage 2 report, section 4 | VERIFIED |
| Cluster profiles/tests | Yes | Yes | K-Means clusters, n=5,605 | Cluster membership | Means, ANOVA, Kruskal-Wallis, Bonferroni, effect sizes | `cluster_profile_summary.csv`; Stage 2 report, section 5 | VERIFIED |
| LVI tertile/cluster cross-tab | Yes | Yes | Stage 2 clustered data | Low/Medium/High sample tertile | `pd.cut` at sample terciles; chi-square | Counts, row/column %, chi-square, Cramer's V, Spearman rho | `outputs/stage2b_visualization/crosstab_cluster_vs_lvi_tertile.csv` | VERIFIED |
| Stage 3 binary models | Yes | Yes | 25 exposure/context features; n=5,605 | High-LVI vs Not-High | Majority, LR, RF, XGBoost; 80/20 split; 5-fold training CV | Accuracy, precision, recall, F1, AUC, Brier, confusion matrix | `outputs/stage3_supervised_ml/supervised_model_results.csv`, binary rows | VERIFIED |
| Stage 3 three-class models | Yes | Yes | Same 25 features | Low/Medium/High | Majority, LR, RF, XGBoost; 80/20 split; 5-fold training CV | Accuracy, macro precision/recall/F1, OvR macro AUC, confusion matrix | Same CSV, three-class rows | VERIFIED |
| Stage 3 SHAP | Yes | Yes | Binary XGBoost held-out test, n=1,121 | High-LVI | TreeExplainer; global and three local plots | Mean absolute SHAP and local log-odds attributions | Five PNGs in `outputs/stage3_supervised_ml/shap_figures/` | VERIFIED |
| Stage 3 Lite binary | Yes | Yes | 39 environmental/engineered features | High-LVI vs Not-High | XGBoost, 5-fold OOF F1-threshold search, soft voting | Accuracy, precision, recall, F1, AUC, Brier | `outputs/stage3_lite/stage3_lite_results.csv`, binary rows | VERIFIED |
| Stage 3 Lite three-class | Yes | Yes | Same 39 features | Low/Medium/High | Nominal XGBoost, Frank-Hall ordinal XGBoost, soft voting | Accuracy, macro precision/recall/F1, OvR macro AUC | Same CSV, three-class rows | VERIFIED |
| Four-tier ablation | Yes | Yes | p=0/39/44/55; one shared binary-stratified split | Binary and three-class LVI tertile | Fixed XGBoost by tier | Accuracy, precision, recall, F1, AUC, binary Brier | `outputs/stage3_ablation/stage3_ablation_results.csv` | VERIFIED |
| Stage 4 algebraic audit | Yes | Yes | Stage 1 indicators, n=5,605 | LVI/exposure score | Closed-form reconstruction and CV models | Identity error, R2, AUC | `leakage_audit_log.json`; first 6 rows of results CSV | VERIFIED |
| Stage 4 corrected socioeconomic target | Yes | Yes | 22 exogenous climate/agri features; 39 groups | High socioeconomic-half tertile | LR/RF/XGBoost; 5-fold stratified and GroupKFold OOF | Accuracy, AUC | `outputs/stage4_leakage_audit/leakage_audit_results.csv`, rows 7-12 | VERIFIED |
| Stage 4 exogenous-to-full-LVI grouped test | Yes | Yes | Same 22 features/groups | High-LVI tertile | 5-fold GroupKFold OOF | Accuracy, AUC | Same CSV, rows 13-15 | VERIFIED |
| Accuracy-optimal XGBoost threshold | Partial | No test result saved | Stage 3 Lite training OOF | High-LVI | `best_t_acc` is computed but unused | None saved | `supervised_ml_stage3_lite.py`, lines 220-227 | CODE_ONLY |
| Balanced-accuracy threshold tuning | No | No | — | High-LVI | Mentioned in docstring/report only; code uses ordinary accuracy | — | Stage 3 Lite docstring vs lines 220-224 | MISSING |

## Execution cautions

- The four Stage 3 Exposure indicators are direct LVI components; Tier 4 contains all LVI inputs. Treat those experiments as circular reconstruction, not external prediction.
- Stage 3/Lite/ablation artifacts prove execution, but models and predictions were not saved, so exact rerun identity cannot be checked without retraining.
- The three SHAP waterfall images prove execution, but their values contradict the narrative in `stage3_report.md`.
- All top-level implemented pipelines have artifacts. The only clear code-only experimental branch is evaluation at `best_t_acc`; balanced-accuracy tuning is not implemented.
