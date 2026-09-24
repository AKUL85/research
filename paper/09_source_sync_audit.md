# Manuscript source synchronization and artifact check

Audit date: 2026-09-24. `paper/main.tex` is authoritative; `paper/07_paper_draft.md` is the reading copy. This audit covers Step 2 only.

## Abstract verification

| Abstract statement | Artifact or source checked | Result |
|---|---|---|
| Cleaned sample: 5,605 records | Cleaned CSV; `paper/08_data_provenance_reconciliation.md` | Matches local input; official 5,604-household discrepancy remains disclosed in Methods. |
| LVI mean 0.4636, SD 0.0646 | `outputs/stage1_lvi/bihs_r3_lvi_scored.csv` | Matches to four decimals. |
| Original full-sample k-means silhouette 0.1521 | `outputs/stage2_pca_clustering/pca_clustering_report.txt` | Matches. E6's 0.1512 uses a 2,000-household silhouette subsample, so it is a different calculation. |
| 22 predictors and 39 reconstructed climate-profile groups | `e1_feature_dictionary.csv`; `e1_metadata.json` | Matches: 16 context and six plot-level predictors; none is a target input. Original wording calling them all exogenous was removed because source dates are unverified. |
| XGBoost AUROC 0.6443, AUPRC 0.4479, balanced accuracy 0.6012, Brier 0.2103 | `e1_model_summary.csv` | All match to four decimals. |
| Largest named high-status switch: 21.59% | `e7_lvi_sensitivity_summary.csv`, `flood_binary_omitted` | 1,210 of 5,605 = 21.5879%, rounded to 21.59%. |
| Restricted descriptive/cross-sectional interpretation | `e1_metadata.json`; `required_experiment_status.csv`; Step 1 provenance audit | Supported as a claim boundary; no external outcome, temporal test, or survey weights were executed. |

## Tables and captions

| Item | Check | Result |
|---|---|---|
| Table I: four dimension/composite rows, all mean/SD/min/max values | Recomputed from `outputs/stage1_lvi/bihs_r3_lvi_scored.csv` and compared with LaTeX | All 16 values match to four decimals. Markdown's stale Sensitivity and Adaptive Capacity ranges were corrected to 0.0000–0.7176 and 0.1303–1.0000. |
| Table II: five models by four metrics | Compared with `e1_model_summary.csv` | All 20 values match to four decimals. The pooled prevalence AUROC 0.5122 is explained by fold-varying constant probabilities; mean within-fold AUROC is 0.5000. |
| Figure 1: workflow boxes/caption | Compared with E0, Stage 1, E6/E7, E1 artifacts and code | Counts and sequence match; “immutable” was removed from the join-box wording because the missing-ID row uses a constructed fingerprint. |
| Figure 2: LVI sensitivity image/caption | Inspected `e7_classification_sensitivity.png` and `e7_lvi_sensitivity_summary.csv` | Three panels and baseline omission match. “Fixed 1st/99th-percentile bounds” was corrected: these bounds are calculated from this sample. |
| Figure 3: model uncertainty image/caption | Inspected `e1_metric_confidence_intervals.png`, `e1_group_bootstrap_ci.csv`, and `e1_metadata.json` | Four panels, pooled points, 2.5th–97.5th percentile intervals, 1,000 replicates, and climate-profile-group resampling match. Markdown's incorrect household-bootstrap label was removed. |

## Other synchronized findings

- The Markdown draft now matches LaTeX on fold-specific socioeconomic target construction, 16 context plus six plot predictors, grouped uncertainty, index ranges, sensitivity results, figure numbering, limitations, and conclusion. Its old standalone E6 method-sensitivity figure was removed to mirror the LaTeX figure sequence.
- E7 contains **11 specifications total**: one baseline and ten alternatives. Both sources now say so; the E7 status is still partial because verified literacy coding and theoretical bounds need source documentation.
- The Gower/average-linkage comparison uses a fixed 1,500-household subsample; both sources now state this. Its ARI is not an all-household estimate.
- The paper now cites IFPRI's official 5,604-household release and says the local extra row is unresolved. The source file was added to `references.bib` and the reading-copy bibliography.
- Marginal interval overlap is not a paired model comparison; both sources now state only that superiority has not been established by a paired comparison.

## Build check

Ran `pdflatex`, `bibtex`, and two additional `pdflatex` passes, then another `pdflatex` pass after the final wording edit. `paper/main.pdf` builds as a five-page A4 manuscript with no undefined citations or references in the final log. Routine TeX underfull-box notices remain; they do not affect the factual checks above.
