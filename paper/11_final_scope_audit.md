# Final methods-submission scope and artifact audit

Date: 2026-09-24. `main.tex` remains the authoritative manuscript; `07_paper_draft.md` is its synchronized reading copy. This audit records the state after Steps 3–6 and supersedes the historical table numbering in `09_source_sync_audit.md`.

## Claim boundaries

- The controlled four-cell experiment is explicitly a **fixed-label leakage diagnostic**. It uses the archived full-sample socioeconomic target and normalized ingredients, which make it unsuitable as an unbiased validation estimate. Its near-perfect ingredient-added AUROC demonstrates mathematical circularity, not substantive prediction.
- The nested E1 experiment has **five outer folds over 39 reconstructed climate-profile groups**. These groups are not authenticated districts, spatial blocks, or official PSUs. Fold-specific training normalization yields target-score cutoffs 0.493580–0.502184, so pooled out-of-fold labels combine distinct fold rules.
- E1 performance concerns a contemporaneous constructed socioeconomic label in this cleaned sample. The manuscript does not infer prospective accuracy, intervention value, a causal climate effect, or a Bangladesh population prevalence. XGBoost is described by point estimates without an inherent-superiority claim.
- Clustering is a short cautionary result: silhouette 0.1521 and geometry-sensitive partitions. The paper does not present clusters as stable household types.
- The questionnaire resolves source response codes but does not authenticate the cleaned literacy aggregate or source-to-field transformations. The 5,605-versus-5,604 row discrepancy, survey design, and climate provenance are disclosed as unresolved.

## Final artifact cross-check

| Manuscript item | Saved evidence | Check |
|---|---|---|
| Abstract controlled AUROC 0.6085→0.9997 and Table II's 12 metric cells | `outputs/stage6_controlled_leakage_audit/controlled_summary.csv`, `controlled_oof.csv`, `controlled_metadata.json` | Four displayed rows match rounded saved pooled AUROC/AUPRC/Brier values; source score reconstruction error is below 1e-10. |
| Abstract E1 XGBoost 0.6443/0.4479/0.2103, baseline Brier 0.2226, Table III's 20 metric cells | `outputs/stage5_required_experiments/e1_model_summary.csv` | All match to four decimals. |
| Five outer folds, 39 groups, target cutoff range, pooled labels | `e1_fold_metrics.csv`, `e1_oof_predictions.csv`, `required_experiments.py` | Five target-score cutoffs range from 0.493580 to 0.502184. The separate `threshold` column in fold metrics is a probability decision threshold, not the target-score cutoff. |
| Abstract 42.0% status variation | `e7_consensus_summary.json` | 2,354/5,605 records vary across the available 11 saved specifications; 15 records have 10 variants because one specification excludes them. |
| Table I, Figure 2, Figure 3 | Stage 1 scored CSV, E7 sensitivity summary/image, E1 metrics and bootstrap image | Values and captions remain those checked in Step 2; Figure 2's baseline wording was changed from “registered” to “saved” to avoid implying preregistration. |

## Build and outstanding dependencies

The final `pdflatex` build produced a five-page A4 PDF with three tables, three figures, and no undefined references or overfull boxes. The remaining TeX notices are underfull spacing in a paragraph, one float page, and the long IFPRI URL. The abstract is 120 whitespace-delimited words in source.

Submission readiness is conditional on the unresolved source-level provenance and independent-outcome limitations. The current paper is a bounded methods case study; it should not be represented as an externally validated food-security prediction system. Before a claim of survey representativeness, geographic transport, or prospective utility, recover the official codebooks/raw modules and design variables, reconcile the unkeyed row, verify climate sourcing, and test an independent outcome.
