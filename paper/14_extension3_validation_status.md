# Extension 3 validation status

**Superseded for welfare outcomes on 2026-09-24:** public BIHS expenditure and official district identifiers were located, the local key misalignment was audited, and a held-out-district welfare comparison was completed. See `15_extension_welfare_results.md`. The X5/FIES branch remains unavailable.

Date: 2026-09-24. Scope: Task 2.3. No independent-outcome or transfer metric exists from the current workspace.

## What can be compared after linkage

`outputs/extension3_outcome_validation/evaluate_fies.py` fixes a five-fold holdout over **official verified PSUs** and computes the eight-item X5 affirmative count only for households with eight Yes/No responses. It compares a training-mean baseline, a demographic baseline, demographics plus the continuous archived LVI, and demographics plus the Extension 2 strict robust-High flag on identical held-out records. It will save pooled MAE/RMSE/R², row-level out-of-fold predictions, and within-PSU MAE differences relative to the demographic baseline. These are cross-sectional criterion comparisons; the archived index scales are full-sample-derived and require a transductive-association caveat.

## Observed results

| Quantity | Result |
|---|---|
| Official-key-matched X5 outcomes in the local workspace | None |
| Verified PSU/location IDs linked to local rows | None |
| Validated FIES response count | Not computable |
| Baseline and LVI/robust-High performance | Not computed |
| Geographic transfer estimate | Not computed |
| Prospective/temporal validation | Not designed from the current Round 3 cross-section |

The script's `--help` path loads successfully. Its missing-data gate exits with: “Matched official outcome extract and linkage audit are required; no validation result can be produced from the local cleaned file alone.” No synthetic or proxy outcome was substituted and no result file was generated.

## Data-linking bottlenecks

The official [Round 3 release](https://bangladesh.ifpri.info/2021/06/bangladesh-integrated-household-survey-2018-19-third-round-dataset/) reports 5,604 households and 325 PSUs, while the local table has 5,605 rows, including one without `hhid2`. The [Dataverse release](https://doi.org/10.7910/DVN/NXKLZJ) catalogs the A identification files, female X5 files, sampling weights, and codebooks, but their contents remain unavailable without a required guestbook response. Local `hhid2` and administrative fields lack an authenticated official mapping; climate sources/reference dates are unknown. These gaps prevent an outcome join, verified location split, and survey-weighted interpretation. The detailed join contract is in `paper/13_extension3_linking_framework.md`.

**Next data input:** official Round 3 A, X5, codebook, and weight/PSU files, or a researcher-provided household-keyed X5 extract plus documented official-key and PSU crosswalk. Only then can the comparison script produce empirical validation results.
