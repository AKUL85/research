# Index coding and specification agreement

Review date: 2026-09-24. This is the Step 4 evidence record. The cleaned analytic data and archived E7 outputs were not modified.

## Questionnaire recovered and what it establishes

The published [BIHS Round 3 English household questionnaire supplement](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10329118/supplementaryFiles) to [Jubayer et al. (2023)](https://doi.org/10.1016/j.heliyon.2023.e17378) was retrieved through Europe PMC. The archive contains `mmc1.pdf`, titled `bangladesh_Bihs_questionnaire_2018-2019_english`. The PDF is a copy of the original survey instrument; the official Dataverse codebooks and raw modules still require a guestbook response and were not inspected.

| Indicator issue | Instrument evidence | Adjudication for this cleaned file |
|---|---|---|
| Adult literacy | Module B1, item B1_08, defines code 1 as “Cannot read and write,” 2 as “Can sign only,” 3 as “Can read only,” and 4 as “Can read and write.” | The direction of the *source question* is clear. The cleaned `adult_literacy_rate` is an upstream aggregate; Stage 1 copies it unchanged into `adult_illiteracy_rate` based on correlation and internal derived counts. Without the raw member records and derivation code, the aggregate's exact meaning and denominator are **not verified**. Retain reversal and omission sensitivity; do not assert that the current coding is definitively correct. |
| Current loan | Module F, F02, asks whether any member currently has a cash loan with an individual or institution; Yes=1 and No=2. The module records source, use, outstanding amount, and interest; F02_a–c distinguish applications, denial, and not needing a loan. | The implemented binary recode `1→1, 2→0` matches the questionnaire's response direction if `has_current_loan` maps to F02. A current loan is neither an unambiguous protective resource nor an unambiguous burden. The baseline's protective direction is a modeling assumption, **not adjudicated** by F02. Report protective, burden, and omitted variants. |
| Flood occurrence and depth | The cleaned fields are perfectly linked: `flood_affected = 1` exactly when `flood_depth_ft_mean > 0` in all 5,605 records. | Their double inclusion places redundant information in half of the Exposure dimension. The instrument alone does not trace the upstream flood-depth construction; retain the one-indicator-omitted analyses. |
| Zero meals | Fifteen cleaned records have `mealdays_total_7d = 0`, anchoring the baseline reverse min–max scale. | The available instrument does not establish whether these particular zeros mean observed non-consumption or missing/invalid aggregate construction. Retain observed, missing, and exclusion variants without declaring one correct. |

The codebook/raw-derivation gap is material: the recovered questionnaire resolves response codes but cannot certify the cleaned field transformations. No baseline indicator was silently recoded or excluded.

## Agreement across the saved E7 specifications

`outputs/stage5_required_experiments/e7_consensus_analysis.py` summarizes the existing `e7_household_variant_scores.csv` without refitting models. Its output is `e7_consensus_summary.json`. The analysis counts High status across all available specifications for each row; 15 rows have 10 instead of 11 because the zero-meal exclusion removes them from that one variant.

| Status across available specifications | Records |
|---|---:|
| Always High | 708 |
| Never High | 2,543 |
| High in some but not all | 2,354 (42.0%) |
| Total | 5,605 |

The archived baseline labels 1,868 records High; 1,160 of those are not High under every available specification. The largest single-change high-status switch remains 1,210 records (21.59%) when flood occurrence is omitted, followed by 1,146 (20.45%) under the loan-burden direction. These are **descriptive agreement counts** over 11 chosen, correlated specifications. They are not posterior probabilities, external validity estimates, or a principled ordering of which coding is correct.

## Reporting decision

The paper now treats household High labels as sample-relative and specification-dependent. The inherited coding remains an explicitly audited baseline. Verification of the cleaned literacy aggregate, current-loan source mapping, and meal/flood transformations requires the official codebooks, raw modules, and cleaning code. Fixed theoretical bounds and theory-derived alternative weights also remain unavailable.
