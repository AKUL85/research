# BIHS Round 3 provenance and row-count reconciliation

**Later correction (2026-09-24):** the local `hhid2` column is independently misordered relative to every other row. A public official-source Module A/weight/expenditure archive shows all 5,605 local rows align exactly by `a01` order across 13 shared fields and household size, while zero received `hhid2` values match the corresponding official row. The missing local ID is therefore an alignment artifact, not evidence by itself of an extra household. The official release's 5,604-household statement versus the 5,605-row weighted source subset remains unresolved. See `15_extension_welfare_results.md` and `outputs/extension3_outcome_validation/public_welfare_link_audit.json` for the current audit; the original Step 1 text below is historical.

Audit date: 2026-09-24. Scope: Step 1 of the revision plan. The cleaned CSV and downstream outputs were read only.

## Source and row-count evidence

| Item | Finding | Evidence |
|---|---|---|
| Official BIHS Round 3 release | IFPRI describes 5,604 households in 325 primary sampling units (PSUs), representative of rural Bangladesh and its seven administrative divisions under the released survey design. | [IFPRI release](https://bangladesh.ifpri.info/2021/06/bangladesh-integrated-household-survey-2018-19-third-round-dataset/); [IFPRI data page](https://rdm.ifpri.info/bangladesh-integrated-household-survey-bihs-2018-2019/) |
| Official dataset | IFPRI's BIHS 2018–2019 dataset, DOI `10.7910/DVN/NXKLZJ`; the Dataverse API currently reports release version 2.1 and lists 158 files. | [Dataset DOI](https://doi.org/10.7910/DVN/NXKLZJ); [Dataverse metadata API](https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/NXKLZJ) |
| Local cleaned input | `bihs_r3_cleaned (1).csv` has 5,605 rows and 137 columns. `hhid2` has 5,604 distinct nonmissing values, zero duplicated nonmissing values, and one missing value at zero-based row index 0. | Direct read-only pandas audit on 2026-09-24; `paper/01_repository_audit.md` |
| Downstream joins | The existing E0 audit matches all 5,605 local records between the cleaned, LVI, and cluster files, using a synthetic fingerprint key for the missing-ID row; it finds no duplicate or unmatched keys. | `outputs/stage5_required_experiments/e0_merge_integrity_summary.csv`; `e0_surrogate_key_audit.csv` |

**Reconciliation status: numerical relationship established, record identity unresolved.** The 5,605 local rows consist of 5,604 distinct nonmissing `hhid2` values plus one row without `hhid2`, matching the one-row difference from IFPRI's published household count. That arithmetic does **not** prove the unkeyed row is extraneous, nor that `hhid2` is the official household key. The E0 fingerprint only tracks the same row through this repository; it cannot establish the row's presence in the official release. Do not delete it or change the analytic denominator without a source-level match.

## Documentation and identifier inventory

The official Dataverse catalog lists `000_Readme.pdf` (file ID 4098413), English household questionnaire `001_bangladesh_ihs_questionnaire_2018-2019_english.pdf` (4097604), male and female codebooks `006_codebook_male.xls` (4098392) and `007_codebook_female.xls` (4098391), community codebook `008_codebook_community.xls` (4098394), household identification module `009_bihs_r3_male_mod_a.tab` (4098252), and `158_r3_bihs_samplingweights.tab` (4367284). These are **located in the official catalog, not downloaded or inspected**. The Dataverse file endpoint responded: `You may not download this file without the required Guestbook response for guestbookID 380.` The guestbook requires a user-supplied response; no identity or contact details were invented or submitted.

The cleaned CSV has `hhid2` but no documented definition of that field and no verified PSU, stratum, sampling-weight, village, or district key. Its administrative fields (`a10`–`a27`) have not been decoded against the questionnaire. The 39 climate-profile groups are fingerprints of repeated climate values, not verified PSUs or districts. No original codebook or source-to-cleaned variable mapping exists in the workspace.

## Climate and survey-design provenance

The local table contains `clim_*` seasonal rain, solar, wind, evapotranspiration, district crop area/production/yield fields, plus plot environmental fields. Their upstream sources, spatial merge keys, aggregation rules, observation windows, and timing relative to household interviews are **not documented**. The BIHS release confirms a community survey exists; that fact alone does not identify the origin of these engineered climate/crop columns. The official catalog contains sampling weights, but the local cleaned CSV has no verified weight, PSU, or stratum columns. Consequently, national or division-level estimates and design-based uncertainty cannot be reconstructed from this workspace.

## Required source-level resolution

1. Access the official Dataverse files through its guestbook with the researcher's own response; retain the readme, questionnaire, relevant codebooks, household identification module, and sampling-weight file with their dataset version and checksums.
2. Confirm the questionnaire/codebook meaning and uniqueness of the official household identifier; map it to local `hhid2` and decode `a10`–`a27`.
3. Join the official household identification module to the local cleaned table by a documented key. Investigate row 0 using source fields without attempting personal re-identification. Record whether the 5,605th row is a genuine additional observation, a malformed import, or a derived/join artifact.
4. Trace each climate/crop variable to its source dataset, reference period, geography, spatial join key, and transformation. Identify any post-interview fields.
5. Map official weights, PSUs, and strata to the analytic rows. Only then choose the justified sample and rerun downstream analyses if the row set changes.

Until those checks finish, manuscript wording should say **“a cleaned table internally described as BIHS Round 3, containing 5,605 records”** and explicitly state that its exact relationship to the official 5,604-household release remains unverified.

## Subsequent Step 4 update (2026-09-24)

An English copy of the original Round-3 household questionnaire was obtained from the published [Europe PMC supplementary archive](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10329118/supplementaryFiles). Its response coding is summarized in `paper/10_index_coding_adjudication.md`. The official Dataverse codebooks, raw modules, sampling-weight contents, and missing-ID source record remain inaccessible without the guestbook response, so the row-count and variable-derivation questions above remain open.
