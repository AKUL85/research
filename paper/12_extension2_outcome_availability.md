# Extension 2: excluded-outcome availability audit

**Superseded for welfare outcomes on 2026-09-24:** a public BIHS expenditure file was subsequently located and linked with an audited correction to the local identifier alignment. See `15_extension_welfare_results.md` for the current criterion-association result. The local cleaned file still lacks FIES/X5 responses.

Date: 2026-09-24. Scope: Task 1.2 only. No outcome validation statistic was computed.

The local cleaned BIHS-labeled file has 5,605 rows and 137 columns. It contains no FIES items or score, dietary-diversity measure, food-consumption value, coping score, poverty-transition flag, later-wave outcome, or other documented criterion that was deliberately reserved from the original LVI for validation. The archived [Round 3 questionnaire](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10329118/supplementaryFiles) includes food-consumption and food-security modules O and X, but those modules are absent from this cleaned analytic file.

Several unused local columns do **not** meet this task's criterion. `mealdays_total_7d` enters the index through meals per person. `income_per_capita_monthly` is another index ingredient, so a crop-revenue association could share its upstream construction; the cleaning code and revenue units are unavailable. `school_fees_monthly_total` depends on school enrollment and is not a household-wide food-security measure. `xxc_*` fields are questionnaire questions on child-marriage law, not welfare outcomes. None is documented as an intentionally held-out validation outcome.

Accordingly, the 705 strict robust-High records remain a **specification-consensus class**, not a validated hardship group. The row-level IDs and variant flags are saved in `outputs/extension2_robust_high/household_robustness.csv` for a future keyed join to an independently measured criterion. An outcome association can be estimated only after a valid outcome and its measurement timing, key mapping, and eligibility rules are recovered.
