#!/usr/bin/env python3
"""
Stage 1 LVI construction for BIHS Round 3.

Pipeline:  indicator preparation -> min-max normalization -> dimension scores
           -> composite Livelihood Vulnerability Index (LVI).

Method follows the composite-index approach in which every indicator is rescaled
to [0, 1], oriented so that HIGHER always means MORE vulnerable, averaged with
equal weights inside each dimension, and the three dimension scores are then
averaged with equal weights into the composite LVI.

Deliberately NOT implemented at this stage: PCA, clustering, ML prediction,
and Low/Medium/High vulnerability classification.

Run:  python lvi_construction.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The script lives in <project>/outputs/stage1_lvi/, so the project root is two
# levels up. Resolving relative to __file__ keeps the run reproducible no matter
# what the current working directory is.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR

# The cleaned file shipped with a browser-download suffix. Prefer the canonical
# name if it is ever restored, otherwise fall back to the name actually on disk.
INPUT_CANDIDATES = [
    PROJECT_ROOT / "bihs_r3_cleaned.csv",
    PROJECT_ROOT / "bihs_r3_cleaned (1).csv",
]

EXPECTED_ROWS = 5605
EXPECTED_COLS = 137

HH_ID = "hhid2"
# BIHS administrative codes retained for context. NOTE: none of these identifies
# the district -- see check 10 in the validation report.
LOCATION_COLS = ["a10", "a11", "a13", "a14", "a15", "a23"]

# Seasonal climate inputs, declared per season so a missing season is reported
# rather than silently dropped from the definition.
SEASONS = ["aus", "aman", "boro"]
WIND_TEMPLATE = "clim_{season}_wind_avg"
EVAPO_TEMPLATE = "clim_{season}_evapotrans_avg"

# Direction labels used throughout.
POSITIVE = "positive"   # higher raw value  -> MORE vulnerable -> forward min-max
NEGATIVE = "negative"   # higher raw value  -> LESS vulnerable -> reverse min-max


class MissingColumnError(RuntimeError):
    """Raised when an indicator required by the framework is absent."""


# --------------------------------------------------------------------------
# Loading and column verification
# --------------------------------------------------------------------------
def resolve_input_path() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(c) for c in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Cleaned BIHS R3 dataset not found. Looked for:\n  " + searched
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cleaned_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input dataset {path} contains no rows.")
    return df


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Fail loudly instead of substituting a different variable."""
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise MissingColumnError(
            f"Required column(s) for {context} are missing from the cleaned "
            f"dataset: {absent}. Stopping rather than substituting another "
            f"variable. Available columns: {len(df.columns)}."
        )


def resolve_seasonal_columns(
    df: pd.DataFrame, template: str, label: str, log: list[str]
) -> list[str]:
    """Return the seasonal columns that exist, reporting any missing season."""
    found, missing = [], []
    for season in SEASONS:
        col = template.format(season=season)
        (found if col in df.columns else missing).append(col)

    if not found:
        raise MissingColumnError(
            f"No seasonal columns found for {label}. Expected any of: "
            f"{[template.format(season=s) for s in SEASONS]}."
        )
    if missing:
        log.append(
            f"WARNING: {label} is averaged over {len(found)} of {len(SEASONS)} "
            f"seasons. Missing season column(s): {missing}. The indicator "
            f"definition is therefore narrower than intended."
        )
    log.append(f"{label}: averaged over {found}")
    return found


# --------------------------------------------------------------------------
# Indicator preparation
# --------------------------------------------------------------------------
def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN (never inf) on a zero denominator."""
    denom = denominator.astype("float64")
    num = numerator.astype("float64")
    valid = denom.notna() & (denom != 0)
    return pd.Series(
        np.where(valid, num.to_numpy() / np.where(valid, denom.to_numpy(), 1.0), np.nan),
        index=numerator.index,
        dtype="float64",
    )


def build_indicators(df: pd.DataFrame, log: list[str]) -> tuple[pd.DataFrame, dict]:
    """Create the four composite indicators plus the recoded credit indicator."""
    out = pd.DataFrame(index=df.index)
    notes: dict[str, str] = {}

    # --- Sensitivity: agricultural occupation dependence -------------------
    require_columns(df, ["n_occ_agri_own", "n_occ_agri_labour"],
                    "agricultural_occupation_dependence")
    out["agricultural_occupation_dependence"] = (
        df["n_occ_agri_own"].astype("float64")
        + df["n_occ_agri_labour"].astype("float64")
    )
    notes["agricultural_occupation_dependence"] = (
        "n_occ_agri_own + n_occ_agri_labour (count of household members whose "
        "occupation is own-farm agriculture or agricultural wage labour)."
    )

    # --- Sensitivity: food pressure ---------------------------------------
    require_columns(df, ["mealdays_total_7d", "hh_size"], "meals_per_person_week")
    n_zero_size = int((df["hh_size"].fillna(0) <= 0).sum())
    if n_zero_size:
        log.append(
            f"WARNING: {n_zero_size} household(s) have hh_size <= 0; "
            f"meals_per_person_week set to missing for them (no division by zero)."
        )
    out["meals_per_person_week"] = safe_ratio(df["mealdays_total_7d"], df["hh_size"])
    notes["meals_per_person_week"] = (
        "mealdays_total_7d / hh_size (person-meal-days per household member over "
        "the 7-day recall). Reverse-normalized: FEWER meals = MORE vulnerable."
    )

    # --- Exposure: seasonal climate averages -------------------------------
    wind_cols = resolve_seasonal_columns(df, WIND_TEMPLATE, "seasonal_wind_avg", log)
    evapo_cols = resolve_seasonal_columns(
        df, EVAPO_TEMPLATE, "seasonal_evapotranspiration_avg", log
    )
    # skipna=True so a household is not lost when one season is unobserved.
    out["seasonal_wind_avg"] = df[wind_cols].astype("float64").mean(axis=1, skipna=True)
    out["seasonal_evapotranspiration_avg"] = (
        df[evapo_cols].astype("float64").mean(axis=1, skipna=True)
    )
    notes["seasonal_wind_avg"] = f"Mean of {wind_cols}."
    notes["seasonal_evapotranspiration_avg"] = f"Mean of {evapo_cols}."

    # --- Adaptive capacity: credit access recode ---------------------------
    # BIHS codes this question 1 = Yes, 2 = No (verified: has_current_loan == 1
    # implies n_loans > 0 and loan_outstanding_total > 0; == 2 implies both are
    # exactly 0). Left raw, reverse min-max normalization would map "has a loan"
    # to 1.0 = MAXIMUM vulnerability, i.e. the exact inverse of the framework,
    # which treats current borrowing as a proxy for credit access. Recoding to a
    # true 0/1 indicator preserves the intended direction.
    require_columns(df, ["has_current_loan"], "has_current_loan_binary")
    loan_raw = df["has_current_loan"]
    observed_codes = set(pd.unique(loan_raw.dropna()))
    if not observed_codes <= {1.0, 2.0}:
        raise ValueError(
            "has_current_loan carries unexpected codes "
            f"{sorted(observed_codes)}; expected the BIHS 1=Yes / 2=No coding. "
            "Refusing to guess the recode."
        )
    out["has_current_loan_binary"] = (
        loan_raw.map({1.0: 1.0, 2.0: 0.0}).astype("float64")
    )
    notes["has_current_loan_binary"] = (
        "Recode of has_current_loan from the BIHS survey coding (1=Yes, 2=No) to "
        "a 0/1 indicator where 1 = household currently holds a loan. Treated as a "
        "proxy for credit access / coping resources, NOT as debt burden."
    )
    log.append(
        "RECODE: has_current_loan (1=Yes, 2=No) -> has_current_loan_binary "
        f"(1=Yes, 0=No). {int(out['has_current_loan_binary'].sum())} of "
        f"{len(out)} households hold a current loan."
    )

    # --- Adaptive capacity: literacy indicator, v2 fix ---------------------
    # The cleaned adult_literacy_rate is inverted: it correlates -0.50 with
    # mean_edu_years_adults and behaves as an ILLITERACY rate (see caveat C1 of
    # the v1 report). v2 resolution (option b): carry the values unchanged under
    # the name adult_illiteracy_rate and FORWARD-normalize them, so a higher
    # value means more vulnerable. The raw column is left untouched in the
    # output for traceability.
    require_columns(df, ["adult_literacy_rate"], "adult_illiteracy_rate")
    out["adult_illiteracy_rate"] = df["adult_literacy_rate"].astype("float64")
    notes["adult_illiteracy_rate"] = (
        "adult_literacy_rate carried over unchanged and renamed: the cleaned "
        "variable is inverted (corr -0.50 with adult schooling) and measures "
        "illiteracy. Forward-normalized: HIGHER = MORE vulnerable."
    )
    r_edu = out["adult_illiteracy_rate"].corr(df["mean_edu_years_adults"])
    log.append(
        "V2 FIX: adult_literacy_rate -> adult_illiteracy_rate (values unchanged, "
        f"direction flipped to forward min-max). corr with mean_edu_years_adults "
        f"= {r_edu:.4f}, consistent with an illiteracy measure."
    )

    return out, notes


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def minmax_normalize(series: pd.Series, direction: str) -> tuple[pd.Series, dict]:
    """
    Min-max rescale one indicator to [0, 1], oriented so higher = more vulnerable.

        positive:  I = (x - min) / (max - min)
        negative:  I = (max - x) / (max - min)

    Missing values are preserved as NaN (never silently replaced with 0).
    A constant indicator (max == min) is returned as 0.0 rather than dividing by
    zero, and is flagged so it can be excluded from the dimension.
    """
    values = series.astype("float64")
    observed = values.dropna()

    meta = {
        "min": np.nan,
        "max": np.nan,
        "n_missing": int(values.isna().sum()),
        "n_observed": int(observed.size),
        "constant": False,
        "note": "",
    }

    if observed.empty:
        meta["note"] = "All values missing; normalized column is entirely NaN."
        return pd.Series(np.nan, index=series.index, dtype="float64"), meta

    lo = float(observed.min())
    hi = float(observed.max())
    meta["min"], meta["max"] = lo, hi
    spread = hi - lo

    if spread == 0:
        # No variation -> no discriminatory power. Set to 0.0 to avoid dividing
        # by zero; recommended treatment is to drop it from the dimension.
        meta["constant"] = True
        meta["note"] = (
            "CONSTANT indicator (max == min): set to 0.0 where observed, NaN "
            "where missing. Contributes a fixed offset and no variation; "
            "recommended treatment is exclusion from the dimension."
        )
        normalized = pd.Series(
            np.where(values.isna(), np.nan, 0.0), index=series.index, dtype="float64"
        )
        return normalized, meta

    if direction == POSITIVE:
        normalized = (values - lo) / spread
    elif direction == NEGATIVE:
        normalized = (hi - values) / spread
    else:
        raise ValueError(f"Unknown direction '{direction}' for {series.name}.")

    return normalized.astype("float64"), meta


# --------------------------------------------------------------------------
# Indicator framework
# --------------------------------------------------------------------------
def indicator_framework() -> list[dict]:
    """
    The finalized framework: 4 Exposure + 4 Sensitivity + 7 Adaptive Capacity.

    'direction' is the relationship between the RAW value and vulnerability.
    Every Adaptive Capacity indicator except adult_illiteracy_rate is NEGATIVE
    (reverse-normalized); adult_illiteracy_rate is POSITIVE since v2. Either
    way a high *_norm value always means high vulnerability.
    """
    return [
        # ---- Exposure (4) -------------------------------------------------
        dict(name="flood_affected", dimension="Exposure", direction=POSITIVE,
             constructed=False,
             description="Household reported being affected by flooding (0/1)."),
        dict(name="flood_depth_ft_mean", dimension="Exposure", direction=POSITIVE,
             constructed=False,
             description="Mean reported flood depth on household plots (feet)."),
        dict(name="seasonal_wind_avg", dimension="Exposure", direction=POSITIVE,
             constructed=True,
             description="Mean district wind speed across aus/aman/boro seasons."),
        dict(name="seasonal_evapotranspiration_avg", dimension="Exposure",
             direction=POSITIVE, constructed=True,
             description="Mean district evapotranspiration across aus/aman/boro."),

        # ---- Sensitivity (4) ----------------------------------------------
        dict(name="agricultural_occupation_dependence", dimension="Sensitivity",
             direction=POSITIVE, constructed=True,
             description="Members in own-farm or agricultural-labour occupations."),
        dict(name="dependency_ratio", dimension="Sensitivity", direction=POSITIVE,
             constructed=False,
             description="Dependents (under 15 / over 59) per working-age member."),
        dict(name="hh_size", dimension="Sensitivity", direction=POSITIVE,
             constructed=False, description="Number of household members."),
        dict(name="meals_per_person_week", dimension="Sensitivity",
             direction=NEGATIVE, constructed=True,
             description="Person-meal-days per member over 7 days; fewer = worse."),

        # ---- Adaptive Capacity (7), reverse-normalized except illiteracy ---
        dict(name="income_per_capita_monthly", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=False,
             description="Monthly income per household member."),
        dict(name="mean_edu_years_adults", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=False,
             description="Mean completed years of schooling among adults 15+."),
        dict(name="adult_illiteracy_rate", dimension="Adaptive Capacity",
             direction=POSITIVE, constructed=True,
             description=("Share of adults 15+ recorded as illiterate (v2: the "
                          "inverted adult_literacy_rate, forward-normalized).")),
        dict(name="livelihood_diversity", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=False,
             description="Count of distinct livelihood activities."),
        dict(name="any_nonfarm_agri_work", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=False,
             description="Any non-farm / non-agricultural work in household (0/1)."),
        dict(name="land_cultivable_decimal", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=False,
             description="Cultivable land in decimals; asset / buffer."),
        dict(name="has_current_loan_binary", dimension="Adaptive Capacity",
             direction=NEGATIVE, constructed=True,
             description="Holds a current loan (1/0); proxy for credit access."),
    ]


DIMENSIONS = {
    "Exposure": ("exposure_score", 4),
    "Sensitivity": ("sensitivity_score", 4),
    "Adaptive Capacity": ("adaptive_capacity_vulnerability_score", 7),
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def normalize_all(
    source: pd.DataFrame, framework: list[dict], log: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize every framework indicator and return values + a summary table."""
    require_columns(source, [i["name"] for i in framework], "the LVI framework")

    normalized = pd.DataFrame(index=source.index)
    rows = []
    for item in framework:
        name = item["name"]
        norm_col = f"{name}_norm"
        norm_values, meta = minmax_normalize(source[name], item["direction"])
        normalized[norm_col] = norm_values

        raw = source[name].astype("float64")
        rows.append({
            "indicator": name,
            "normalized_column": norm_col,
            "dimension": item["dimension"],
            "constructed": item["constructed"],
            "raw_direction_vs_vulnerability": (
                "higher = MORE vulnerable" if item["direction"] == POSITIVE
                else "higher = LESS vulnerable (reverse-normalized)"
            ),
            "normalization_formula": (
                "(x - min) / (max - min)" if item["direction"] == POSITIVE
                else "(max - x) / (max - min)"
            ),
            "norm_min_used": meta["min"],
            "norm_max_used": meta["max"],
            "n_missing": meta["n_missing"],
            "n_observed": meta["n_observed"],
            "raw_mean": raw.mean(), "raw_std": raw.std(),
            "norm_mean": norm_values.mean(), "norm_std": norm_values.std(),
            "norm_min_observed": norm_values.min(),
            "norm_max_observed": norm_values.max(),
            "is_constant": meta["constant"],
            "description": item["description"],
            "note": meta["note"],
        })

        if meta["constant"]:
            log.append(f"WARNING: indicator '{name}' is constant. {meta['note']}")
        if meta["n_missing"]:
            log.append(
                f"NOTE: indicator '{name}' has {meta['n_missing']} missing value(s).")

    return normalized, pd.DataFrame(rows)


def compute_dimension_scores(
    normalized: pd.DataFrame, framework: list[dict], log: list[str]
) -> pd.DataFrame:
    """
    Equal-weighted arithmetic mean of the normalized indicators in each dimension.

    Missing-value policy: a household's dimension score is the mean of the
    indicators it actually has (skipna). The number of indicators used is stored
    alongside each score so partial scores are never silent. A dimension with no
    observed indicator yields NaN rather than 0.
    """
    scores = pd.DataFrame(index=normalized.index)

    for dimension, (score_col, expected_n) in DIMENSIONS.items():
        cols = [f"{i['name']}_norm" for i in framework if i["dimension"] == dimension]
        if len(cols) != expected_n:
            raise ValueError(
                f"{dimension} resolved to {len(cols)} indicators but the framework "
                f"specifies {expected_n}: {cols}"
            )
        block = normalized[cols]
        count_col = score_col.replace("_score", "") + "_n_indicators"
        scores[score_col] = block.mean(axis=1, skipna=True)
        scores[count_col] = block.notna().sum(axis=1).astype("int64")

        n_partial = int((scores[count_col] < expected_n).sum())
        if n_partial:
            log.append(
                f"WARNING: {n_partial} household(s) have an incomplete {dimension} "
                f"score (fewer than {expected_n} normalized indicators available). "
                f"See the *_n_indicators column."
            )
        log.append(f"{dimension}: {len(cols)} indicators, equal weights -> {score_col}")

    # Composite LVI: equal-weighted mean of the three dimension scores. Requires
    # all three to be present so the composite is never built on a narrower base.
    score_cols = [c for c, _ in DIMENSIONS.values()]
    complete = scores[score_cols].notna().all(axis=1)
    scores["LVI"] = np.where(
        complete, scores[score_cols].mean(axis=1, skipna=False), np.nan
    )
    n_incomplete = int((~complete).sum())
    if n_incomplete:
        log.append(
            f"WARNING: {n_incomplete} household(s) lack at least one dimension "
            f"score; their LVI is set to NaN (households are NOT dropped)."
        )
    return scores


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def spearman(a: pd.Series, b: pd.Series) -> float:
    """
    Spearman rank correlation without SciPy.

    Spearman is Pearson on ranks, so ranking both series and taking the Pearson
    correlation is exact. pandas delegates method='spearman' to SciPy, which is
    not a dependency of this project.
    """
    return float(a.rank().corr(b.rank()))


def _fmt(value, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NaN"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def build_validation_report(
    raw_df: pd.DataFrame,
    result: pd.DataFrame,
    summary: pd.DataFrame,
    framework: list[dict],
    prep_log: list[str],
    input_path: Path,
    input_hash: str,
) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 78)
    add("BIHS ROUND 3 - STAGE 1 LIVELIHOOD VULNERABILITY INDEX (LVI)")
    add("Validation and data-quality report")
    add("=" * 78)
    add(f"Generated             : {datetime.now():%Y-%m-%d %H:%M:%S}")
    add(f"Input file            : {input_path.name}")
    add(f"Input SHA-256         : {input_hash}")
    add(f"pandas / numpy        : {pd.__version__} / {np.__version__}")
    add(f"Python                : {sys.version.split()[0]}")
    add("")
    add("Scope: indicator preparation -> min-max normalization -> dimension")
    add("scores -> composite LVI. No PCA, clustering, ML, or Low/Medium/High")
    add("classification is performed at this stage, by design.")
    add("")

    # ---- 1. Row counts ----------------------------------------------------
    add("-" * 78)
    add("CHECK 1 | INPUT AND OUTPUT ROW COUNTS")
    add("-" * 78)
    add(f"Input rows                 : {len(raw_df)}")
    add(f"Input columns              : {raw_df.shape[1]}")
    add(f"Output rows                : {len(result)}")
    add(f"Output columns             : {result.shape[1]}")
    add(f"Expected rows ({EXPECTED_ROWS})       : "
        f"{'PASS' if len(raw_df) == EXPECTED_ROWS else 'MISMATCH'}")
    add(f"Expected columns ({EXPECTED_COLS})     : "
        f"{'PASS' if raw_df.shape[1] == EXPECTED_COLS else 'MISMATCH'}")
    add(f"Rows preserved end-to-end  : "
        f"{'PASS - no household dropped' if len(raw_df) == len(result) else 'FAIL'}")
    add("")

    # ---- 2. Household identifier -----------------------------------------
    add("-" * 78)
    add("CHECK 2 | HOUSEHOLD IDENTIFIER")
    add("-" * 78)
    ids = result[HH_ID]
    n_missing_id = int(ids.isna().sum())
    n_dup = int(ids.dropna().duplicated().sum())
    add(f"Identifier column          : {HH_ID} (dtype={ids.dtype})")
    add(f"Missing identifiers        : {n_missing_id}")
    add(f"Duplicated (non-missing)   : {n_dup}")
    add(f"Distinct non-missing IDs   : {ids.nunique()}")
    add(f"Uniqueness                 : {'PASS' if n_dup == 0 else 'FAIL'}")
    if n_missing_id:
        idx = result.index[ids.isna()].tolist()
        add(f"WARNING: {n_missing_id} row(s) carry no household identifier "
            f"(0-based row index {idx}).")
        add("         These rows were RETAINED and scored; they simply cannot be")
        add("         linked back to the roster. Recommend recovering the ID before")
        add("         any merge or panel linkage.")
    add("")

    # ---- 3. Missing values ------------------------------------------------
    add("-" * 78)
    add("CHECK 3 | MISSING VALUES IN INDICATOR AND SCORE COLUMNS")
    add("-" * 78)
    add(f"{'column':<46}{'n_missing':>12}{'pct':>10}")
    checked = (
        [i["name"] for i in framework]
        + [f"{i['name']}_norm" for i in framework]
        + [c for c, _ in DIMENSIONS.values()]
        + ["LVI"]
    )
    total_missing = 0
    for col in checked:
        n = int(result[col].isna().sum())
        total_missing += n
        add(f"{col:<46}{n:>12}{100 * n / len(result):>9.2f}%")
    add("")
    add(f"Total missing cells across indicator and score columns: {total_missing}")
    add("Missing-value policy:")
    add("  - Raw missing values are PRESERVED as NaN through normalization")
    add("    (never silently replaced with zero).")
    add("  - A dimension score is the mean of the indicators actually observed")
    add("    for that household; the count used is written to the matching")
    add("    *_n_indicators column so partial scores are always visible.")
    add("  - LVI requires all three dimension scores; otherwise it is NaN.")
    add("  - No household is dropped for missingness.")
    add("")

    L.extend(checks_4_to_9(result, summary, framework))

    # ---- 10. Climate constant within district ----------------------------
    add("-" * 78)
    add("CHECK 10 | CLIMATE INDICATORS CONSTANT WITHIN DISTRICTS")
    add("-" * 78)
    clim_cols = [WIND_TEMPLATE.format(season=s) for s in SEASONS
                 if WIND_TEMPLATE.format(season=s) in raw_df.columns]
    clim_cols += [EVAPO_TEMPLATE.format(season=s) for s in SEASONS
                  if EVAPO_TEMPLATE.format(season=s) in raw_df.columns]
    add("The cleaned dataset carries NO explicit district identifier column: no")
    add("variable in the file partitions the climate series (the BIHS a10/a11/")
    add("a13/a14/a15/a23 codes all mix multiple climate profiles). The check is")
    add("therefore run against a district key reconstructed from the distinct")
    add("combinations of the seasonal climate series themselves.")
    add("")
    profile = raw_df[clim_cols].round(9).astype(str).agg("|".join, axis=1)
    n_profiles = int(profile.nunique())
    add(f"Distinct climate profiles (reconstructed districts): {n_profiles}")
    sizes = profile.value_counts()
    add(f"Households per profile: min={int(sizes.min())}, "
        f"median={int(sizes.median())}, max={int(sizes.max())}")
    add("")
    add("Per-column distinct values within each reconstructed district:")
    clim_ok = True
    for col in clim_cols + ["seasonal_wind_avg", "seasonal_evapotranspiration_avg"]:
        series = raw_df[col] if col in raw_df.columns else result[col]
        max_distinct = int(series.groupby(profile).nunique().max())
        ok = max_distinct == 1
        clim_ok &= ok
        add(f"  {col:<40} max distinct = {max_distinct}  {'PASS' if ok else 'FAIL'}")
    add("")
    add(f"Climate constancy check: {'PASS' if clim_ok else 'FAIL'}")
    add("Confirms the climate variables are district-level contextual data merged")
    add("onto households, not household-specific weather measurements. Two")
    add("households in the same district necessarily share identical exposure on")
    add("these two indicators.")
    add("")

    L.extend(caveats_section(raw_df, result, summary, framework, prep_log,
                             input_path, n_profiles))
    return "\n".join(L)


def checks_4_to_9(
    result: pd.DataFrame, summary: pd.DataFrame, framework: list[dict]
) -> list[str]:
    """Checks 4-9: the checks that depend on normalization and scoring.

    Kept separate so they can be re-run on their own (e.g. the v2 changelog)
    without repeating checks 1-3 and 10, which do not depend on the scores.
    """
    L: list[str] = []
    add = L.append

    # ---- 4. Normalized indicator distributions ---------------------------
    add("-" * 78)
    add("CHECK 4 | NORMALIZED INDICATOR DISTRIBUTIONS")
    add("-" * 78)
    add(f"{'indicator':<38}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
    for _, row in summary.iterrows():
        add(f"{row['indicator']:<38}"
            f"{_fmt(row['norm_min_observed'], 4):>9}"
            f"{_fmt(row['norm_max_observed'], 4):>9}"
            f"{_fmt(row['norm_mean'], 4):>9}"
            f"{_fmt(row['norm_std'], 4):>9}")
    add("")
    add("Raw min/max actually used for each rescaling:")
    add(f"{'indicator':<38}{'raw min':>14}{'raw max':>14}")
    for _, row in summary.iterrows():
        add(f"{row['indicator']:<38}"
            f"{_fmt(row['norm_min_used'], 4):>14}"
            f"{_fmt(row['norm_max_used'], 4):>14}")
    add("")

    # ---- 5. Dimension and composite distributions -------------------------
    add("-" * 78)
    add("CHECK 5 | DIMENSION SCORES AND COMPOSITE LVI")
    add("-" * 78)
    add(f"{'score':<44}{'min':>8}{'max':>8}{'mean':>8}{'std':>8}{'n':>7}")
    for col in [c for c, _ in DIMENSIONS.values()] + ["LVI"]:
        s = result[col]
        add(f"{col:<44}{_fmt(s.min(), 4):>8}{_fmt(s.max(), 4):>8}"
            f"{_fmt(s.mean(), 4):>8}{_fmt(s.std(), 4):>8}{int(s.notna().sum()):>7}")
    add("")
    add("LVI percentiles:")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        add(f"  p{int(q * 100):<3} : {_fmt(result['LVI'].quantile(q), 4)}")
    add("")
    add("Correlation between dimension scores and LVI (Pearson):")
    corr_cols = [c for c, _ in DIMENSIONS.values()] + ["LVI"]
    corr = result[corr_cols].corr()
    add(f"{'':<44}" + "".join(f"{c[:12]:>14}" for c in corr_cols))
    for c in corr_cols:
        add(f"{c:<44}" + "".join(f"{corr.loc[c, o]:>14.4f}" for o in corr_cols))
    add("")

    # ---- 6. Range check ---------------------------------------------------
    add("-" * 78)
    add("CHECK 6 | ALL VALID SCORES WITHIN [0, 1]")
    add("-" * 78)
    range_ok = True
    for col in ([f"{i['name']}_norm" for i in framework]
                + [c for c, _ in DIMENSIONS.values()] + ["LVI"]):
        s = result[col].dropna()
        if s.empty:
            add(f"{col:<46} no observed values")
            continue
        ok = bool(s.min() >= -1e-9 and s.max() <= 1 + 1e-9)
        range_ok &= ok
        add(f"{col:<46}[{s.min():.6f}, {s.max():.6f}]  {'PASS' if ok else 'FAIL'}")
    add("")
    add(f"Overall range check: {'PASS' if range_ok else 'FAIL'}")
    add("")

    # ---- 7. Direction check ----------------------------------------------
    add("-" * 78)
    add("CHECK 7 | DIRECTION - HIGHER ADAPTIVE CAPACITY => LOWER VULNERABILITY")
    add("-" * 78)
    add("For every reverse-normalized indicator the raw value must correlate")
    add("-1.00 with its own normalized value, and negatively with the Adaptive")
    add("Capacity vulnerability score and the composite LVI.")
    add("")
    add(f"{'indicator':<38}{'r(raw,norm)':>13}{'r(raw,AC)':>12}"
        f"{'r(raw,LVI)':>12}  verdict")
    direction_ok = True
    ac_score = result["adaptive_capacity_vulnerability_score"]
    for item in framework:
        if item["direction"] != NEGATIVE:
            continue
        raw = result[item["name"]].astype("float64")
        r_self = raw.corr(result[f"{item['name']}_norm"])
        r_ac = raw.corr(ac_score)
        r_lvi = raw.corr(result["LVI"])
        ok = bool(np.isclose(r_self, -1.0, atol=1e-6)) and (r_ac < 0)
        direction_ok &= ok
        add(f"{item['name']:<38}{_fmt(r_self, 4):>13}{_fmt(r_ac, 4):>12}"
            f"{_fmt(r_lvi, 4):>12}  {'PASS' if ok else 'REVIEW'}")
    add("")
    add("Forward-normalized indicators (higher raw = higher vulnerability):")
    add(f"{'indicator':<38}{'r(raw,norm)':>13}")
    for item in framework:
        if item["direction"] != POSITIVE:
            continue
        raw = result[item["name"]].astype("float64")
        r_self = raw.corr(result[f"{item['name']}_norm"])
        ok = bool(np.isclose(r_self, 1.0, atol=1e-6))
        direction_ok &= ok
        add(f"{item['name']:<38}{_fmt(r_self, 4):>13}  {'PASS' if ok else 'REVIEW'}")
    add("")
    add(f"Overall direction check: {'PASS' if direction_ok else 'REVIEW REQUIRED'}")
    add("")
    add("NOTE on adult_illiteracy_rate (v2): the inverted adult_literacy_rate is")
    add("now carried as an illiteracy rate and FORWARD-normalized, so it sits in")
    add("the forward table above. Its raw value should correlate POSITIVELY with")
    add("the Adaptive Capacity vulnerability score and the LVI:")
    ill = result["adult_illiteracy_rate"].astype("float64")
    add(f"  r(adult_illiteracy_rate, AC score) = {_fmt(ill.corr(ac_score), 4)}")
    add(f"  r(adult_illiteracy_rate, LVI)      = {_fmt(ill.corr(result['LVI']), 4)}")
    add(f"  r(adult_illiteracy_rate, mean_edu_years_adults) = "
        f"{_fmt(ill.corr(result['mean_edu_years_adults']), 4)}")
    add("")

    # ---- 8. Variation -----------------------------------------------------
    add("-" * 78)
    add("CHECK 8 | CONSTANT OR LOW-VARIATION INDICATORS")
    add("-" * 78)
    add(f"{'indicator':<38}{'n_unique':>10}{'raw std':>14}{'norm std':>11}  status")
    low_variation = []
    for item in framework:
        name = item["name"]
        raw = result[name].astype("float64")
        nunique = int(raw.nunique(dropna=True))
        nstd = float(result[f"{name}_norm"].std())
        if nunique <= 1:
            status = "CONSTANT"
        elif nstd < 0.05:
            status = "LOW VARIATION"
            low_variation.append(name)
        else:
            status = "ok"
        add(f"{name:<38}{nunique:>10}{_fmt(raw.std(), 4):>14}"
            f"{_fmt(nstd, 4):>11}  {status}")
    add("")
    constants = summary.loc[summary["is_constant"], "indicator"].tolist()
    add(f"Constant indicators: {constants if constants else 'none'}")
    add(f"Low-variation (normalized std < 0.05): "
        f"{low_variation if low_variation else 'none'}")
    add("No second winsorization or outlier capping was applied: the input is")
    add("already cleaned and winsorized, so re-capping would compound the")
    add("treatment. Normalization uses the observed min/max as they stand.")
    add("")

    # ---- 9. Completeness --------------------------------------------------
    add("-" * 78)
    add("CHECK 9 | HOUSEHOLDS WITH INCOMPLETE DIMENSION SCORES OR LVI")
    add("-" * 78)
    for dimension, (score_col, expected_n) in DIMENSIONS.items():
        count_col = score_col.replace("_score", "") + "_n_indicators"
        counts = result[count_col].value_counts().sort_index()
        add(f"{dimension} (expects {expected_n} indicators):")
        for n_used, n_hh in counts.items():
            flag = "" if n_used == expected_n else "   <-- INCOMPLETE"
            add(f"    {int(n_used)} indicator(s) used : {int(n_hh)} household(s){flag}")
    add("")
    n_lvi_missing = int(result["LVI"].isna().sum())
    add(f"Households with a complete LVI : {int(result['LVI'].notna().sum())}")
    add(f"Households with LVI = NaN      : {n_lvi_missing}")
    add(f"Completeness: "
        f"{'PASS - every household fully scored' if n_lvi_missing == 0 else 'REVIEW'}")
    add("")
    return L


def caveats_section(
    raw_df: pd.DataFrame,
    result: pd.DataFrame,
    summary: pd.DataFrame,
    framework: list[dict],
    prep_log: list[str],
    input_path: Path,
    n_profiles: int,
) -> list[str]:
    L: list[str] = []
    add = L.append

    # ---- Preparation log --------------------------------------------------
    add("-" * 78)
    add("INDICATOR PREPARATION LOG")
    add("-" * 78)
    for line in prep_log:
        add(f"  {line}")
    add("")

    # ---- Caveats ----------------------------------------------------------
    add("=" * 78)
    add("WARNINGS AND METHODOLOGICAL CAVEATS")
    add("=" * 78)
    add("")
    add("[C1] adult_literacy_rate IS INVERTED - RESOLVED IN v2")
    add("-" * 78)
    add("  The cleaned adult_literacy_rate measures the OPPOSITE of literacy.")
    add("  Evidence from the cleaned file:")
    r_edu = raw_df["adult_literacy_rate"].corr(raw_df["mean_edu_years_adults"])
    r_max = raw_df["adult_literacy_rate"].corr(raw_df["max_edu_years"])
    r_inc = raw_df["adult_literacy_rate"].corr(raw_df["income_per_capita_monthly"])
    add(f"    - corr(adult_literacy_rate, mean_edu_years_adults) = {r_edu:.4f}")
    add(f"    - corr(adult_literacy_rate, max_edu_years)         = {r_max:.4f}")
    add(f"    - corr(adult_literacy_rate, income_per_capita)     = {r_inc:.4f}")
    add("    - adult_literacy_rate == n_literate_15plus / n_adults_15plus exactly,")
    add("      so the inversion originates in n_literate_15plus (head_literate")
    add("      codes 1/2 carry ~0 years of schooling, code 4 carries ~6.9).")
    add("")
    add("  ACTION TAKEN (v2): the variable is carried over UNCHANGED under the")
    add("  name adult_illiteracy_rate and FORWARD-normalized, (x - min)/(max - min),")
    add("  so higher = more vulnerable. It remains one of the 7 Adaptive Capacity")
    add("  indicators. The raw adult_literacy_rate column is kept in the output")
    add("  for traceability but no longer feeds any score. This is resolution (b)")
    add("  of the v1 report; re-deriving n_literate_15plus from the raw BIHS codes")
    add("  (resolution a) remains the better fix if the raw roster is available.")
    add("")
    add("[C2] has_current_loan WAS RECODED - PLEASE CONFIRM")
    add("-" * 78)
    add("  In the cleaned file has_current_loan carries the BIHS survey coding")
    add("  1 = Yes and 2 = No, not a 0/1 dummy. Verified:")
    grp = raw_df.groupby("has_current_loan").agg(
        n=("has_current_loan", "size"),
        mean_n_loans=("n_loans", "mean"),
        mean_outstanding=("loan_outstanding_total", "mean"))
    for code, row in grp.iterrows():
        add(f"    has_current_loan={int(code)}: n={int(row['n']):<5} "
            f"mean n_loans={row['mean_n_loans']:.2f}  "
            f"mean outstanding={row['mean_outstanding']:,.0f}")
    add("  Left on the raw 1/2 scale, reverse min-max normalization would map")
    add("  'has a loan' to 1.0 = MAXIMUM vulnerability and 'no loan' to 0.0, the")
    add("  exact inverse of the framework, which treats current borrowing as a")
    add("  proxy for credit access under Adaptive Capacity.")
    add("")
    add("  ACTION TAKEN: recoded to has_current_loan_binary (1 = Yes, 0 = No) and")
    add("  that binary indicator is what is reverse-normalized. This preserves the")
    add("  stated framework rather than changing it; the original column is")
    add("  untouched and is carried through to the output for inspection.")
    add("")
    add("  SIMPLIFICATION TO DOCUMENT IN THE PAPER: current borrowing is used as a")
    add("  proxy for credit ACCESS / coping resources, not as debt burden. The two")
    add("  readings are opposed - a household deep in high-interest debt is not")
    add("  more resilient - so this is a genuine limitation, not merely a coding")
    add("  choice. loan_outstanding_total and loan_interest_rate_mean are present")
    add("  in the data if you later want a burden-sensitive alternative.")
    add("")
    add("[C3] FLOOD INDICATORS ARE STRUCTURALLY COLLINEAR")
    add("-" * 78)
    n_both = int(((raw_df["flood_affected"] == 1)
                  == (raw_df["flood_depth_ft_mean"] > 0)).sum())
    add("  flood_affected == 1 if and only if flood_depth_ft_mean > 0, for all")
    add(f"  {n_both} of {len(raw_df)} households. flood_depth_ft_mean is zero for")
    add("  every unaffected household, so the two Exposure indicators share the")
    add("  same zero/non-zero structure and flooding effectively carries 2 of the")
    add("  4 Exposure slots, i.e. 50% of the dimension. The remaining two slots")
    add("  are district-level climate. Consider whether this weighting is what you")
    add("  intend before the paper is written.")
    add("")
    add("[C4] CLIMATE INDICATORS ARE DISTRICT-LEVEL CONTEXT")
    add("-" * 78)
    add("  seasonal_wind_avg and seasonal_evapotranspiration_avg take only")
    add(f"  {n_profiles} distinct values each - one per district - and are merged")
    add("  onto households. They are contextual, not household-specific weather.")
    add("  Half of the Exposure dimension therefore varies only between districts,")
    add("  and standard errors in any later district-level analysis should account")
    add("  for this clustering.")
    add("")
    add("[C5] ZERO-MEAL HOUSEHOLDS")
    add("-" * 78)
    n_zero_meals = int((raw_df["mealdays_total_7d"] == 0).sum())
    add(f"  {n_zero_meals} household(s) record mealdays_total_7d = 0, with all four")
    add("  demographic meal-day components at zero. Zero meals over a 7-day recall")
    add("  is implausible as a genuine measurement and most likely reflects")
    add("  non-response or an absent household.")
    add("  These households receive meals_per_person_week = 0, which after reverse")
    add("  normalization becomes 1.0 - the MAXIMUM food-pressure vulnerability -")
    add("  and they also set the lower anchor of the min-max range for that")
    add("  indicator, compressing everyone else.")
    add("  ACTION TAKEN: none; the values are left exactly as cleaned and the")
    add("  households are retained. RECOMMENDATION: decide whether these should be")
    add("  treated as missing, which would raise the observed minimum and rescale")
    add("  the indicator for all households.")
    add("")
    add("[C6] SKEW IN meals_per_person_week")
    add("-" * 78)
    mpp = result["meals_per_person_week"]
    add(f"  Distribution: min={mpp.min():.2f}, p25={mpp.quantile(.25):.2f}, "
        f"median={mpp.median():.2f}, p75={mpp.quantile(.75):.2f}, "
        f"max={mpp.max():.2f}")
    add("  The upper tail sits far above the median, so min-max normalization")
    add("  compresses the bulk of households into a narrow band near the top of")
    add("  the reversed scale. This is a known property of min-max on skewed")
    add("  variables and is reported rather than corrected, since the brief")
    add("  prohibits a second capping pass.")
    add("")
    add("[C7] LAND VALUE 99 IS GENUINE, NOT A SURVEY CODE")
    add("-" * 78)
    vc = raw_df["land_cultivable_decimal"].value_counts()
    add(f"  land_cultivable_decimal == 99 occurs {int(vc.get(99, 0))} times. Checked")
    add("  against neighbouring values to rule out a 99 = 'not applicable' code:")
    add("    " + ", ".join(f"{v}:{int(vc.get(v, 0))}"
                           for v in [95, 96, 97, 98, 99, 100, 101, 102]))
    add("  The frequency at 99 is in line with its neighbours and 99 decimals is")
    add("  about one acre, so it is treated as a genuine measurement.")
    n_zero_land = int((raw_df["land_cultivable_decimal"] == 0).sum())
    add(f"  Separately, {n_zero_land} households report zero cultivable land")
    add(f"  ({100 * n_zero_land / len(raw_df):.1f}%), which is substantively")
    add("  meaningful - landlessness - and is retained as a true zero.")
    add("")
    add("[C8] EQUAL WEIGHTING")
    add("-" * 78)
    add("  Indicators are equally weighted within each dimension and the three")
    add("  dimensions are equally weighted in the composite, per the brief. Because")
    add("  the dimensions hold 4, 4 and 7 indicators, a single Adaptive Capacity")
    add("  indicator carries 1/21 of the composite while a single Exposure")
    add("  indicator carries 1/12. This is the standard balanced-weighted-average")
    add("  LVI convention (Hahn et al.), but it is a choice worth stating.")
    add("")
    add("[C9] INPUT FILENAME")
    add("-" * 78)
    add("  The brief specifies bihs_r3_cleaned.csv; the file present is")
    add(f"  '{input_path.name}'. It matches the expected shape")
    add(f"  ({EXPECTED_ROWS} x {EXPECTED_COLS}) and was read unmodified. The script")
    add("  prefers the canonical name if you rename it.")
    add("")
    add("=" * 78)
    add("END OF REPORT")
    add("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------
# Methodology note
# --------------------------------------------------------------------------
def build_methodology_note(summary: pd.DataFrame, result: pd.DataFrame) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Stage 1 - Livelihood Vulnerability Index (LVI) construction")
    add("")
    add(f"BIHS Round 3 | Generated {datetime.now():%Y-%m-%d} | "
        f"n = {len(result)} households")
    add("")
    add("## 1. Approach")
    add("")
    add("The LVI follows the balanced weighted-average composite-index approach.")
    add("Every indicator is rescaled to `[0, 1]`, oriented so a higher value always")
    add("means greater vulnerability, averaged with equal weights within its")
    add("dimension, and the three dimension scores are averaged with equal weights")
    add("into the composite.")
    add("")
    add("## 2. Normalization")
    add("")
    add("For an indicator where a higher raw value means **greater** vulnerability:")
    add("")
    add("```")
    add("I = (x - min(x)) / (max(x) - min(x))")
    add("```")
    add("")
    add("For an indicator where a higher raw value means **lower** vulnerability")
    add("(all Adaptive Capacity indicators except `adult_illiteracy_rate`, plus")
    add("`meals_per_person_week`):")
    add("")
    add("```")
    add("I = (max(x) - x) / (max(x) - min(x))")
    add("```")
    add("")
    add("`min(x)` and `max(x)` are the observed sample minimum and maximum, listed")
    add("per indicator in `lvi_indicator_summary.csv`. Missing values are preserved")
    add("as `NaN`. A constant indicator (`max == min`) is set to `0.0` rather than")
    add("dividing by zero and is flagged; no indicator in this dataset is constant.")
    add("No second winsorization is applied - the input is already cleaned.")
    add("")
    add("## 3. Dimension scores")
    add("")
    add("```")
    add("D_exposure    = mean(4 normalized Exposure indicators)")
    add("D_sensitivity = mean(4 normalized Sensitivity indicators)")
    add("D_adaptive    = mean(7 normalized Adaptive Capacity indicators)")
    add("```")
    add("")
    add("Written as `exposure_score`, `sensitivity_score` and")
    add("`adaptive_capacity_vulnerability_score`. Each is accompanied by a")
    add("`*_n_indicators` column recording how many indicators were available for")
    add("that household, so partial scores are never silent.")
    add("")
    add("## 4. Composite")
    add("")
    add("```")
    add("LVI = (exposure_score + sensitivity_score")
    add("       + adaptive_capacity_vulnerability_score) / 3")
    add("```")
    add("")
    add("Range `[0, 1]`; higher = more vulnerable. Households are not classified")
    add("into Low/Medium/High at this stage.")
    add("")
    add("## 5. Indicator placement")
    add("")
    for dimension in DIMENSIONS:
        block = summary[summary["dimension"] == dimension]
        add(f"### {dimension} ({len(block)} indicators)")
        add("")
        add("| Indicator | Direction | Normalization | Min | Max |")
        add("|---|---|---|---|---|")
        for _, row in block.iterrows():
            direction = ("higher = more vulnerable"
                         if row["normalization_formula"].startswith("(x")
                         else "higher = less vulnerable (reversed)")
            add(f"| `{row['indicator']}` | {direction} | "
                f"`{row['normalization_formula']}` | "
                f"{row['norm_min_used']:.4f} | {row['norm_max_used']:.4f} |")
        add("")
    add("## 6. Constructed indicators")
    add("")
    add("```")
    add("agricultural_occupation_dependence = n_occ_agri_own + n_occ_agri_labour")
    add("meals_per_person_week              = mealdays_total_7d / hh_size")
    add("seasonal_wind_avg                  = mean(clim_{aus,aman,boro}_wind_avg)")
    add("seasonal_evapotranspiration_avg    = mean(clim_{aus,aman,boro}_evapotrans_avg)")
    add("has_current_loan_binary            = 1 if has_current_loan == 1 else 0")
    add("adult_illiteracy_rate              = adult_literacy_rate  (renamed, v2)")
    add("```")
    add("")
    add("## 7. Caveats carried into the paper")
    add("")
    add("1. **`adult_literacy_rate` is inverted (resolved in v2).** It correlates")
    add("   **negatively** with adult schooling (r = -0.50) and with income, i.e. it")
    add("   measures illiteracy. Since v2 it enters the index as")
    add("   `adult_illiteracy_rate`, forward-normalized (higher = more vulnerable).")
    add("   See `lvi_validation_report.txt`, caveat C1 and the v2 changelog.")
    add("2. **`has_current_loan` was recoded** from the BIHS 1=Yes/2=No coding to a")
    add("   0/1 indicator. Without this the framework would have been inverted.")
    add("   Current borrowing proxies credit **access**, not debt burden.")
    add("3. **Flood indicators are structurally collinear**: `flood_affected == 1`")
    add("   exactly when `flood_depth_ft_mean > 0`, so flooding carries half the")
    add("   Exposure dimension.")
    add("4. **Climate variables are district-level context** (39 districts), not")
    add("   household weather; they vary only between districts.")
    add("5. **Equal weighting** means an Adaptive Capacity indicator carries 1/21 of")
    add("   the composite against 1/12 for an Exposure indicator.")
    add("")
    add("## 8. Reproducing")
    add("")
    add("```bash")
    add("python outputs/stage1_lvi/lvi_construction.py")
    add("```")
    add("")
    add("The script is deterministic: no sampling, no randomness, no seed needed.")
    add("It reads the cleaned CSV read-only and never writes back to it.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def assemble_output(
    raw_df: pd.DataFrame,
    constructed: pd.DataFrame,
    normalized: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Identifiers + location + raw inputs + constructed + normalized + scores."""
    parts: list[pd.DataFrame] = []

    id_cols = [HH_ID] + [c for c in LOCATION_COLS if c in raw_df.columns]
    parts.append(raw_df[id_cols])

    # Raw source variables that feed the framework, including the inputs to the
    # constructed indicators, so every score is traceable from the output alone.
    raw_inputs = [
        "flood_affected", "flood_depth_ft_mean",
        *[WIND_TEMPLATE.format(season=s) for s in SEASONS
          if WIND_TEMPLATE.format(season=s) in raw_df.columns],
        *[EVAPO_TEMPLATE.format(season=s) for s in SEASONS
          if EVAPO_TEMPLATE.format(season=s) in raw_df.columns],
        "n_occ_agri_own", "n_occ_agri_labour", "dependency_ratio", "hh_size",
        "mealdays_total_7d", "income_per_capita_monthly", "mean_edu_years_adults",
        "adult_literacy_rate", "livelihood_diversity", "any_nonfarm_agri_work",
        "land_cultivable_decimal", "has_current_loan",
    ]
    parts.append(raw_df[[c for c in raw_inputs if c in raw_df.columns]])
    parts.append(constructed)
    parts.append(normalized)

    score_cols = []
    for score_col, _n in DIMENSIONS.values():
        score_cols += [score_col, score_col.replace("_score", "") + "_n_indicators"]
    parts.append(scores[score_cols + ["LVI"]])

    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def build_v2_changelog(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    framework: list[dict],
    old_scores: pd.DataFrame | None,
    prep_log: list[str],
) -> str:
    """Short v2 changelog appended to the existing validation report."""
    L: list[str] = []
    add = L.append
    add("")
    add("")
    add("=" * 78)
    add("V2 CHANGELOG - LITERACY INDICATOR FIX")
    add("=" * 78)
    add(f"Generated             : {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("Change:")
    add("  adult_literacy_rate was confirmed inverted (corr with")
    add("  mean_edu_years_adults = -0.50; it behaves as an illiteracy rate).")
    add("  It is now carried as adult_illiteracy_rate with values UNCHANGED and")
    add("  FORWARD-normalized, (x - min)/(max - min), instead of reverse")
    add("  min-max. It stays in Adaptive Capacity; the dimension still has 7")
    add("  indicators. Column adult_literacy_rate_norm is replaced by")
    add("  adult_illiteracy_rate_norm. No other indicator was touched. The flood")
    add("  collinearity (C3) is unchanged and remains an accepted limitation.")
    add("")
    add("Recomputed: exposure_score, sensitivity_score,")
    add("adaptive_capacity_vulnerability_score, LVI. Exposure and Sensitivity")
    add("are numerically identical to v1 (no input to them changed).")
    add("")
    for line in prep_log:
        if line.startswith("V2 FIX"):
            add(f"  {line}")
    add("")
    add("Old (v1) vs new (v2) score distributions:")
    add(f"{'score':<44}{'v1 mean':>10}{'v1 std':>10}{'v2 mean':>10}{'v2 std':>10}")
    for col in [c for c, _ in DIMENSIONS.values()] + ["LVI"]:
        new = result[col]
        if old_scores is not None and col in old_scores.columns:
            old = old_scores[col]
            add(f"{col:<44}{_fmt(old.mean(), 4):>10}{_fmt(old.std(), 4):>10}"
                f"{_fmt(new.mean(), 4):>10}{_fmt(new.std(), 4):>10}")
        else:
            add(f"{col:<44}{'n/a':>10}{'n/a':>10}"
                f"{_fmt(new.mean(), 4):>10}{_fmt(new.std(), 4):>10}")
    if (old_scores is not None and "LVI" in old_scores.columns
            and len(old_scores) == len(result)):
        old_lvi = old_scores["LVI"].reset_index(drop=True)
        new_lvi = result["LVI"].reset_index(drop=True)
        diff = new_lvi - old_lvi
        add("")
        add("Household-level change in LVI (v2 - v1), same row order:")
        add(f"  mean change            : {diff.mean():+.4f}")
        add(f"  mean absolute change   : {diff.abs().mean():.4f}")
        add(f"  max  absolute change   : {diff.abs().max():.4f}")
        add(f"  Pearson  corr(v1, v2)  : {old_lvi.corr(new_lvi):.4f}")
        add(f"  Spearman corr(v1, v2)  : {spearman(old_lvi, new_lvi):.4f}")
    add("")
    add("Row with missing hhid2 (0-based row index 0), retained and scored.")
    add("Identifying columns for manual ID recovery:")
    missing = result.index[result[HH_ID].isna()].tolist()
    for i in missing:
        vals = ", ".join(f"{c}={result.at[i, c]}" for c in LOCATION_COLS
                         if c in result.columns)
        add(f"  row {i}: {vals}")
    add("")
    add("Validation checks 4-9 re-run on the v2 scores (checks 1, 2, 3 and 10")
    add("are unaffected by the change and were not re-run):")
    add("")
    L.extend(checks_4_to_9(result, summary, framework))
    add("=" * 78)
    add("END OF V2 CHANGELOG")
    add("=" * 78)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    append_changelog = "--append-v2-changelog" in argv
    print("=" * 70)
    print("BIHS R3 - Stage 1 LVI construction")
    print("=" * 70)

    try:
        input_path = resolve_input_path()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Reading   : {input_path}")
    raw_df = load_cleaned_data(input_path)
    input_hash = file_sha256(input_path)
    print(f"Loaded    : {len(raw_df)} rows x {raw_df.shape[1]} columns")
    if raw_df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        print(f"NOTE      : expected {EXPECTED_ROWS} x {EXPECTED_COLS}; "
              f"continuing with the actual shape.")

    prep_log: list[str] = []
    framework = indicator_framework()

    try:
        require_columns(raw_df, [HH_ID], "the household identifier")
        constructed, _notes = build_indicators(raw_df, prep_log)
        source = pd.concat([raw_df, constructed], axis=1)
        source = source.loc[:, ~source.columns.duplicated(keep="last")]

        normalized, summary = normalize_all(source, framework, prep_log)
        scores = compute_dimension_scores(normalized, framework, prep_log)
    except (MissingColumnError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    result = assemble_output(raw_df, constructed, normalized, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_path = OUTPUT_DIR / "bihs_r3_lvi_scored.csv"
    summary_path = OUTPUT_DIR / "lvi_indicator_summary.csv"
    report_path = OUTPUT_DIR / "lvi_validation_report.txt"
    method_path = OUTPUT_DIR / "lvi_methodology.md"

    # Capture the previous scores BEFORE overwriting, for the v2 changelog.
    old_scores = None
    if scored_path.exists():
        old_scores = pd.read_csv(scored_path, dtype={HH_ID: str}, low_memory=False)

    result.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    if append_changelog and report_path.exists():
        # Keep the existing v1 report (checks 1-3 and 10 unaffected) and
        # append the v2 changelog with checks 4-9 re-run.
        with report_path.open("a", encoding="utf-8") as fh:
            fh.write(build_v2_changelog(result, summary, framework,
                                        old_scores, prep_log))
    else:
        report = build_validation_report(
            raw_df, result, summary, framework, prep_log, input_path, input_hash)
        report_path.write_text(report, encoding="utf-8")
    method_path.write_text(build_methodology_note(summary, result), encoding="utf-8")

    # ---- console summary --------------------------------------------------
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Households scored : {len(result)} (input {len(raw_df)}, none dropped)")
    print("Indicators        : "
          f"{', '.join(f'{d} {n}' for d, (_c, n) in DIMENSIONS.items())}")
    print()
    print(f"{'score':<40}{'min':>8}{'max':>8}{'mean':>8}{'std':>8}")
    for col in [c for c, _ in DIMENSIONS.values()] + ["LVI"]:
        s = result[col]
        print(f"{col:<40}{s.min():>8.4f}{s.max():>8.4f}"
              f"{s.mean():>8.4f}{s.std():>8.4f}")
    print()
    in_range = all(
        (result[c].dropna().between(-1e-9, 1 + 1e-9)).all()
        for c in [c for c, _ in DIMENSIONS.values()] + ["LVI"])
    print(f"All scores within [0, 1]   : {'PASS' if in_range else 'FAIL'}")
    print(f"Households with LVI = NaN  : {int(result['LVI'].isna().sum())}")
    print(f"Missing household IDs      : {int(result[HH_ID].isna().sum())}")
    for i in result.index[result[HH_ID].isna()].tolist():
        vals = ", ".join(f"{c}={result.at[i, c]}" for c in LOCATION_COLS
                         if c in result.columns)
        print(f"  row {i} (no hhid2)        : {vals}")
    print("Constant indicators        : "
          f"{summary.loc[summary['is_constant'], 'indicator'].tolist() or 'none'}")
    print()
    warnings = [line for line in prep_log if line.startswith("WARNING")]
    print(f"Preparation warnings       : {len(warnings)}")
    for line in warnings:
        print(f"  - {line}")
    print()
    print("Open issues requiring your decision (detail in the validation report):")
    print("  C1  adult_literacy_rate inverted -> RESOLVED in v2 as")
    print("      adult_illiteracy_rate, forward-normalized")
    print("  C2  has_current_loan recoded from 1=Yes/2=No to 1/0 - please confirm")
    print("  C5  15 households record zero meal-days over 7 days")
    print()
    print("-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)
    for path in [scored_path, summary_path, report_path, method_path]:
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
