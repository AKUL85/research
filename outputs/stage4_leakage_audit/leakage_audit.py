#!/usr/bin/env python3
"""
Stage 4: Leakage audit of the Stage 3 supervised experiments, plus a corrected
(non-circular, spatially-blocked) experiment.

READ-ONLY with respect to every pre-existing artifact. Writes only into
outputs/stage4_leakage_audit/.

Three questions:

Q1  Is the Stage 3 "zero-leakage" exposure-only model actually leakage-free?
    The four Exposure indicators handed to it are, by construction, the four
    terms of exposure_score, and exposure_score is exactly one third of the
    LVI target. We quantify how much of the reported performance is recovered
    by the closed-form index with no model fitted at all.

Q2  What does the 4-tier ablation actually measure? Tier 4 supplies all 15
    indicators whose equal-weighted mean *is* the target. We verify this is an
    algebraic identity, not an empirical finding.

Q3  Corrected experiment. Target = the socioeconomic half of vulnerability
    (sensitivity + adaptive capacity), which contains no climate input.
    Features = only climate/agricultural variables that never enter the index.
    Validation = district-held-out GroupKFold, because every climate predictor
    is (near-)constant within the 39 reconstructed districts.

Reproducibility: RANDOM_STATE = 42.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "stage4_leakage_audit"
OUT.mkdir(parents=True, exist_ok=True)

LVI_CSV = ROOT / "outputs" / "stage1_lvi" / "bihs_r3_lvi_scored.csv"
RAW_CSV = next(p for p in [ROOT / "bihs_r3_cleaned.csv",
                           ROOT / "bihs_r3_cleaned (1).csv"] if p.exists())

# The four Exposure indicators == the four terms of exposure_score.
F_EXP4 = ["flood_affected_norm", "flood_depth_ft_mean_norm",
          "seasonal_wind_avg_norm", "seasonal_evapotranspiration_avg_norm"]

# All fifteen indicators whose equal-weighted dimension means define the LVI.
F_ALL15 = F_EXP4 + [
    "agricultural_occupation_dependence_norm", "dependency_ratio_norm",
    "hh_size_norm", "meals_per_person_week_norm",
    "income_per_capita_monthly_norm", "mean_edu_years_adults_norm",
    "adult_illiteracy_rate_norm", "livelihood_diversity_norm",
    "any_nonfarm_agri_work_norm", "land_cultivable_decimal_norm",
    "has_current_loan_binary_norm",
]

# Climate / agricultural variables that never enter the LVI at any stage.
F_EXOG = [
    "clim_aus_rain_total_mm", "clim_aman_rain_total_mm", "clim_boro_rain_total_mm",
    "clim_aus_solar_rad_avg", "clim_aman_solar_rad_avg", "clim_boro_solar_rad_avg",
    "clim_aus_district_yield_ratio", "clim_aman_district_yield_ratio",
    "clim_boro_district_yield_ratio", "mean_district_yield_ratio",
    "clim_aus_district_area_ha", "clim_aman_district_area_ha", "clim_boro_district_area_ha",
    "clim_aus_district_production_mt", "clim_aman_district_production_mt",
    "clim_boro_district_production_mt",
    "mean_plot_temperature", "max_plot_temperature", "mean_plot_humidity",
    "total_plot_rainfall", "mean_plot_solar_rad", "plot_distance_m_mean",
]

# Columns whose distinct combinations reconstruct the district climate profile.
CLIM_PROFILE = ["clim_aus_wind_avg", "clim_aman_wind_avg", "clim_boro_wind_avg",
                "clim_aus_evapotrans_avg", "clim_aman_evapotrans_avg",
                "clim_boro_evapotrans_avg", "clim_aus_rain_total_mm",
                "clim_aman_rain_total_mm", "clim_boro_rain_total_mm"]


def models():
    """The same three families used in Stage 3, at the Stage 3 settings."""
    return {
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(C=0.01, max_iter=1000,
                                                      random_state=RANDOM_STATE))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5,
                                               min_samples_split=2,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=1.0, random_state=RANDOM_STATE,
                                 eval_metric="logloss", n_jobs=-1),
    }


def cv_eval(X, y, groups, scheme, label, feat_label):
    """Out-of-fold evaluation under either a stratified or a district-grouped CV."""
    rows = []
    for name, mdl in models().items():
        if scheme == "grouped":
            cv = GroupKFold(n_splits=5)
            splitter = cv.split(X, y, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            splitter = cv.split(X, y)
        prob = cross_val_predict(mdl, X, y, cv=list(splitter),
                                 method="predict_proba", n_jobs=1)[:, 1]
        rows.append({
            "target": label, "features": feat_label, "cv": scheme, "model": name,
            "n": len(y), "p": X.shape[1],
            "accuracy": accuracy_score(y, (prob >= 0.5).astype(int)),
            "roc_auc": roc_auc_score(y, prob),
        })
    return rows


def auc_rank(score, label):
    """Mann-Whitney ROC-AUC of a fixed score. No model, nothing fitted."""
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum()); n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    lvi = pd.read_csv(LVI_CSV)
    raw = pd.read_csv(RAW_CSV)
    assert len(lvi) == len(raw) == 5605

    for c in F_EXOG:
        lvi[c] = raw[c]
    district = raw.groupby(CLIM_PROFILE).ngroup().to_numpy()

    log = {}

    # ---- Structural identities -------------------------------------------
    exp_recon = lvi[F_EXP4].mean(axis=1)
    lvi_recon = lvi[["exposure_score", "sensitivity_score",
                     "adaptive_capacity_vulnerability_score"]].mean(axis=1)
    log["identity_exposure_score_max_abs_err"] = float(
        (exp_recon - lvi["exposure_score"]).abs().max())
    log["identity_LVI_max_abs_err"] = float((lvi_recon - lvi["LVI"]).abs().max())
    log["n_districts"] = int(pd.Series(district).nunique())

    # ---- Targets ----------------------------------------------------------
    qf = lvi["LVI"].quantile([1 / 3, 2 / 3]).to_numpy()
    y_full = (lvi["LVI"] > qf[1]).astype(int).to_numpy()

    lvi["socio_half"] = lvi[["sensitivity_score",
                             "adaptive_capacity_vulnerability_score"]].mean(axis=1)
    qs = lvi["socio_half"].quantile([1 / 3, 2 / 3]).to_numpy()
    y_socio = (lvi["socio_half"] > qs[1]).astype(int).to_numpy()

    log["lvi_tertile_cuts"] = [float(v) for v in qf]
    log["socio_tertile_cuts"] = [float(v) for v in qs]
    log["majority_baseline_acc"] = float(1 - y_full.mean())
    log["corr_socio_half_vs_LVI"] = float(lvi["socio_half"].corr(lvi["LVI"]))

    # ---- Q1: closed-form index vs fitted model, same information ----------
    log["closed_form_exposure_score_auc_on_y_full"] = float(
        auc_rank(lvi["exposure_score"].to_numpy(), y_full))
    log["closed_form_flood_depth_auc_on_y_full"] = float(
        auc_rank(lvi["flood_depth_ft_mean"].to_numpy(), y_full))

    rows = []
    rows += cv_eval(lvi[F_EXP4].to_numpy(), y_full, district, "stratified",
                    "High-LVI tertile", "4 Exposure indicators (= 1/3 of target)")
    # ---- Q2: all 15 components -> the identity ----------------------------
    rows += cv_eval(lvi[F_ALL15].to_numpy(), y_full, district, "stratified",
                    "High-LVI tertile", "all 15 LVI components (= the target)")
    # ---- Q3: corrected experiment ----------------------------------------
    rows += cv_eval(lvi[F_EXOG].to_numpy(), y_socio, district, "stratified",
                    "High socioeconomic-half tertile", "exogenous climate/agri only")
    rows += cv_eval(lvi[F_EXOG].to_numpy(), y_socio, district, "grouped",
                    "High socioeconomic-half tertile", "exogenous climate/agri only")
    rows += cv_eval(lvi[F_EXOG].to_numpy(), y_full, district, "grouped",
                    "High-LVI tertile", "exogenous climate/agri only")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "leakage_audit_results.csv", index=False)

    # ---- R^2 of reconstructing exposure_score from its own four terms -----
    from sklearn.linear_model import LinearRegression
    r2 = LinearRegression().fit(lvi[F_EXP4], lvi["exposure_score"]).score(
        lvi[F_EXP4], lvi["exposure_score"])
    log["R2_exposure_score_from_its_4_terms"] = float(r2)

    with open(OUT / "leakage_audit_log.json", "w") as fh:
        json.dump(log, fh, indent=2)

    pd.set_option("display.width", 200, "display.max_colwidth", 46)
    print("\n=== STRUCTURAL IDENTITIES ===")
    for k in ["identity_exposure_score_max_abs_err", "identity_LVI_max_abs_err",
              "R2_exposure_score_from_its_4_terms", "n_districts",
              "corr_socio_half_vs_LVI", "majority_baseline_acc"]:
        print(f"  {k:42s} {log[k]}")
    print("\n=== Q1: SAME INFORMATION, NO MODEL FITTED ===")
    print(f"  ROC-AUC of raw exposure_score (closed form) : "
          f"{log['closed_form_exposure_score_auc_on_y_full']:.4f}")
    print(f"  ROC-AUC of raw flood_depth_ft_mean          : "
          f"{log['closed_form_flood_depth_auc_on_y_full']:.4f}")
    print("\n=== OUT-OF-FOLD RESULTS ===")
    print(res.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {OUT/'leakage_audit_results.csv'}")


if __name__ == "__main__":
    main()
