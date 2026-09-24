#!/usr/bin/env python3
"""Fixed-label 2x2 diagnostic: target ingredients x climate-profile blocking.

This is an intentional leakage stress test, not a prospective validation result.
The target and its normalized ingredients are frozen from the archived full-sample
index so that all four cells share exactly the same labels and model settings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "bihs_r3_cleaned (1).csv"
SCORED = ROOT / "outputs/stage1_lvi/bihs_r3_lvi_scored.csv"
CONTEXT = [
    "clim_aus_rain_total_mm", "clim_aman_rain_total_mm", "clim_boro_rain_total_mm",
    "clim_aus_solar_rad_avg", "clim_aman_solar_rad_avg", "clim_boro_solar_rad_avg",
    "clim_aus_district_yield_ratio", "clim_aman_district_yield_ratio",
    "clim_boro_district_yield_ratio", "mean_district_yield_ratio",
    "clim_aus_district_area_ha", "clim_aman_district_area_ha",
    "clim_boro_district_area_ha", "clim_aus_district_production_mt",
    "clim_aman_district_production_mt", "clim_boro_district_production_mt",
    "mean_plot_temperature", "max_plot_temperature", "mean_plot_humidity",
    "total_plot_rainfall", "mean_plot_solar_rad", "plot_distance_m_mean",
]
PROFILE = [
    "clim_aus_wind_avg", "clim_aman_wind_avg", "clim_boro_wind_avg",
    "clim_aus_evapotrans_avg", "clim_aman_evapotrans_avg",
    "clim_boro_evapotrans_avg", "clim_aus_rain_total_mm",
    "clim_aman_rain_total_mm", "clim_boro_rain_total_mm",
]
INGREDIENTS = [
    "agricultural_occupation_dependence_norm", "dependency_ratio_norm",
    "hh_size_norm", "meals_per_person_week_norm",
    "income_per_capita_monthly_norm", "mean_edu_years_adults_norm",
    "adult_illiteracy_rate_norm", "livelihood_diversity_norm",
    "any_nonfarm_agri_work_norm", "land_cultivable_decimal_norm",
    "has_current_loan_binary_norm",
]


def main() -> None:
    raw, scored = pd.read_csv(RAW), pd.read_csv(SCORED)
    assert len(raw) == len(scored) == 5605
    assert raw.hhid2.fillna("MISSING").astype(str).equals(
        scored.hhid2.fillna("MISSING").astype(str))
    assert not raw[CONTEXT + PROFILE].isna().all(axis=1).any()
    socio = (scored.sensitivity_score + scored.adaptive_capacity_vulnerability_score) / 2
    reconstructed = (scored[INGREDIENTS[:4]].mean(axis=1)
                     + scored[INGREDIENTS[4:]].mean(axis=1)) / 2
    error = float((socio - reconstructed).abs().max())
    assert error < 1e-10, error
    threshold = float(socio.quantile(2 / 3))
    y = (socio > threshold).astype(int).to_numpy()
    groups = pd.factorize(pd.MultiIndex.from_frame(raw[PROFILE]), sort=True)[0]
    assert np.unique(groups).size == 39

    records, predictions = [], []
    for split_name, splitter in [
        ("household_random", StratifiedKFold(5, shuffle=True, random_state=42)),
        ("climate_profile_blocked", StratifiedGroupKFold(5, shuffle=True, random_state=42)),
    ]:
        splits = list(splitter.split(raw, y, groups) if split_name == "climate_profile_blocked"
                      else splitter.split(raw, y))
        assert sorted(np.concatenate([te for _, te in splits]).tolist()) == list(range(len(y)))
        for feature_name, columns in [
            ("nonoverlapping", CONTEXT),
            ("target_ingredients_added", CONTEXT + INGREDIENTS),
        ]:
            X = pd.concat([raw[CONTEXT].reset_index(drop=True),
                           scored[INGREDIENTS].reset_index(drop=True)], axis=1)[columns]
            prob = np.full(len(y), np.nan)
            for fold, (tr, te) in enumerate(splits, 1):
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=1.0, max_iter=3000, random_state=42))
                model.fit(X.iloc[tr], y[tr])
                prob[te] = model.predict_proba(X.iloc[te])[:, 1]
                predictions.extend({"row_index": int(i), "split": split_name,
                                    "features": feature_name, "fold": fold,
                                    "group": int(groups[i]), "label": int(y[i]),
                                    "probability": float(prob[i])} for i in te)
            assert np.isfinite(prob).all()
            records.append({"split": split_name, "features": feature_name,
                            "n": len(y), "n_features": len(columns),
                            "auroc": roc_auc_score(y, prob),
                            "auprc": average_precision_score(y, prob),
                            "brier": brier_score_loss(y, prob)})
    pd.DataFrame(records).to_csv(OUT / "controlled_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "controlled_oof.csv", index=False)
    (OUT / "controlled_metadata.json").write_text(json.dumps({
        "target": "fixed archived full-sample socioeconomic upper tertile",
        "target_cutoff": threshold, "positive_n": int(y.sum()),
        "score_reconstruction_max_abs_error": error,
        "groups": int(np.unique(groups).size), "folds_per_split": 5,
        "model": "median imputation, standardization, logistic regression C=1",
        "interpretation": "diagnostic paired feature contrasts within each split; full-sample target/normalization make this unsuitable as an unbiased validation estimate",
    }, indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
