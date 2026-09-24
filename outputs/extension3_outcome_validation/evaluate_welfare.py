#!/usr/bin/env python3
"""Held-out-district comparison for separately prepared household expenditure.

The outcome is contemporaneous monthly per-capita expenditure, analyzed on
the log scale. This is a criterion and geographic-transfer test, not a future
forecast. All model choices are fixed before reading outcome values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path(__file__).resolve().parent
DATA = OUT / "public_welfare_link.csv"
SPECIFICATIONS = {
    "training_mean": [],
    "lvi_only": ["LVI"],
    "demographic_baseline": ["head_age", "head_sex", "hh_size"],
    "demographic_plus_lvi": ["head_age", "head_sex", "hh_size", "LVI"],
    "demographic_plus_robust_high": ["head_age", "head_sex", "hh_size", "strict_robust_high"],
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                tr: np.ndarray, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.full(len(test), float(y[tr].mean()))
    numeric = [c for c in cols if c != "head_sex"]
    categorical = [c for c in cols if c == "head_sex"]
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("sex", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
         categorical),
    ])
    model = make_pipeline(preprocess, Ridge(alpha=1.0))
    model.fit(train[cols], y[tr])
    return model.predict(test[cols])


def main() -> None:
    data = pd.read_csv(DATA)
    assert len(data) == 5605 and data.official_a01.is_unique
    data = data.loc[data.monthly_per_capita_expenditure.notna()].copy()
    assert len(data) == 5604 and data.monthly_per_capita_expenditure.gt(0).all()
    assert data.district.nunique() == 64
    y = np.log(data.monthly_per_capita_expenditure.to_numpy())
    groups = data.district.to_numpy()
    records = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(data, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])
        for name, cols in SPECIFICATIONS.items():
            pred = fit_predict(data.iloc[tr], data.iloc[te], y, tr, cols)
            records.extend({"row_index": int(data.iloc[pos].row_index),
                            "district": int(groups[pos]), "fold": fold,
                            "model": name, "log_outcome": float(y[pos]),
                            "log_prediction": float(value)}
                           for pos, value in zip(te, pred))
    oof = pd.DataFrame(records)
    assert oof.groupby("model").row_index.nunique().eq(len(data)).all()
    metrics = []
    for name, d in oof.groupby("model", sort=False):
        metrics.append({"model": name, "n": len(d), "held_out_districts": d.district.nunique(),
                        "mae_log": mean_absolute_error(d.log_outcome, d.log_prediction),
                        "rmse_log": mean_squared_error(d.log_outcome, d.log_prediction) ** 0.5,
                        "r2_log": r2_score(d.log_outcome, d.log_prediction),
                        "mae_monthly_currency": mean_absolute_error(np.exp(d.log_outcome),
                                                                      np.exp(d.log_prediction))})
    met = pd.DataFrame(metrics)
    oof.to_csv(OUT / "welfare_oof_predictions.csv", index=False)
    met.to_csv(OUT / "welfare_baseline_comparison.csv", index=False)

    by_district = (oof.assign(abs_error=lambda d: (d.log_outcome - d.log_prediction).abs())
                   .groupby(["district", "model"], as_index=False)
                   .agg(n=("row_index", "size"), mae_log=("abs_error", "mean")))
    wide = by_district.pivot(index="district", columns="model", values="mae_log")
    wide["lvi_increment_vs_demographic"] = wide.demographic_plus_lvi - wide.demographic_baseline
    wide["robust_increment_vs_demographic"] = (wide.demographic_plus_robust_high
                                                - wide.demographic_baseline)
    wide.reset_index().to_csv(OUT / "welfare_district_comparison.csv", index=False)
    rng = np.random.default_rng(42)
    summary = {}
    for column in ["lvi_increment_vs_demographic", "robust_increment_vs_demographic"]:
        values = wide[column].to_numpy()
        bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        summary[column] = {"mean_across_districts": float(values.mean()),
                           "district_bootstrap_95pct_ci": [float(x) for x in np.quantile(
                               bootstrap, [0.025, 0.975])],
                           "districts_improved": int((values < 0).sum())}
    result = {
        "outcome": "log of separate same-wave household monthly per-capita expenditure",
        "analyzed_households": len(data), "verified_districts": int(data.district.nunique()),
        "cv": "five-fold GroupKFold on official Module A district code",
        "models": list(SPECIFICATIONS), "ridge_alpha": 1.0,
        "paired_district_mae_increments": summary,
        "limits": "Same-wave criterion and held-out-district transfer only; archived full-sample LVI and robust labels are transductive, predictor timing and climate provenance are not verified, and no PSU/stratum design inference or later outcome is used.",
    }
    (OUT / "welfare_validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(met.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
