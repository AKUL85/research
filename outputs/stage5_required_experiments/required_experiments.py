#!/usr/bin/env python3
"""Execute only the E0/E1/E6/E7 experiments required by paper/05.

Existing research artifacts are read-only. All outputs are written beside this
script. E2, E3 and E5 are recorded as not executable because their required
external outcome, longitudinal outcome, and survey-design variables are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calinski_harabasz_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "bihs_r3_cleaned (1).csv"
LVI_PATH = ROOT / "outputs/stage1_lvi/bihs_r3_lvi_scored.csv"
CLUSTER_PATH = ROOT / "outputs/stage2_pca_clustering/bihs_r3_clustered.csv"
SUMMARY_PATH = ROOT / "outputs/stage1_lvi/lvi_indicator_summary.csv"

HHID = "hhid2"
CLIM_PROFILE = [
    "clim_aus_wind_avg", "clim_aman_wind_avg", "clim_boro_wind_avg",
    "clim_aus_evapotrans_avg", "clim_aman_evapotrans_avg",
    "clim_boro_evapotrans_avg", "clim_aus_rain_total_mm",
    "clim_aman_rain_total_mm", "clim_boro_rain_total_mm",
]
F_EXOG = [
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
F_CONTEXT = F_EXOG[:16]
F_PLOT = F_EXOG[16:]
SENS = [
    "agricultural_occupation_dependence", "dependency_ratio", "hh_size",
    "meals_per_person_week",
]
ADAPT = [
    "income_per_capita_monthly", "mean_edu_years_adults", "adult_illiteracy_rate",
    "livelihood_diversity", "any_nonfarm_agri_work", "land_cultivable_decimal",
    "has_current_loan_binary",
]
SOCIO_DIRECTIONS = {
    "agricultural_occupation_dependence": 1, "dependency_ratio": 1, "hh_size": 1,
    "meals_per_person_week": -1, "income_per_capita_monthly": -1,
    "mean_edu_years_adults": -1, "adult_illiteracy_rate": 1,
    "livelihood_diversity": -1, "any_nonfarm_agri_work": -1,
    "land_cultivable_decimal": -1, "has_current_loan_binary": -1,
}


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "random_seed": str(SEED),
    }


def stable_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating, int, np.integer)):
        return format(float(value), ".12g")
    return str(value)


def add_integrity_key(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    key = out[HHID].astype("string")
    records: list[dict] = []
    fp_cols = [c for c in ["a10", "a11", "a13", "a14", "a15", "a23",
                           "hh_size", "flood_affected", "flood_depth_ft_mean"]
               if c in out.columns]
    for idx in out.index[key.isna()]:
        payload = "|".join(f"{c}={stable_value(out.at[idx, c])}" for c in fp_cols)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        surrogate = f"SURROGATE_MISSING_HHID_{digest}"
        key.at[idx] = surrogate
        records.append({"row_index": int(idx), "surrogate_key": surrogate,
                        "fingerprint_columns": ";".join(fp_cols),
                        "fingerprint_payload": payload})
    out["_integrity_key"] = key
    return out, records


def compare_frames(left: pd.DataFrame, right: pd.DataFrame,
                   left_name: str, right_name: str) -> tuple[dict, list[dict]]:
    common = sorted((set(left.columns) & set(right.columns)) - {HHID, "_integrity_key"})
    merged = left.merge(right, on="_integrity_key", how="outer", validate="one_to_one",
                        suffixes=("__left", "__right"), indicator=True)
    rows = []
    for col in common:
        a = merged[f"{col}__left"]
        b = merged[f"{col}__right"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            ok = np.isclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            max_diff = float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0
        else:
            av = a.astype("string").fillna("<NA>")
            bv = b.astype("string").fillna("<NA>")
            ok = (av == bv).to_numpy()
            max_diff = np.nan
        rows.append({"left": left_name, "right": right_name, "column": col,
                     "n_compared": int(len(merged)), "n_mismatched": int((~ok).sum()),
                     "max_abs_difference": max_diff})
    summary = {
        "left": left_name, "right": right_name, "left_rows": len(left),
        "right_rows": len(right), "left_unique_keys": left["_integrity_key"].nunique(),
        "right_unique_keys": right["_integrity_key"].nunique(),
        "left_duplicate_keys": int(left["_integrity_key"].duplicated().sum()),
        "right_duplicate_keys": int(right["_integrity_key"].duplicated().sum()),
        "matched_keys": int((merged["_merge"] == "both").sum()),
        "left_only_keys": int((merged["_merge"] == "left_only").sum()),
        "right_only_keys": int((merged["_merge"] == "right_only").sum()),
        "shared_columns_checked": len(common),
        "columns_with_mismatch": int(sum(r["n_mismatched"] > 0 for r in rows)),
    }
    return summary, rows


def run_e0() -> None:
    print("E0: validating one-to-one household identity and field preservation", flush=True)
    frames = {
        "raw": pd.read_csv(RAW_PATH, dtype={HHID: "string"}),
        "stage1_lvi": pd.read_csv(LVI_PATH, dtype={HHID: "string"}),
        "stage2_cluster": pd.read_csv(CLUSTER_PATH, dtype={HHID: "string"}),
    }
    keyed, key_rows = {}, []
    for name, frame in frames.items():
        keyed[name], recs = add_integrity_key(frame)
        for rec in recs:
            rec["dataset"] = name
            key_rows.append(rec)
    summaries, columns = [], []
    for a, b in [("raw", "stage1_lvi"), ("stage1_lvi", "stage2_cluster")]:
        s, c = compare_frames(keyed[a], keyed[b], a, b)
        summaries.append(s)
        columns.extend(c)
    pd.DataFrame(summaries).to_csv(OUT / "e0_merge_integrity_summary.csv", index=False)
    pd.DataFrame(columns).to_csv(OUT / "e0_shared_column_checks.csv", index=False)
    pd.DataFrame(key_rows).to_csv(OUT / "e0_surrogate_key_audit.csv", index=False)
    assert all(s["left_duplicate_keys"] == 0 and s["right_duplicate_keys"] == 0
               and s["left_only_keys"] == 0 and s["right_only_keys"] == 0
               for s in summaries)
    metadata = {"experiment": "E0", "status": "EXECUTED", "versions": versions(),
                "surrogate_rule": "SHA-256 prefix over nine stable shared fields",
                "summary": summaries}
    (OUT / "e0_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def group_ids(raw: pd.DataFrame) -> np.ndarray:
    return pd.factorize(pd.MultiIndex.from_frame(raw[CLIM_PROFILE]), sort=True)[0]


def normalize_train_apply(train: pd.DataFrame, apply: pd.DataFrame,
                          columns: list[str], directions: dict[str, int]) -> pd.DataFrame:
    out = pd.DataFrame(index=apply.index)
    for col in columns:
        lo, hi = float(train[col].min()), float(train[col].max())
        if hi == lo:
            out[col] = 0.0
        elif directions[col] == 1:
            out[col] = (apply[col] - lo) / (hi - lo)
        else:
            out[col] = (hi - apply[col]) / (hi - lo)
    return out


def socioeconomic_score(train_ref: pd.DataFrame, apply: pd.DataFrame) -> pd.Series:
    norm = normalize_train_apply(train_ref, apply, SENS + ADAPT, SOCIO_DIRECTIONS)
    return (norm[SENS].mean(axis=1) + norm[ADAPT].mean(axis=1)) / 2


def model_specs() -> dict:
    return {
        "LogisticRegression": (
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
            {"model__n_estimators": [200], "model__max_depth": [4, 8, None],
             "model__min_samples_leaf": [1, 5]},
        ),
        "XGBoost": (
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", XGBClassifier(random_state=SEED, n_jobs=-1,
                                              objective="binary:logistic",
                                              eval_metric="logloss", verbosity=0))]),
            {"model__n_estimators": [150, 300], "model__max_depth": [2, 3],
             "model__learning_rate": [0.03, 0.08], "model__subsample": [0.8],
             "model__colsample_bytree": [0.8]},
        ),
        "ContextOnlyLogistic": (
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.array([balanced_accuracy_score(y, prob >= t) for t in candidates])
    best = np.flatnonzero(scores == scores.max())
    idx = int(best[np.argmin(np.abs(candidates[best] - 0.5))])
    return float(candidates[idx]), float(scores[idx])


def calibration_coefficients(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit[:, None], y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def binary_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ci, cs = calibration_coefficients(y, p)
    return {
        "n": len(y), "positive_n": int(y.sum()), "positive_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1]),
        "calibration_intercept": ci, "calibration_slope": cs,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_e1() -> None:
    print("E1: nested spatial non-circular validation", flush=True)
    raw, lvi = pd.read_csv(RAW_PATH), pd.read_csv(LVI_PATH)
    raw_k, _ = add_integrity_key(raw)
    lvi_k, _ = add_integrity_key(lvi)
    target_cols = SENS + ADAPT
    raw_feature_cols = list(dict.fromkeys(F_EXOG + CLIM_PROFILE))
    joined = raw_k[["_integrity_key"] + raw_feature_cols].merge(
        lvi_k[["_integrity_key"] + target_cols], on="_integrity_key", validate="one_to_one")
    X = joined[F_EXOG]
    target_raw = joined[target_cols]
    groups = group_ids(joined)
    global_score = socioeconomic_score(target_raw, target_raw)
    global_y = (global_score > global_score.quantile(2 / 3)).astype(int).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(X, global_y, groups))
    fold_id = np.full(len(joined), -1, dtype=int)
    oof_rows, fold_rows, param_rows = [], [], []
    specs = model_specs()

    for fold, (tr, te) in enumerate(outer_splits, 1):
        fold_id[te] = fold
        score_tr = socioeconomic_score(target_raw.iloc[tr], target_raw.iloc[tr])
        score_te = socioeconomic_score(target_raw.iloc[tr], target_raw.iloc[te])
        cut = float(score_tr.quantile(2 / 3))
        ytr = (score_tr > cut).astype(int).to_numpy()
        yte = (score_te > cut).astype(int).to_numpy()
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + fold)
        inner_splits = list(inner.split(X.iloc[tr], ytr, groups[tr]))
        print(f"  outer fold {fold}: train={len(tr)}, test={len(te)}, "
              f"train groups={np.unique(groups[tr]).size}, test groups={np.unique(groups[te]).size}",
              flush=True)
        for name, (estimator, grid) in specs.items():
            feature_cols = F_CONTEXT if name == "ContextOnlyLogistic" else F_EXOG
            search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner_splits,
                                  n_jobs=1, refit=True, return_train_score=False)
            search.fit(X.iloc[tr][feature_cols], ytr)
            inner_prob = cross_val_predict(clone(search.best_estimator_), X.iloc[tr][feature_cols], ytr,
                                           cv=inner_splits, method="predict_proba", n_jobs=1)[:, 1]
            threshold, inner_bal = choose_threshold(ytr, inner_prob)
            best = clone(search.best_estimator_).fit(X.iloc[tr][feature_cols], ytr)
            prob = best.predict_proba(X.iloc[te][feature_cols])[:, 1]
            pred = (prob >= threshold).astype(int)
            met = binary_metrics(yte, prob, pred)
            met.update({"fold": fold, "model": name, "threshold": threshold,
                        "target_train_cut": cut, "inner_best_auc": search.best_score_,
                        "inner_threshold_balanced_accuracy": inner_bal,
                        "train_n": len(tr), "test_n": len(te),
                        "train_groups": np.unique(groups[tr]).size,
                        "test_groups": np.unique(groups[te]).size})
            fold_rows.append(met)
            param_rows.append({"fold": fold, "model": name,
                               "best_params": json.dumps(search.best_params_, sort_keys=True),
                               "feature_set": "district_context_only" if name == "ContextOnlyLogistic" else "all_22_exogenous",
                               "inner_best_auc": search.best_score_, "threshold": threshold,
                               "target_train_cut": cut})
            for pos, idx in enumerate(te):
                oof_rows.append({"row_index": int(idx), "integrity_key": joined.iloc[idx]["_integrity_key"],
                                 "climate_profile_group": int(groups[idx]), "outer_fold": fold,
                                 "model": name, "target": int(yte[pos]),
                                 "target_score": float(score_te.iloc[pos]),
                                 "target_train_cut": cut, "probability": float(prob[pos]),
                                 "threshold": threshold, "prediction": int(pred[pos])})

        prevalence = float(ytr.mean())
        base_prob = np.full(len(te), prevalence)
        base_pred = np.zeros(len(te), dtype=int)
        met = binary_metrics(yte, base_prob, base_pred)
        met.update({"fold": fold, "model": "PrevalenceBaseline", "threshold": 0.5,
                    "target_train_cut": cut, "inner_best_auc": np.nan,
                    "inner_threshold_balanced_accuracy": np.nan,
                    "train_n": len(tr), "test_n": len(te),
                    "train_groups": np.unique(groups[tr]).size,
                    "test_groups": np.unique(groups[te]).size})
        fold_rows.append(met)
        for pos, idx in enumerate(te):
            oof_rows.append({"row_index": int(idx), "integrity_key": joined.iloc[idx]["_integrity_key"],
                             "climate_profile_group": int(groups[idx]), "outer_fold": fold,
                             "model": "PrevalenceBaseline", "target": int(yte[pos]),
                             "target_score": float(score_te.iloc[pos]),
                             "target_train_cut": cut, "probability": prevalence,
                             "threshold": 0.5, "prediction": 0})

    assert (fold_id >= 1).all()
    oof = pd.DataFrame(oof_rows).sort_values(["model", "row_index"])
    folds = pd.DataFrame(fold_rows)
    summaries = []
    for model, d in oof.groupby("model", sort=False):
        row = binary_metrics(d.target.to_numpy(), d.probability.to_numpy(), d.prediction.to_numpy())
        row.update({"model": model, "outer_folds": 5,
                    "group_count": d.climate_profile_group.nunique(),
                    "threshold_mean": d.groupby("outer_fold").threshold.first().mean(),
                    "threshold_sd": d.groupby("outer_fold").threshold.first().std(ddof=1),
                    "fold_auc_mean": folds.loc[folds.model == model, "roc_auc"].mean(),
                    "fold_auc_sd": folds.loc[folds.model == model, "roc_auc"].std(ddof=1)})
        summaries.append(row)
    summary = pd.DataFrame(summaries)

    rng = np.random.default_rng(SEED)
    ci_rows = []
    metric_names = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity",
                    "specificity", "precision", "f1", "brier", "log_loss"]
    for model, d in oof.groupby("model", sort=False):
        unique_groups = np.sort(d.climate_profile_group.unique())
        boots = {m: [] for m in metric_names}
        for _ in range(1000):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            bd = pd.concat([d[d.climate_profile_group == g] for g in sampled], ignore_index=True)
            if bd.target.nunique() < 2:
                continue
            bm = binary_metrics(bd.target.to_numpy(), bd.probability.to_numpy(),
                                bd.prediction.to_numpy())
            for m in metric_names:
                boots[m].append(bm[m])
        point = summary.set_index("model").loc[model]
        for m in metric_names:
            vals = np.asarray(boots[m], float)
            ci_rows.append({"model": model, "metric": m, "point_estimate": point[m],
                            "ci_lower_2_5": np.quantile(vals, 0.025),
                            "ci_upper_97_5": np.quantile(vals, 0.975),
                            "bootstrap_replicates": len(vals), "resampling_unit": "climate_profile"})

    cal_rows = []
    for model, d in oof.groupby("model", sort=False):
        obs, pred = calibration_curve(d.target, d.probability, n_bins=10, strategy="quantile")
        for i, (pp, oo) in enumerate(zip(pred, obs), 1):
            cal_rows.append({"model": model, "bin": i, "mean_predicted": pp,
                             "observed_fraction": oo})

    oof.to_csv(OUT / "e1_oof_predictions.csv", index=False)
    folds.to_csv(OUT / "e1_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "e1_model_summary.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(OUT / "e1_group_bootstrap_ci.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "e1_tuned_parameters.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "e1_calibration_curve.csv", index=False)
    pd.DataFrame([
        {"feature": col, "measurement_level": "district_context" if col in F_CONTEXT else "plot",
         "used_by_all_feature_models": True,
         "used_by_context_only_baseline": col in F_CONTEXT,
         "enters_target": False}
        for col in F_EXOG
    ]).to_csv(OUT / "e1_feature_dictionary.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(joined)),
                  "integrity_key": joined["_integrity_key"],
                  "climate_profile_group": groups, "outer_fold": fold_id}).to_csv(
                      OUT / "e1_fold_assignments.csv", index=False)
    meta = {"experiment": "E1", "status": "EXECUTED", "versions": versions(),
            "target": "outer-training-fold normalized socioeconomic-only high tertile",
            "features": F_EXOG, "outer_cv": "5-fold StratifiedGroupKFold",
            "inner_cv": "4-fold StratifiedGroupKFold", "groups": int(np.unique(groups).size),
            "bootstrap_replicates": 1000, "bootstrap_unit": "climate_profile"}
    (OUT / "e1_metadata.json").write_text(json.dumps(meta, indent=2))
    print(summary[["model", "roc_auc", "pr_auc", "balanced_accuracy", "brier"]].to_string(index=False),
          flush=True)


def pca_scores(X: np.ndarray, robust: bool = False) -> tuple[np.ndarray, int, object, PCA]:
    scaler = RobustScaler() if robust else StandardScaler()
    z = scaler.fit_transform(X)
    full = PCA(svd_solver="full", random_state=SEED).fit(z)
    n_keep = int(np.argmax(np.cumsum(full.explained_variance_ratio_) >= 0.80) + 1)
    pca = PCA(n_components=n_keep, svd_solver="full", random_state=SEED)
    return pca.fit_transform(z), n_keep, scaler, pca


def matched_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = np.unique(a), np.unique(b)
    scores = np.zeros((len(ua), len(ub)))
    for i, ca in enumerate(ua):
        A = a == ca
        for j, cb in enumerate(ub):
            B = b == cb
            scores[i, j] = (A & B).sum() / (A | B).sum()
    ri, ci = linear_sum_assignment(-scores)
    return float(scores[ri, ci].mean())


def run_e6(bootstrap_reps: int = 100) -> None:
    print(f"E6: cluster stability with {bootstrap_reps} household and group bootstraps", flush=True)
    df = pd.read_csv(LVI_PATH)
    raw = pd.read_csv(RAW_PATH)
    summ = pd.read_csv(SUMMARY_PATH)
    cols = summ.normalized_column.tolist()
    X = df[cols].to_numpy(float)
    groups = group_ids(raw)
    scores, n_keep, _, _ = pca_scores(X)
    baseline, base_rows = {}, []
    for k in range(2, 9):
        km = KMeans(k, n_init=20, max_iter=500, random_state=SEED).fit(scores)
        baseline[k] = km.labels_
        base_rows.append({"k": k, "n_components": n_keep,
                          "silhouette": silhouette_score(scores, km.labels_, sample_size=2000,
                                                         random_state=SEED),
                          "davies_bouldin": davies_bouldin_score(scores, km.labels_),
                          "calinski_harabasz": calinski_harabasz_score(scores, km.labels_),
                          "smallest_cluster": int(np.bincount(km.labels_).min()),
                          "largest_cluster": int(np.bincount(km.labels_).max())})
    pd.DataFrame(base_rows).to_csv(OUT / "e6_baseline_k_metrics.csv", index=False)

    rng = np.random.default_rng(SEED)
    audit_idx = np.sort(rng.choice(len(X), size=750, replace=False))
    consensus = {k: np.zeros((len(audit_idx), len(audit_idx)), dtype=np.uint16)
                 for k in range(2, 9)}
    stability = []
    unique_groups = np.unique(groups)
    for scheme in ["household_bootstrap", "climate_profile_bootstrap"]:
        for rep in range(bootstrap_reps):
            if scheme == "household_bootstrap":
                idx = rng.choice(len(X), size=len(X), replace=True)
            else:
                gs = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([np.flatnonzero(groups == g) for g in gs])
            scaler = StandardScaler().fit(X[idx])
            zb = scaler.transform(X[idx])
            za = scaler.transform(X)
            pf = PCA(svd_solver="full", random_state=SEED).fit(zb)
            nk = int(np.argmax(np.cumsum(pf.explained_variance_ratio_) >= 0.80) + 1)
            pca = PCA(n_components=nk, svd_solver="full", random_state=SEED).fit(zb)
            sb, sa = pca.transform(zb), pca.transform(za)
            for k in range(2, 9):
                km = KMeans(k, n_init=10, max_iter=500,
                            random_state=SEED + rep).fit(sb)
                lab = km.predict(sa)
                stability.append({"scheme": scheme, "replicate": rep + 1, "k": k,
                                  "bootstrap_n": len(idx), "n_components": nk,
                                  "ari_vs_baseline": adjusted_rand_score(baseline[k], lab),
                                  "matched_jaccard_vs_baseline": matched_jaccard(baseline[k], lab),
                                  "silhouette": silhouette_score(sa, lab, sample_size=1000,
                                                                 random_state=SEED + rep)})
                sub = lab[audit_idx]
                consensus[k] += (sub[:, None] == sub[None, :]).astype(np.uint16)
            if (rep + 1) % 10 == 0:
                print(f"  {scheme}: {rep + 1}/{bootstrap_reps}", flush=True)

    stab = pd.DataFrame(stability)
    stab.to_csv(OUT / "e6_bootstrap_stability.csv", index=False)
    agg = stab.groupby(["scheme", "k"]).agg(
        replicates=("replicate", "count"), ari_mean=("ari_vs_baseline", "mean"),
        ari_sd=("ari_vs_baseline", "std"), ari_p025=("ari_vs_baseline", lambda x: x.quantile(.025)),
        ari_p975=("ari_vs_baseline", lambda x: x.quantile(.975)),
        jaccard_mean=("matched_jaccard_vs_baseline", "mean"),
        jaccard_p025=("matched_jaccard_vs_baseline", lambda x: x.quantile(.025)),
        jaccard_p975=("matched_jaccard_vs_baseline", lambda x: x.quantile(.975)),
        silhouette_mean=("silhouette", "mean")).reset_index()
    agg.to_csv(OUT / "e6_stability_summary.csv", index=False)

    np.savez_compressed(OUT / "e6_consensus_matrices.npz", audit_indices=audit_idx,
                        **{f"k{k}": consensus[k] / (2 * bootstrap_reps) for k in range(2, 9)})
    con_rows = []
    for k in range(2, 9):
        con = consensus[k].astype(float) / (2 * bootstrap_reps)
        tri = np.triu_indices_from(con, 1)
        vals = con[tri]
        same = baseline[k][audit_idx][:, None] == baseline[k][audit_idx][None, :]
        con_rows.append({"k": k, "audit_subsample_n": len(audit_idx),
                         "mean_consensus_within_baseline": con[np.triu(same, 1)].mean(),
                         "mean_consensus_between_baseline": con[np.triu(~same, 1)].mean(),
                         "pac_0_1_to_0_9": float(((vals > .1) & (vals < .9)).mean())})
    pd.DataFrame(con_rows).to_csv(OUT / "e6_consensus_summary.csv", index=False)

    sens_rows = []
    for label, keep, robust in [
        ("remove_flood_affected", [c for c in cols if c != "flood_affected_norm"], False),
        ("remove_flood_depth", [c for c in cols if c != "flood_depth_ft_mean_norm"], False),
        ("robust_scaler", cols, True),
    ]:
        sx, nk, _, _ = pca_scores(df[keep].to_numpy(float), robust=robust)
        for k in range(2, 9):
            lab = KMeans(k, n_init=20, max_iter=500, random_state=SEED).fit_predict(sx)
            sens_rows.append({"method": label, "sample": "all", "n": len(X), "k": k,
                              "n_components": nk, "ari_vs_baseline": adjusted_rand_score(baseline[k], lab),
                              "matched_jaccard_vs_baseline": matched_jaccard(baseline[k], lab),
                              "silhouette": silhouette_score(sx, lab, sample_size=2000,
                                                             random_state=SEED)})

    gower_idx = np.sort(np.random.default_rng(SEED + 1).choice(len(X), size=1500, replace=False))
    gdist = pdist(X[gower_idx], metric="cityblock") / X.shape[1]
    Z = linkage(gdist, method="average")
    square = squareform(gdist)
    for k in range(2, 9):
        lab = fcluster(Z, t=k, criterion="maxclust")
        sens_rows.append({"method": "gower_average_hierarchical", "sample": "fixed_random_1500",
                          "n": len(gower_idx), "k": k, "n_components": np.nan,
                          "ari_vs_baseline": adjusted_rand_score(baseline[k][gower_idx], lab),
                          "matched_jaccard_vs_baseline": matched_jaccard(baseline[k][gower_idx], lab),
                          "silhouette": silhouette_score(square, lab, metric="precomputed")})
    pd.DataFrame(sens_rows).to_csv(OUT / "e6_method_sensitivity.csv", index=False)
    meta = {"experiment": "E6", "status": "EXECUTED", "versions": versions(),
            "bootstrap_replicates_per_scheme": bootstrap_reps,
            "consensus_subsample_n": 750, "gower_hierarchical_subsample_n": 1500,
            "note": "Gower average-linkage used; PAM unavailable in installed stack."}
    (OUT / "e6_metadata.json").write_text(json.dumps(meta, indent=2))
    print(agg.to_string(index=False), flush=True)


def dimension_lvi(norm: pd.DataFrame, equal_indicator: bool = False,
                  excluded_rows: pd.Series | None = None) -> pd.Series:
    exp = ["flood_affected_norm", "flood_depth_ft_mean_norm", "seasonal_wind_avg_norm",
           "seasonal_evapotranspiration_avg_norm"]
    sens = [f"{c}_norm" for c in SENS]
    adapt = [f"{c}_norm" for c in ADAPT]
    available = [c for c in exp + sens + adapt if c in norm.columns]
    if equal_indicator:
        score = norm[available].mean(axis=1)
    else:
        dims = [norm[[c for c in block if c in norm.columns]].mean(axis=1)
                for block in [exp, sens, adapt]]
        score = pd.concat(dims, axis=1).mean(axis=1)
    if excluded_rows is not None:
        score = score.mask(excluded_rows)
    return score


def robust_normalized(df: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for row in summ.itertuples():
        raw = df[row.indicator].astype(float)
        lo, hi = raw.quantile([.01, .99])
        clipped = raw.clip(lo, hi)
        forward = "(x - min)" in row.normalization_formula
        out[row.normalized_column] = ((clipped - lo) / (hi - lo) if forward
                                      else (hi - clipped) / (hi - lo))
    return out


def classes(score: pd.Series) -> tuple[pd.Series, float, float]:
    valid = score.dropna()
    q1, q2 = valid.quantile([1 / 3, 2 / 3])
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    out.loc[score.notna()] = np.where(score[score.notna()] <= q1, 0,
                                     np.where(score[score.notna()] <= q2, 1, 2))
    return out, float(q1), float(q2)


def run_e7() -> None:
    print("E7: LVI specification sensitivity", flush=True)
    df = pd.read_csv(LVI_PATH)
    raw = pd.read_csv(RAW_PATH)
    summ = pd.read_csv(SUMMARY_PATH)
    norm_cols = summ.normalized_column.tolist()
    base_norm = df[norm_cols].copy()
    variants: dict[str, tuple[pd.DataFrame, bool, pd.Series | None]] = {
        "baseline_current": (base_norm.copy(), False, None),
        "literacy_direction_reversed": (base_norm.assign(
            adult_illiteracy_rate_norm=1 - base_norm.adult_illiteracy_rate_norm), False, None),
        "literacy_omitted": (base_norm.drop(columns="adult_illiteracy_rate_norm"), False, None),
        "loan_burden_direction": (base_norm.assign(
            has_current_loan_binary_norm=1 - base_norm.has_current_loan_binary_norm), False, None),
        "loan_omitted": (base_norm.drop(columns="has_current_loan_binary_norm"), False, None),
        "zero_meals_as_missing": (base_norm.assign(
            meals_per_person_week_norm=base_norm.meals_per_person_week_norm.mask(
                df.meals_per_person_week == 0)), False, None),
        "zero_meal_households_excluded": (base_norm.copy(), False, df.meals_per_person_week == 0),
        "flood_binary_omitted": (base_norm.drop(columns="flood_affected_norm"), False, None),
        "flood_depth_omitted": (base_norm.drop(columns="flood_depth_ft_mean_norm"), False, None),
        "equal_indicator_weights": (base_norm.copy(), True, None),
        "robust_p01_p99_bounds": (robust_normalized(df, summ), False, None),
    }
    base_score = dimension_lvi(*variants["baseline_current"][:2],
                               excluded_rows=variants["baseline_current"][2])
    base_class, _, _ = classes(base_score)
    base_cluster_scores, _, _, _ = pca_scores(base_norm.to_numpy(float))
    base_cluster = KMeans(2, n_init=20, random_state=SEED).fit_predict(base_cluster_scores)
    rows, household_rows, downstream_rows = [], [], []

    fold_file = OUT / "e1_fold_assignments.csv"
    fold_assign = pd.read_csv(fold_file) if fold_file.exists() else None
    groups = group_ids(raw)
    Xpred = raw[F_EXOG]
    if fold_assign is None:
        gy = (base_score > base_score.quantile(2 / 3)).astype(int)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xpred, gy, groups))
    else:
        splits = [(np.flatnonzero(fold_assign.outer_fold.to_numpy() != f),
                   np.flatnonzero(fold_assign.outer_fold.to_numpy() == f)) for f in range(1, 6)]

    for name, (norm, equal_ind, excluded) in variants.items():
        score = dimension_lvi(norm, equal_ind, excluded)
        cls, q1, q2 = classes(score)
        common = score.notna() & base_score.notna()
        kappa = cohen_kappa_score(base_class[common].astype(int), cls[common].astype(int))
        high_switch = ((base_class[common] == 2) != (cls[common] == 2)).sum()
        any_switch = (base_class[common].astype(int) != cls[common].astype(int)).sum()

        complete = norm.notna().all(axis=1) & score.notna()
        vx, nk, _, _ = pca_scores(norm.loc[complete].to_numpy(float))
        vlab = KMeans(2, n_init=20, random_state=SEED).fit_predict(vx)
        ari = adjusted_rand_score(base_cluster[complete], vlab)
        rows.append({"variant": name, "n_scored": int(score.notna().sum()),
                     "n_excluded": int(score.isna().sum()), "mean": score.mean(),
                     "std": score.std(ddof=1), "min": score.min(), "max": score.max(),
                     "tertile_cut_low_medium": q1, "tertile_cut_medium_high": q2,
                     "spearman_vs_baseline": score[common].corr(base_score[common], method="spearman"),
                     "tertile_kappa_vs_baseline": kappa,
                     "any_class_switch_n": int(any_switch),
                     "any_class_switch_pct": 100 * any_switch / common.sum(),
                     "high_status_switch_n": int(high_switch),
                     "high_status_switch_pct": 100 * high_switch / common.sum(),
                     "cluster_complete_n": int(complete.sum()), "cluster_n_components": nk,
                     "k2_cluster_ari_vs_baseline": ari})
        for i in range(len(df)):
            household_rows.append({"row_index": i, "hhid2": df.at[i, HHID], "variant": name,
                                   "lvi": score.iat[i], "tertile_class": cls.iat[i]})

        oof_y, oof_p = [], []
        for tr, te in splits:
            train_scores = score.iloc[tr].dropna()
            cut = train_scores.quantile(2 / 3)
            keep_tr = score.iloc[tr].notna().to_numpy()
            keep_te = score.iloc[te].notna().to_numpy()
            tr2, te2 = tr[keep_tr], te[keep_te]
            ytr = (score.iloc[tr2] > cut).astype(int).to_numpy()
            yte = (score.iloc[te2] > cut).astype(int).to_numpy()
            model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler()),
                              ("model", LogisticRegression(C=.1, max_iter=3000,
                                                           random_state=SEED))])
            model.fit(Xpred.iloc[tr2], ytr)
            oof_y.extend(yte.tolist())
            oof_p.extend(model.predict_proba(Xpred.iloc[te2])[:, 1].tolist())
        yv, pv = np.asarray(oof_y), np.asarray(oof_p)
        downstream_rows.append({"variant": name, "n": len(yv),
                                "model": "fixed LogisticRegression C=0.1",
                                "cv": "E1 outer climate-profile folds",
                                "roc_auc": roc_auc_score(yv, pv),
                                "pr_auc": average_precision_score(yv, pv),
                                "brier": brier_score_loss(yv, pv)})

    pd.DataFrame(rows).to_csv(OUT / "e7_lvi_sensitivity_summary.csv", index=False)
    pd.DataFrame(household_rows).to_csv(OUT / "e7_household_variant_scores.csv", index=False)
    pd.DataFrame(downstream_rows).to_csv(OUT / "e7_downstream_model_sensitivity.csv", index=False)
    unresolved = pd.DataFrame([
        {"specification": "verified literacy recode", "status": "NOT_EXECUTABLE",
         "missing_input": "questionnaire/codebook or raw literacy response categories"},
        {"specification": "theoretical/fixed normalization bounds", "status": "NOT_EXECUTABLE",
         "missing_input": "prespecified defensible bounds"},
        {"specification": "theory-derived alternative dimension weights", "status": "NOT_EXECUTABLE",
         "missing_input": "prespecified weights and justification"},
    ])
    unresolved.to_csv(OUT / "e7_unresolved_specifications.csv", index=False)
    meta = {"experiment": "E7", "status": "PARTIALLY_EXECUTED",
            "versions": versions(), "executed_variants": list(variants),
            "unresolved_variants": unresolved.to_dict("records")}
    (OUT / "e7_metadata.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(rows)[["variant", "spearman_vs_baseline", "tertile_kappa_vs_baseline",
                              "high_status_switch_pct", "k2_cluster_ari_vs_baseline"]].to_string(index=False),
          flush=True)


def write_required_status() -> None:
    rows = [
        {"experiment": "E0", "requirement": "merge-integrity audit", "status": "EXECUTED",
         "reason": "current files contain required keys and shared fields"},
        {"experiment": "E1", "requirement": "nested spatial non-circular validation",
         "status": "EXECUTED", "reason": "current files contain target components, predictors, and 39 groups"},
        {"experiment": "E2", "requirement": "external-outcome validation",
         "status": "NOT_EXECUTABLE", "reason": "no independent observed vulnerability outcome is present"},
        {"experiment": "E3", "requirement": "temporal/longitudinal forecasting",
         "status": "NOT_EXECUTABLE", "reason": "no later outcome or documented predictor-outcome time ordering is present"},
        {"experiment": "E5", "requirement": "survey-design sensitivity",
         "status": "NOT_EXECUTABLE", "reason": "survey weights, strata, and PSU variables/documentation are absent"},
        {"experiment": "E6", "requirement": "cluster stability", "status": "EXECUTED",
         "reason": "current normalized indicators and reconstructed climate-profile groups are sufficient"},
        {"experiment": "E7", "requirement": "LVI specification sensitivity",
         "status": "PARTIALLY_EXECUTED", "reason": "data-supported variants run; verified literacy and theoretical bounds require missing documentation"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "required_experiment_status.csv", index=False)
    (OUT / "execution_environment.json").write_text(json.dumps(versions(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["e0", "e1", "e6", "e7", "status"])
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    args = parser.parse_args()
    np.random.seed(SEED)
    if args.experiment == "e0": run_e0()
    elif args.experiment == "e1": run_e1()
    elif args.experiment == "e6": run_e6(args.bootstrap_reps)
    elif args.experiment == "e7": run_e7()
    else: write_required_status()


if __name__ == "__main__":
    main()
