#!/usr/bin/env python3
"""
Stage 3: Supervised Machine Learning for the BIHS Livelihood Vulnerability Paper.

RESEARCH QUESTION:
    Can climate and agro-environmental exposure variables alone predict household
    vulnerability class, without using the socioeconomic variables that were used
    to construct the vulnerability label?

PIPELINE:
    1. Target definition:
       - y_3class: LVI tertile class (Low, Medium, High).
       - y_binary: High vs Not-High (primary target).
    2. Feature space (X):
       - ONLY exposure and environmental context variables:
         * flood_affected, flood_depth_ft_mean
         * clim_* seasonal variables (rain, solar radiation, wind, evapotranspiration, district area/yield/production)
         * district/location identifiers (reconstructed 39 climate profile districts)
       - EXCLUSION OF TARGET LEAKAGE:
         Explicitly excludes all 11 sensitivity and adaptive capacity indicators (and their raw inputs)
         that fed into LVI construction (income, education, literacy, land, livelihood diversity,
         agri dependence, meals, dependency ratio, loan access), plus LVI itself, dimension scores,
         and Stage 2 clustering/PCA outputs.
    3. Train/test split:
       - 80/20 stratified split (random_state=42).
       - 5-fold cross-validation on the training set for hyperparameter tuning.
    4. Models:
       - Baseline: Naive majority class predictor.
       - Model 1: Logistic Regression (interpretable linear baseline with StandardScaler).
       - Model 2: Random Forest.
       - Model 3: XGBoost.
    5. Evaluation:
       - Accuracy, Precision, Recall, F1-score, ROC-AUC, Brier score, Confusion Matrix.
       - Both binary and 3-class evaluations.
    6. Interpretability (SHAP):
       - TreeExplainer on test set for best-performing model.
       - Global feature importance bar chart (mean |SHAP|).
       - SHAP summary/beeswarm plot.
       - 3 representative individual household waterfall plots (High, Low, Borderline).
    7. Outputs:
       - supervised_model_results.csv
       - shap_figures/ (300 DPI publication-ready PNGs)
       - stage3_report.md

Reproducibility: RANDOM_STATE = 42 throughout.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
import shap

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SCRIPT_DIR = Path(__file__).resolve().parent
# Allow running from either project root or outputs/stage3_supervised_ml/
if SCRIPT_DIR.name == "stage3_supervised_ml":
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    OUT_DIR = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR
    OUT_DIR = PROJECT_ROOT / "outputs" / "stage3_supervised_ml"

INPUT_CLUSTERED = PROJECT_ROOT / "outputs" / "stage2_pca_clustering" / "bihs_r3_clustered.csv"
INPUT_CLEANED_CANDIDATES = [
    PROJECT_ROOT / "bihs_r3_cleaned.csv",
    PROJECT_ROOT / "bihs_r3_cleaned (1).csv",
]

SHAP_DIR = OUT_DIR / "shap_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data and Construct Targets
# -----------------------------------------------------------------------------
def load_and_prepare_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    if not INPUT_CLUSTERED.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CLUSTERED}")
    
    cleaned_path = None
    for cand in INPUT_CLEANED_CANDIDATES:
        if cand.exists():
            cleaned_path = cand
            break
    if cleaned_path is None:
        raise FileNotFoundError("Cleaned BIHS dataset not found in candidates.")
    
    df_clustered = pd.read_csv(INPUT_CLUSTERED)
    df_cleaned = pd.read_csv(cleaned_path)
    
    # Define LVI tertiles (Low, Medium, High)
    q = df_clustered["LVI"].quantile([1 / 3, 2 / 3]).values
    df_clustered["lvi_class"] = pd.cut(
        df_clustered["LVI"],
        bins=[-np.inf, q[0], q[1], np.inf],
        labels=["Low", "Medium", "High"]
    )
    
    # Primary target: Binary (High vs Not-High)
    df_clustered["y_binary"] = (df_clustered["lvi_class"] == "High").astype(int)
    # Secondary target: 3-Class (Low=0, Medium=1, High=2)
    df_clustered["y_3class"] = df_clustered["lvi_class"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
    
    # Merge additional seasonal clim_* variables from cleaned data if absent in clustered
    clim_cols = [
        c for c in df_cleaned.columns
        if c.startswith("clim_") or c in ["mean_district_yield_ratio", "seasonal_wind_avg", "seasonal_evapotranspiration_avg"]
    ]
    for c in clim_cols:
        if c not in df_clustered.columns:
            df_clustered[c] = df_cleaned[c]
            
    # Flood exposure variables
    flood_cols = ["flood_affected", "flood_depth_ft_mean"]
    
    # Reconstructed District identifier (39 distinct climate profiles)
    clim_profile_cols = [
        "clim_aus_wind_avg", "clim_aman_wind_avg", "clim_boro_wind_avg",
        "clim_aus_evapotrans_avg", "clim_aman_evapotrans_avg", "clim_boro_evapotrans_avg",
        "clim_aus_rain_total_mm", "clim_aman_rain_total_mm", "clim_boro_rain_total_mm"
    ]
    df_clustered["district_id"] = df_clustered.groupby(clim_profile_cols).ngroup()
    
    # Define feature set X (EXPOSURE / CONTEXT ONLY)
    feature_cols = flood_cols + clim_cols + ["district_id"]
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Excluded variable list (for audit / report documentation)
    excluded_vars = [
        # Sensitivity indicators and raw components
        "agricultural_occupation_dependence", "agricultural_occupation_dependence_norm",
        "n_occ_agri_own", "n_occ_agri_labour", "dependency_ratio", "dependency_ratio_norm",
        "hh_size", "hh_size_norm", "meals_per_person_week", "meals_per_person_week_norm", "mealdays_total_7d",
        # Adaptive Capacity indicators and raw components
        "income_per_capita_monthly", "income_per_capita_monthly_norm",
        "mean_edu_years_adults", "mean_edu_years_adults_norm", "max_edu_years",
        "adult_illiteracy_rate", "adult_illiteracy_rate_norm", "adult_literacy_rate", "n_literate_15plus", "n_adults_15plus",
        "livelihood_diversity", "livelihood_diversity_norm", "crop_diversity", "n_activities_7d",
        "any_nonfarm_agri_work", "any_nonfarm_agri_work_norm", "n_nonfarm_agri_activities",
        "land_cultivable_decimal", "land_cultivable_decimal_norm", "land_homestead_decimal", "land_fallow_decimal", "n_plots",
        "has_current_loan_binary", "has_current_loan_binary_norm", "has_current_loan", "ever_had_loan", "n_loans",
        # Target and Composite Scores
        "LVI", "exposure_score", "sensitivity_score", "adaptive_capacity_vulnerability_score",
        "exposure_n_indicators", "sensitivity_n_indicators", "adaptive_capacity_vulnerability_n_indicators",
        # Stage 2 PCA & Clustering labels
        "cluster_kmeans", "cluster_ward", "cluster_name",
        "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"
    ]
    
    return df_clustered, feature_cols, excluded_vars

# -----------------------------------------------------------------------------
# 2. Main Training and Evaluation Routine
# -----------------------------------------------------------------------------
def run_stage3():
    t_start = time.time()
    print("=" * 80)
    print("BIHS ROUND 3 -- STAGE 3: SUPERVISED ML VULNERABILITY PREDICTION")
    print("=" * 80)
    
    df, feature_cols, excluded_vars = load_and_prepare_data()
    print(f"Sample size: n = {len(df)} households")
    print(f"Feature count (Exposure/Context only): {len(feature_cols)} features")
    print(f"Excluded variables (Target-Leakage prevention): {len(excluded_vars)} variables")
    
    X = df[feature_cols]
    y_bin = df["y_binary"]
    y_3c = df["y_3class"]
    
    # 80/20 Stratified Split
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.20, stratify=y_bin, random_state=RANDOM_STATE
    )
    X_train_3c, X_test_3c, y_train_3c, y_test_3c = train_test_split(
        X, y_3c, test_size=0.20, stratify=y_3c, random_state=RANDOM_STATE
    )
    
    cv_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_3c = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    # -------------------------------------------------------------------------
    # Task 1: Binary Classification (High vs Not-High)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TASK 1: BINARY CLASSIFICATION (High vs Not-High)")
    print("-" * 80)
    
    # 1. Naive Baseline (Predict Class 0)
    y_pred_naive_b = np.zeros_like(y_test_b)
    y_prob_naive_b = np.full(shape=(len(y_test_b), 2), fill_value=[1 - y_train_b.mean(), y_train_b.mean()])
    cm_naive_b = confusion_matrix(y_test_b, y_pred_naive_b)
    results.append({
        "task": "Binary (High vs Not-High)",
        "model": "Naive Baseline (Majority)",
        "best_hyperparameters": "None (Predict Class 0)",
        "accuracy": accuracy_score(y_test_b, y_pred_naive_b),
        "precision": precision_score(y_test_b, y_pred_naive_b, zero_division=0),
        "recall": recall_score(y_test_b, y_pred_naive_b, zero_division=0),
        "f1_score": f1_score(y_test_b, y_pred_naive_b, zero_division=0),
        "roc_auc": roc_auc_score(y_test_b, y_prob_naive_b[:, 1]),
        "brier_score": brier_score_loss(y_test_b, y_prob_naive_b[:, 1]),
        "confusion_matrix": str(cm_naive_b.tolist())
    })
    print(f"Naive Baseline   | Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 2. Logistic Regression
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    param_grid_lr = {'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    grid_lr_b = GridSearchCV(pipe_lr, param_grid_lr, cv=cv_b, scoring='roc_auc', n_jobs=-1)
    grid_lr_b.fit(X_train_b, y_train_b)
    best_lr_b = grid_lr_b.best_estimator_
    y_pred_lr_b = best_lr_b.predict(X_test_b)
    y_prob_lr_b = best_lr_b.predict_proba(X_test_b)[:, 1]
    cm_lr_b = confusion_matrix(y_test_b, y_pred_lr_b)
    results.append({
        "task": "Binary (High vs Not-High)",
        "model": "Logistic Regression",
        "best_hyperparameters": str(grid_lr_b.best_params_),
        "accuracy": accuracy_score(y_test_b, y_pred_lr_b),
        "precision": precision_score(y_test_b, y_pred_lr_b),
        "recall": recall_score(y_test_b, y_pred_lr_b),
        "f1_score": f1_score(y_test_b, y_pred_lr_b),
        "roc_auc": roc_auc_score(y_test_b, y_prob_lr_b),
        "brier_score": brier_score_loss(y_test_b, y_prob_lr_b),
        "confusion_matrix": str(cm_lr_b.tolist())
    })
    print(f"Logistic Reg     | Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 3. Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf_b = GridSearchCV(rf, param_grid_rf, cv=cv_b, scoring='roc_auc', n_jobs=-1)
    grid_rf_b.fit(X_train_b, y_train_b)
    best_rf_b = grid_rf_b.best_estimator_
    y_pred_rf_b = best_rf_b.predict(X_test_b)
    y_prob_rf_b = best_rf_b.predict_proba(X_test_b)[:, 1]
    cm_rf_b = confusion_matrix(y_test_b, y_pred_rf_b)
    results.append({
        "task": "Binary (High vs Not-High)",
        "model": "Random Forest",
        "best_hyperparameters": str(grid_rf_b.best_params_),
        "accuracy": accuracy_score(y_test_b, y_pred_rf_b),
        "precision": precision_score(y_test_b, y_pred_rf_b),
        "recall": recall_score(y_test_b, y_pred_rf_b),
        "f1_score": f1_score(y_test_b, y_pred_rf_b),
        "roc_auc": roc_auc_score(y_test_b, y_prob_rf_b),
        "brier_score": brier_score_loss(y_test_b, y_prob_rf_b),
        "confusion_matrix": str(cm_rf_b.tolist())
    })
    print(f"Random Forest    | Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 4. XGBoost
    xgb_b = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_xgb_b = GridSearchCV(xgb_b, param_grid_xgb, cv=cv_b, scoring='roc_auc', n_jobs=-1)
    grid_xgb_b.fit(X_train_b, y_train_b)
    best_xgb_b = grid_xgb_b.best_estimator_
    y_pred_xgb_b = best_xgb_b.predict(X_test_b)
    y_prob_xgb_b = best_xgb_b.predict_proba(X_test_b)[:, 1]
    cm_xgb_b = confusion_matrix(y_test_b, y_pred_xgb_b)
    results.append({
        "task": "Binary (High vs Not-High)",
        "model": "XGBoost",
        "best_hyperparameters": str(grid_xgb_b.best_params_),
        "accuracy": accuracy_score(y_test_b, y_pred_xgb_b),
        "precision": precision_score(y_test_b, y_pred_xgb_b),
        "recall": recall_score(y_test_b, y_pred_xgb_b),
        "f1_score": f1_score(y_test_b, y_pred_xgb_b),
        "roc_auc": roc_auc_score(y_test_b, y_prob_xgb_b),
        "brier_score": brier_score_loss(y_test_b, y_prob_xgb_b),
        "confusion_matrix": str(cm_xgb_b.tolist())
    })
    print(f"XGBoost          | Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # -------------------------------------------------------------------------
    # Task 2: 3-Class Classification (Low vs Medium vs High)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TASK 2: 3-CLASS CLASSIFICATION (Low vs Medium vs High)")
    print("-" * 80)
    
    # 1. Naive Baseline (Predict Majority Class in 3-class)
    majority_class_3c = y_train_3c.mode()[0]
    y_pred_naive_3c = np.full_like(y_test_3c, fill_value=majority_class_3c)
    class_probs_3c = [(y_train_3c == c).mean() for c in [0, 1, 2]]
    y_prob_naive_3c = np.tile(class_probs_3c, (len(y_test_3c), 1))
    cm_naive_3c = confusion_matrix(y_test_3c, y_pred_naive_3c)
    results.append({
        "task": "3-Class (Low/Med/High)",
        "model": "Naive Baseline (Majority)",
        "best_hyperparameters": f"None (Predict Class {majority_class_3c})",
        "accuracy": accuracy_score(y_test_3c, y_pred_naive_3c),
        "precision": precision_score(y_test_3c, y_pred_naive_3c, average='macro', zero_division=0),
        "recall": recall_score(y_test_3c, y_pred_naive_3c, average='macro', zero_division=0),
        "f1_score": f1_score(y_test_3c, y_pred_naive_3c, average='macro', zero_division=0),
        "roc_auc": roc_auc_score(y_test_3c, y_prob_naive_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan,
        "confusion_matrix": str(cm_naive_3c.tolist())
    })
    print(f"Naive Baseline   | Acc: {results[-1]['accuracy']:.4f} | Macro F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 2. Logistic Regression (3-class)
    pipe_lr_3c = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    grid_lr_3c = GridSearchCV(pipe_lr_3c, param_grid_lr, cv=cv_3c, scoring='accuracy', n_jobs=-1)
    grid_lr_3c.fit(X_train_3c, y_train_3c)
    best_lr_3c = grid_lr_3c.best_estimator_
    y_pred_lr_3c = best_lr_3c.predict(X_test_3c)
    y_prob_lr_3c = best_lr_3c.predict_proba(X_test_3c)
    cm_lr_3c = confusion_matrix(y_test_3c, y_pred_lr_3c)
    results.append({
        "task": "3-Class (Low/Med/High)",
        "model": "Logistic Regression",
        "best_hyperparameters": str(grid_lr_3c.best_params_),
        "accuracy": accuracy_score(y_test_3c, y_pred_lr_3c),
        "precision": precision_score(y_test_3c, y_pred_lr_3c, average='macro'),
        "recall": recall_score(y_test_3c, y_pred_lr_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, y_pred_lr_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, y_prob_lr_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan,
        "confusion_matrix": str(cm_lr_3c.tolist())
    })
    print(f"Logistic Reg     | Acc: {results[-1]['accuracy']:.4f} | Macro F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 3. Random Forest (3-class)
    rf_3c = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_rf_3c = GridSearchCV(rf_3c, param_grid_rf, cv=cv_3c, scoring='accuracy', n_jobs=-1)
    grid_rf_3c.fit(X_train_3c, y_train_3c)
    best_rf_3c = grid_rf_3c.best_estimator_
    y_pred_rf_3c = best_rf_3c.predict(X_test_3c)
    y_prob_rf_3c = best_rf_3c.predict_proba(X_test_3c)
    cm_rf_3c = confusion_matrix(y_test_3c, y_pred_rf_3c)
    results.append({
        "task": "3-Class (Low/Med/High)",
        "model": "Random Forest",
        "best_hyperparameters": str(grid_rf_3c.best_params_),
        "accuracy": accuracy_score(y_test_3c, y_pred_rf_3c),
        "precision": precision_score(y_test_3c, y_pred_rf_3c, average='macro'),
        "recall": recall_score(y_test_3c, y_pred_rf_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, y_pred_rf_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, y_prob_rf_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan,
        "confusion_matrix": str(cm_rf_3c.tolist())
    })
    print(f"Random Forest    | Acc: {results[-1]['accuracy']:.4f} | Macro F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # 4. XGBoost (3-class)
    xgb_3c = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', objective='multi:softprob', num_class=3)
    grid_xgb_3c = GridSearchCV(xgb_3c, param_grid_xgb, cv=cv_3c, scoring='accuracy', n_jobs=-1)
    grid_xgb_3c.fit(X_train_3c, y_train_3c)
    best_xgb_3c = grid_xgb_3c.best_estimator_
    y_pred_xgb_3c = best_xgb_3c.predict(X_test_3c)
    y_prob_xgb_3c = best_xgb_3c.predict_proba(X_test_3c)
    cm_xgb_3c = confusion_matrix(y_test_3c, y_pred_xgb_3c)
    results.append({
        "task": "3-Class (Low/Med/High)",
        "model": "XGBoost",
        "best_hyperparameters": str(grid_xgb_3c.best_params_),
        "accuracy": accuracy_score(y_test_3c, y_pred_xgb_3c),
        "precision": precision_score(y_test_3c, y_pred_xgb_3c, average='macro'),
        "recall": recall_score(y_test_3c, y_pred_xgb_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, y_pred_xgb_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, y_prob_xgb_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan,
        "confusion_matrix": str(cm_xgb_3c.tolist())
    })
    print(f"XGBoost          | Acc: {results[-1]['accuracy']:.4f} | Macro F1: {results[-1]['f1_score']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    csv_out = OUT_DIR / "supervised_model_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\nSaved model results to: {csv_out}")
    
    # -------------------------------------------------------------------------
    # 3. SHAP Interpretability Analysis (Best Binary Model: XGBoost)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("COMPUTING SHAP VALUES FOR BEST MODEL (XGBoost Binary)")
    print("-" * 80)
    
    explainer = shap.TreeExplainer(best_xgb_b)
    shap_values = explainer(X_test_b)
    
    # Pretty labels for publication
    clean_names = {
        'flood_affected': 'Flood affected (0/1)',
        'flood_depth_ft_mean': 'Flood depth (mean ft)',
        'clim_aus_rain_total_mm': 'Aus rainfall (mm)',
        'clim_aus_solar_rad_avg': 'Aus solar radiation',
        'clim_aus_wind_avg': 'Aus wind speed',
        'clim_aus_evapotrans_avg': 'Aus evapotranspiration',
        'clim_aus_district_area_ha': 'Aus district area (ha)',
        'clim_aus_district_production_mt': 'Aus district prod (MT)',
        'clim_aus_district_yield_ratio': 'Aus district yield ratio',
        'clim_aman_rain_total_mm': 'Aman rainfall (mm)',
        'clim_aman_solar_rad_avg': 'Aman solar radiation',
        'clim_aman_wind_avg': 'Aman wind speed',
        'clim_aman_evapotrans_avg': 'Aman evapotranspiration',
        'clim_aman_district_area_ha': 'Aman district area (ha)',
        'clim_aman_district_production_mt': 'Aman district prod (MT)',
        'clim_aman_district_yield_ratio': 'Aman district yield ratio',
        'clim_boro_rain_total_mm': 'Boro rainfall (mm)',
        'clim_boro_solar_rad_avg': 'Boro solar radiation',
        'clim_boro_wind_avg': 'Boro wind speed',
        'clim_boro_evapotrans_avg': 'Boro evapotranspiration',
        'clim_boro_district_area_ha': 'Boro district area (ha)',
        'clim_boro_district_production_mt': 'Boro district prod (MT)',
        'clim_boro_district_yield_ratio': 'Boro district yield ratio',
        'mean_district_yield_ratio': 'Mean district yield ratio',
        'district_id': 'District climate zone ID'
    }
    
    shap_values_clean = explainer(X_test_b)
    shap_values_clean.feature_names = [clean_names.get(c, c) for c in X_test_b.columns]
    
    # (a) Global feature importance bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.bar(shap_values_clean, max_display=15, show=False)
    plt.title("Figure 1: Global Feature Importance (Mean |SHAP| Value)\nXGBoost Binary Vulnerability Model", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = SHAP_DIR / "shap_bar_global_importance.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")
    
    # (b) Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_clean, max_display=15, show=False)
    plt.title("Figure 2: SHAP Beeswarm Summary Plot\nFeature Impact on Predicting High Livelihood Vulnerability", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    bee_path = SHAP_DIR / "shap_beeswarm_summary.png"
    plt.savefig(bee_path, dpi=300)
    plt.close()
    print(f"Saved: {bee_path}")
    
    # (c) Representative Waterfall plots
    probs = y_prob_xgb_b
    high_idx = int(np.where((y_test_b.values == 1) & (probs > 0.85))[0][0])
    low_idx = int(np.where((y_test_b.values == 0) & (probs < 0.10))[0][0])
    border_idx = int(np.where((probs >= 0.45) & (probs <= 0.55))[0][0])
    
    waterfalls = [
        (high_idx, f"High-Vulnerability Household (True High, P(High)={probs[high_idx]:.2f})", "shap_waterfall_high_vulnerability.png"),
        (low_idx, f"Low-Vulnerability Household (True Not-High, P(High)={probs[low_idx]:.2f})", "shap_waterfall_low_vulnerability.png"),
        (border_idx, f"Borderline Household (True {'High' if y_test_b.values[border_idx]==1 else 'Not-High'}, P(High)={probs[border_idx]:.2f})", "shap_waterfall_borderline_case.png")
    ]
    
    for idx, title, fname in waterfalls:
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(shap_values_clean[idx], max_display=12, show=False)
        plt.title(f"Figure: SHAP Waterfall Plot\n{title}", fontsize=11, fontweight='bold', pad=15)
        plt.tight_layout()
        wf_path = SHAP_DIR / fname
        plt.savefig(wf_path, dpi=300)
        plt.close()
        print(f"Saved waterfall plot: {wf_path}")
        
    print(f"\nStage 3 completed in {time.time() - t_start:.1f} seconds.")
    return df_results

if __name__ == "__main__":
    run_stage3()
