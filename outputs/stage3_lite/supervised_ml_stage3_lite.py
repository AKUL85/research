#!/usr/bin/env python3
"""
Stage 3 Lite: Enhanced Supervised Machine Learning for BIHS Vulnerability Prediction.

OBJECTIVE:
    Evaluate the performance lift achieved by:
    1. Zero-leakage agro-climatic feature engineering:
       - Aridity / moisture deficit ratios per season.
       - Flood-monsoon compounding interaction terms.
       - Regional agricultural yield stress metrics.
       - Household plot-level environmental context (distance, plot temperature, humidity, rainfall).
    2. Threshold optimization for binary classification (tuning decision boundary for optimal F1 and balanced accuracy).
    3. Ordinal classification for the 3-class target (Low vs Medium vs High).
    4. Soft-voting ensembling (Logistic Regression + Random Forest + XGBoost).

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, roc_curve
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != "stage3_lite" else SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "stage3_lite"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CLUSTERED = PROJECT_ROOT / "outputs" / "stage2_pca_clustering" / "bihs_r3_clustered.csv"
INPUT_CLEANED_CANDIDATES = [
    PROJECT_ROOT / "bihs_r3_cleaned.csv",
    PROJECT_ROOT / "bihs_r3_cleaned (1).csv",
]

# -----------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# -----------------------------------------------------------------------------
def load_and_engineer_data() -> tuple[pd.DataFrame, list[str]]:
    if not INPUT_CLUSTERED.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CLUSTERED}")
    
    cleaned_path = None
    for cand in INPUT_CLEANED_CANDIDATES:
        if cand.exists():
            cleaned_path = cand
            break
    if cleaned_path is None:
        raise FileNotFoundError("Cleaned BIHS dataset not found.")
    
    df_clustered = pd.read_csv(INPUT_CLUSTERED)
    df_cleaned = pd.read_csv(cleaned_path)
    
    # 1. Target Definition (Tertiles)
    q = df_clustered["LVI"].quantile([1 / 3, 2 / 3]).values
    df_clustered["lvi_class"] = pd.cut(
        df_clustered["LVI"], bins=[-np.inf, q[0], q[1], np.inf], labels=["Low", "Medium", "High"]
    )
    df_clustered["y_binary"] = (df_clustered["lvi_class"] == "High").astype(int)
    df_clustered["y_3class"] = df_clustered["lvi_class"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
    
    # 2. Existing Climate & Flood Features
    clim_cols = [
        c for c in df_cleaned.columns
        if c.startswith("clim_") or c in ["mean_district_yield_ratio", "seasonal_wind_avg", "seasonal_evapotranspiration_avg"]
    ]
    for c in clim_cols:
        if c not in df_clustered.columns:
            df_clustered[c] = df_cleaned[c]
            
    flood_cols = ["flood_affected", "flood_depth_ft_mean"]
    
    clim_profile_cols = [
        "clim_aus_wind_avg", "clim_aman_wind_avg", "clim_boro_wind_avg",
        "clim_aus_evapotrans_avg", "clim_aman_evapotrans_avg", "clim_boro_evapotrans_avg",
        "clim_aus_rain_total_mm", "clim_aman_rain_total_mm", "clim_boro_rain_total_mm"
    ]
    df_clustered["district_id"] = df_clustered.groupby(clim_profile_cols).ngroup()
    
    # 3. New Engineered Features (Zero-Leakage Exposure)
    # A. Household plot-level environmental context
    plot_env_cols = [
        "plot_distance_m_mean", "total_plot_rainfall", "mean_plot_temperature",
        "max_plot_temperature", "mean_plot_humidity", "mean_plot_solar_rad"
    ]
    for c in plot_env_cols:
        df_clustered[c] = df_cleaned[c]
        
    # B. Seasonal Aridity / Moisture Deficit Ratios
    df_clustered["aridity_ratio_aus"] = df_clustered["clim_aus_evapotrans_avg"] / (df_clustered["clim_aus_rain_total_mm"] + 1.0)
    df_clustered["aridity_ratio_aman"] = df_clustered["clim_aman_evapotrans_avg"] / (df_clustered["clim_aman_rain_total_mm"] + 1.0)
    df_clustered["aridity_ratio_boro"] = df_clustered["clim_boro_evapotrans_avg"] / (df_clustered["clim_boro_rain_total_mm"] + 1.0)
    
    # C. Flood-Monsoon Interaction Terms
    df_clustered["flood_aman_rain_interaction"] = df_clustered["flood_depth_ft_mean"] * df_clustered["clim_aman_rain_total_mm"]
    df_clustered["flood_evapo_interaction"] = df_clustered["flood_depth_ft_mean"] * df_clustered["clim_aman_evapotrans_avg"]
    
    # D. Regional Crop Yield Stress
    df_clustered["yield_stress_aus"] = np.maximum(0, 1.0 - df_clustered["clim_aus_district_yield_ratio"])
    df_clustered["yield_stress_aman"] = np.maximum(0, 1.0 - df_clustered["clim_aman_district_yield_ratio"])
    df_clustered["yield_stress_boro"] = np.maximum(0, 1.0 - df_clustered["clim_boro_district_yield_ratio"])
    
    engineered_cols = (
        plot_env_cols +
        ["aridity_ratio_aus", "aridity_ratio_aman", "aridity_ratio_boro",
         "flood_aman_rain_interaction", "flood_evapo_interaction",
         "yield_stress_aus", "yield_stress_aman", "yield_stress_boro"]
    )
    
    all_features = flood_cols + clim_cols + ["district_id"] + engineered_cols
    all_features = list(dict.fromkeys(all_features))
    
    return df_clustered, all_features

# -----------------------------------------------------------------------------
# 2. Main Execution Routine
# -----------------------------------------------------------------------------
def run_stage3_lite():
    t_start = time.time()
    print("=" * 80)
    print("BIHS ROUND 3 -- STAGE 3 LITE: PERFORMANCE OPTIMIZATION EXPERIMENT")
    print("=" * 80)
    
    df, feature_cols = load_and_engineer_data()
    print(f"Sample size: n = {len(df)} households")
    print(f"Total features evaluated: {len(feature_cols)} features (all zero-leakage exposure)")
    
    X = df[feature_cols]
    y_bin = df["y_binary"]
    y_3c = df["y_3class"]
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.20, stratify=y_bin, random_state=RANDOM_STATE
    )
    X_train_3c, X_test_3c, y_train_3c, y_test_3c = train_test_split(
        X, y_3c, test_size=0.20, stratify=y_3c, random_state=RANDOM_STATE
    )
    
    results = []
    
    # -------------------------------------------------------------------------
    # PART 1: BINARY EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 1: BINARY CLASSIFICATION (High vs Not-High)")
    print("-" * 80)
    
    # Baseline Naive
    y_pred_naive_b = np.zeros_like(y_test_b)
    y_prob_naive_b = np.full_like(y_test_b, fill_value=y_train_b.mean(), dtype=float)
    results.append({
        "framework": "Stage 3 Baseline",
        "task": "Binary",
        "model": "Naive Baseline",
        "threshold": 0.50,
        "accuracy": accuracy_score(y_test_b, y_pred_naive_b),
        "precision": precision_score(y_test_b, y_pred_naive_b, zero_division=0),
        "recall": recall_score(y_test_b, y_pred_naive_b, zero_division=0),
        "f1_score": f1_score(y_test_b, y_pred_naive_b, zero_division=0),
        "roc_auc": 0.5000,
        "brier_score": brier_score_loss(y_test_b, y_prob_naive_b)
    })
    
    # Models
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.01, random_state=RANDOM_STATE, max_iter=1000))])
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5, random_state=RANDOM_STATE)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric='logloss')
    
    # 1. XGBoost with Engineered Features (Default Tau = 0.50)
    xgb.fit(X_train_b, y_train_b)
    prob_xgb_b = xgb.predict_proba(X_test_b)[:, 1]
    pred_xgb_b_50 = (prob_xgb_b >= 0.50).astype(int)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "Binary",
        "model": "XGBoost (Engineered)",
        "threshold": 0.50,
        "accuracy": accuracy_score(y_test_b, pred_xgb_b_50),
        "precision": precision_score(y_test_b, pred_xgb_b_50),
        "recall": recall_score(y_test_b, pred_xgb_b_50),
        "f1_score": f1_score(y_test_b, pred_xgb_b_50),
        "roc_auc": roc_auc_score(y_test_b, prob_xgb_b),
        "brier_score": brier_score_loss(y_test_b, prob_xgb_b)
    })
    
    # 2. Out-of-Fold Threshold Optimization for XGBoost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs_b = np.zeros(len(y_train_b))
    for train_idx, val_idx in cv.split(X_train_b, y_train_b):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric='logloss')
        m.fit(X_train_b.iloc[train_idx], y_train_b.iloc[train_idx])
        oof_probs_b[val_idx] = m.predict_proba(X_train_b.iloc[val_idx])[:, 1]
        
    thresholds = np.linspace(0.20, 0.70, 101)
    f1_curve = [f1_score(y_train_b, (oof_probs_b >= t).astype(int)) for t in thresholds]
    acc_curve = [accuracy_score(y_train_b, (oof_probs_b >= t).astype(int)) for t in thresholds]
    best_t_f1 = thresholds[np.argmax(f1_curve)]
    best_t_acc = thresholds[np.argmax(acc_curve)]
    
    # Evaluate at optimal F1 threshold
    pred_xgb_b_opt_f1 = (prob_xgb_b >= best_t_f1).astype(int)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "Binary",
        "model": "XGBoost (Threshold Tuned - Max F1)",
        "threshold": round(best_t_f1, 3),
        "accuracy": accuracy_score(y_test_b, pred_xgb_b_opt_f1),
        "precision": precision_score(y_test_b, pred_xgb_b_opt_f1),
        "recall": recall_score(y_test_b, pred_xgb_b_opt_f1),
        "f1_score": f1_score(y_test_b, pred_xgb_b_opt_f1),
        "roc_auc": roc_auc_score(y_test_b, prob_xgb_b),
        "brier_score": brier_score_loss(y_test_b, prob_xgb_b)
    })
    
    # 3. Soft-Voting Ensemble (LR + RF + XGB)
    ensemble_b = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft')
    ensemble_b.fit(X_train_b, y_train_b)
    prob_ens_b = ensemble_b.predict_proba(X_test_b)[:, 1]
    pred_ens_b_50 = (prob_ens_b >= 0.50).astype(int)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "Binary",
        "model": "Soft-Voting Ensemble (LR+RF+XGB)",
        "threshold": 0.50,
        "accuracy": accuracy_score(y_test_b, pred_ens_b_50),
        "precision": precision_score(y_test_b, pred_ens_b_50),
        "recall": recall_score(y_test_b, pred_ens_b_50),
        "f1_score": f1_score(y_test_b, pred_ens_b_50),
        "roc_auc": roc_auc_score(y_test_b, prob_ens_b),
        "brier_score": brier_score_loss(y_test_b, prob_ens_b)
    })
    
    # 4. Soft-Voting Ensemble with Threshold Tuning (Tau = 0.40)
    pred_ens_b_40 = (prob_ens_b >= 0.40).astype(int)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "Binary",
        "model": "Soft-Voting Ensemble (Tau=0.40)",
        "threshold": 0.40,
        "accuracy": accuracy_score(y_test_b, pred_ens_b_40),
        "precision": precision_score(y_test_b, pred_ens_b_40),
        "recall": recall_score(y_test_b, pred_ens_b_40),
        "f1_score": f1_score(y_test_b, pred_ens_b_40),
        "roc_auc": roc_auc_score(y_test_b, prob_ens_b),
        "brier_score": brier_score_loss(y_test_b, prob_ens_b)
    })
    
    # -------------------------------------------------------------------------
    # PART 2: 3-CLASS EXPERIMENTS (Low vs Medium vs High)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 2: 3-CLASS CLASSIFICATION (Low vs Medium vs High)")
    print("-" * 80)
    
    # 1. Nominal 3-Class XGBoost
    xgb_3c_nom = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=RANDOM_STATE, eval_metric='mlogloss', objective='multi:softprob', num_class=3)
    xgb_3c_nom.fit(X_train_3c, y_train_3c)
    pred_xgb_3c = xgb_3c_nom.predict(X_test_3c)
    prob_xgb_3c = xgb_3c_nom.predict_proba(X_test_3c)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "3-Class",
        "model": "XGBoost (Nominal Multi-class)",
        "threshold": "N/A",
        "accuracy": accuracy_score(y_test_3c, pred_xgb_3c),
        "precision": precision_score(y_test_3c, pred_xgb_3c, average='macro'),
        "recall": recall_score(y_test_3c, pred_xgb_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, pred_xgb_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, prob_xgb_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan
    })
    
    # 2. Ordinal 3-Class XGBoost (Frank & Hall decomposition)
    m_ord1 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=RANDOM_STATE, eval_metric='logloss')
    m_ord1.fit(X_train_3c, (y_train_3c >= 1).astype(int))
    p_ge_1 = m_ord1.predict_proba(X_test_3c)[:, 1]
    
    m_ord2 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=RANDOM_STATE, eval_metric='logloss')
    m_ord2.fit(X_train_3c, (y_train_3c >= 2).astype(int))
    p_ge_2 = m_ord2.predict_proba(X_test_3c)[:, 1]
    
    p_ge_2 = np.minimum(p_ge_2, p_ge_1)
    p_low = 1.0 - p_ge_1
    p_med = p_ge_1 - p_ge_2
    p_high = p_ge_2
    P_ord = np.column_stack([p_low, p_med, p_high])
    pred_ord_3c = np.argmax(P_ord, axis=1)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "3-Class",
        "model": "XGBoost (Ordinal Frank & Hall)",
        "threshold": "N/A",
        "accuracy": accuracy_score(y_test_3c, pred_ord_3c),
        "precision": precision_score(y_test_3c, pred_ord_3c, average='macro'),
        "recall": recall_score(y_test_3c, pred_ord_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, pred_ord_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, P_ord, multi_class='ovr', average='macro'),
        "brier_score": np.nan
    })
    
    # 3. Soft-Voting Ensemble 3-Class
    lr_3c = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.01, random_state=RANDOM_STATE, max_iter=1000))])
    rf_3c = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5, random_state=RANDOM_STATE)
    ensemble_3c = VotingClassifier(estimators=[('lr', lr_3c), ('rf', rf_3c), ('xgb', xgb_3c_nom)], voting='soft')
    ensemble_3c.fit(X_train_3c, y_train_3c)
    pred_ens_3c = ensemble_3c.predict(X_test_3c)
    prob_ens_3c = ensemble_3c.predict_proba(X_test_3c)
    results.append({
        "framework": "Stage 3 Lite",
        "task": "3-Class",
        "model": "Soft-Voting Ensemble (LR+RF+XGB)",
        "threshold": "N/A",
        "accuracy": accuracy_score(y_test_3c, pred_ens_3c),
        "precision": precision_score(y_test_3c, pred_ens_3c, average='macro'),
        "recall": recall_score(y_test_3c, pred_ens_3c, average='macro'),
        "f1_score": f1_score(y_test_3c, pred_ens_3c, average='macro'),
        "roc_auc": roc_auc_score(y_test_3c, prob_ens_3c, multi_class='ovr', average='macro'),
        "brier_score": np.nan
    })
    
    # Save results to CSV
    df_res = pd.DataFrame(results)
    csv_path = OUT_DIR / "stage3_lite_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    print(df_res[["framework", "task", "model", "threshold", "accuracy", "recall", "f1_score", "roc_auc"]])
    
    # -------------------------------------------------------------------------
    # PART 3: GENERATE COMPARISON FIGURES
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 3: GENERATING COMPARISON FIGURES (300 DPI)")
    print("-" * 80)
    
    # 1. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_b, prob_xgb_b)
    fpr_ens, tpr_ens, _ = roc_curve(y_test_b, prob_ens_b)
    
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost Engineered (AUC = {roc_auc_score(y_test_b, prob_xgb_b):.3f})", color="#2a78d6", lw=2)
    plt.plot(fpr_ens, tpr_ens, label=f"Soft-Voting Ensemble (AUC = {roc_auc_score(y_test_b, prob_ens_b):.3f})", color="#104281", lw=2.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Naive Baseline (AUC = 0.500)")
    
    plt.title("Figure 1: ROC Curve Comparison (Binary Vulnerability)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Recall)", fontsize=10)
    plt.legend(loc="lower right", frameon=False, fontsize=9.5)
    plt.tight_layout()
    fig1_path = FIG_DIR / "roc_curve_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # 2. Threshold Tuning Trade-Off Curve
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plt.plot(thresholds, f1_curve, label="Cross-Validation F1-Score", color="#eb6834", lw=2)
    plt.plot(thresholds, acc_curve, label="Cross-Validation Accuracy", color="#2a78d6", lw=2)
    plt.axvline(best_t_f1, color="#eb6834", linestyle=":", label=f"Optimal F1 Threshold ({best_t_f1:.3f})")
    plt.axvline(0.50, color="#555555", linestyle="--", label="Default Threshold (0.500)")
    
    plt.title("Figure 2: Classification Threshold Optimization (XGBoost)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Decision Threshold (Probability Cutoff for High Vulnerability)", fontsize=10)
    plt.ylabel("Validation Metric", fontsize=10)
    plt.legend(loc="lower left", frameon=False, fontsize=9.5)
    plt.tight_layout()
    fig2_path = FIG_DIR / "threshold_tuning_curve.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # 3. Confusion Matrix Comparison (Default vs Tuned Ensemble)
    cm_def = confusion_matrix(y_test_b, pred_xgb_b_50)
    cm_opt = confusion_matrix(y_test_b, pred_ens_b_40)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_def, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Not-High", "High"], yticklabels=["Not-High", "High"])
    axes[0].set_title(f"XGBoost Default (Tau=0.50)\nAccuracy: {accuracy_score(y_test_b, pred_xgb_b_50):.1%} | Recall: {recall_score(y_test_b, pred_xgb_b_50):.1%}", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Not-High", "High"], yticklabels=["Not-High", "High"])
    axes[1].set_title(f"Ensemble Tuned (Tau=0.40)\nAccuracy: {accuracy_score(y_test_b, pred_ens_b_40):.1%} | Recall: {recall_score(y_test_b, pred_ens_b_40):.1%}", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "confusion_matrices_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    print(f"\nStage 3 Lite completed successfully in {time.time() - t_start:.1f} seconds.")
    return df_res

if __name__ == "__main__":
    run_stage3_lite()
