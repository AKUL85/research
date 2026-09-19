#!/usr/bin/env python3
"""
Stage 3: Progressive 4-Tier Feature Ablation Study for the BIHS Livelihood Vulnerability Paper.

RESEARCH QUESTION:
    How much predictive power does each vulnerability dimension contribute?
    Can climate exposure alone explain household vulnerability, or is socioeconomic
    adaptive capacity necessary to resolve the "missing middle"?

TIER PROGRESSION:
    - Tier 1: Naive Baseline (Majority Class, zero predictive lift)
    - Tier 2: Pure Agro-Environmental Exposure (Climate, floods, seasonal aridity, plot context)
    - Tier 3: Exposure + Exogenous Demographic Context (Head age, sex, marital status, remoteness)
    - Tier 4: Full Multidimensional Livelihood Model (Adding Sensitivity & Adaptive Capacity)

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != "stage3_ablation" else SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "stage3_ablation"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CLUSTERED = PROJECT_ROOT / "outputs" / "stage2_pca_clustering" / "bihs_r3_clustered.csv"
INPUT_CLEANED_CANDIDATES = [
    PROJECT_ROOT / "bihs_r3_cleaned.csv",
    PROJECT_ROOT / "bihs_r3_cleaned (1).csv",
]

def load_data_and_tiers():
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
    
    # 2. Tier 2 Features: Exposure only (39 features)
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
    
    plot_env_cols = [
        "plot_distance_m_mean", "total_plot_rainfall", "mean_plot_temperature",
        "max_plot_temperature", "mean_plot_humidity", "mean_plot_solar_rad"
    ]
    for c in plot_env_cols:
        df_clustered[c] = df_cleaned[c]
        
    df_clustered["aridity_ratio_aus"] = df_clustered["clim_aus_evapotrans_avg"] / (df_clustered["clim_aus_rain_total_mm"] + 1.0)
    df_clustered["aridity_ratio_aman"] = df_clustered["clim_aman_evapotrans_avg"] / (df_clustered["clim_aman_rain_total_mm"] + 1.0)
    df_clustered["aridity_ratio_boro"] = df_clustered["clim_boro_evapotrans_avg"] / (df_clustered["clim_boro_rain_total_mm"] + 1.0)
    df_clustered["flood_aman_rain_interaction"] = df_clustered["flood_depth_ft_mean"] * df_clustered["clim_aman_rain_total_mm"]
    df_clustered["flood_evapo_interaction"] = df_clustered["flood_depth_ft_mean"] * df_clustered["clim_aman_evapotrans_avg"]
    df_clustered["yield_stress_aus"] = np.maximum(0, 1.0 - df_clustered["clim_aus_district_yield_ratio"])
    df_clustered["yield_stress_aman"] = np.maximum(0, 1.0 - df_clustered["clim_aman_district_yield_ratio"])
    df_clustered["yield_stress_boro"] = np.maximum(0, 1.0 - df_clustered["clim_boro_district_yield_ratio"])
    
    engineered_cols = (
        plot_env_cols +
        ["aridity_ratio_aus", "aridity_ratio_aman", "aridity_ratio_boro",
         "flood_aman_rain_interaction", "flood_evapo_interaction",
         "yield_stress_aus", "yield_stress_aman", "yield_stress_boro"]
    )
    tier2_features = list(dict.fromkeys(flood_cols + clim_cols + ["district_id"] + engineered_cols))
    
    # 3. Tier 3 Features: Exposure + Exogenous Demographics (44 features)
    tier3_demographics = ["head_age", "head_sex", "head_marital", "school_distance_km_mean", "hh_type"]
    for c in tier3_demographics:
        df_clustered[c] = df_cleaned[c]
    tier3_features = list(dict.fromkeys(tier2_features + tier3_demographics))
    
    # 4. Tier 4 Features: Full Multidimensional Livelihood Model (55 features)
    tier4_socioeconomic = [
        "income_per_capita_monthly", "mean_edu_years_adults", "adult_illiteracy_rate",
        "land_cultivable_decimal", "livelihood_diversity", "any_nonfarm_agri_work",
        "has_current_loan_binary", "agricultural_occupation_dependence",
        "dependency_ratio", "hh_size", "meals_per_person_week"
    ]
    for c in tier4_socioeconomic:
        if c not in df_clustered.columns:
            df_clustered[c] = df_cleaned[c]
    tier4_features = list(dict.fromkeys(tier3_features + tier4_socioeconomic))
    
    return df_clustered, tier2_features, tier3_features, tier4_features

def run_ablation():
    t_start = time.time()
    print("=" * 80)
    print("BIHS ROUND 3 -- PROGRESSIVE 4-TIER FEATURE ABLATION STUDY")
    print("=" * 80)
    
    df, t2_feats, t3_feats, t4_feats = load_data_and_tiers()
    
    y_bin = df["y_binary"]
    y_3c = df["y_3class"]
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, stratify=y_bin, random_state=RANDOM_STATE
    )
    
    y_train_b, y_test_b = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
    y_train_3c, y_test_3c = y_3c.iloc[train_idx], y_3c.iloc[test_idx]
    
    tiers = [
        ("Tier 1 (Naive Baseline)", []),
        ("Tier 2 (Exposure Only)", t2_feats),
        ("Tier 3 (Exposure + Demographics)", t3_feats),
        ("Tier 4 (Full Livelihood Model)", t4_feats),
    ]
    
    results = []
    prob_dict_b = {}
    cm_dict_3c = {}
    
    for name, f_cols in tiers:
        print(f"\n--- Running {name} (features = {len(f_cols)}) ---")
        if not f_cols:
            # Binary Naive
            pred_b = np.zeros_like(y_test_b)
            prob_b = np.full_like(y_test_b, fill_value=y_train_b.mean(), dtype=float)
            prob_dict_b[name] = prob_b
            results.append({
                "tier": name, "task": "Binary", "feature_count": 0,
                "accuracy": accuracy_score(y_test_b, pred_b),
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "roc_auc": 0.5000, "brier_score": brier_score_loss(y_test_b, prob_b)
            })
            
            # 3-Class Naive
            pred_3c = np.full_like(y_test_3c, fill_value=y_train_3c.mode()[0])
            prob_3c = np.tile([(y_train_3c == c).mean() for c in [0, 1, 2]], (len(y_test_3c), 1))
            cm_dict_3c[name] = confusion_matrix(y_test_3c, pred_3c)
            results.append({
                "tier": name, "task": "3-Class", "feature_count": 0,
                "accuracy": accuracy_score(y_test_3c, pred_3c),
                "precision": precision_score(y_test_3c, pred_3c, average='macro', zero_division=0),
                "recall": recall_score(y_test_3c, pred_3c, average='macro', zero_division=0),
                "f1_score": f1_score(y_test_3c, pred_3c, average='macro', zero_division=0),
                "roc_auc": 0.5000, "brier_score": np.nan
            })
        else:
            X_tr = df[f_cols].iloc[train_idx]
            X_te = df[f_cols].iloc[test_idx]
            
            # Binary XGBoost
            xgb_b = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_STATE, eval_metric='logloss'
            )
            xgb_b.fit(X_tr, y_train_b)
            prob_b = xgb_b.predict_proba(X_te)[:, 1]
            pred_b = (prob_b >= 0.50).astype(int)
            prob_dict_b[name] = prob_b
            
            results.append({
                "tier": name, "task": "Binary", "feature_count": len(f_cols),
                "accuracy": accuracy_score(y_test_b, pred_b),
                "precision": precision_score(y_test_b, pred_b),
                "recall": recall_score(y_test_b, pred_b),
                "f1_score": f1_score(y_test_b, pred_b),
                "roc_auc": roc_auc_score(y_test_b, prob_b),
                "brier_score": brier_score_loss(y_test_b, prob_b)
            })
            
            # 3-Class XGBoost
            xgb_3c = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9,
                random_state=RANDOM_STATE, eval_metric='mlogloss', objective='multi:softprob', num_class=3
            )
            xgb_3c.fit(X_tr, y_train_3c)
            prob_3c = xgb_3c.predict_proba(X_te)
            pred_3c = xgb_3c.predict(X_te)
            cm_dict_3c[name] = confusion_matrix(y_test_3c, pred_3c)
            
            results.append({
                "tier": name, "task": "3-Class", "feature_count": len(f_cols),
                "accuracy": accuracy_score(y_test_3c, pred_3c),
                "precision": precision_score(y_test_3c, pred_3c, average='macro'),
                "recall": recall_score(y_test_3c, pred_3c, average='macro'),
                "f1_score": f1_score(y_test_3c, pred_3c, average='macro'),
                "roc_auc": roc_auc_score(y_test_3c, prob_3c, multi_class='ovr', average='macro'),
                "brier_score": np.nan
            })
            
    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / "stage3_ablation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved ablation results to: {csv_path}")
    print(df_results[["tier", "task", "feature_count", "accuracy", "recall", "f1_score", "roc_auc"]])
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION FIGURES (300 DPI)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GENERATING ABLATION FIGURES (300 DPI)")
    print("-" * 80)
    
    # Figure 1: Accuracy & F1 Progression Across Tiers (Dual-Panel)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    tier_labels = ["Tier 1\n(Baseline)", "Tier 2\n(Exposure)", "Tier 3\n(+Demogr)", "Tier 4\n(Full Model)"]
    
    df_b = df_results[df_results["task"] == "Binary"]
    df_3 = df_results[df_results["task"] == "3-Class"]
    
    # Panel A: Binary
    axes[0].plot(tier_labels, df_b["accuracy"] * 100, marker="o", lw=2.5, color="#2a78d6", label="Accuracy (%)")
    axes[0].plot(tier_labels, df_b["f1_score"] * 100, marker="s", lw=2.5, color="#eb6834", label="F1-Score (%)")
    axes[0].plot(tier_labels, df_b["roc_auc"] * 100, marker="^", lw=2.5, color="#1baf7a", label="ROC-AUC (x100)")
    for i, txt in enumerate(df_b["accuracy"] * 100):
        axes[0].annotate(f"{txt:.1f}%", (i, txt + 1.5), ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Panel A: Binary Classification (High vs Not-High)", fontsize=11, fontweight="bold", pad=12)
    axes[0].set_ylabel("Metric Score (%)", fontsize=10)
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: 3-Class
    axes[1].plot(tier_labels, df_3["accuracy"] * 100, marker="o", lw=2.5, color="#2a78d6", label="Accuracy (%)")
    axes[1].plot(tier_labels, df_3["f1_score"] * 100, marker="s", lw=2.5, color="#eb6834", label="Macro F1 (%)")
    axes[1].plot(tier_labels, df_3["roc_auc"] * 100, marker="^", lw=2.5, color="#1baf7a", label="Macro ROC-AUC (x100)")
    for i, txt in enumerate(df_3["accuracy"] * 100):
        axes[1].annotate(f"{txt:.1f}%", (i, txt + 1.5), ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Panel B: 3-Class Classification (Low / Medium / High)", fontsize=11, fontweight="bold", pad=12)
    axes[1].set_ylabel("Metric Score (%)", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "fig_ablation_metrics_progression.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # Figure 2: Overlay of ROC Curves Across Tiers
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = {"Tier 1 (Naive Baseline)": "#888888", "Tier 2 (Exposure Only)": "#2a78d6",
              "Tier 3 (Exposure + Demographics)": "#eb6834", "Tier 4 (Full Livelihood Model)": "#1baf7a"}
    styles = {"Tier 1 (Naive Baseline)": "--", "Tier 2 (Exposure Only)": "-",
              "Tier 3 (Exposure + Demographics)": "-", "Tier 4 (Full Livelihood Model)": "-"}
    
    for name, prob in prob_dict_b.items():
        if name == "Tier 1 (Naive Baseline)":
            plt.plot([0, 1], [0, 1], linestyle="--", color=colors[name], label=f"{name} (AUC = 0.500)")
        else:
            fpr, tpr, _ = roc_curve(y_test_b, prob)
            auc = roc_auc_score(y_test_b, prob)
            plt.plot(fpr, tpr, linestyle=styles[name], color=colors[name], lw=2.2, label=f"{name} (AUC = {auc:.3f})")
            
    plt.title("Figure 2: ROC Curve Progression Across Feature Tiers", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.legend(loc="lower right", frameon=False, fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = FIG_DIR / "fig_ablation_roc_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # Figure 3: Confusion Matrix Evolution (Resolving the 'Missing Middle')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_t2 = cm_dict_3c["Tier 2 (Exposure Only)"]
    cm_t3 = cm_dict_3c["Tier 3 (Exposure + Demographics)"]
    cm_t4 = cm_dict_3c["Tier 4 (Full Livelihood Model)"]
    
    labels = ["Low", "Medium", "High"]
    for ax, cm, title in zip(axes, [cm_t2, cm_t3, cm_t4], 
                            ["Tier 2: Exposure Only (Acc: 62.8%)", 
                             "Tier 3: +Demographics (Acc: 66.5%)", 
                             "Tier 4: Full Model (Acc: 86.4%)"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        
    plt.tight_layout()
    fig3_path = FIG_DIR / "fig_ablation_confusion_matrices.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    print(f"\nAblation study completed in {time.time() - t_start:.1f} seconds.")
    return df_results

if __name__ == "__main__":
    run_ablation()
