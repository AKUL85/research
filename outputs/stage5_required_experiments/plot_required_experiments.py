#!/usr/bin/env python3
"""Render figures from saved Stage 5 CSV artifacts without refitting models."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_e0() -> None:
    data = rows("e0_merge_integrity_summary.csv")
    labels = [f"{r['left']} → {r['right']}" for r in data]
    matched = [int(r["matched_keys"]) for r in data]
    unmatched = [int(r["left_only_keys"]) + int(r["right_only_keys"]) for r in data]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - .18, matched, .36, label="Matched keys", color="#2a9d8f")
    ax.bar(x + .18, unmatched, .36, label="Unmatched keys", color="#e76f51")
    for i, (m, u) in enumerate(zip(matched, unmatched)):
        ax.text(i - .18, m, f"{m:,}", ha="center", va="bottom")
        ax.text(i + .18, u, str(u), ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Household keys")
    ax.set_title("E0 one-to-one merge-integrity audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "e0_merge_integrity.png", dpi=300)
    plt.close(fig)


def empirical_roc(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    distinct = np.r_[True, p[order][1:] != p[order][:-1]]
    tp = np.cumsum(ys)[distinct]
    fp = np.cumsum(1 - ys)[distinct]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def plot_e1() -> None:
    oof = rows("e1_oof_predictions.csv")
    summary = {r["model"]: r for r in rows("e1_model_summary.csv")}
    calibration = rows("e1_calibration_curve.csv")
    models = list(dict.fromkeys(r["model"] for r in oof))
    colors = dict(zip(models, ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8a5cf5"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model in models:
        selected = [r for r in oof if r["model"] == model]
        y = np.array([int(r["target"]) for r in selected])
        p = np.array([float(r["probability"]) for r in selected])
        fpr, tpr = empirical_roc(y, p)
        axes[0].plot(fpr, tpr, color=colors[model],
                     label=f"{model} ({float(summary[model]['roc_auc']):.3f})")
        cal = [r for r in calibration if r["model"] == model]
        axes[1].plot([f(r, "mean_predicted") for r in cal],
                     [f(r, "observed_fraction") for r in cal],
                     marker="o", color=colors[model], label=model)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title="Outer-fold OOF ROC")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction",
                title="OOF calibration")
    fig.tight_layout()
    fig.savefig(OUT / "e1_discrimination_calibration.png", dpi=300)
    plt.close(fig)

    ci = rows("e1_group_bootstrap_ci.csv")
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "brier"]
    display_models = ["XGBoost", "RandomForest", "LogisticRegression",
                      "ContextOnlyLogistic", "PrevalenceBaseline"]
    display_labels = ["XGBoost", "Random forest", "Logistic regression",
                      "Context-only logistic", "Prevalence baseline"]
    metric_titles = ["AUROC", "AUPRC", "Balanced accuracy", "Brier score"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.65), sharey=True)
    for ax, metric, title in zip(axes, metrics, metric_titles):
        d = {r["model"]: r for r in ci if r["metric"] == metric}
        points = np.array([f(d[m], "point_estimate") for m in display_models])
        lower = np.array([f(d[m], "ci_lower_2_5") for m in display_models])
        upper = np.array([f(d[m], "ci_upper_97_5") for m in display_models])
        y = np.arange(len(display_models))
        ax.errorbar(points, y, xerr=[points - lower, upper - points],
                    fmt="o", markersize=3.5, linewidth=1,
                    color="#264653", capsize=2)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=.2, linewidth=.5)
    axes[0].set_yticks(y, display_labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Nested group-aware out-of-fold performance", fontsize=9)
    fig.tight_layout(pad=.6, rect=(0, 0, 1, .94))
    fig.savefig(OUT / "e1_metric_confidence_intervals.png", dpi=300)
    plt.close(fig)


def plot_e6() -> None:
    data = rows("e6_stability_summary.csv")
    schemes = list(dict.fromkeys(r["scheme"] for r in data))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for scheme in schemes:
        d = sorted((r for r in data if r["scheme"] == scheme), key=lambda r: int(r["k"]))
        k = [int(r["k"]) for r in d]
        axes[0].plot(k, [f(r, "ari_mean") for r in d], marker="o", label=scheme)
        axes[1].plot(k, [f(r, "jaccard_mean") for r in d], marker="o", label=scheme)
    axes[0].set(xlabel="k", ylabel="Mean ARI vs baseline", title="Assignment stability")
    axes[1].set(xlabel="k", ylabel="Mean matched Jaccard", title="Matched-cluster stability")
    axes[0].set_xticks(range(2, 9))
    axes[1].set_xticks(range(2, 9))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "e6_bootstrap_stability.png", dpi=300)
    plt.close(fig)

    data = rows("e6_method_sensitivity.csv")
    methods = list(dict.fromkeys(r["method"] for r in data))
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        d = sorted((r for r in data if r["method"] == method), key=lambda r: int(r["k"]))
        ax.plot([int(r["k"]) for r in d], [f(r, "ari_vs_baseline") for r in d],
                marker="o", label=method)
    ax.set(xlabel="k", ylabel="ARI vs baseline K-Means",
           title="E6 method/specification sensitivity")
    ax.set_xticks(range(2, 9))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "e6_method_sensitivity.png", dpi=300)
    plt.close(fig)


def plot_e7() -> None:
    result = rows("e7_lvi_sensitivity_summary.csv")
    plot = sorted((r for r in result if r["variant"] != "baseline_current"),
                  key=lambda r: f(r, "high_status_switch_pct"), reverse=True)
    labels = {
        "flood_binary_omitted": "Flood occurrence omitted",
        "loan_burden_direction": "Loan direction reversed",
        "literacy_direction_reversed": "Literacy direction reversed",
        "loan_omitted": "Loan omitted",
        "equal_indicator_weights": "Equal indicator weights",
        "robust_p01_p99_bounds": "1st--99th percentile bounds",
        "literacy_omitted": "Literacy omitted",
        "flood_depth_omitted": "Flood depth omitted",
        "zero_meals_as_missing": "Zero meals treated as missing",
        "zero_meal_households_excluded": "Zero-meal households excluded",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.35), sharey=True)
    y = np.arange(len(plot))
    for ax, col, title in [
        (axes[0], "spearman_vs_baseline", "Spearman rank correlation"),
        (axes[1], "tertile_kappa_vs_baseline", "Tertile Cohen's kappa"),
        (axes[2], "high_status_switch_pct", "High-class switches (%)"),
    ]:
        ax.barh(y, [f(r, col) for r in plot], color="#2a9d8f")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=.2, linewidth=.5)
    axes[0].set_xlim(0, 1.05)
    axes[1].set_xlim(0, 1.05)
    axes[2].set_xlim(0, 23)
    axes[0].set_yticks(y, [labels[r["variant"]] for r in plot], fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("LVI specification sensitivity relative to the baseline", fontsize=9)
    fig.tight_layout(pad=.6, rect=(0, 0, 1, .95))
    fig.savefig(OUT / "e7_classification_sensitivity.png", dpi=300)
    plt.close(fig)

    values: dict[str, list[float]] = defaultdict(list)
    for r in rows("e7_household_variant_scores.csv"):
        if r["lvi"]:
            values[r["variant"]].append(float(r["lvi"]))
    order = [r["variant"] for r in result]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([values[v] for v in order],
               labels=[v.replace("_", " ") for v in order], showfliers=False)
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.set(ylabel="LVI score", title="E7 score distributions by specification")
    fig.tight_layout()
    fig.savefig(OUT / "e7_score_distributions.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_e0()
    plot_e1()
    plot_e6()
    plot_e7()
    print("Rendered E0, E1, E6, and E7 figures.")


if __name__ == "__main__":
    main()
