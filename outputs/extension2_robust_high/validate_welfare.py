#!/usr/bin/env python3
"""Descriptive criterion association of robust High with separate expenditure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/extension3_outcome_validation/public_welfare_link.csv"
OUT = Path(__file__).resolve().parent


def main() -> None:
    data = pd.read_csv(SOURCE)
    assert len(data) == 5605 and data.official_a01.is_unique
    status = np.select([data.strict_robust_high, data.never_high, data.switcher],
                       ["strict_robust_high", "never_high", "switcher"],
                       default="incomplete_uncertified")
    data["status"] = status
    assert data.status.value_counts().to_dict() == {
        "strict_robust_high": 705, "never_high": 2543,
        "switcher": 2354, "incomplete_uncertified": 3,
    }
    data = data.loc[data.monthly_per_capita_expenditure.notna()].copy()
    assert data.monthly_per_capita_expenditure.gt(0).all()
    data["log_expenditure"] = np.log(data.monthly_per_capita_expenditure)
    group = (data.groupby("status", as_index=False)
             .agg(n=("row_index", "size"),
                  mean_monthly_expenditure=("monthly_per_capita_expenditure", "mean"),
                  median_monthly_expenditure=("monthly_per_capita_expenditure", "median"),
                  mean_log_expenditure=("log_expenditure", "mean")))
    group.to_csv(OUT / "welfare_by_robustness.csv", index=False)
    by = group.set_index("status")
    diff = float(by.loc["strict_robust_high", "mean_log_expenditure"]
                 - by.loc["never_high", "mean_log_expenditure"])

    # Cluster resample whole source districts; this is uncertainty for the
    # cleaned matched sample, not a design-based population interval.
    d = data[data.status.isin(["strict_robust_high", "never_high"])].groupby(
        ["district", "status"]).log_expenditure.agg(["size", "sum"]).unstack(fill_value=0)
    districts = d.index.to_numpy()
    nr = d[("size", "strict_robust_high")].to_numpy()
    nn = d[("size", "never_high")].to_numpy()
    sr = d[("sum", "strict_robust_high")].to_numpy()
    sn = d[("sum", "never_high")].to_numpy()
    rng = np.random.default_rng(42)
    picks = rng.integers(0, len(districts), size=(2000, len(districts)))
    boot = sr[picks].sum(axis=1) / nr[picks].sum(axis=1) - sn[picks].sum(axis=1) / nn[picks].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    rho = float(spearmanr(data.LVI, data.log_expenditure).statistic)
    result = {
        "outcome": "separate contemporaneous household monthly per-capita expenditure aggregate",
        "source_link": str(SOURCE.relative_to(ROOT)),
        "matched_outcome_n": len(data), "missing_outcome_n": 5605 - len(data),
        "district_count": int(data.district.nunique()),
        "strict_robust_high_vs_never_high_log_mean_difference": diff,
        "district_bootstrap_95pct_ci": [float(lo), float(hi)],
        "geometric_mean_ratio": float(np.exp(diff)),
        "lvi_vs_log_expenditure_spearman": rho,
        "bootstrap_replicates": len(boot),
        "interpretation": "Cross-sectional criterion association only; index ingredients include income and meals, and expenditure is not a later hardship outcome. This is not external prediction, causal validation, or population inference.",
    }
    (OUT / "welfare_criterion_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(group.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
