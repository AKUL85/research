#!/usr/bin/env python3
"""Audit high-status agreement and one-at-a-time switch drivers in saved E7 variants.

The 11 variants are a chosen sensitivity set, not a probability distribution.
Rows omitted by a variant are not treated as non-High. Strict robust High
requires observed High status in all 11 variants.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/stage5_required_experiments/e7_household_variant_scores.csv"
OUT = Path(__file__).resolve().parent
BASE = "baseline_current"


def main() -> None:
    long = pd.read_csv(SOURCE)
    assert long.columns.tolist() == ["row_index", "hhid2", "variant", "lvi", "tertile_class"]
    assert not long.duplicated(["row_index", "variant"]).any()
    assert set(long.tertile_class.dropna().astype(int).unique()) == {0, 1, 2}
    variants = [BASE] + sorted(set(long.variant) - {BASE})
    assert len(variants) == 11 and long.row_index.nunique() == 5605
    labels = long.pivot(index="row_index", columns="variant", values="tertile_class")[variants]
    ids = long[long.variant.eq(BASE)].set_index("row_index").hhid2.reindex(labels.index)
    assert labels[BASE].notna().all()
    observed = labels.notna()
    high = labels.eq(2)
    n_observed = observed.sum(axis=1)
    n_high = high.sum(axis=1)
    baseline_high = high[BASE]
    all_available_high = n_high.eq(n_observed)
    strict_robust_high = n_observed.eq(11) & all_available_high
    never_high = n_high.eq(0)
    switcher = n_high.gt(0) & n_high.lt(n_observed)

    comparison = high.drop(columns=BASE).ne(baseline_high, axis=0).where(
        observed.drop(columns=BASE), False)
    by_variant = []
    for name in variants[1:]:
        eligible = observed[name]
        loss = baseline_high & ~high[name] & eligible
        gain = ~baseline_high & high[name] & eligible
        by_variant.append({"variant": name, "n_scored": int(eligible.sum()),
                           "n_high": int((high[name] & eligible).sum()),
                           "baseline_high_to_nonhigh": int(loss.sum()),
                           "baseline_nonhigh_to_high": int(gain.sum()),
                           "any_high_switch": int(comparison[name].sum()),
                           "switch_pct_of_scored": float(100 * comparison[name].sum() / eligible.sum())})

    driver_lists = comparison.apply(
        lambda row: [name for name in variants[1:] if bool(row[name])], axis=1)
    household = pd.DataFrame({"row_index": labels.index, "hhid2": ids.to_numpy(),
                              "baseline_high": baseline_high.to_numpy(),
                              "n_specifications": n_observed.to_numpy(),
                              "n_high": n_high.to_numpy(),
                              "strict_robust_high": strict_robust_high.to_numpy(),
                              "high_in_all_available": all_available_high.to_numpy(),
                              "never_high": never_high.to_numpy(),
                              "switcher": switcher.to_numpy(),
                              "n_baseline_switches": comparison.sum(axis=1).to_numpy(),
                              "switch_drivers": driver_lists.map("|".join).to_numpy()})
    for name in variants[1:]:
        household[f"switch__{name}"] = comparison[name].to_numpy()
    household.to_csv(OUT / "household_robustness.csv", index=False)

    patterns = (household.loc[household.n_baseline_switches.gt(0)]
                .groupby(["baseline_high", "switch_drivers"], dropna=False)
                .size().rename("n_records").reset_index()
                .sort_values("n_records", ascending=False))
    patterns.to_csv(OUT / "switch_driver_patterns.csv", index=False)
    pd.DataFrame(by_variant).to_csv(OUT / "variant_switch_drivers.csv", index=False)

    unique = household.loc[household.n_baseline_switches.eq(1)].switch_drivers.value_counts()
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "definition": "Strict robust High is High in all 11 observed saved specifications; incomplete records are unclassified for strict robustness.",
        "variant_count": len(variants), "n_records": len(household),
        "complete_cases": int(n_observed.eq(11).sum()),
        "incomplete_cases": int(n_observed.lt(11).sum()),
        "strict_robust_high": int(strict_robust_high.sum()),
        "high_in_all_available": int(all_available_high.sum()),
        "incomplete_high_in_all_available": int((all_available_high & n_observed.lt(11)).sum()),
        "never_high": int(never_high.sum()),
        "switcher": int(switcher.sum()),
        "baseline_high": int(baseline_high.sum()),
        "baseline_high_not_strict_robust": int((baseline_high & ~strict_robust_high).sum()),
        "any_baseline_switch": int(comparison.any(axis=1).sum()),
        "exactly_one_baseline_switch": int(comparison.sum(axis=1).eq(1).sum()),
        "unique_switch_driver_counts": {str(k): int(v) for k, v in unique.items()},
        "caution": "Drivers are one-at-a-time contrasts to the baseline; correlated coding choices are not independent causes, and the set is not exhaustive or externally validated."
    }
    assert result["strict_robust_high"] + result["never_high"] + result["switcher"] + result["incomplete_high_in_all_available"] == len(household)
    (OUT / "robust_high_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(by_variant).to_string(index=False))


if __name__ == "__main__":
    main()
