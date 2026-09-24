#!/usr/bin/env python3
"""Summarize high-status agreement across the saved E7 index specifications.

This is descriptive agreement over a fixed, non-probabilistic set of coding
choices. It does not select a preferred index or validate vulnerability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "e7_household_variant_scores.csv"
OUT = HERE / "e7_consensus_summary.json"


def main() -> None:
    rows = pd.read_csv(SOURCE, usecols=["row_index", "variant", "tertile_class"])
    assert not rows.duplicated(["row_index", "variant"]).any()
    variants = sorted(rows["variant"].unique().tolist())
    assert len(variants) == 11 and "baseline_current" in variants
    wide = rows.pivot(index="row_index", columns="variant", values="tertile_class")
    assert len(wide) == 5605
    assert wide["baseline_current"].notna().all()
    high = wide.eq(2).where(wide.notna())
    available = high.notna().sum(axis=1)
    votes = high.fillna(False).sum(axis=1).astype(int)
    baseline = high["baseline_current"].astype(bool)
    always = votes.eq(available)
    never = votes.eq(0)
    ambiguous = ~(always | never)

    assert available.min() == 10 and available.max() == 11
    assert int(baseline.sum()) == 1868
    assert int(always.sum() + never.sum() + ambiguous.sum()) == len(wide)

    result = {
        "source": SOURCE.name,
        "interpretation": "Descriptive agreement among 11 chosen, correlated index specifications; not an uncertainty probability or a validated classification.",
        "variant_count": len(variants),
        "variants": variants,
        "households": len(wide),
        "available_specifications_per_household": {
            str(int(k)): int(v) for k, v in available.value_counts().sort_index().items()
        },
        "baseline_high_n": int(baseline.sum()),
        "always_high_n": int(always.sum()),
        "never_high_n": int(never.sum()),
        "ambiguous_high_n": int(ambiguous.sum()),
        "ambiguous_high_pct": round(float(100 * ambiguous.mean()), 2),
        "baseline_high_not_always_high_n": int((baseline & ~always).sum()),
        "baseline_not_high_ever_high_n": int((~baseline & ~never).sum()),
        "high_vote_histogram": {
            str(int(k)): int(v) for k, v in votes.value_counts().sort_index().items()
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
