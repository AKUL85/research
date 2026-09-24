#!/usr/bin/env python3
"""Compare LVI and robust-High against held-out-PSU FIES response counts.

Input must already be keyed to local row_index using the *official* household
identifier. This script does not infer that mapping or a PSU from climate data.
No result is written unless all required fields and provenance checks pass.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "bihs_r3_cleaned (1).csv"
SCORED = ROOT / "outputs/stage1_lvi/bihs_r3_lvi_scored.csv"
ROBUST = ROOT / "outputs/extension2_robust_high/household_robustness.csv"
FIES = [f"x5_{n:02d}" for n in range(1, 9)]
REQUIRED = ["row_index", "official_household_id", "verified_psu", *FIES]


def model(columns: list[str]):
    numeric = [c for c in columns if c != "head_sex"]
    categorical = [c for c in columns if c == "head_sex"]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("sex", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    return make_pipeline(transform, Ridge(alpha=1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched_csv", type=Path,
                        help="Official-key-verified household extract; see paper/13_extension3_linking_framework.md")
    parser.add_argument("--linkage-audit", type=Path, required=True,
                        help="JSON documenting official key and verified PSU provenance")
    args = parser.parse_args()
    if not args.matched_csv.exists() or not args.linkage_audit.exists():
        parser.error("Matched official outcome extract and linkage audit are required; no validation result can be produced from the local cleaned file alone")

    audit = json.loads(args.linkage_audit.read_text())
    for field in ["source_dataset_doi", "source_version", "key_definition", "psu_definition",
                  "outcome_module", "join_method", "matched_households"]:
        if not audit.get(field):
            parser.error(f"Linkage audit missing required provenance field: {field}")
    if audit["source_dataset_doi"].upper() != "10.7910/DVN/NXKLZJ":
        parser.error("This script requires the official BIHS Round 3 outcome source")
    if audit["join_method"] != "documented_official_household_key":
        parser.error("Outcome link must use the documented official household key")

    source = pd.read_csv(args.matched_csv, dtype={"official_household_id": str,
                                                   "verified_psu": str})
    absent = sorted(set(REQUIRED) - set(source.columns))
    if absent:
        parser.error(f"Matched extract lacks required columns: {absent}")
    if source.row_index.isna().any() or source.row_index.duplicated().any():
        parser.error("row_index must be present and unique in the matched extract")
    if source.official_household_id.isna().any() or source.official_household_id.duplicated().any():
        parser.error("official_household_id must be present and unique")
    if source.verified_psu.isna().any() or source.verified_psu.nunique() < 5:
        parser.error("At least five nonmissing, official verified PSUs are required")
    if int(audit["matched_households"]) != len(source):
        parser.error("Matched row count differs from the linkage audit")
    if not source.row_index.between(0, 5604).all():
        parser.error("Local row_index outside the audited 0..5604 range")

    raw = pd.read_csv(RAW, usecols=["head_age", "head_sex", "hh_size"])
    scored = pd.read_csv(SCORED, usecols=["LVI"])
    robust = pd.read_csv(ROBUST, usecols=["row_index", "strict_robust_high"])
    local = pd.DataFrame({"row_index": np.arange(5605), "head_age": raw.head_age,
                          "head_sex": raw.head_sex, "hh_size": raw.hh_size,
                          "LVI": scored.LVI}).merge(robust, on="row_index", validate="one_to_one")
    joined = source.merge(local, on="row_index", validate="one_to_one", indicator=True)
    assert joined._merge.eq("both").all()
    codes = joined[FIES].apply(pd.to_numeric, errors="coerce")
    valid = codes.isin([1, 2]).all(axis=1)
    excluded = int((~valid).sum())
    data = joined.loc[valid].copy()
    if len(data) < 50 or data.verified_psu.nunique() < 5:
        parser.error("Too few complete-response households or verified PSUs after excluding refusals/missing items")
    data["fies_yes_count"] = codes.loc[valid].eq(1).sum(axis=1).astype(int)
    y = data.fies_yes_count.to_numpy(dtype=float)
    groups = data.verified_psu.to_numpy()
    specs = {
        "training_mean": [],
        "demographic_baseline": ["head_age", "head_sex", "hh_size"],
        "demographic_plus_lvi": ["head_age", "head_sex", "hh_size", "LVI"],
        "demographic_plus_robust_high": ["head_age", "head_sex", "hh_size", "strict_robust_high"],
    }
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(data, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])
        for name, columns in specs.items():
            if columns:
                fitted = model(columns).fit(data.iloc[tr][columns], y[tr])
                pred = np.clip(fitted.predict(data.iloc[te][columns]), 0, 8)
            else:
                pred = np.full(len(te), y[tr].mean())
            rows.extend({"row_index": int(data.iloc[i].row_index),
                         "verified_psu": str(groups[i]), "fold": fold,
                         "model": name, "outcome": int(y[i]),
                         "prediction": float(p)} for i, p in zip(te, pred))
    oof = pd.DataFrame(rows)
    metrics = []
    for name, frame in oof.groupby("model", sort=False):
        metrics.append({"model": name, "n": len(frame),
                        "n_psu": frame.verified_psu.nunique(),
                        "mae": mean_absolute_error(frame.outcome, frame.prediction),
                        "rmse": mean_squared_error(frame.outcome, frame.prediction) ** 0.5,
                        "r2": r2_score(frame.outcome, frame.prediction)})
    met = pd.DataFrame(metrics)
    assert met.n.eq(len(data)).all() and met.n_psu.eq(data.verified_psu.nunique()).all()
    by_psu = (oof.assign(abs_error=lambda d: (d.outcome - d.prediction).abs())
              .groupby(["verified_psu", "model"], as_index=False)
              .agg(n=("row_index", "size"), mae=("abs_error", "mean")))
    paired = by_psu.pivot(index="verified_psu", columns="model", values="mae")
    paired["lvi_mae_minus_demographic"] = (paired["demographic_plus_lvi"]
                                           - paired["demographic_baseline"])
    paired["robust_mae_minus_demographic"] = (paired["demographic_plus_robust_high"]
                                              - paired["demographic_baseline"])
    OUT.mkdir(exist_ok=True)
    oof.to_csv(OUT / "fies_oof_predictions.csv", index=False)
    met.to_csv(OUT / "fies_baseline_comparison.csv", index=False)
    paired.reset_index().to_csv(OUT / "fies_psu_comparison.csv", index=False)
    (OUT / "fies_validation_audit.json").write_text(json.dumps({
        "source_dataset_doi": audit["source_dataset_doi"],
        "source_version": audit["source_version"], "outcome_module": audit["outcome_module"],
        "matched_input_n": len(source), "excluded_incomplete_or_refused_n": excluded,
        "analyzed_n": len(data), "verified_psu_n": int(data.verified_psu.nunique()),
        "outcome": "eight-item X5 affirmative response count, complete cases only",
        "validation": "five-fold verified-PSU GroupKFold, same-wave criterion; no temporal forecast",
        "index_caveat": "archived LVI and robust-High status use full-sample index specifications; results are criterion associations and are transductive with respect to index scaling",
    }, indent=2) + "\n")
    print(met.to_string(index=False))


if __name__ == "__main__":
    main()
