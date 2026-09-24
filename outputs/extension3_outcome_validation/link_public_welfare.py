#!/usr/bin/env python3
"""Audit and link the public BIHS Round 3 expenditure file to local LVI rows.

This leaves the original cleaned CSV untouched. Positional mapping is accepted
only after every row agrees on thirteen independent Module A fields and on
household size from the separately released expenditure file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/extension3_public_source"
OUT = Path(__file__).resolve().parent
FIELDS = ["hh_type", "a10", "a11", "a13", "a14", "a15", "a23", "a24",
          "a25", "a26", "a27", "a16_1_mm", "a16_1_yy"]
EXP = "pcm1_fx_nfx_hrentq_uval"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = {
        "identification": SRC / "009_bihs_r3_male_mod_a.dta",
        "weights": SRC / "158_BIHS sampling weights_r3.dta",
        "expenditure": SRC / "BIHS_FtF_GFSS_hhexpenditure_r3.dta",
    }
    assert all(p.exists() for p in files.values()), "Public source files are missing"
    raw = pd.read_csv(ROOT / "bihs_r3_cleaned (1).csv")
    scored = pd.read_csv(ROOT / "outputs/stage1_lvi/bihs_r3_lvi_scored.csv")
    robust = pd.read_csv(ROOT / "outputs/extension2_robust_high/household_robustness.csv")
    a = pd.read_stata(files["identification"], convert_categoricals=False)
    w = pd.read_stata(files["weights"], convert_categoricals=False)
    e = pd.read_stata(files["expenditure"], convert_categoricals=False)
    assert len(raw) == len(scored) == len(robust) == 5605
    assert w.a01.is_unique and e.a01.is_unique and a.a01.is_unique
    a = a[a.a01.isin(w.a01)].sort_values("a01").reset_index(drop=True)
    e = e[e.a01.isin(w.a01)].sort_values("a01").reset_index(drop=True)
    w = w.sort_values("a01").reset_index(drop=True)
    assert len(a) == len(e) == len(w) == len(raw) == 5605
    assert a.a01.equals(e.a01) and a.a01.equals(w.a01)
    assert a.hhid2.equals(e.hhid2)

    matches = {}
    for field in FIELDS:
        left = raw[field].fillna(-999999).to_numpy()
        right = a[field].fillna(-999999).to_numpy()
        matches[field] = int((left == right).sum())
    matches["hh_size_vs_expenditure_hhsize"] = int((raw.hh_size.to_numpy()
                                                     == e.hhsize.to_numpy()).sum())
    assert all(v == len(raw) for v in matches.values()), matches
    assert scored.hhid2.fillna("MISSING").equals(raw.hhid2.fillna("MISSING"))
    assert robust.row_index.equals(pd.Series(np.arange(len(raw)), name="row_index"))

    out = pd.DataFrame({
        "row_index": np.arange(len(raw)), "official_a01": a.a01,
        "corrected_hhid2": a.hhid2, "local_hhid2_as_received": raw.hhid2,
        "district": a.district, "community_id": a.community_id,
        "interview_month": a.a16_1_mm, "interview_year": a.a16_1_yy,
        "hhweight": w.hhweight, "head_age": raw.head_age,
        "head_sex": raw.head_sex, "hh_size": raw.hh_size,
        "LVI": scored.LVI,
        "strict_robust_high": robust.strict_robust_high,
        "never_high": robust.never_high,
        "switcher": robust.switcher,
        "monthly_per_capita_expenditure": e[EXP],
    })
    assert out.official_a01.is_unique and out.corrected_hhid2.is_unique
    assert out.monthly_per_capita_expenditure.notna().sum() == 5604
    out.to_csv(OUT / "public_welfare_link.csv", index=False)
    audit = {
        "public_archive": "https://reproducibility.worldbank.org/catalog/540",
        "archive_doi": "10.60572/1z28-h024",
        "underlying_dataset_doi": "10.7910/DVN/NXKLZJ",
        "source_sha256": {key: sha256(path) for key, path in files.items()},
        "local_rows": len(raw), "official_sample_rows": len(a),
        "direct_local_hhid2_equal_corrected": int(raw.hhid2.fillna("MISSING").eq(
            a.hhid2.fillna("MISSING")).sum()),
        "local_hhid2_missing": int(raw.hhid2.isna().sum()),
        "rowwise_checks": matches,
        "district_count": int(a.district.nunique()),
        "community_count": int(a.community_id.nunique()),
        "expenditure_variable": EXP,
        "expenditure_nonmissing": int(out.monthly_per_capita_expenditure.notna().sum()),
        "expenditure_missing": int(out.monthly_per_capita_expenditure.isna().sum()),
        "mapping_note": "All local rows align with the officially weighted identification sample sorted by numeric a01; received local hhid2 is independently misordered and must not be used for outcome joins. No source file was modified.",
        "location_note": "district and community_id are from official Module A; neither is asserted to be the survey PSU without a codebook crosswalk.",
        "outcome_note": "Archived separate household per-capita monthly expenditure aggregate, not a later outcome or FIES score.",
    }
    (OUT / "public_welfare_link_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
