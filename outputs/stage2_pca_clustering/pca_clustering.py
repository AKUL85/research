#!/usr/bin/env python3
"""
Stage 2 of the BIHS Round 3 livelihood vulnerability analysis:
PCA + K-Means clustering (with Ward hierarchical robustness check).

Pipeline
--------
  15 normalized indicators (*_norm, Stage 1 v2 / post literacy-fix)
    -> z-score standardization (correlation-matrix PCA)
    -> PCA, retain components up to the first that reaches >= 80 % cumulative
       explained variance (target band 80-85 %)
    -> K-Means on retained PC scores for k = 2..8, scored with silhouette,
       Davies-Bouldin and Calinski-Harabasz; k chosen by majority vote
    -> final K-Means at chosen k; Ward hierarchical clustering at the same k
       on the same PC scores; adjusted Rand index between the two labelings
    -> cluster profiles (n, %, mean LVI, mean dimension scores, mean raw
       indicators) with one-way ANOVA AND Kruskal-Wallis per raw indicator
    -> three outputs: bihs_r3_clustered.csv, cluster_profile_summary.csv,
       pca_clustering_report.txt

Inputs that are deliberately NOT used as clustering features: LVI and the three
dimension scores. They are equal-weight averages of the same 15 indicators and
would make the clustering circular.

Reproducibility: every stochastic step uses RANDOM_STATE = 42.

Run:  python pca_clustering.py
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STAGE1_DIR = PROJECT_ROOT / "outputs" / "stage1_lvi"
OUTPUT_DIR = SCRIPT_DIR

INPUT_SCORED = STAGE1_DIR / "bihs_r3_lvi_scored.csv"
INPUT_INDICATOR_SUMMARY = STAGE1_DIR / "lvi_indicator_summary.csv"

OUT_CLUSTERED = OUTPUT_DIR / "bihs_r3_clustered.csv"
OUT_PROFILE = OUTPUT_DIR / "cluster_profile_summary.csv"
OUT_REPORT = OUTPUT_DIR / "pca_clustering_report.txt"

EXPECTED_ROWS = 5605
EXPECTED_N_INDICATORS = 15

HH_ID = "hhid2"
LVI_COL = "LVI"
DIM_SCORE_COLS = [
    "exposure_score",
    "sensitivity_score",
    "adaptive_capacity_vulnerability_score",
]
# Anything that is a composite of the indicators is excluded from features.
COMPOSITE_COLS = [LVI_COL] + DIM_SCORE_COLS + [
    "exposure_n_indicators",
    "sensitivity_n_indicators",
    "adaptive_capacity_vulnerability_n_indicators",
]

# PCA retention: keep the smallest number of components whose cumulative
# explained variance reaches this threshold (target band 0.80-0.85).
PCA_CUMVAR_TARGET = 0.80

K_RANGE = list(range(2, 9))          # k = 2 .. 8
KMEANS_N_INIT = 20                   # restarts per k; best inertia kept
KMEANS_MAX_ITER = 500

# Loadings with |value| >= this are called "heavy" when naming components.
HEAVY_LOADING = 0.40

ALPHA = 0.05

# Short, paper-friendly labels for the 15 indicators.
SHORT_LABEL = {
    "flood_affected": "flood affected (0/1)",
    "flood_depth_ft_mean": "flood depth (ft)",
    "seasonal_wind_avg": "seasonal wind",
    "seasonal_evapotranspiration_avg": "seasonal evapotranspiration",
    "agricultural_occupation_dependence": "agri-occupation dependence",
    "dependency_ratio": "dependency ratio",
    "hh_size": "household size",
    "meals_per_person_week": "meals per person/week",
    "income_per_capita_monthly": "income per capita",
    "mean_edu_years_adults": "adult education (yrs)",
    "adult_illiteracy_rate": "adult illiteracy rate",
    "livelihood_diversity": "livelihood diversity",
    "any_nonfarm_agri_work": "non-farm work (0/1)",
    "land_cultivable_decimal": "cultivable land (dec)",
    "has_current_loan_binary": "current loan (0/1)",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_p(p: float) -> str:
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def df_to_text(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Fixed-width table for the plain-text report."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    return out.to_string(index=False)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h1(self, title: str) -> None:
        self.lines += ["", "=" * 96, title, "=" * 96]

    def h2(self, title: str) -> None:
        self.lines += ["", "-" * 96, title, "-" * 96]

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, df: pd.DataFrame, float_fmt: str = "{:.4f}") -> None:
        self.lines.append(df_to_text(df, float_fmt))

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_SCORED.exists():
        sys.exit(f"ERROR: input not found: {INPUT_SCORED}")
    if not INPUT_INDICATOR_SUMMARY.exists():
        sys.exit(f"ERROR: indicator summary not found: {INPUT_INDICATOR_SUMMARY}")
    # hhid2 kept as string; row 0 has a blank id (documented in Stage 1
    # validation check) and must survive the round trip unchanged.
    df = pd.read_csv(INPUT_SCORED, dtype={HH_ID: str}, keep_default_na=True)
    summ = pd.read_csv(INPUT_INDICATOR_SUMMARY)
    return df, summ


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    np.random.seed(RANDOM_STATE)
    rep = Report()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rep.h1("BIHS ROUND 3 -- STAGE 2: PCA + CLUSTERING REPORT")
    rep.p(f"Generated            : {run_ts}")
    rep.p(f"Script               : {Path(__file__).name}")
    rep.p(f"RANDOM SEED          : random_state = {RANDOM_STATE} "
          "(K-Means initialisation, PCA solver; Ward linkage is deterministic)")
    rep.p(f"Python               : {platform.python_version()}")
    rep.p(f"numpy / pandas       : {np.__version__} / {pd.__version__}")
    rep.p(f"scikit-learn / scipy : {sklearn.__version__} / {scipy.__version__}")

    # ------------------------------------------------------------------
    # 0. Load and validate input
    # ------------------------------------------------------------------
    df, summ = load_data()
    rep.h2("0. INPUT")
    rep.p(f"File     : {INPUT_SCORED}")
    rep.p(f"SHA-256  : {sha256_of(INPUT_SCORED)}")
    rep.p(f"Shape    : {df.shape[0]} rows x {df.shape[1]} columns")
    if df.shape[0] != EXPECTED_ROWS:
        sys.exit(f"ERROR: expected {EXPECTED_ROWS} rows, got {df.shape[0]}")
    n_blank_id = int(df[HH_ID].isna().sum())
    rep.p(f"Households with blank {HH_ID}: {n_blank_id} (retained; row order is the key)")

    # ------------------------------------------------------------------
    # 1. Feature selection
    # ------------------------------------------------------------------
    rep.h2("1. CLUSTERING FEATURES (15 normalized indicators)")
    # Use the Stage 1 indicator summary as the authoritative list so the raw
    # <-> normalized mapping and dimension membership are not re-derived here.
    summ = summ.set_index("normalized_column")
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    norm_cols = [c for c in norm_cols if c in summ.index]
    if len(norm_cols) != EXPECTED_N_INDICATORS:
        sys.exit(f"ERROR: expected {EXPECTED_N_INDICATORS} *_norm columns, found "
                 f"{len(norm_cols)}: {norm_cols}")
    raw_cols = [summ.loc[c, "indicator"] for c in norm_cols]
    dim_of = {c: summ.loc[c, "dimension"] for c in norm_cols}
    raw_of = dict(zip(norm_cols, raw_cols))
    missing_raw = [r for r in raw_cols if r not in df.columns]
    if missing_raw:
        sys.exit(f"ERROR: raw indicator columns missing from input: {missing_raw}")
    leaked = [c for c in norm_cols if c in COMPOSITE_COLS]
    assert not leaked, f"composite columns leaked into features: {leaked}"
    if "adult_illiteracy_rate_norm" not in norm_cols:
        sys.exit("ERROR: expected post literacy-fix column adult_illiteracy_rate_norm")

    X_raw01 = df[norm_cols].to_numpy(dtype=float)
    n_missing = int(np.isnan(X_raw01).sum())
    rep.p(f"Features used ({len(norm_cols)}), all on [0, 1] and oriented so that "
          "higher = more vulnerable (Stage 1 v2, post literacy-fix):")
    feat_tbl = pd.DataFrame({
        "normalized_feature": norm_cols,
        "dimension": [dim_of[c] for c in norm_cols],
        "raw_indicator": raw_cols,
        "mean": X_raw01.mean(axis=0),
        "std": X_raw01.std(axis=0, ddof=1),
        "variance": X_raw01.var(axis=0, ddof=1),
    })
    rep.table(feat_tbl)
    rep.p(f"Missing values in feature matrix: {n_missing}")
    if n_missing:
        sys.exit("ERROR: feature matrix contains missing values; Stage 1 should have none")
    rep.p()
    rep.p("EXCLUDED from features (composites of the same indicators -> circular): "
          + ", ".join(COMPOSITE_COLS))

    # ------------------------------------------------------------------
    # 2. Standardization + PCA
    # ------------------------------------------------------------------
    rep.h2("2. PRINCIPAL COMPONENT ANALYSIS")
    var = feat_tbl.set_index("normalized_feature")["variance"]
    ratio = var.max() / var.min()
    rep.p("Standardization: although every feature is min-max scaled to [0, 1], their")
    rep.p("variances differ by a factor of "
          f"{ratio:.1f} (largest: {var.idxmax()} = {var.max():.4f}; smallest: "
          f"{var.idxmin()} = {var.min():.4f}).")
    rep.p("Covariance-matrix PCA would therefore be dominated by the 0/1 indicators.")
    rep.p("Each feature is z-scored (mean 0, SD 1) before PCA, i.e. PCA is run on the")
    rep.p("correlation matrix, giving every indicator equal a-priori weight. The same")
    rep.p("standardized matrix feeds the clustering.")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw01)

    pca_full = PCA(n_components=len(norm_cols), svd_solver="full",
                   random_state=RANDOM_STATE)
    pca_full.fit(X)
    evr = pca_full.explained_variance_ratio_
    eig = pca_full.explained_variance_
    cum = np.cumsum(evr)
    pc_names = [f"PC{i + 1}" for i in range(len(norm_cols))]

    var_tbl = pd.DataFrame({
        "component": pc_names,
        "eigenvalue": eig,
        "explained_var_pct": evr * 100,
        "cumulative_pct": cum * 100,
    })
    rep.p()
    rep.p("Explained variance (all 15 components):")
    rep.table(var_tbl)

    n_keep = int(np.argmax(cum >= PCA_CUMVAR_TARGET) + 1)
    n_kaiser = int((eig > 1.0).sum())
    kept_pct = cum[n_keep - 1] * 100
    rep.p()
    rep.p(f"Retention rule: smallest number of components with cumulative explained "
          f"variance >= {PCA_CUMVAR_TARGET * 100:.0f} % (target band 80-85 %).")
    rep.p(f"RETAINED: {n_keep} components (PC1-PC{n_keep}) explaining "
          f"{kept_pct:.2f} % of total variance "
          f"(PC1-PC{n_keep - 1}: {cum[n_keep - 2] * 100:.2f} %, below threshold).")
    rep.p(f"For reference, the Kaiser criterion (eigenvalue > 1) would retain "
          f"{n_kaiser} components ({cum[n_kaiser - 1] * 100:.2f} %).")
    rep.p("The 80 % rule is used because the aim is to feed clustering with a compact but")
    rep.p("nearly information-complete representation, not to interpret only the")
    rep.p("dominant axes.")

    # Loadings: correlation between original standardized variables and PCs.
    # components_ rows are unit eigenvectors; scaling by sqrt(eigenvalue) gives
    # variable-PC correlations, the conventional "loading" for interpretation.
    loadings = pca_full.components_.T * np.sqrt(eig)
    load_df = pd.DataFrame(loadings, index=norm_cols, columns=pc_names)
    load_df.index.name = "feature"

    rep.p()
    rep.p("Component loading matrix (variable-component correlations = eigenvector x "
          "sqrt(eigenvalue)); ALL 15 components:")
    lt = load_df.reset_index()
    lt["dimension"] = lt["feature"].map(dim_of)
    lt = lt[["feature", "dimension"] + pc_names]
    rep.table(lt, "{:.3f}")

    rep.p()
    rep.p(f"Heavy loadings (|loading| >= {HEAVY_LOADING}) on the retained components; "
          "sign shows direction relative to the 'more vulnerable' orientation:")
    pc_descr: dict[str, str] = {}
    for j in range(n_keep):
        pc = pc_names[j]
        col = load_df[pc]
        heavy = col[col.abs() >= HEAVY_LOADING].sort_values(key=np.abs, ascending=False)
        if heavy.empty:
            heavy = col.reindex(col.abs().sort_values(ascending=False).index[:2])
        items = [f"{SHORT_LABEL[raw_of[f]]} ({v:+.2f})" for f, v in heavy.items()]
        pc_descr[pc] = "; ".join(items)
        rep.p(f"  {pc} ({evr[j] * 100:5.2f} %): " + pc_descr[pc])

    # Suggested names for the leading PCs, derived from the dimension mix.
    rep.p()
    rep.p("Suggested component names (for the paper; derived from heavy loadings):")
    for j in range(min(n_keep, 6)):
        pc = pc_names[j]
        col = load_df[pc]
        heavy = col[col.abs() >= HEAVY_LOADING]
        if heavy.empty:
            heavy = col.reindex(col.abs().sort_values(ascending=False).index[:2])
        dims = pd.Series([dim_of[f] for f in heavy.index]).value_counts()
        dom = dims.index[0]
        lead = SHORT_LABEL[raw_of[heavy.abs().idxmax()]]
        rep.p(f"  {pc}: {dom} axis, led by {lead} "
              f"({len(heavy)} heavy loading(s); dimensions: "
              + ", ".join(f"{d}={n}" for d, n in dims.items()) + ")")

    pca = PCA(n_components=n_keep, svd_solver="full", random_state=RANDOM_STATE)
    scores = pca.fit_transform(X)
    kept_pc_names = pc_names[:n_keep]
    # Sanity: reduced fit must reproduce the leading components of the full fit.
    assert np.allclose(pca.explained_variance_ratio_, evr[:n_keep])

    # ------------------------------------------------------------------
    # 3. K-Means model selection
    # ------------------------------------------------------------------
    rep.h2("3. K-MEANS MODEL SELECTION (k = 2..8)")
    rep.p(f"Clustering space : {n_keep} retained PC scores ({kept_pct:.2f} % variance)")
    rep.p(f"Algorithm        : K-Means (k-means++ init, n_init={KMEANS_N_INIT}, "
          f"max_iter={KMEANS_MAX_ITER}, random_state={RANDOM_STATE})")
    rep.p("Metrics          : silhouette (higher better, range -1..1), Davies-Bouldin "
          "(lower better), Calinski-Harabasz (higher better)")
    rep.p(f"Silhouette computed on all {len(df)} households (no subsampling).")

    rows = []
    km_models: dict[int, KMeans] = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE)
        lab = km.fit_predict(scores)
        km_models[k] = km
        sizes = np.bincount(lab)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(scores, lab),
            "davies_bouldin": davies_bouldin_score(scores, lab),
            "calinski_harabasz": calinski_harabasz_score(scores, lab),
            "inertia": km.inertia_,
            "n_iter": km.n_iter_,
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
        })
    ktab = pd.DataFrame(rows)
    ktab["sil_rank"] = ktab["silhouette"].rank(ascending=False).astype(int)
    ktab["db_rank"] = ktab["davies_bouldin"].rank(ascending=True).astype(int)
    ktab["ch_rank"] = ktab["calinski_harabasz"].rank(ascending=False).astype(int)
    ktab["mean_rank"] = ktab[["sil_rank", "db_rank", "ch_rank"]].mean(axis=1)

    rep.p()
    rep.p("Full comparison table (every k tested):")
    rep.table(ktab, "{:.4f}")

    best_sil = int(ktab.loc[ktab["silhouette"].idxmax(), "k"])
    best_db = int(ktab.loc[ktab["davies_bouldin"].idxmin(), "k"])
    best_ch = int(ktab.loc[ktab["calinski_harabasz"].idxmax(), "k"])
    votes = pd.Series([best_sil, best_db, best_ch]).value_counts()
    if votes.iloc[0] >= 2:
        best_k = int(votes.index[0])
        rule = (f"majority vote: {votes.iloc[0]} of 3 metrics prefer k={best_k} "
                f"(silhouette -> k={best_sil}, Davies-Bouldin -> k={best_db}, "
                f"Calinski-Harabasz -> k={best_ch})")
    else:
        best_k = int(ktab.loc[ktab["mean_rank"].idxmin(), "k"])
        rule = (f"no majority (silhouette -> k={best_sil}, Davies-Bouldin -> k={best_db}, "
                f"Calinski-Harabasz -> k={best_ch}); tie broken by lowest mean rank "
                f"across the three metrics -> k={best_k}")

    brow = ktab.set_index("k").loc[best_k]
    # Deltas vs. the runner-up on each metric, so the paragraph has numbers.
    others = ktab[ktab["k"] != best_k]
    sil_runner = others.loc[others["silhouette"].idxmax()]
    db_runner = others.loc[others["davies_bouldin"].idxmin()]
    ch_runner = others.loc[others["calinski_harabasz"].idxmax()]

    rep.p()
    rep.p(f"CHOSEN k = {best_k}  ({rule}).")
    rep.p()
    rep.p("Selection reasoning:")
    para = (
        f"Across k = 2..8, the silhouette index peaks at k={best_sil} "
        f"({ktab.set_index('k').loc[best_sil, 'silhouette']:.4f}), Davies-Bouldin is "
        f"lowest at k={best_db} ({ktab.set_index('k').loc[best_db, 'davies_bouldin']:.4f}) "
        f"and Calinski-Harabasz is highest at k={best_ch} "
        f"({ktab.set_index('k').loc[best_ch, 'calinski_harabasz']:.1f}). "
        f"k={best_k} is selected by {rule.split(':')[0]}. At k={best_k} the metrics are "
        f"silhouette {brow['silhouette']:.4f} (runner-up k={int(sil_runner['k'])}: "
        f"{sil_runner['silhouette']:.4f}), Davies-Bouldin {brow['davies_bouldin']:.4f} "
        f"(runner-up k={int(db_runner['k'])}: {db_runner['davies_bouldin']:.4f}) and "
        f"Calinski-Harabasz {brow['calinski_harabasz']:.1f} "
        f"(runner-up k={int(ch_runner['k'])}: {ch_runner['calinski_harabasz']:.1f}). "
        f"The smallest cluster at k={best_k} holds {int(brow['smallest_cluster'])} households "
        f"({brow['smallest_cluster'] / len(df) * 100:.1f} %), so no degenerate micro-cluster "
        f"is produced. Silhouette values are modest in absolute terms, which is expected for "
        f"survey data in a {n_keep}-dimensional PC space where household types form a "
        f"continuum rather than well-separated blobs; the relative ordering across k, not the "
        f"absolute level, drives the choice."
    )
    rep.p(para)

    # ------------------------------------------------------------------
    # 4. Final K-Means + Ward robustness check
    # ------------------------------------------------------------------
    rep.h2(f"4. FINAL K-MEANS (k={best_k}) AND WARD HIERARCHICAL ROBUSTNESS CHECK")
    km_final = km_models[best_k]
    km_lab_raw = km_final.labels_
    # Relabel clusters by ascending mean LVI so cluster 1 = least vulnerable.
    lvi = df[LVI_COL].to_numpy()
    order = np.argsort([lvi[km_lab_raw == c].mean() for c in range(best_k)])
    remap = {old: new + 1 for new, old in enumerate(order)}
    km_lab = np.array([remap[c] for c in km_lab_raw])
    rep.p(f"K-Means clusters relabelled 1..{best_k} in ascending order of mean LVI "
          "(labels are arbitrary in K-Means; this makes them readable).")
    rep.p(f"Final inertia: {km_final.inertia_:.2f}; iterations to converge: {km_final.n_iter_}")

    Z = linkage(scores, method="ward")
    ward_lab_raw = fcluster(Z, t=best_k, criterion="maxclust")
    ari = adjusted_rand_score(km_lab, ward_lab_raw)
    # Ward on identical feature space; also report agreement after best label matching.
    ct = pd.crosstab(pd.Series(km_lab, name="kmeans"),
                     pd.Series(ward_lab_raw, name="ward"))
    # Greedy one-to-one matching of Ward clusters to K-Means clusters by overlap.
    ct_work = ct.copy()
    matched = 0
    for _ in range(min(ct.shape)):
        i, j = np.unravel_index(np.argmax(ct_work.to_numpy()), ct_work.shape)
        matched += int(ct_work.iat[i, j])
        ct_work = ct_work.drop(index=ct_work.index[i]).drop(columns=ct_work.columns[j])
    agree_pct = matched / len(df) * 100

    rep.p()
    rep.p("Ward hierarchical clustering (Euclidean distance on the same retained PC scores,")
    rep.p(f"scipy linkage method='ward', cut at k={best_k} with criterion='maxclust').")
    rep.p(f"Adjusted Rand Index (K-Means vs Ward) : {ari:.4f}")
    rep.p(f"Households in matched clusters        : {matched} / {len(df)} ({agree_pct:.1f} %) "
          "after greedy one-to-one label matching")
    rep.p("ARI = 1 means identical partitions, 0 means chance-level agreement.")
    rep.p()
    rep.p("Cross-tabulation (rows: K-Means cluster, cols: Ward cluster):")
    rep.p(ct.to_string())
    ward_sizes = np.bincount(ward_lab_raw)[1:]
    rep.p()
    rep.p("Ward cluster sizes: " + ", ".join(f"W{i + 1}={s}" for i, s in enumerate(ward_sizes)))
    ward_sil = silhouette_score(scores, ward_lab_raw)
    rep.p(f"Ward silhouette at k={best_k}: {ward_sil:.4f} (K-Means: {brow['silhouette']:.4f})")

    # ------------------------------------------------------------------
    # 5. Cluster profiling + tests
    # ------------------------------------------------------------------
    rep.h2("5. CLUSTER PROFILES")
    df_out = df.copy()
    df_out["cluster_kmeans"] = km_lab
    df_out["cluster_ward"] = ward_lab_raw
    for j, pc in enumerate(kept_pc_names):
        df_out[pc] = scores[:, j]

    cl_ids = list(range(1, best_k + 1))
    prof_rows = []

    # (a) size, LVI, dimension scores
    size_tbl = (df_out.groupby("cluster_kmeans")
                .agg(n=("cluster_kmeans", "size"))
                .assign(pct=lambda t: t["n"] / len(df_out) * 100))
    means_lvi = df_out.groupby("cluster_kmeans")[[LVI_COL] + DIM_SCORE_COLS].mean()
    head_tbl = size_tbl.join(means_lvi)
    head_tbl.loc["ALL", "n"] = len(df_out)
    head_tbl.loc["ALL", "pct"] = 100.0
    for c in [LVI_COL] + DIM_SCORE_COLS:
        head_tbl.loc["ALL", c] = df_out[c].mean()
    head_tbl["n"] = head_tbl["n"].astype(int)
    rep.p("Cluster sizes, mean LVI and mean dimension scores:")
    rep.table(head_tbl.reset_index().rename(columns={"index": "cluster",
                                                     "cluster_kmeans": "cluster"}))

    # Tests on composite scores (context only; not used to define clusters)
    rep.p()
    rep.p("One-way tests across clusters on LVI and dimension scores (context only):")
    for c in [LVI_COL] + DIM_SCORE_COLS:
        groups = [df_out.loc[df_out["cluster_kmeans"] == g, c].to_numpy() for g in cl_ids]
        F, pF = stats.f_oneway(*groups)
        H, pH = stats.kruskal(*groups)
        rep.p(f"  {c:42s} ANOVA F={F:10.2f} p={fmt_p(pF):>9s} | KW H={H:10.2f} p={fmt_p(pH)}")
        prof_rows.append(_profile_row(c, "composite", "composite", df_out, cl_ids,
                                      F, pF, H, pH, None, None))

    # (b) raw indicators: means, normality check, ANOVA + Kruskal-Wallis
    rep.p()
    rep.p("Mean of each RAW indicator by cluster, with one-way ANOVA and Kruskal-Wallis.")
    rep.p("Normality is assessed per indicator with D'Agostino-Pearson K^2 on the pooled")
    rep.p("residuals (value minus own-cluster mean). Because BIHS indicators are counts,")
    rep.p("0/1 flags and right-skewed monetary/land amounts, normality is rejected for")
    rep.p("essentially all of them at n = 5605, so Kruskal-Wallis is designated the PRIMARY")
    rep.p("test and ANOVA is reported alongside for readers who expect F-statistics.")
    rep.p("Effect sizes: eta^2 = SS_between / SS_total (ANOVA); epsilon^2 = "
          "(H - k + 1) / (n - k) (Kruskal-Wallis).")
    rep.p(f"Bonferroni-adjusted alpha for {len(raw_cols)} indicator tests: "
          f"{ALPHA / len(raw_cols):.5f}.")
    rep.p()

    raw_mean_tbl = df_out.groupby("cluster_kmeans")[raw_cols].mean().T
    raw_mean_tbl.columns = [f"C{c}_mean" for c in raw_mean_tbl.columns]
    raw_mean_tbl["ALL_mean"] = df_out[raw_cols].mean()
    raw_mean_tbl.index.name = "raw_indicator"

    test_rows = []
    n = len(df_out)
    for rc in raw_cols:
        x = df_out[rc].to_numpy(dtype=float)
        groups = [x[km_lab == g] for g in cl_ids]
        F, pF = stats.f_oneway(*groups)
        H, pH = stats.kruskal(*groups)
        resid = np.concatenate([g - g.mean() for g in groups])
        K2, pK2 = stats.normaltest(resid)
        grand = x.mean()
        ss_tot = ((x - grand) ** 2).sum()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        eta2 = ss_b / ss_tot if ss_tot > 0 else np.nan
        eps2 = (H - best_k + 1) / (n - best_k)
        test_rows.append({
            "raw_indicator": rc,
            "normality_K2": K2, "normality_p": pK2,
            "normal": pK2 >= ALPHA,
            "anova_F": F, "anova_p": pF, "eta2": eta2,
            "kruskal_H": H, "kruskal_p": pH, "epsilon2": eps2,
        })
        prof_rows.append(_profile_row(rc, raw_of_inv(norm_cols, raw_of)[rc],
                                      dim_of[raw_of_inv(norm_cols, raw_of)[rc]],
                                      df_out, cl_ids, F, pF, H, pH, eta2, eps2,
                                      K2=K2, pK2=pK2))
    test_tbl = pd.DataFrame(test_rows).set_index("raw_indicator")
    n_nonnormal = int((~test_tbl["normal"]).sum())
    primary = "kruskal" if n_nonnormal >= len(raw_cols) / 2 else "anova"
    rep.p(f"Normality rejected (p < {ALPHA}) for {n_nonnormal} of {len(raw_cols)} indicators "
          f"-> PRIMARY test: {'Kruskal-Wallis' if primary == 'kruskal' else 'ANOVA'}.")
    rep.p()

    full_tbl = raw_mean_tbl.join(test_tbl)
    full_tbl.insert(0, "dimension", [dim_of[raw_of_inv(norm_cols, raw_of)[r]]
                                     for r in full_tbl.index])
    show = full_tbl.reset_index()
    show["anova_p"] = show["anova_p"].map(fmt_p)
    show["kruskal_p"] = show["kruskal_p"].map(fmt_p)
    show["normality_p"] = show["normality_p"].map(fmt_p)
    show = show.drop(columns=["normal"])
    rep.table(show, "{:.3f}")

    n_sig_kw = int((test_tbl["kruskal_p"] < ALPHA / len(raw_cols)).sum())
    n_sig_an = int((test_tbl["anova_p"] < ALPHA / len(raw_cols)).sum())
    rep.p()
    rep.p(f"Indicators differing significantly across clusters at Bonferroni alpha "
          f"{ALPHA / len(raw_cols):.5f}: Kruskal-Wallis {n_sig_kw}/{len(raw_cols)}, "
          f"ANOVA {n_sig_an}/{len(raw_cols)}.")
    ns = test_tbl[test_tbl["kruskal_p"] >= ALPHA / len(raw_cols)].index.tolist()
    if ns:
        rep.p("Not significant (Kruskal-Wallis, Bonferroni): " + ", ".join(ns))
    big = test_tbl["epsilon2"].sort_values(ascending=False)
    rep.p("Largest Kruskal-Wallis effect sizes (epsilon^2): "
          + ", ".join(f"{SHORT_LABEL[i]}={v:.3f}" for i, v in big.head(5).items()))
    rep.p("Smallest: "
          + ", ".join(f"{SHORT_LABEL[i]}={v:.3f}" for i, v in big.tail(3).items()))

    # ------------------------------------------------------------------
    # 6. Cluster narratives / names
    # ------------------------------------------------------------------
    rep.h2("6. CLUSTER DESCRIPTIONS AND NAMES")
    rep.p("Each cluster is characterised by how far its mean on each NORMALIZED indicator")
    rep.p("(higher = more vulnerable) sits from the sample mean, in pooled-SD units")
    rep.p("(standardized mean difference, SMD). |SMD| >= 0.30 is treated as a defining")
    rep.p("trait; the name is built from the largest-|SMD| traits plus the LVI rank.")
    rep.p()

    grand_norm = df_out[norm_cols].mean()
    sd_norm = df_out[norm_cols].std(ddof=1)
    lvi_all = df_out[LVI_COL].mean()
    names: dict[int, str] = {}
    for g in cl_ids:
        sub = df_out[df_out["cluster_kmeans"] == g]
        smd = ((sub[norm_cols].mean() - grand_norm) / sd_norm)
        smd = smd.sort_values(key=np.abs, ascending=False)
        defining = smd[smd.abs() >= 0.30]
        if defining.empty:
            defining = smd.head(3)
        name = _cluster_name(defining, raw_of, sub[LVI_COL].mean(), lvi_all,
                             sub[DIM_SCORE_COLS].mean(), df_out[DIM_SCORE_COLS].mean())
        names[g] = name
        n_g = len(sub)
        rep.p(f"CLUSTER {g}: \"{name}\"")
        rep.p(f"  n = {n_g} households ({n_g / n * 100:.1f} %); mean LVI = "
              f"{sub[LVI_COL].mean():.4f} (sample {lvi_all:.4f}); "
              f"exposure {sub['exposure_score'].mean():.3f}, sensitivity "
              f"{sub['sensitivity_score'].mean():.3f}, adaptive-capacity vulnerability "
              f"{sub['adaptive_capacity_vulnerability_score'].mean():.3f}.")
        traits = []
        for f, v in defining.items():
            rc = raw_of[f]
            traits.append(f"{SHORT_LABEL[rc]} raw mean {sub[rc].mean():.2f} vs "
                          f"{df_out[rc].mean():.2f} (SMD {v:+.2f} toward "
                          f"{'more' if v > 0 else 'less'} vulnerable)")
        rep.p("  Defining traits: " + "; ".join(traits) + ".")
        rep.p("  " + _narrative(g, name, sub, df_out, defining, raw_of))
        rep.p()

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    df_out["cluster_name"] = df_out["cluster_kmeans"].map(names)
    df_out.to_csv(OUT_CLUSTERED, index=False)

    prof_df = pd.DataFrame(prof_rows)
    # attach cluster names as a header block via extra columns for readability
    for g in cl_ids:
        prof_df[f"C{g}_name"] = names[g]
        prof_df[f"C{g}_n"] = int((km_lab == g).sum())
    prof_df["primary_test"] = "kruskal_wallis" if primary == "kruskal" else "anova"
    prof_df["bonferroni_alpha"] = ALPHA / len(raw_cols)
    prof_df.to_csv(OUT_PROFILE, index=False)

    rep.h2("7. OUTPUT FILES")
    rep.p(f"(a) {OUT_CLUSTERED.name}: {df_out.shape[0]} rows x {df_out.shape[1]} cols = "
          f"original {df.shape[1]} columns + cluster_kmeans, cluster_ward, cluster_name, "
          f"{', '.join(kept_pc_names)}")
    rep.p(f"(b) {OUT_PROFILE.name}: {prof_df.shape[0]} rows (4 composites + "
          f"{len(raw_cols)} raw indicators) x {prof_df.shape[1]} cols")
    rep.p(f"(c) {OUT_REPORT.name}: this report")
    rep.p()
    rep.p("Sanity checks:")
    rep.p(f"  cluster label counts sum to n: {int(np.bincount(km_lab)[1:].sum()) == n}")
    rep.p(f"  PC scores are mean-zero: max |mean| = "
          f"{np.abs(scores.mean(axis=0)).max():.2e}")
    rep.p(f"  PC score variances equal eigenvalues: "
          f"{np.allclose(scores.var(axis=0, ddof=1), eig[:n_keep])}")
    rep.p(f"  re-running KMeans with the same seed reproduces labels: "
          f"{_reproducible(scores, best_k, km_lab_raw)}")
    rep.write(OUT_REPORT)

    print(f"PCA: retained {n_keep} components ({kept_pct:.2f} %)")
    print(ktab[["k", "silhouette", "davies_bouldin", "calinski_harabasz"]].to_string(index=False))
    print(f"Chosen k = {best_k}; ARI (K-Means vs Ward) = {ari:.4f}")
    for g in cl_ids:
        print(f"  Cluster {g}: n={int((km_lab == g).sum())}  {names[g]}")
    print(f"Wrote {OUT_CLUSTERED.name}, {OUT_PROFILE.name}, {OUT_REPORT.name}")


# --------------------------------------------------------------------------
# Profiling helpers
# --------------------------------------------------------------------------
def raw_of_inv(norm_cols: list[str], raw_of: dict[str, str]) -> dict[str, str]:
    return {raw_of[c]: c for c in norm_cols}


def _profile_row(col, norm_col, dimension, df_out, cl_ids, F, pF, H, pH, eta2, eps2,
                 K2=np.nan, pK2=np.nan) -> dict:
    row = {"variable": col, "normalized_column": norm_col, "dimension": dimension,
           "ALL_mean": df_out[col].mean(), "ALL_sd": df_out[col].std(ddof=1)}
    for g in cl_ids:
        sub = df_out.loc[df_out["cluster_kmeans"] == g, col]
        row[f"C{g}_mean"] = sub.mean()
        row[f"C{g}_sd"] = sub.std(ddof=1)
        row[f"C{g}_median"] = sub.median()
    row.update({
        "normality_K2": K2, "normality_p": pK2,
        "anova_F": F, "anova_p": pF, "eta2": eta2,
        "kruskal_H": H, "kruskal_p": pH, "epsilon2": eps2,
    })
    return row


def _cluster_name(defining: pd.Series, raw_of: dict[str, str], lvi_c: float,
                  lvi_all: float, dims_c: pd.Series, dims_all: pd.Series) -> str:
    """Build a descriptive name from the top defining traits.

    Trait phrases describe the RAW direction (e.g. 'low-income' when the
    normalized income indicator is high = more vulnerable)."""
    phrase = {
        "flood_affected": ("flood-exposed", "flood-spared"),
        "flood_depth_ft_mean": ("deep-flooded", "shallow-flood"),
        "seasonal_wind_avg": ("high-wind", "low-wind"),
        "seasonal_evapotranspiration_avg": ("high-evapotranspiration", "low-evapotranspiration"),
        "agricultural_occupation_dependence": ("agriculture-dependent", "off-farm"),
        "dependency_ratio": ("high-dependency", "low-dependency"),
        "hh_size": ("large-household", "small-household"),
        "meals_per_person_week": ("food-short", "food-secure"),
        "income_per_capita_monthly": ("low-income", "higher-income"),
        "mean_edu_years_adults": ("low-education", "better-educated"),
        "adult_illiteracy_rate": ("high-illiteracy", "literate"),
        "livelihood_diversity": ("single-livelihood", "diversified-livelihood"),
        "any_nonfarm_agri_work": ("no-non-farm-work", "non-farm-employed"),
        "land_cultivable_decimal": ("land-poor", "land-rich"),
        "has_current_loan_binary": ("credit-excluded", "credit-holding"),
    }
    parts = []
    for f, v in defining.head(3).items():
        more, less = phrase[raw_of[f]]
        parts.append(more if v > 0 else less)
    rel = lvi_c - lvi_all
    if rel > 0.02:
        tail = "high-LVI"
    elif rel < -0.02:
        tail = "low-LVI"
    else:
        tail = "average-LVI"
    return ", ".join(parts) + f" ({tail})"


def _narrative(g: int, name: str, sub: pd.DataFrame, all_df: pd.DataFrame,
               defining: pd.Series, raw_of: dict[str, str]) -> str:
    n = len(sub)
    lvi_c, lvi_all = sub["LVI"].mean(), all_df["LVI"].mean()
    dim_lab = {"exposure_score": "exposure",
               "sensitivity_score": "sensitivity",
               "adaptive_capacity_vulnerability_score": "adaptive-capacity vulnerability"}
    dim_diff = {dim_lab[d]: sub[d].mean() - all_df[d].mean() for d in dim_lab}
    hi = max(dim_diff, key=dim_diff.get)
    lo = min(dim_diff, key=dim_diff.get)
    s = (f"Cluster {g} groups {n} households ({n / len(all_df) * 100:.1f} % of the sample) "
         f"and has a mean LVI of {lvi_c:.3f}, {abs(lvi_c - lvi_all):.3f} "
         f"{'above' if lvi_c > lvi_all else 'below'} the sample mean of {lvi_all:.3f}. "
         f"Relative to the whole sample its {hi} score is highest ({dim_diff[hi]:+.3f}) "
         f"and its {lo} score is lowest ({dim_diff[lo]:+.3f}). ")
    # Key raw facts
    facts = []
    fa = sub["flood_affected"].mean() * 100
    facts.append(f"{fa:.0f} % report flooding (sample {all_df['flood_affected'].mean() * 100:.0f} %)")
    facts.append(f"income per capita {sub['income_per_capita_monthly'].mean():.0f} Tk/month "
                 f"(sample {all_df['income_per_capita_monthly'].mean():.0f})")
    facts.append(f"cultivable land {sub['land_cultivable_decimal'].mean():.0f} decimals "
                 f"(sample {all_df['land_cultivable_decimal'].mean():.0f})")
    facts.append(f"adult illiteracy {sub['adult_illiteracy_rate'].mean() * 100:.0f} % "
                 f"(sample {all_df['adult_illiteracy_rate'].mean() * 100:.0f} %)")
    facts.append(f"{sub['any_nonfarm_agri_work'].mean() * 100:.0f} % have non-farm work "
                 f"(sample {all_df['any_nonfarm_agri_work'].mean() * 100:.0f} %)")
    facts.append(f"{sub['has_current_loan_binary'].mean() * 100:.0f} % hold a loan "
                 f"(sample {all_df['has_current_loan_binary'].mean() * 100:.0f} %)")
    s += "Key raw values: " + "; ".join(facts) + ". "
    s += (f"The descriptive name \"{name}\" summarises the "
          f"{len(defining)} indicator(s) with |SMD| >= 0.30.")
    return s


def _reproducible(scores: np.ndarray, k: int, labels: np.ndarray) -> bool:
    km = KMeans(n_clusters=k, init="k-means++", n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE)
    return bool(np.array_equal(km.fit_predict(scores), labels))


if __name__ == "__main__":
    main()
