"""
BIHS Round 3 -- Stage 2b: visualization and interpretation of the PCA + K-Means clusters.

Inputs  (outputs/stage2_pca_clustering/):
    bihs_r3_clustered.csv, cluster_profile_summary.csv, pca_clustering_report.txt
Outputs (outputs/stage2b_visualization/):
    figures/fig1_pca_scatter.png ... fig6_crosstab_heatmap.png   (300 dpi)
    crosstab_cluster_vs_lvi_tertile.csv
    stage2b_summary_stats.csv
    results_interpretation.md
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "stage2_pca_clustering"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- style
# Categorical slots 1-2 of the validated default dataviz palette (blue, orange).
CLUSTER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
})


def short_name(c):
    """Short cluster label for legends/ticks."""
    return {1: "C1: Off-farm, flood-spared", 2: "C2: Agri-dependent, flood-exposed"}.get(c, f"Cluster {c}")


# ----------------------------------------------------------------------------- load
df = pd.read_csv(IN / "bihs_r3_clustered.csv")
prof = pd.read_csv(IN / "cluster_profile_summary.csv")
report = (IN / "pca_clustering_report.txt").read_text(encoding="utf-8")

df["cluster"] = df["cluster_kmeans"].astype(int)
clusters = sorted(df["cluster"].unique())
n_total = len(df)

# % variance explained for PC1/PC2 from the report
ev = {}
for m in re.finditer(r"^\s+(PC\d+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$", report, flags=re.M):
    ev[m.group(1)] = float(m.group(2))
pc1_pct, pc2_pct = ev["PC1"], ev["PC2"]

# LVI tertile classes (Stage 1 did not classify; tertiles defined here on the full sample)
q = df["LVI"].quantile([1 / 3, 2 / 3]).values
df["lvi_class"] = pd.cut(df["LVI"], bins=[-np.inf, q[0], q[1], np.inf], labels=["Low", "Medium", "High"])

# indicator metadata (order = Stage 1 dimension order)
IND = [  # (raw column, normalized column, dimension, pretty label)
    ("flood_affected", "flood_affected_norm", "Exposure", "Flood affected (0/1)"),
    ("flood_depth_ft_mean", "flood_depth_ft_mean_norm", "Exposure", "Flood depth (ft)"),
    ("seasonal_wind_avg", "seasonal_wind_avg_norm", "Exposure", "Seasonal wind"),
    ("seasonal_evapotranspiration_avg", "seasonal_evapotranspiration_avg_norm", "Exposure", "Seasonal evapotransp."),
    ("agricultural_occupation_dependence", "agricultural_occupation_dependence_norm", "Sensitivity", "Agri-occupation dependence"),
    ("dependency_ratio", "dependency_ratio_norm", "Sensitivity", "Dependency ratio"),
    ("hh_size", "hh_size_norm", "Sensitivity", "Household size"),
    ("meals_per_person_week", "meals_per_person_week_norm", "Sensitivity", "Meals per person/week"),
    ("income_per_capita_monthly", "income_per_capita_monthly_norm", "Adaptive capacity", "Income per capita"),
    ("mean_edu_years_adults", "mean_edu_years_adults_norm", "Adaptive capacity", "Adult education (yrs)"),
    ("adult_illiteracy_rate", "adult_illiteracy_rate_norm", "Adaptive capacity", "Adult illiteracy rate"),
    ("livelihood_diversity", "livelihood_diversity_norm", "Adaptive capacity", "Livelihood diversity"),
    ("any_nonfarm_agri_work", "any_nonfarm_agri_work_norm", "Adaptive capacity", "Non-farm work (0/1)"),
    ("land_cultivable_decimal", "land_cultivable_decimal_norm", "Adaptive capacity", "Cultivable land (dec)"),
    ("has_current_loan_binary", "has_current_loan_binary_norm", "Adaptive capacity", "Current loan (0/1)"),
]


# ----------------------------------------------------------------------------- summary stats per cluster
def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


summ = (df.groupby("cluster")["LVI"]
          .agg(n="size", mean="mean", sd="std", median="median",
               q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), ci95=ci95, min="min", max="max")
          .reset_index())
summ["pct"] = 100 * summ["n"] / n_total
summ["iqr"] = summ["q3"] - summ["q1"]
summ["cv_pct"] = 100 * summ["sd"] / summ["mean"]
summ["name"] = summ["cluster"].map(lambda c: df.loc[df.cluster == c, "cluster_name"].iloc[0])
summ = summ.sort_values("mean").reset_index(drop=True)
summ.to_csv(OUT / "stage2b_summary_stats.csv", index=False)
order = summ["cluster"].tolist()          # lowest -> highest mean LVI
mu = df["LVI"].mean()

# ============================================================================= FIG 1: PCA scatter
fig, ax = plt.subplots(figsize=(7.2, 5.6))
rng = np.random.default_rng(42)
idx = rng.permutation(n_total)            # shuffle so neither cluster is painted on top
sub = df.iloc[idx]
ax.scatter(sub["PC1"], sub["PC2"], c=sub["cluster"].map(CLUSTER_COLORS), s=7, alpha=0.35,
           linewidths=0, rasterized=True)
for c in clusters:
    cx, cy = df.loc[df.cluster == c, ["PC1", "PC2"]].mean()
    ax.scatter(cx, cy, s=260, marker="X", c=CLUSTER_COLORS[c], edgecolors=SURFACE, linewidths=2, zorder=5)
    ax.annotate(f"C{c} centroid\n({cx:.2f}, {cy:.2f})", (cx, cy), xytext=(12, 12), textcoords="offset points",
                fontsize=8.5, color=INK, ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, alpha=0.9), zorder=6)
ax.axhline(0, color=GRID, lw=0.8)
ax.axvline(0, color=GRID, lw=0.8)
ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%): land, livelihood diversity, non-farm work (+) vs "
              f"agri-dependence, flooding, household size (−)", fontsize=9)
ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%): evapotranspiration, dependency ratio, income (+) vs wind (−)", fontsize=9)
ax.set_title(f"Households in PC1–PC2 space by K-Means cluster (k = 2, N = {n_total:,})")
handles = [plt.Line2D([], [], marker="o", ls="", color=CLUSTER_COLORS[c], markersize=7,
                      label=f"{short_name(c)} (n = {int((df.cluster == c).sum()):,})") for c in clusters]
handles.append(plt.Line2D([], [], marker="X", ls="", color=INK2, markersize=10, label="Cluster centroid"))
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
fig.savefig(FIG / "fig1_pca_scatter.png")
plt.close(fig)

# ============================================================================= FIG 2: radar chart
# Axis values = cluster mean of each Stage-1 normalized indicator (0-1, higher = more vulnerable),
# so all 15 indicators share one scale regardless of raw units.
norm_cols = [n for _, n, _, _ in IND]
radar = df.groupby("cluster")[norm_cols].mean()
radar_all = df[norm_cols].mean()
labels = [lab for _, _, _, lab in IND]
dims = [d for _, _, d, _ in IND]
K = len(IND)
ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])

fig = plt.figure(figsize=(8.4, 8.0))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=INK2)
ax.yaxis.grid(color=GRID, lw=0.7)
ax.xaxis.grid(color=GRID, lw=0.7)
ax.spines["polar"].set_color(GRID)
ax.set_xticks(ang)
ax.set_xticklabels([])
# dimension shading wedges + dimension headers
dim_fill = {"Exposure": "#f3f1ec", "Sensitivity": "#ffffff", "Adaptive capacity": "#f3f1ec"}
start = 0
for d in ["Exposure", "Sensitivity", "Adaptive capacity"]:
    cnt = dims.count(d)
    theta0 = ang[start] - np.pi / K
    centre = theta0 + cnt * np.pi / K
    ax.bar(centre, 1, width=cnt * 2 * np.pi / K, bottom=0, color=dim_fill[d], zorder=0, edgecolor="none")
    ax.text(centre, 1.46, d.upper(), ha="center", va="center", fontsize=8.5, color=INK2, fontweight="bold")
    start += cnt
for a, lab in zip(ang, labels):
    ha = "left" if 0.05 < a < np.pi - 0.05 else ("right" if a > np.pi + 0.05 else "center")
    ax.text(a, 1.10, lab, ha=ha, va="center", fontsize=8.3, color=INK)
vals_all = np.concatenate([radar_all.values, radar_all.values[:1]])
ax.plot(ang_c, vals_all, color=INK2, lw=1.2, ls="--", label="Sample mean", zorder=3)
for c in clusters:
    v = np.concatenate([radar.loc[c].values, radar.loc[c].values[:1]])
    ax.plot(ang_c, v, color=CLUSTER_COLORS[c], lw=2.2, label=short_name(c), zorder=4)
    ax.fill(ang_c, v, color=CLUSTER_COLORS[c], alpha=0.10, zorder=2)
ax.set_title("Cluster profiles on the 15 LVI indicators\n(min-max normalized 0–1; farther from centre = more vulnerable)",
             pad=58, loc="center")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
fig.savefig(FIG / "fig2_radar_profiles.png")
plt.close(fig)

# ============================================================================= FIG 3: mean LVI by cluster (95% CI)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
x = np.arange(len(order))
for i, c in enumerate(order):
    r = summ.loc[summ.cluster == c].iloc[0]
    ax.bar(i, r["mean"], width=0.55, color=CLUSTER_COLORS[c], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(i, r["mean"], yerr=r["ci95"], color=INK, capsize=5, lw=1.4, zorder=5)
    ax.text(i, r["mean"] + r["ci95"] + 0.003, f"{r['mean']:.3f}\n(±{r['ci95']:.3f})",
            ha="center", va="bottom", fontsize=9, color=INK)
ax.axhline(mu, color=INK2, ls="--", lw=1)

ax.set_xticks(x)
ax.set_xticklabels([short_name(c).replace(": ", ":\n") for c in order])
ax.set_ylim(0.40, float(max(summ["mean"] + summ["ci95"])) + 0.03)
ax.set_ylabel("Mean Livelihood Vulnerability Index (LVI, 0–1)")
ax.set_xlabel("K-Means cluster (ordered from lowest to highest mean LVI)")
ax.set_title("Mean LVI by cluster with 95% confidence intervals")
ax.text(0.0, -0.27, f"Dashed line: sample mean LVI = {mu:.3f}. Note: y-axis starts at 0.40 so the confidence intervals are legible; the LVI range is 0–1.",
        transform=ax.transAxes, fontsize=8, color=INK2)
fig.savefig(FIG / "fig3_mean_lvi_by_cluster.png")
plt.close(fig)

# ============================================================================= FIG 4: cluster sizes (n and %)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
for ax, col, ylab, fmt in [(axes[0], "n", "Households (n)", "{:,.0f}"),
                           (axes[1], "pct", "Share of sample (%)", "{:.1f}%")]:
    for i, c in enumerate(order):
        v = summ.loc[summ.cluster == c, col].iloc[0]
        ax.bar(i, v, width=0.55, color=CLUSTER_COLORS[c], edgecolor=SURFACE, linewidth=2)
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"C{c}" for c in order])
    ax.set_ylabel(ylab)
    ax.set_xlabel("K-Means cluster")
    ax.set_ylim(0, summ[col].max() * 1.15)
axes[1].axhline(100 / len(order), color=INK2, ls="--", lw=1)
axes[1].text(len(order) - 0.55, 100 / len(order) + 1.2, f"Equal split ({100 / len(order):.0f}%)",
             ha="right", fontsize=8.5, color=INK2)
fig.suptitle(f"Cluster sizes: number and share of households (N = {n_total:,})",
             x=0.02, ha="left", fontsize=11.5, fontweight="bold")
fig.legend(handles=[Patch(color=CLUSTER_COLORS[c], label=short_name(c)) for c in order],
           loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(FIG / "fig4_cluster_sizes.png")
plt.close(fig)

# ============================================================================= FIG 5: LVI boxplots
fig, ax = plt.subplots(figsize=(6.8, 4.9))
data = [df.loc[df.cluster == c, "LVI"].values for c in order]
bp = ax.boxplot(data, positions=range(len(order)), widths=0.5, notch=True, showfliers=True, patch_artist=True,
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor=INK2, markeredgecolor="none", alpha=0.35),
                medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2, lw=1), capprops=dict(color=INK2, lw=1))
for patch, c in zip(bp["boxes"], order):
    patch.set_facecolor(CLUSTER_COLORS[c])
    patch.set_alpha(0.45)
    patch.set_edgecolor(CLUSTER_COLORS[c])
    patch.set_linewidth(1.4)
for i, c in enumerate(order):
    r = summ.loc[summ.cluster == c].iloc[0]
    ax.scatter(i, r["mean"], marker="D", s=34, color=SURFACE, edgecolor=INK, zorder=6, label="Mean" if i == 0 else None)
    ax.text(i + 0.30, r["q3"], f"IQR {r['iqr']:.3f}\nSD {r['sd']:.3f}", fontsize=8, color=INK2, va="center")
ax.axhline(mu, color=INK2, ls="--", lw=1)
ax.text(-0.48, mu + 0.004, f"Sample mean = {mu:.3f}", ha="left", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlim(-0.5, len(order) - 0.3)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([short_name(c).replace(": ", ":\n") for c in order])
ax.set_ylabel("Livelihood Vulnerability Index (LVI, 0–1)")
ax.set_xlabel("K-Means cluster (ordered by mean LVI)")
ax.set_title("Within-cluster distribution of LVI (median, IQR, 1.5×IQR whiskers, outliers)")
ax.legend(loc="upper left")
fig.savefig(FIG / "fig5_lvi_boxplots.png")
plt.close(fig)

# ============================================================================= FIG 6 + table: cluster x LVI tertile cross-tab
ct = pd.crosstab(df["cluster"], df["lvi_class"]).loc[order]
row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
col_pct = ct.div(ct.sum(axis=0), axis=1) * 100
chi2, p_chi, dof, _ = stats.chi2_contingency(ct.values)
cramers_v = float(np.sqrt(chi2 / (n_total * (min(ct.shape) - 1))))
rank_map = {c: i for i, c in enumerate(order)}
rho, p_rho = stats.spearmanr(df["cluster"].map(rank_map), df["lvi_class"].cat.codes)

tab = ct.copy()
tab.columns = [f"{c}_n" for c in tab.columns]
for c in ct.columns:
    tab[f"{c}_row_pct"] = row_pct[c].round(1)
for c in ct.columns:
    tab[f"{c}_col_pct"] = col_pct[c].round(1)
tab["total_n"] = ct.sum(axis=1)
tab.index.name = "cluster"
tab.to_csv(OUT / "crosstab_cluster_vs_lvi_tertile.csv")
with open(OUT / "crosstab_cluster_vs_lvi_tertile.csv", "a", encoding="utf-8") as f:
    f.write(f"\n# LVI tertile cut-points (sample terciles): Low <= {q[0]:.4f} < Medium <= {q[1]:.4f} < High\n")
    f.write(f"# chi-square = {chi2:.2f}, df = {dof}, p = {p_chi:.3g}; Cramer's V = {cramers_v:.3f}; "
            f"Spearman rho (cluster LVI-rank vs tertile) = {rho:.3f}, p = {p_rho:.3g}\n")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", ["#ffffff"] + SEQ_BLUE[:9])
im = ax.imshow(row_pct.values, cmap=cmap, vmin=0, vmax=60, aspect="auto")
ax.grid(False)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = row_pct.values[i, j]
        ax.text(j, i, f"{v:.1f}%\n(n = {ct.values[i, j]:,})", ha="center", va="center", fontsize=10,
                color=SURFACE if v > 38 else INK)
ax.set_xticks(range(ct.shape[1]))
ax.set_xticklabels([f"{c} LVI" for c in ct.columns])
ax.set_yticks(range(ct.shape[0]))
ax.set_yticklabels([f"{short_name(c)}\n(n = {int(ct.loc[c].sum()):,})" for c in order], fontsize=9)
ax.set_xlabel(f"LVI tertile class (sample terciles: Low ≤ {q[0]:.3f}, Medium ≤ {q[1]:.3f}, High > {q[1]:.3f})")
ax.set_ylabel("K-Means cluster")
ax.set_title("Cluster membership vs LVI tertile class (row %: share of each cluster)")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Row %", fontsize=9)
cb.outline.set_visible(False)
ax.text(0, -0.30, f"χ²({dof}) = {chi2:.1f}, p = {p_chi:.2g}; Cramér's V = {cramers_v:.2f}; "
        f"Spearman ρ = {rho:.2f}. A uniform 33.3% in every cell would mean no association.",
        transform=ax.transAxes, fontsize=8, color=INK2)
fig.savefig(FIG / "fig6_crosstab_heatmap.png")
plt.close(fig)

# ============================================================================= results_interpretation.md
raw_cols = [r for r, _, _, _ in IND]
raw_means = df.groupby("cluster")[raw_cols].mean()
all_means = df[raw_cols].mean()

# defining traits per cluster: SMD on normalized indicators (as in the Stage 2 report)
pooled_sd = df[norm_cols].std(ddof=1)
smd = (radar.sub(radar_all, axis=1)).div(pooled_sd, axis=1)

policy = {
    1: ("Policy relevance: because this group's vulnerability stems from a thin asset and livelihood base rather than "
        "from hazard exposure, it is better served by asset-building, skills and off-farm income-diversification "
        "programmes than by flood-protection investment."),
    2: ("Policy relevance: this group is the natural target for flood-risk reduction, crop insurance and "
        "climate-resilient agriculture, because its higher LVI is driven by exposure and sensitivity while its "
        "adaptive capacity is comparatively strong."),
}

pcs_retained = sum(list(ev.values())[:9])
lines = []
lines.append("# Results: household vulnerability clusters (BIHS Round 3)\n")
lines.append(f"*Draft results-section source generated by `visualize_clusters.py` on {pd.Timestamp.today():%Y-%m-%d}. "
             f"Figures in `figures/` (300 dpi). N = {n_total:,} households; K-Means, k = 2, on 9 retained PCs "
             f"({pcs_retained:.1f}% of variance).*\n")
lines.append("## Cluster profiles\n")
for c in order:
    r = summ.loc[summ.cluster == c].iloc[0]
    name = r["name"]
    s = smd.loc[c].sort_values(key=np.abs, ascending=False)
    traits = []
    for ncol, val in s.items():
        if abs(val) < 0.30:
            break
        raw = next(t for t in IND if t[1] == ncol)
        direction = "more" if val > 0 else "less"
        traits.append(f"{raw[3].lower()} {raw_means.loc[c, raw[0]]:.2f} vs sample {all_means[raw[0]]:.2f} "
                      f"(SMD {val:+.2f}, {direction} vulnerable)")
    ex, se, ac = (df.loc[df.cluster == c, k].mean()
                  for k in ["exposure_score", "sensitivity_score", "adaptive_capacity_vulnerability_score"])
    lvi_dist = row_pct.loc[c]
    para = (f"**Cluster {c} — “{name}”.** This cluster contains {int(r['n']):,} households "
            f"({r['pct']:.1f}% of the sample), with a mean LVI of {r['mean']:.3f} (95% CI ±{r['ci95']:.3f}; "
            f"median {r['median']:.3f}, SD {r['sd']:.3f}, IQR {r['iqr']:.3f}), "
            f"{'below' if r['mean'] < mu else 'above'} the sample mean of {mu:.3f} (Fig. 3). "
            f"Its dimension scores are exposure {ex:.3f}, sensitivity {se:.3f} and adaptive-capacity vulnerability {ac:.3f} "
            f"(sample: {df['exposure_score'].mean():.3f}, {df['sensitivity_score'].mean():.3f}, "
            f"{df['adaptive_capacity_vulnerability_score'].mean():.3f}). "
            f"Its defining traits (|SMD| ≥ 0.30 on the normalized indicators; Fig. 2) are: " + "; ".join(traits) + ". "
            f"By LVI tertile, {lvi_dist['Low']:.1f}% of its households are Low, {lvi_dist['Medium']:.1f}% Medium and "
            f"{lvi_dist['High']:.1f}% High (Fig. 6). " + policy.get(c, ""))
    lines.append(para + "\n")

lo, hi = order[0], order[-1]
lines.append("## Cluster-based versus index-based classification\n")
lines.append(
    f"The two vulnerability measures overlap only partially. Cluster membership and LVI tertile class are statistically "
    f"associated (χ²({dof}) = {chi2:.1f}, p = {p_chi:.2g}), but the association is weak: Cramér's V = {cramers_v:.2f} "
    f"and the Spearman correlation between cluster LVI-rank and tertile class is ρ = {rho:.2f}. "
    f"The lower-LVI cluster (C{lo}) holds {row_pct.loc[lo, 'Low']:.1f}% Low, {row_pct.loc[lo, 'Medium']:.1f}% Medium and "
    f"{row_pct.loc[lo, 'High']:.1f}% High households, whereas the higher-LVI cluster (C{hi}) holds "
    f"{row_pct.loc[hi, 'Low']:.1f}%, {row_pct.loc[hi, 'Medium']:.1f}% and {row_pct.loc[hi, 'High']:.1f}% respectively "
    f"(a uniform 33.3% would indicate no relationship). Put differently, {col_pct.loc[hi, 'High']:.1f}% of all High-LVI "
    f"households fall in C{hi} and {col_pct.loc[lo, 'Low']:.1f}% of all Low-LVI households fall in C{lo}. "
    f"The gap in mean LVI between clusters ({summ['mean'].iloc[-1] - summ['mean'].iloc[0]:.3f}) is small relative to the "
    f"within-cluster spread (SD {summ['sd'].iloc[0]:.3f} and {summ['sd'].iloc[-1]:.3f}; Fig. 5), and the boxplots show the "
    f"two LVI distributions overlapping across almost their entire range. "
    f"This divergence is informative rather than a failure of either measure. The LVI aggregates exposure, sensitivity and "
    f"adaptive capacity additively, so a household that is highly flood-exposed but asset-rich can receive the same score as "
    f"one that is flood-spared but land-poor and single-livelihood. The clustering separates precisely these two *types* of "
    f"vulnerability: C{hi} is a hazard-exposed, agriculture-dependent group whose vulnerability is driven by exposure and "
    f"sensitivity, while C{lo} is a flood-spared but capacity-constrained group whose vulnerability is driven by low adaptive "
    f"capacity. The ordering of the clusters by mean LVI agrees with the index, which validates the direction of the LVI, but "
    f"the composition of vulnerability, and therefore the appropriate policy response, differs between the two groups in a way "
    f"that a single composite score cannot reveal. We therefore recommend reporting both: the LVI for ranking and targeting "
    f"intensity, and the cluster typology for choosing the type of intervention.\n")

lines.append("## Methodological notes\n")
lines.append(f"- LVI tertile classes were not produced in Stage 1; they are defined here as sample terciles of the LVI "
             f"(Low ≤ {q[0]:.4f} < Medium ≤ {q[1]:.4f} < High), giving n = {int((df.lvi_class == 'Low').sum()):,} / "
             f"{int((df.lvi_class == 'Medium').sum()):,} / {int((df.lvi_class == 'High').sum()):,}.")
lines.append("- Radar axes (Fig. 2) use the Stage 1 min-max normalized indicators (0–1, oriented so higher = more vulnerable), "
             "so raw units (Tk, decimals, counts, 0/1) are directly comparable; the dashed line is the sample mean.")
lines.append("- Error bars in Fig. 3 are 95% t-based confidence intervals of the cluster mean; because n is large the CIs are narrow, "
             "so the boxplots in Fig. 5 should be used to judge the *practical* separation of the clusters.")
lines.append("- Cluster labels are numbered in ascending order of mean LVI (Stage 2 convention). Cluster names come from the Stage 2 report.")
lines.append("- Silhouette at k = 2 is 0.152 and the Ward robustness check gives ARI = 0.28 (Stage 2 report), so the clusters should be "
             "described as a two-type continuum rather than as sharply separated groups; Fig. 1 shows the overlap in PC space.")
lines.append("\n## Figure list\n")
lines += [
    "- **Fig. 1** `figures/fig1_pca_scatter.png` — Households in PC1–PC2 space coloured by cluster, with centroids.",
    "- **Fig. 2** `figures/fig2_radar_profiles.png` — Cluster mean profiles on the 15 normalized LVI indicators.",
    "- **Fig. 3** `figures/fig3_mean_lvi_by_cluster.png` — Mean LVI by cluster with 95% CI, ordered by vulnerability.",
    "- **Fig. 4** `figures/fig4_cluster_sizes.png` — Cluster sizes (n and % of sample).",
    "- **Fig. 5** `figures/fig5_lvi_boxplots.png` — Within-cluster LVI distributions.",
    "- **Fig. 6** `figures/fig6_crosstab_heatmap.png` — Cluster × LVI-tertile cross-tabulation (row %); "
    "table in `crosstab_cluster_vs_lvi_tertile.csv`.",
]
(OUT / "results_interpretation.md").write_text("\n".join(lines), encoding="utf-8")

print("PC1/PC2 %:", pc1_pct, pc2_pct)
print(summ[["cluster", "n", "pct", "mean", "sd", "ci95", "median", "iqr"]].to_string(index=False))
print("\nTertile cuts:", q)
print(ct)
print(row_pct.round(1))
print(f"chi2={chi2:.1f} p={p_chi:.3g} V={cramers_v:.3f} rho={rho:.3f}")
print("\nSMD by cluster (|SMD|>=0.3):")
for c in order:
    s = smd.loc[c]
    print(c, s[s.abs() >= 0.3].round(2).to_dict())
