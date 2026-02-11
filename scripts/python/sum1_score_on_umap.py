#!/usr/bin/env python3
"""
Compute Sum1 regulon score per sample and overlay on an existing UMAP.

Inputs:
  - yeast_ORF_TPM_matrix_labeled_with_categories.csv
      columns: systematic_name, common_name, gene_category, <samples...>
  - info/sum1_target_genes.txt
      one gene per line (systematic or common)
  - proj_umap_samples.csv
      produced earlier; must contain UMAP1, UMAP2, temp_C, time_min, rep

Outputs:
  - sum1_scores.csv
  - proj_UMAP_sum1_score.png
  - sum1_score_vs_time.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TPM_CSV  = Path("yeast_ORF_TPM_matrix_labeled_with_categories.csv")
SUM1_TXT = Path("info/sum1_target_genes.txt")
UMAP_CSV = Path("proj_umap_samples.csv")

OUT_SCORES = Path("sum1_scores.csv")
OUT_UMAP   = Path("proj_UMAP_sum1_score.png")
OUT_TIME   = Path("sum1_score_vs_time.png")


def canon(x: str) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s.upper()


def load_gene_list(path: Path) -> set[str]:
    genes = set()
    for line in path.read_text().splitlines():
        t = canon(line)
        if t:
            genes.add(t)
    return genes


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Z-score each row (gene) across columns (samples)."""
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def main():
    for p in [TPM_CSV, SUM1_TXT, UMAP_CSV]:
        if not p.exists():
            raise SystemExit(f"ERROR: missing required file: {p}")

    # Load TPM matrix
    df = pd.read_csv(TPM_CSV)
    needed = {"systematic_name", "common_name"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"ERROR: {TPM_CSV} must contain columns: {sorted(needed)}")

    meta_cols = [c for c in ["systematic_name", "common_name", "gene_category"] if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    # log2(TPM+1)
    X = df[sample_cols].to_numpy(dtype=float)
    X_log = np.log2(X + 1.0)

    # z-score per gene across samples
    X_z = zscore_rows(X_log)

    sys_names = df["systematic_name"].astype(str).map(canon)
    common_names = df["common_name"].astype(str).map(canon)

    # Load Sum1 gene list and match by either systematic or common
    sum1 = load_gene_list(SUM1_TXT)
    in_set = [(s in sum1) or (c in sum1) for s, c in zip(sys_names, common_names)]
    in_set = np.array(in_set, dtype=bool)

    if in_set.sum() == 0:
        raise SystemExit("ERROR: No genes matched Sum1 list. Check name format in info/sum1_target_genes.txt")

    # Sum1 score per sample: mean z across Sum1 genes
    sum1_scores = X_z[in_set, :].mean(axis=0)
    score_df = pd.DataFrame({"sample": sample_cols, "sum1_score": sum1_scores}).set_index("sample")

    # Load UMAP and join
    um = pd.read_csv(UMAP_CSV, index_col=0)
    for col in ["UMAP1", "UMAP2", "temp_C", "time_min", "rep"]:
        if col not in um.columns:
            raise SystemExit(f"ERROR: {UMAP_CSV} missing column {col}")

    out = um.join(score_df, how="left")
    if out["sum1_score"].isna().any():
        missing = out.index[out["sum1_score"].isna()].tolist()[:10]
        raise SystemExit(f"ERROR: Some UMAP samples not found in TPM columns (name mismatch). Examples: {missing}")

    out.to_csv(OUT_SCORES)
    print(f"Wrote {OUT_SCORES} (n={out.shape[0]})")

    # ---- Plot 1: UMAP colored by Sum1 score + arrows per replicate trajectory ----
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # markers by timepoint
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    out["time_min"] = pd.to_numeric(out["time_min"], errors="coerce")
    out["temp_C"] = pd.to_numeric(out["temp_C"], errors="coerce")

    unique_times = sorted(out["time_min"].dropna().unique())
    time_to_marker = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(unique_times)}

    # color by Sum1 score
    vals = out["sum1_score"].to_numpy()
    norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
    cmap = plt.cm.viridis

    # points
    for t in unique_times:
        sub = out[out["time_min"] == t]
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=70, alpha=0.9,
            marker=time_to_marker[t],
            c=cmap(norm(sub["sum1_score"])),
            edgecolors="none",
            label=f"{int(t)} min" if float(t).is_integer() else f"{t} min"
        )

    # arrows per (temp, rep)
    for (temp, rep), d in out.groupby(["temp_C", "rep"]):
        d = d.dropna(subset=["time_min"]).sort_values("time_min")
        if d.shape[0] < 2:
            continue
        xs = d["UMAP1"].to_numpy()
        ys = d["UMAP2"].to_numpy()
        for i in range(len(xs) - 1):
            ax.annotate(
                "", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.55, color="0.25")
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Sum1 regulon score (mean z)")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Timepoint", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_UMAP, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_UMAP}")

    # ---- Plot 2: Sum1 score vs time, separate lines per temperature (mean +/- not shown) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # aggregate across replicates
    agg = out.groupby(["temp_C", "time_min"], as_index=False)["sum1_score"].mean()
    for temp, d in agg.groupby("temp_C"):
        d = d.sort_values("time_min")
        ax.plot(d["time_min"], d["sum1_score"], marker="o", label=f"{temp:g}°C")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Sum1 regulon score (mean z)")
    ax.legend(title="Temperature", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_TIME, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_TIME}")

    print(f"Matched {in_set.sum()} Sum1 genes out of {df.shape[0]} total genes.")


if __name__ == "__main__":
    main()
