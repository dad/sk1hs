#!/usr/bin/env python3
"""
Rank genes whose expression aligns with a "detour" coordinate, controlling for temperature and time.

Inputs
------
--expr: expression matrix CSV/TSV with first column = gene name, other columns = samples
--detour: CSV with at least: sample, detour, temperature, time_min (column names can vary; see below)

Outputs
-------
ranked_genes_detour.csv (sorted by |beta_detour|)
top_genes_pos.txt, top_genes_neg.txt (for enrichment)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # sniff delimiter
    with open(p, "r") as f:
        head = f.readline()
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(p, sep=sep)


def _standardize_sample_name(s: str) -> str:
    """Normalize sample IDs lightly so joins succeed."""
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace(".fastq", "").replace(".fq", "").replace(".bam", "")
    return s


def _find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", default="expression_matrix.csv",
                    help="Gene x sample matrix (CSV/TSV). First col gene, other cols samples.")
    ap.add_argument("--detour", default="umap_detour_metrics.csv",
                    help="CSV with detour + metadata per sample.")
    ap.add_argument("--out", default="ranked_genes_detour.csv")
    ap.add_argument("--top", type=int, default=200,
                    help="How many genes to write into top gene lists.")
    ap.add_argument("--no_zscore", action="store_true",
                    help="Skip gene-wise z-scoring across samples (not recommended).")
    args = ap.parse_args()

    print("Loading expression matrix...")
    expr = _read_table(args.expr)

    gene_col = expr.columns[0]
    expr = expr.rename(columns={gene_col: "gene"})
    expr["gene"] = expr["gene"].astype(str)

    # Keep only numeric sample columns
    sample_cols = [c for c in expr.columns if c != "gene"]
    # Coerce to numeric; non-numeric -> NaN
    expr[sample_cols] = expr[sample_cols].apply(pd.to_numeric, errors="coerce")

    # Drop genes with all-NaN
    expr = expr.dropna(subset=sample_cols, how="all").reset_index(drop=True)

    # log2(x+1)
    expr_log = np.log2(expr[sample_cols].fillna(0.0) + 1.0)
    expr_log.index = expr["gene"].values

    if not args.no_zscore:
        # z-score each gene across samples
        mu = expr_log.mean(axis=1)
        sd = expr_log.std(axis=1).replace(0, np.nan)
        expr_log = (expr_log.sub(mu, axis=0)).div(sd, axis=0)
        expr_log = expr_log.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # long form
    long = expr_log.reset_index().melt(id_vars="index", var_name="sample", value_name="expr_z")
    long = long.rename(columns={"index": "gene"})
    long["sample_norm"] = long["sample"].map(_standardize_sample_name)

    print("Loading detour metrics...")
    det = _read_table(args.detour)
    det.columns = [c.strip() for c in det.columns]
    # Try to locate key columns robustly
    c_sample = _find_col(det, ["sample", "sample_id", "Sample", "name"])
    c_detour = _find_col(det, ["detour", "Detour", "umap_detour", "detour_umap", "Detour (UMAP orth distance)"])
    c_temp = _find_col(det, ["temperature", "temp", "Temperature", "temp_c"])
    c_time = _find_col(det, ["time_min", "time", "Time (min)", "timepoint", "minutes"])

    if c_sample is None or c_detour is None:
        raise ValueError(
            f"Could not find required columns in {args.detour}. "
            f"Need a sample column and a detour column. Found: {list(det.columns)}"
        )

    det = det.rename(columns={c_sample: "sample", c_detour: "detour"})
    if c_temp is not None:
        det = det.rename(columns={c_temp: "temperature"})
    if c_time is not None:
        det = det.rename(columns={c_time: "time_min"})

    det["sample_norm"] = det["sample"].map(_standardize_sample_name)

    # Coerce covariates
    det["detour"] = pd.to_numeric(det["detour"], errors="coerce")
    if "temperature" in det.columns:
        det["temperature"] = pd.to_numeric(det["temperature"], errors="coerce")
    if "time_min" in det.columns:
        det["time_min"] = pd.to_numeric(det["time_min"], errors="coerce")

    # Merge
    m = long.merge(det[["sample_norm", "detour"] + [c for c in ["temperature", "time_min"] if c in det.columns]],
                   on="sample_norm", how="inner")

    if m.empty:
        raise ValueError("No overlapping samples between expression matrix and detour table after normalization.")

    print(f"Merged rows: {len(m):,} (genes x samples)")

    # Build design matrix
    X_cols = ["detour"]
    if "temperature" in m.columns:
        X_cols.append("temperature")
    if "time_min" in m.columns:
        X_cols.append("time_min")

    # To reduce collinearity scale covariates (does not change p-values materially for detour rank)
    for c in X_cols:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["expr_z", "detour"])
    for c in [c for c in X_cols if c != "detour"]:
        if c in m.columns:
            s = m[c].std()
            if np.isfinite(s) and s > 0:
                m[c] = (m[c] - m[c].mean()) / s

    # Per-gene regression
    out_rows = []
    genes = m["gene"].unique()

    print(f"Fitting per-gene models for {len(genes):,} genes...")
    for g in genes:
        sub = m[m["gene"] == g]
        if sub.shape[0] < 6:
            continue
        y = sub["expr_z"].astype(float).values
        X = sub[X_cols].astype(float)
        X = sm.add_constant(X, has_constant="add")
        try:
            fit = sm.OLS(y, X).fit()
            beta = fit.params.get("detour", np.nan)
            pval = fit.pvalues.get("detour", np.nan)
            r2 = fit.rsquared
            out_rows.append((g, beta, pval, r2, sub.shape[0]))
        except Exception:
            continue

    res = pd.DataFrame(out_rows, columns=["gene", "beta_detour", "p_detour", "r2", "n"])
    res = res.dropna(subset=["beta_detour", "p_detour"])
    res["abs_beta"] = res["beta_detour"].abs()
    res = res.sort_values(["abs_beta", "p_detour"], ascending=[False, True]).reset_index(drop=True)

    res.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  (n={len(res):,} genes)")

    # Top gene lists for enrichment
    top_pos = res.sort_values(["beta_detour", "p_detour"], ascending=[False, True]).head(args.top)["gene"].tolist()
    top_neg = res.sort_values(["beta_detour", "p_detour"], ascending=[True, True]).head(args.top)["gene"].tolist()

    Path("top_genes_pos.txt").write_text("\n".join(top_pos) + "\n")
    Path("top_genes_neg.txt").write_text("\n".join(top_neg) + "\n")
    print(f"Wrote: top_genes_pos.txt, top_genes_neg.txt (top {args.top})")


if __name__ == "__main__":
    main()
