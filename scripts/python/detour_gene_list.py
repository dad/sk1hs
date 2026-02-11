#!/usr/bin/env python3
"""
Rank genes by association with the detour coordinate.

Uses linear model per gene:
  expr_z ~ detour + temperature + time_min

Inputs (defaults should match your project):
  - yeast_ORF_TPM_matrix_labeled.csv   (gene rows, sample columns, plus annotations)
  - umap_detour_metrics.csv            (per-sample detour + temp_C + time_min)

Outputs (written to info/):
  - info/ranked_genes_detour.csv
  - info/detour_up_genes.txt
  - info/detour_down_genes.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

TPM = Path("yeast_ORF_TPM_matrix_labeled.csv")
DETOUR = Path("umap_detour_metrics.csv")

OUT_DIR = Path("info")
OUT_DIR.mkdir(exist_ok=True)

OUT_RANK = OUT_DIR / "ranked_genes_detour.csv"
OUT_UP = OUT_DIR / "detour_up_genes.txt"
OUT_DOWN = OUT_DIR / "detour_down_genes.txt"

TOP_N = 300  # change if you want


def canon(x):
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none"} else s


def main():
    if not TPM.exists():
        raise SystemExit(f"Missing TPM matrix: {TPM}")
    if not DETOUR.exists():
        raise SystemExit(f"Missing detour metrics: {DETOUR}")

    # ---- Load detour table ----
    det = pd.read_csv(DETOUR)
    need = ["temp_C", "time_min", "detour"]
    for c in need:
        if c not in det.columns:
            raise SystemExit(f"{DETOUR} missing required column: {c}")
    # sample ID is the index from earlier, but could also be a column
    if "sample" in det.columns:
        det = det.set_index("sample")
    else:
        # if saved without sample column, assume first unnamed column is index-like
        if det.columns[0].startswith("Unnamed"):
            det = det.set_index(det.columns[0])

    det.index = det.index.map(canon)

    # ---- Load TPM matrix ----
    df = pd.read_csv(TPM)

    # pick gene columns
    if "systematic_name" in df.columns:
        gene_col = "systematic_name"
    else:
        gene_col = df.columns[0]

    common_col = "common_name" if "common_name" in df.columns else None

    df[gene_col] = df[gene_col].astype(str).map(canon)
    if common_col:
        df[common_col] = df[common_col].astype(str).map(canon)

    # numeric sample columns = everything not in annotation cols
    anno_cols = {gene_col}
    if common_col:
        anno_cols.add(common_col)
    if "gene_category" in df.columns:
        anno_cols.add("gene_category")

    sample_cols = [c for c in df.columns if c not in anno_cols]

    X = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X.index = df[gene_col].values

    # ---- Align samples (intersection) ----
    samples = sorted(set(sample_cols) & set(det.index))
    if len(samples) < 10:
        missing = sorted(set(sample_cols) - set(det.index))[:10]
        raise SystemExit(
            f"Too few overlapping samples between TPM columns and detour table (n={len(samples)}).\n"
            f"Example TPM-only samples: {missing}\n"
            f"Tip: check whether detour index matches TPM column names exactly."
        )

    X = X[samples]
    det_use = det.loc[samples, ["detour", "temp_C", "time_min"]].copy()

    # ---- Transform expression ----
    X_log = np.log2(X + 1.0)

    # gene-wise zscore across samples
    mu = X_log.mean(axis=1)
    sd = X_log.std(axis=1).replace(0, np.nan)
    X_z = (X_log.sub(mu, axis=0)).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # standardize covariates (stability)
    for c in ["temp_C", "time_min"]:
        s = det_use[c].std()
        if np.isfinite(s) and s > 0:
            det_use[c] = (det_use[c] - det_use[c].mean()) / s

    # ---- Per-gene regression ----
    rows = []
    design = sm.add_constant(det_use[["detour", "temp_C", "time_min"]], has_constant="add")

    genes = X_z.index.tolist()
    for g in genes:
        y = X_z.loc[g].values.astype(float)
        fit = sm.OLS(y, design.values).fit()
        beta = float(fit.params[1])   # detour coefficient
        p = float(fit.pvalues[1])
        r2 = float(fit.rsquared)
        rows.append((g, beta, p, r2))

    res = pd.DataFrame(rows, columns=["systematic_name", "beta_detour", "p_detour", "r2"])
    res["abs_beta"] = res["beta_detour"].abs()
    res = res.sort_values(["abs_beta", "p_detour"], ascending=[False, True]).reset_index(drop=True)

    # add common name if present
    if common_col:
        sys_to_common = dict(zip(df[gene_col].values, df[common_col].values))
        res["common_name"] = res["systematic_name"].map(lambda x: sys_to_common.get(x, x))

    res.to_csv(OUT_RANK, index=False)
    print(f"Wrote: {OUT_RANK} (n={len(res)})")

    # top lists
    up = res.sort_values(["beta_detour", "p_detour"], ascending=[False, True]).head(TOP_N)
    down = res.sort_values(["beta_detour", "p_detour"], ascending=[True, True]).head(TOP_N)

    OUT_UP.write_text("\n".join(up["systematic_name"].astype(str)) + "\n")
    OUT_DOWN.write_text("\n".join(down["systematic_name"].astype(str)) + "\n")
    print(f"Wrote: {OUT_UP} (top {TOP_N} detour-up genes)")
    print(f"Wrote: {OUT_DOWN} (top {TOP_N} detour-down genes)")

    # quick peek
    print("\nTop detour-up genes:")
    print(up[["systematic_name", "common_name" if common_col else "systematic_name", "beta_detour", "p_detour"]].head(15))
    print("\nTop detour-down genes:")
    print(down[["systematic_name", "common_name" if common_col else "systematic_name", "beta_detour", "p_detour"]].head(15))


if __name__ == "__main__":
    main()
